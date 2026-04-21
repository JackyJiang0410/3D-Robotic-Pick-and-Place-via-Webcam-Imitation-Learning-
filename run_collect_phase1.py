from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from stage1_sensory_input import Stage1Config, Stage1Extractor
from stage2_mujoco.panda_env import PandaPickPlaceEnv
from stage2_mujoco.zarr_logger import ZarrTrajectoryLogger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1: webcam teleop + collect (obs, act) to Zarr.")
    p.add_argument("--viewer", action="store_true", help="Open MuJoCo passive viewer.")
    p.add_argument("--camera-id", type=int, default=0)
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument(
        "--delta-scale",
        type=float,
        default=0.005,
        help="Meters per step for full-scale x/y gesture (maps to dx/dy).",
    )
    p.add_argument("--dataset", type=str, default="data/phase1_demos.zarr")
    p.add_argument("--episode-name", type=str, default=None, help="Optional name like ep_000123.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--print-hz",
        type=float,
        default=4.0,
        help="Terminal status print rate (Hz).",
    )
    p.add_argument(
        "--preview",
        action="store_true",
        help="Show an OpenCV webcam preview window (may conflict with MuJoCo viewer on some macOS setups).",
    )
    p.add_argument(
        "--verbose-numbers",
        action="store_true",
        help="Also print numeric x,y and dx,dy (default is compact plain-language).",
    )
    p.add_argument(
        "--auto-time-scale",
        type=float,
        default=10.0,
        help="Slow down auto pick-place sequence by this multiplier (>=1 is slower).",
    )
    p.add_argument(
        "--auto-ease",
        type=str,
        default="smoothstep",
        choices=["linear", "smoothstep"],
        help="Interpolation profile for auto sequence motion.",
    )
    p.add_argument(
        "--auto-grasp-offset",
        type=float,
        default=0.10,
        help="Target z above object center during auto descend (meters). Larger means less downward motion.",
    )
    p.add_argument(
        "--auto-table-top-z",
        type=float,
        default=0.30,
        help="Estimated table top z in world frame.",
    )
    p.add_argument(
        "--auto-table-clearance-z",
        type=float,
        default=0.11,
        help="Minimum hand-body z above table top during descend to avoid collision.",
    )
    return p.parse_args()


def gesture_to_dxdyg(action_human: np.ndarray, delta_scale: float) -> np.ndarray:
    x, y, g = action_human.astype(np.float32)
    # Mirror X so hand-right moves the robot in the opposite world-X direction.
    dx = float(-delta_scale * x)
    dy = float(-delta_scale * y)
    return np.array([dx, dy, float(g)], dtype=np.float32)


def _format_cmd(dx: float, dy: float, g_cmd: float, scales: Tuple[float, float]) -> str:
    dx_s, dy_s = scales
    eps_x = max(0.25 * abs(dx_s), 1e-6)
    eps_y = max(0.25 * abs(dy_s), 1e-6)

    active = (abs(dx) > eps_x) or (abs(dy) > eps_y)
    if not active:
        motion = "hold"
    else:
        # Prefer the strongest axis so the message matches what you feel most.
        mag = [abs(dx), abs(dy)]
        idx = int(np.argmax(mag))
        order = [idx] + [i for i in range(2) if i != idx]
        ordered: List[str] = []
        for i in order:
            if i == 0 and abs(dx) > eps_x:
                ordered.append("world +X" if dx > 0 else "world -X")
            if i == 1 and abs(dy) > eps_y:
                ordered.append("world +Y" if dy > 0 else "world -Y")
        motion = " + ".join(ordered) if ordered else "hold"

    grip = "CLOSE" if g_cmd > 0.5 else "OPEN"
    return f"cmd={motion} | grip={grip}"


@dataclass
class AutoPickPlaceSequence:
    start_t: float
    pre_xyz: np.ndarray
    pre_hover_z: float
    home_xy: np.ndarray
    grasp_z: float
    lift_z: float
    place_z: float
    place_xy: np.ndarray
    phase_name: str = "descend"
    phase_idx: int = 0
    phase_start_t: float = 0.0
    success_lifted: bool = False
    time_scale: float = 2.0
    ease_mode: str = "smoothstep"

    # (name, duration_sec)
    PHASES = [
        ("descend", 0.7),
        ("close", 0.35),
        ("lift", 0.8),
        ("return_pre", 0.9),
        ("move_home", 1.0),
        ("lower_home", 0.9),
        ("open", 0.35),
        ("raise_home", 0.8),
        ("back_to_pre", 1.0),
        ("settle", 0.35),
    ]

    def __post_init__(self) -> None:
        self.phase_start_t = self.start_t
        self.phase_name = self.PHASES[0][0]

    def _interp(self, a: float, b: float, alpha: float) -> float:
        return (1.0 - alpha) * a + alpha * b

    def _interp_vec(self, a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
        return (1.0 - alpha) * a + alpha * b

    def _ease(self, alpha: float) -> float:
        a = float(np.clip(alpha, 0.0, 1.0))
        if self.ease_mode == "linear":
            return a
        # smoothstep: 3a^2 - 2a^3
        return a * a * (3.0 - 2.0 * a)

    def step(self, env: PandaPickPlaceEnv, now: float, obs) -> tuple[np.ndarray, bool]:
        name, base_dur = self.PHASES[self.phase_idx]
        dur = max(base_dur * max(self.time_scale, 0.1), 1e-3)
        elapsed = now - self.phase_start_t
        alpha = float(np.clip(elapsed / max(dur, 1e-6), 0.0, 1.0))
        alpha_eased = self._ease(alpha)

        pre_hover_xyz = np.array([self.pre_xyz[0], self.pre_xyz[1], self.pre_hover_z], dtype=np.float32)
        pre_grasp_xyz = np.array([self.pre_xyz[0], self.pre_xyz[1], self.grasp_z], dtype=np.float32)
        pre_lift_xyz = np.array([self.pre_xyz[0], self.pre_xyz[1], self.lift_z], dtype=np.float32)
        home_lift_xyz = np.array([self.home_xy[0], self.home_xy[1], self.lift_z], dtype=np.float32)
        home_place_xyz = np.array([self.home_xy[0], self.home_xy[1], self.place_z], dtype=np.float32)

        if name == "descend":
            descend_xyz = np.array(
                [self.pre_xyz[0], self.pre_xyz[1], self._interp(self.pre_hover_z, self.grasp_z, alpha_eased)],
                dtype=np.float32,
            )
            env.set_ee_target(descend_xyz)
            act = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        elif name == "close":
            env.set_ee_target(pre_grasp_xyz)
            act = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        elif name == "lift":
            lift_xyz = np.array(
                [self.pre_xyz[0], self.pre_xyz[1], self._interp(self.grasp_z, self.lift_z, alpha_eased)],
                dtype=np.float32,
            )
            env.set_ee_target(lift_xyz)
            act = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            if float(obs.obj_pos[2]) > (self.grasp_z + 0.03):
                self.success_lifted = True
        elif name == "return_pre":
            env.set_ee_target(self._interp_vec(pre_lift_xyz, pre_hover_xyz, alpha_eased))
            act = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        elif name == "move_home":
            env.set_ee_target(self._interp_vec(pre_hover_xyz, home_lift_xyz, alpha_eased))
            act = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        elif name == "lower_home":
            env.set_ee_target(self._interp_vec(home_lift_xyz, home_place_xyz, alpha_eased))
            act = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        elif name == "open":
            env.set_ee_target(home_place_xyz)
            act = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        elif name == "raise_home":
            env.set_ee_target(self._interp_vec(home_place_xyz, home_lift_xyz, alpha_eased))
            act = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        elif name == "back_to_pre":
            env.set_ee_target(self._interp_vec(home_lift_xyz, pre_hover_xyz, alpha_eased))
            act = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        else:  # settle
            env.set_ee_target(pre_hover_xyz)
            act = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        if elapsed >= dur:
            self.phase_idx += 1
            if self.phase_idx >= len(self.PHASES):
                return act, True
            self.phase_name = self.PHASES[self.phase_idx][0]
            self.phase_start_t = now

        return act, False


def run_loop(env: PandaPickPlaceEnv, viewer: Optional[object], args: argparse.Namespace) -> None:
    extractor = Stage1Extractor(Stage1Config(camera_id=args.camera_id))
    cap = extractor.open_camera()

    logger = ZarrTrajectoryLogger(args.dataset)
    obs0 = env.reset(seed=args.seed)
    obs = obs0
    ep = logger.start_episode(obs_dim=obs0.as_vector().shape[0], act_dim=env.action_dim, name=args.episode_name)
    start = time.time()
    end = start + args.seconds
    last_print = 0.0
    min_print_dt = 1.0 / max(args.print_hz, 1e-6)
    hand_seen_once = False
    last_g_state = 0.0
    auto_seq: Optional[AutoPickPlaceSequence] = None

    print(f"Writing dataset to: {Path(args.dataset).expanduser().resolve()}")
    print(f"Episode: {ep.group.name}  (press Ctrl+C to stop early)")
    print("Gesture -> Panda action mapping:")
    print("  Move hand RIGHT  : +x -> -dx (ee moves -X)")
    print("  Move hand LEFT   : -x -> +dx (ee moves +X)")
    print("  Move hand UP     : -y -> +dy  (dy uses inverted y)")
    print("  Move hand DOWN   : +y -> -dy")
    print("  Pinch fingers    : g=1 -> CLOSE gripper")
    print("  Open fingers     : g=0 -> OPEN gripper")
    print("Tips: keep one hand centered in camera, move slowly, adjust --delta-scale.")
    print(
        "Auto sequence: descend -> close -> lift -> return_pre -> move_home -> "
        "lower_home -> open -> raise_home -> back_to_pre."
    )
    print(
        f"Auto timing: --auto-time-scale={args.auto_time_scale}  easing={args.auto_ease}  "
        f"--auto-grasp-offset={args.auto_grasp_offset}  "
        f"table_z={args.auto_table_top_z}+{args.auto_table_clearance_z}"
    )
    if args.preview and viewer is not None:
        print("NOTE: --preview + MuJoCo viewer can crash OpenCV on some macOS setups. If it crashes, omit --preview.")

    try:
        while time.time() < end:
            now = time.time()
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Failed to read frame from webcam.")

            vis, signal = extractor.process_frame(frame)

            if signal is not None:
                g_rise = (last_g_state <= 0.5) and (signal.g > 0.5)
                last_g_state = signal.g
            else:
                g_rise = False
                last_g_state = 0.0

            if auto_seq is None and g_rise:
                ws_min, ws_max = env.workspace_bounds
                ee_t = env.ee_target
                pre_xyz = ee_t.copy()
                home_xy = obs0.ee_pos[:2].astype(np.float32).copy()
                hover_z = float(ee_t[2])
                # Keep safe margins from table and workspace bounds.
                table_safe_z = float(args.auto_table_top_z + args.auto_table_clearance_z)
                grasp_z = float(
                    np.clip(
                        max(obs.obj_pos[2] + float(args.auto_grasp_offset), table_safe_z),
                        ws_min[2] + 0.03,
                        ws_max[2] - 0.05,
                    )
                )
                lift_z = float(np.clip(max(hover_z, grasp_z) + 0.12, ws_min[2] + 0.05, ws_max[2] - 0.02))
                place_z = grasp_z
                auto_seq = AutoPickPlaceSequence(
                    start_t=now,
                    pre_xyz=pre_xyz,
                    pre_hover_z=hover_z,
                    home_xy=home_xy,
                    grasp_z=grasp_z,
                    lift_z=lift_z,
                    place_z=place_z,
                    place_xy=home_xy,
                    time_scale=float(args.auto_time_scale),
                    ease_mode=str(args.auto_ease),
                )
                print(
                    f"[AUTO] Triggered: pre=({pre_xyz[0]:.3f},{pre_xyz[1]:.3f},{hover_z:.3f}) "
                    f"home_xy=({home_xy[0]:.3f},{home_xy[1]:.3f}) "
                    f"grasp_z={grasp_z:.3f} lift_z={lift_z:.3f}"
                )

            if auto_seq is not None:
                act, done = auto_seq.step(env, now, obs)
                if done:
                    seq_success = auto_seq.success_lifted
                    print(f"[AUTO] Completed sequence. success_lifted={seq_success}")
                    auto_seq = None
            elif signal is not None:
                act = gesture_to_dxdyg(signal.as_vector(), args.delta_scale)
            else:
                act = np.array([0.0, 0.0, 0.0], dtype=np.float32)

            obs = env.step(act)
            now = time.time()
            logger.append(ep, obs.as_vector(), act, now)

            if viewer is not None:
                viewer.sync()

            if args.preview:
                cmd_txt = _format_cmd(float(act[0]), float(act[1]), float(act[2]), (args.delta_scale, args.delta_scale))
                if signal is None:
                    hud = "No hand"
                else:
                    pinch = "PINCH" if signal.g > 0.5 else "OPEN"
                    if auto_seq is not None:
                        hud = f"{cmd_txt} | hand {pinch} | AUTO:{auto_seq.phase_name}"
                    else:
                        hud = f"{cmd_txt} | hand {pinch}"
                cv2.putText(vis, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                try:
                    cv2.imshow("Webcam (teleop)", vis)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        print("Quit requested from webcam preview (q).")
                        break
                except cv2.error as exc:
                    print("OpenCV preview failed; continuing without preview.")
                    print(f"Reason: {exc}")
                    args.preview = False

            if now - last_print >= min_print_dt:
                if signal is not None:
                    hand_seen_once = True
                    pinch_label = "CLOSE" if signal.g > 0.5 else "OPEN"
                    obj_to_goal = float(np.linalg.norm(obs.obj_pos - obs.target_pos))
                    cmd_txt = _format_cmd(float(act[0]), float(act[1]), float(act[2]), (args.delta_scale, args.delta_scale))
                    base = (
                        f"{cmd_txt} | pinch={pinch_label} | sim_grip_open={float(obs.gripper_open[0]):.3f} | "
                        f"|obj-goal|={obj_to_goal:.2f}m"
                    )
                    if args.verbose_numbers:
                        base += (
                            f" | hand_xyg=({signal.x:+.2f},{signal.y:+.2f},{signal.g:.0f})"
                            f" | dxy=({act[0]:+.4f},{act[1]:+.4f})"
                        )
                    if auto_seq is not None:
                        base += f" | auto_phase={auto_seq.phase_name}"
                    print(base)
                else:
                    if not hand_seen_once:
                        print("No hand detected: put ONE hand fully in frame and face palm to camera.")
                    else:
                        print("Hand lost: hold hand still in frame for ~1 second to reacquire tracking.")
                last_print = now

            time.sleep(env.model.opt.timestep)
    finally:
        cap.release()
        extractor.close()
        if args.preview:
            cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    env = PandaPickPlaceEnv()

    if not args.viewer:
        run_loop(env, None, args)
        return

    from mujoco import viewer as mj_viewer

    print("Launching viewer. On macOS, run this script with `mjpython`.")
    with mj_viewer.launch_passive(env.model, env.data) as viewer:
        run_loop(env, viewer, args)


if __name__ == "__main__":
    main()

