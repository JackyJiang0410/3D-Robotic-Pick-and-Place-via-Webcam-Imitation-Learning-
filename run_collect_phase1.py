from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
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
        default=0.0,
        help=(
            "Target z relative to object CENTER during auto descend (meters). "
            "0.0 means TCP aims at the cube center (best grip). Positive lifts the grip toward the cube top."
        ),
    )
    p.add_argument(
        "--snap-to-cube",
        dest="snap_to_cube",
        action="store_true",
        default=True,
        help=(
            "On pinch trigger, snap the descend XY target to the cube's current XY position "
            "so the grasp is centered (default ON for early data collection)."
        ),
    )
    p.add_argument(
        "--no-snap-to-cube",
        dest="snap_to_cube",
        action="store_false",
        help="Disable the XY-snap so demonstrations follow your hand position exactly (noisier).",
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
        default=0.015,
        help="Minimum hand-body z above table top during descend to avoid collision.",
    )
    p.add_argument(
        "--success-lift",
        type=float,
        default=0.05,
        help="Minimum lift height (meters) above the cube's rest z to count an attempt as successful.",
    )
    p.add_argument(
        "--save-failures",
        action="store_true",
        help="Also save episodes where the pick failed (default: discard).",
    )
    p.add_argument(
        "--reset-between-attempts",
        action="store_true",
        default=True,
        help="Reset env (cube respawns, arm goes home) after each auto pick-place sequence.",
    )
    p.add_argument(
        "--save-images",
        action="store_true",
        help=(
            "Also record an RGB image per step into each saved episode's `img` dataset. "
            "Off by default (lower disk + faster). Recommended for vision-based IL pipelines."
        ),
    )
    p.add_argument(
        "--image-camera",
        type=str,
        default="agent_view",
        help="Name of the MuJoCo camera to render from (defined in the scene XML).",
    )
    p.add_argument(
        "--image-size",
        type=str,
        default="128x128",
        help="WxH for recorded images (e.g. 128x128, 256x256). Smaller = much smaller dataset.",
    )
    return p.parse_args()


def _parse_image_size(s: str) -> tuple[int, int]:
    """Parse a 'WxH' string into (width, height)."""
    parts = s.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Bad --image-size {s!r}; expected like '128x128'.")
    return int(parts[0]), int(parts[1])


@dataclass
class EpisodeBuffer:
    """In-memory rolling buffer for a single pick attempt.

    Steps are appended every control tick during one auto-sequence and are
    only persisted to Zarr if the attempt is judged successful.
    """

    obs: List[np.ndarray] = field(default_factory=list)
    act: List[np.ndarray] = field(default_factory=list)
    t: List[float] = field(default_factory=list)
    images: List[np.ndarray] = field(default_factory=list)
    image_camera: Optional[str] = None
    success: bool = False
    peak_lift_m: float = 0.0

    def append(
        self,
        obs_vec: np.ndarray,
        act_vec: np.ndarray,
        t: float,
        img: Optional[np.ndarray] = None,
    ) -> None:
        self.obs.append(np.asarray(obs_vec, dtype=np.float32).copy())
        self.act.append(np.asarray(act_vec, dtype=np.float32).copy())
        self.t.append(float(t))
        if img is not None:
            self.images.append(np.asarray(img, dtype=np.uint8).copy())

    def __len__(self) -> int:
        return len(self.obs)

    def commit(self, logger: ZarrTrajectoryLogger, name: Optional[str] = None) -> str:
        if not self.obs:
            raise RuntimeError("Cannot commit an empty episode.")
        # If we recorded images they MUST line up with the obs/act stream so downstream code can
        # index them step-by-step. Truncate to the shortest just in case render lagged a step.
        has_images = len(self.images) > 0
        if has_images and len(self.images) != len(self.obs):
            n = min(len(self.images), len(self.obs))
            self.obs = self.obs[:n]
            self.act = self.act[:n]
            self.t = self.t[:n]
            self.images = self.images[:n]
        image_shape = tuple(int(x) for x in self.images[0].shape) if has_images else None

        ep = logger.start_episode(
            obs_dim=self.obs[0].shape[0],
            act_dim=self.act[0].shape[0],
            name=name,
            image_shape=image_shape,
            image_camera=self.image_camera,
        )
        ep.group.attrs["success"] = bool(self.success)
        ep.group.attrs["peak_lift_m"] = float(self.peak_lift_m)
        for i in range(len(self.obs)):
            img = self.images[i] if has_images else None
            logger.append(ep, self.obs[i], self.act[i], self.t[i], img=img)
        return ep.group.name


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
        ("hold", 0.4),
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
        elif name == "hold":
            # Sit still with the gripper clamped so the contact forces fully settle and the cube
            # stops wobbling before we start moving the arm.
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
    home_xy = obs0.ee_pos[:2].astype(np.float32).copy()
    home_seed = int(args.seed)

    # Optional offscreen renderer for vision-based IL data collection.
    renderer = None
    if args.save_images:
        try:
            import mujoco as _mj  # local import to avoid breaking non-image runs
            img_w, img_h = _parse_image_size(args.image_size)
            renderer = _mj.Renderer(env.model, height=img_h, width=img_w)
            print(
                f"[IMG] Recording {img_w}x{img_h} RGB frames from camera '{args.image_camera}' "
                f"into each saved episode (img dataset, uint8)."
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[IMG] WARNING: could not init Renderer ({exc!r}); continuing without images.")
            renderer = None

    start = time.time()
    end = start + args.seconds
    last_print = 0.0
    min_print_dt = 1.0 / max(args.print_hz, 1e-6)
    hand_seen_once = False
    last_g_state = 0.0
    auto_seq: Optional[AutoPickPlaceSequence] = None

    # Per-episode buffer + counters.
    # NOTE: The buffer is created at episode start (right after env.reset) so the TELEOP APPROACH
    # (the part the model needs to learn -- "how to move ee_xy to above cube_xy") is recorded too.
    # Recording STOPS at pinch trigger (start of auto sequence); the scripted auto phase is not
    # appended. We still evaluate grasp success during the auto phase to decide save/discard.
    buffer: EpisodeBuffer = EpisodeBuffer(
        image_camera=args.image_camera if renderer is not None else None,
    )
    buffer_grasp_locked = False  # set True when grasp is detected -> stop appending further steps
    attempts = 0
    saved = 0
    discarded = 0

    print(f"Writing dataset to: {Path(args.dataset).expanduser().resolve()}")
    print("Each EPISODE starts at env.reset() and ends when the auto pick-place sequence completes.")
    print("Recording window = from reset (teleop approach) UNTIL pinch trigger.")
    print("After pinch, the auto phase still plays so you can see it, but is NOT recorded.")
    print(
        f"Episodes are SAVED to Zarr only when the cube is lifted >= {args.success_lift:.3f} m "
        f"AND held by both fingers (use --save-failures to keep failed attempts too)."
    )
    print("Gesture -> Panda action mapping:")
    print("  Move hand RIGHT  : +x -> -dx (ee moves -X)")
    print("  Move hand LEFT   : -x -> +dx (ee moves +X)")
    print("  Move hand UP     : -y -> +dy  (dy uses inverted y)")
    print("  Move hand DOWN   : +y -> -dy")
    print("  Pinch fingers    : g=1 -> trigger AUTO pick-place sequence")
    print("  Open fingers     : g=0 -> OPEN gripper / no trigger")
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
                hover_z = float(ee_t[2])
                # Snap the descend XY target to the cube's current XY (clamped to workspace) so the
                # grasp is well-centered. The user only needs to hover roughly above the cube and pinch.
                if args.snap_to_cube:
                    snap_xy = np.clip(
                        obs.obj_pos[:2].astype(np.float32),
                        ws_min[:2],
                        ws_max[:2],
                    )
                    pre_xyz[0] = float(snap_xy[0])
                    pre_xyz[1] = float(snap_xy[1])
                    env.set_ee_target(pre_xyz)
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
                attempts += 1
                if not buffer_grasp_locked:
                    buffer_grasp_locked = True
                    print(
                        f"[REC ] Pinch trigger at step {len(buffer)}. Recording stopped; "
                        f"auto sequence will run for validation only."
                    )
                print(
                    f"[AUTO #{attempts}] Triggered after {len(buffer)} teleop steps. "
                    f"pre=({pre_xyz[0]:.3f},{pre_xyz[1]:.3f},{hover_z:.3f}) "
                    f"home_xy=({home_xy[0]:.3f},{home_xy[1]:.3f}) "
                    f"grasp_z={grasp_z:.3f} lift_z={lift_z:.3f} | obj_xy=({obs.obj_pos[0]:.3f},{obs.obj_pos[1]:.3f})"
                )

            if auto_seq is not None:
                act, done = auto_seq.step(env, now, obs)
            elif signal is not None:
                act = gesture_to_dxdyg(signal.as_vector(), args.delta_scale)
                done = False
            else:
                act = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                done = False

            obs = env.step(act)
            now = time.time()

            # Append to the episode buffer. We append every tick from episode start (teleop included)
            # UNTIL pinch trigger (buffer_grasp_locked=True), at which point we stop. The auto
            # sequence keeps running for validation, but those frames are NOT recorded because it's
            # scripted motion.
            if not buffer_grasp_locked:
                step_img = None
                if renderer is not None:
                    try:
                        renderer.update_scene(env.data, camera=args.image_camera)
                        step_img = renderer.render()  # (H, W, 3) uint8
                    except Exception as exc:  # noqa: BLE001
                        # Render failures shouldn't kill the data collection loop.
                        if not getattr(args, "_img_warn_emitted", False):
                            print(f"[IMG] WARNING: render failed ({exc!r}); dropping image stream.")
                            args._img_warn_emitted = True  # type: ignore[attr-defined]
                        renderer = None
                buffer.append(obs.as_vector(), act, now, img=step_img)

            # Keep success evaluation active even after recording has stopped, so we can still
            # save only successful attempts while excluding scripted auto frames from the dataset.
            lift = env.object_lift_height()
            if lift > buffer.peak_lift_m:
                buffer.peak_lift_m = lift
            if (not buffer.success) and env.is_object_grasped(min_lift_m=args.success_lift):
                buffer.success = True
                print(
                    f"[AUTO #{attempts}] Grasp confirmed during auto phase "
                    f"(peak_lift={buffer.peak_lift_m:.3f}m)."
                )

            if auto_seq is not None and done:
                # Sequence finished. Persist the buffer if grasp succeeded; discard otherwise.
                final_success = buffer.success
                if final_success or args.save_failures:
                    name = logger_episode_name(args.episode_name, attempts)
                    written_name = buffer.commit(logger, name=name)
                    saved += 1
                    tag = "SAVED" if final_success else "SAVED(FAIL)"
                    print(
                        f"[AUTO #{attempts}] {tag} -> {written_name}  "
                        f"steps={len(buffer)}  peak_lift={buffer.peak_lift_m:.3f}m  success={final_success}"
                    )
                else:
                    discarded += 1
                    print(
                        f"[AUTO #{attempts}] DISCARDED (no successful grasp)  "
                        f"steps={len(buffer)}  peak_lift={buffer.peak_lift_m:.3f}m"
                    )
                auto_seq = None

                if args.reset_between_attempts:
                    home_seed += 1
                    obs = env.reset(seed=home_seed)
                    home_xy = obs.ee_pos[:2].astype(np.float32).copy()
                    print(
                        f"[ENV] Reset for next attempt. obj_xy=({obs.obj_pos[0]:.3f},{obs.obj_pos[1]:.3f})  "
                        f"saved={saved}  discarded={discarded}  attempts={attempts}"
                    )

                # Fresh buffer for the next episode. Recording resumes on the very next tick so
                # the teleop approach phase is captured for the upcoming attempt as well.
                buffer = EpisodeBuffer(
                    image_camera=args.image_camera if renderer is not None else None,
                )
                buffer_grasp_locked = False

            if viewer is not None:
                viewer.sync()

            if args.preview:
                cmd_txt = _format_cmd(float(act[0]), float(act[1]), float(act[2]), (args.delta_scale, args.delta_scale))
                if signal is None:
                    hud = "No hand"
                else:
                    pinch = "PINCH" if signal.g > 0.5 else "OPEN"
                    if auto_seq is not None:
                        hud = f"{cmd_txt} | hand {pinch} | AUTO:{auto_seq.phase_name} lift={env.object_lift_height():.3f}"
                    else:
                        hud = f"{cmd_txt} | hand {pinch} | saved={saved} disc={discarded}"
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
                        f"|obj-goal|={obj_to_goal:.2f}m | lift={env.object_lift_height():+.3f}m"
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
        # If we were mid-episode when the loop ended, persist only if grasp was already confirmed.
        # An in-flight buffer with no grasp yet (e.g. user was still teleoperating) is just dropped.
        if len(buffer) > 0 and buffer.success:
            name = logger_episode_name(args.episode_name, max(attempts, 1))
            written_name = buffer.commit(logger, name=name)
            saved += 1
            print(f"[AUTO #{max(attempts, 1)}] (on exit) SAVED -> {written_name}  success=True")
        elif len(buffer) > 0 and args.save_failures:
            name = logger_episode_name(args.episode_name, max(attempts, 1))
            written_name = buffer.commit(logger, name=name)
            saved += 1
            print(f"[AUTO #{max(attempts, 1)}] (on exit) SAVED(FAIL) -> {written_name}")
        elif len(buffer) > 0:
            discarded += 1
            print(f"[AUTO #{max(attempts, 1)}] (on exit) DISCARDED (no successful grasp; {len(buffer)} steps)")

        print(
            f"[SUMMARY] attempts={attempts}  saved={saved}  discarded={discarded}  "
            f"dataset={Path(args.dataset).expanduser().resolve()}"
        )
        cap.release()
        extractor.close()
        if args.preview:
            cv2.destroyAllWindows()


def logger_episode_name(base_name: Optional[str], attempt_idx: int) -> Optional[str]:
    """Return a per-attempt episode name. If user provided --episode-name, suffix with attempt idx."""
    if base_name is None:
        return None
    return f"{base_name}_a{attempt_idx:03d}"


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

