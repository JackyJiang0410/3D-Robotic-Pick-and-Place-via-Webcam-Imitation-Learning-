from __future__ import annotations

import argparse
import time
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
    p.add_argument(
        "--z-scale",
        type=float,
        default=None,
        help="Meters per step for full-scale z (maps to dz). Defaults to --delta-scale if omitted.",
    )
    p.add_argument(
        "--z-mode",
        type=str,
        default="mp_z_spread",
        choices=["palm_scale", "mp_z_spread", "mp_z_wrist", "reach_2d"],
        help="How Stage 1 computes z from MediaPipe (try mp_z_spread for better in/out feel).",
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
        help="Also print numeric x,y,z and dx,dy,dz (default is compact plain-language).",
    )
    return p.parse_args()


def gesture_to_dxdyzdg(action_human: np.ndarray, delta_scale: float, z_scale: float) -> np.ndarray:
    x, y, z, g = action_human.astype(np.float32)
    dx = float(delta_scale * x)
    dy = float(-delta_scale * y)
    dz = float(z_scale * z)
    return np.array([dx, dy, dz, float(g)], dtype=np.float32)


def _format_cmd(dx: float, dy: float, dz: float, g_cmd: float, scales: Tuple[float, float, float]) -> str:
    dx_s, dy_s, dz_s = scales
    eps_x = max(0.25 * abs(dx_s), 1e-6)
    eps_y = max(0.25 * abs(dy_s), 1e-6)
    eps_z = max(0.25 * abs(dz_s), 1e-6)

    active = (abs(dx) > eps_x) or (abs(dy) > eps_y) or (abs(dz) > eps_z)
    if not active:
        motion = "hold"
    else:
        # Prefer the strongest axis so the message matches what you feel most.
        mag = [abs(dx), abs(dy), abs(dz)]
        idx = int(np.argmax(mag))
        order = [idx] + [i for i in range(3) if i != idx]
        ordered: List[str] = []
        for i in order:
            if i == 0 and abs(dx) > eps_x:
                ordered.append("world +X" if dx > 0 else "world -X")
            if i == 1 and abs(dy) > eps_y:
                ordered.append("world +Y" if dy > 0 else "world -Y")
            if i == 2 and abs(dz) > eps_z:
                ordered.append("world +Z (up)" if dz > 0 else "world -Z (down)")
        motion = " + ".join(ordered) if ordered else "hold"

    grip = "CLOSE" if g_cmd > 0.5 else "OPEN"
    return f"cmd={motion} | grip={grip}"


def run_loop(env: PandaPickPlaceEnv, viewer: Optional[object], args: argparse.Namespace) -> None:
    z_scale = float(args.z_scale) if args.z_scale is not None else float(args.delta_scale)
    extractor = Stage1Extractor(
        Stage1Config(camera_id=args.camera_id, z_mode=args.z_mode)
    )
    cap = extractor.open_camera()

    logger = ZarrTrajectoryLogger(args.dataset)
    obs0 = env.reset(seed=args.seed)
    ep = logger.start_episode(obs_dim=obs0.as_vector().shape[0], act_dim=env.action_dim, name=args.episode_name)
    start = time.time()
    end = start + args.seconds
    last_print = 0.0
    min_print_dt = 1.0 / max(args.print_hz, 1e-6)
    hand_seen_once = False

    print(f"Writing dataset to: {Path(args.dataset).expanduser().resolve()}")
    print(f"Episode: {ep.group.name}  (press Ctrl+C to stop early)")
    print("Gesture -> Panda action mapping:")
    print("  Move hand RIGHT  : +x -> +dx (ee moves +X)")
    print("  Move hand LEFT   : -x -> -dx")
    print("  Move hand UP     : -y -> +dy  (dy uses inverted y)")
    print("  Move hand DOWN   : +y -> -dy")
    print(f"  In/out (z) mode : {args.z_mode}")
    print("    - palm_scale: apparent palm size (legacy)")
    print("    - mp_z_spread: fingertip-vs-wrist MediaPipe z (often best)")
    print("    - mp_z_wrist: wrist z vs neutral")
    print("    - reach_2d: index reach in image (2D-only)")
    print(f"  z sensitivity : dz = z * --z-scale (default matches --delta-scale = {z_scale})")
    print("  Pinch fingers    : g=1 -> CLOSE gripper")
    print("  Open fingers     : g=0 -> OPEN gripper")
    print("Tips: keep one hand centered in camera, move slowly, adjust --delta-scale / --z-scale.")
    if args.preview and viewer is not None:
        print("NOTE: --preview + MuJoCo viewer can crash OpenCV on some macOS setups. If it crashes, omit --preview.")

    try:
        while time.time() < end:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Failed to read frame from webcam.")

            vis, signal = extractor.process_frame(frame)
            if signal is not None:
                act = gesture_to_dxdyzdg(signal.as_vector(), args.delta_scale, z_scale)
            else:
                act = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

            obs = env.step(act)
            now = time.time()
            logger.append(ep, obs.as_vector(), act, now)

            if viewer is not None:
                viewer.sync()

            if args.preview:
                cmd_txt = _format_cmd(float(act[0]), float(act[1]), float(act[2]), float(act[3]), (args.delta_scale, args.delta_scale, z_scale))
                if signal is None:
                    hud = "No hand"
                else:
                    pinch = "PINCH" if signal.g > 0.5 else "OPEN"
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
                    cmd_txt = _format_cmd(float(act[0]), float(act[1]), float(act[2]), float(act[3]), (args.delta_scale, args.delta_scale, z_scale))
                    base = (
                        f"{cmd_txt} | pinch={pinch_label} | sim_grip_open={float(obs.gripper_open[0]):.3f} | "
                        f"|obj-goal|={obj_to_goal:.2f}m"
                    )
                    if args.verbose_numbers:
                        base += (
                            f" | hand_xyzg=({signal.x:+.2f},{signal.y:+.2f},{signal.z:+.2f},{signal.g:.0f})"
                            f" | dxyz=({act[0]:+.4f},{act[1]:+.4f},{act[2]:+.4f})"
                        )
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

