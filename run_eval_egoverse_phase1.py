from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from run_train_egoverse_phase1 import EgoVerseStyleBC
from stage2_mujoco.panda_env import PandaPickPlaceEnv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate EgoVerse-style BC policy in MuJoCo.")
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--policy", type=str, default="data/policies/egoverse_bc_policy.pt")
    p.add_argument(
        "--session-num",
        type=int,
        default=10,
        help="Number of evaluation sessions (reset cube + home arm each session; end after scripted post-pinch or failure).",
    )
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--image-camera", type=str, default="agent_view")
    p.add_argument("--device", type=str, default="auto", help="auto|cpu|mps|cuda")
    p.add_argument(
        "--trigger-distance",
        type=float,
        default=0.03,
        help="Start scripted post-pinch when ||ee_xy - obj_xy|| is below this (meters).",
    )
    p.add_argument(
        "--trigger-dwell-steps",
        type=int,
        default=5,
        help="Require this many consecutive near-target control steps before triggering scripted phase.",
    )
    p.add_argument(
        "--act-scale-xy",
        type=float,
        default=0.5,
        help="Scale factor applied to policy dx,dy during approach (lower is more conservative).",
    )
    p.add_argument(
        "--act-max-xy",
        type=float,
        default=0.02,
        help="Clamp magnitude of each XY action component to [-act-max-xy, +act-max-xy].",
    )
    p.add_argument(
        "--act-deadband-xy",
        type=float,
        default=0.001,
        help="Set |dx|,|dy| below this threshold to zero to reduce drift from tiny residuals.",
    )
    p.add_argument("--auto-time-scale", type=float, default=2.0, help="Slow down scripted post-pinch sequence.")
    p.add_argument(
        "--auto-ease",
        type=str,
        default="smoothstep",
        choices=["linear", "smoothstep"],
        help="Interpolation profile for scripted post-pinch sequence.",
    )
    p.add_argument(
        "--auto-grasp-offset",
        type=float,
        default=0.0,
        help="Target z relative to object center during scripted descend.",
    )
    p.add_argument(
        "--snap-to-cube",
        dest="snap_to_cube",
        action="store_true",
        default=True,
        help="At trigger time, snap descend XY target to current cube XY.",
    )
    p.add_argument(
        "--no-snap-to-cube",
        dest="snap_to_cube",
        action="store_false",
        help="Disable XY snap and use current ee_target XY.",
    )
    p.add_argument("--auto-table-top-z", type=float, default=0.30, help="Estimated table top z in world frame.")
    p.add_argument(
        "--auto-table-clearance-z",
        type=float,
        default=0.015,
        help="Minimum grasp target z above table top.",
    )
    p.add_argument(
        "--spawn-x-range",
        type=str,
        default="0.45,0.65",
        help='Cube spawn X range in meters as "lo,hi" (world depth strip, right side of workspace).',
    )
    p.add_argument(
        "--spawn-y-range",
        type=str,
        default="0.06,0.10",
        help='Cube spawn Y range in meters as "lo,hi" (narrow strip to the right of base centerline).',
    )
    p.add_argument(
        "--frozen-min-steps",
        type=int,
        default=30,
        help="Minimum sim steps before declaring a frozen (no-motion) failure during approach only.",
    )
    p.add_argument(
        "--frozen-path-threshold",
        type=float,
        default=0.005,
        help="If cumulative EE path length stays below this (m) after frozen-min-steps (approach only), abort session.",
    )
    p.add_argument(
        "--frozen-min-dist-xy",
        type=float,
        default=0.06,
        help="Only treat as frozen if ||ee_xy-obj_xy|| exceeds this (m); avoids aborting while nearly aligned above cube.",
    )
    p.add_argument(
        "--session-max-steps",
        type=int,
        default=30_000,
        help="Safety cap on sim steps per session if scripted phase never completes.",
    )
    p.add_argument(
        "--stall-step-eps",
        type=float,
        default=1e-4,
        help="If per-step TCP motion is below this (m) for many consecutive steps while far from cube, count as stall.",
    )
    p.add_argument(
        "--stall-consecutive",
        type=int,
        default=150,
        help="Consecutive sub-threshold steps (see --stall-step-eps) to abort approach as stuck.",
    )
    return p.parse_args()


def _parse_range(s: str, name: str) -> tuple[float, float]:
    parts = s.replace(" ", "").split(",")
    if len(parts) != 2:
        raise ValueError(f"Bad --{name} {s!r}; expected 'lo,hi' like '0.45,0.65'.")
    return float(parts[0]), float(parts[1])


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_policy(path: Path, device: torch.device):
    # PyTorch 2.6+ defaults to weights_only=True, which breaks loading older
    # checkpoints that contain numpy metadata. This project expects trusted,
    # local checkpoints, so we explicitly load full checkpoint contents.
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = EgoVerseStyleBC(
        obs_dim=int(ckpt["obs_dim"]),
        act_dim=int(ckpt["act_dim"]),
        use_images=bool(ckpt["use_images"]),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    obs_mean = np.asarray(ckpt["obs_mean"], dtype=np.float32)
    obs_std = np.asarray(ckpt["obs_std"], dtype=np.float32)
    return model, obs_mean, obs_std, bool(ckpt["use_images"])


@dataclass
class AutoPickPlaceSequence:
    start_t: float
    pre_xyz: np.ndarray
    pre_hover_z: float
    home_xy: np.ndarray
    grasp_z: float
    lift_z: float
    place_z: float
    phase_idx: int = 0
    phase_start_t: float = 0.0
    time_scale: float = 2.0
    ease_mode: str = "smoothstep"

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

    def _interp(self, a: float, b: float, alpha: float) -> float:
        return (1.0 - alpha) * a + alpha * b

    def _interp_vec(self, a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
        return (1.0 - alpha) * a + alpha * b

    def _ease(self, alpha: float) -> float:
        a = float(np.clip(alpha, 0.0, 1.0))
        if self.ease_mode == "linear":
            return a
        return a * a * (3.0 - 2.0 * a)  # smoothstep

    def step(self, env: PandaPickPlaceEnv, now: float) -> tuple[np.ndarray, bool]:
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
            z = self._interp(self.pre_hover_z, self.grasp_z, alpha_eased)
            env.set_ee_target(np.array([self.pre_xyz[0], self.pre_xyz[1], z], dtype=np.float32))
            act = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        elif name == "close":
            env.set_ee_target(pre_grasp_xyz)
            act = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        elif name == "hold":
            env.set_ee_target(pre_grasp_xyz)
            act = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        elif name == "lift":
            z = self._interp(self.grasp_z, self.lift_z, alpha_eased)
            env.set_ee_target(np.array([self.pre_xyz[0], self.pre_xyz[1], z], dtype=np.float32))
            act = np.array([0.0, 0.0, 1.0], dtype=np.float32)
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
            self.phase_start_t = now
        return act, False


def run(env: PandaPickPlaceEnv, viewer: Optional[object], args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    model, obs_mean, obs_std, use_images = load_policy(Path(args.policy).expanduser().resolve(), device)
    print(f"Loaded policy: {args.policy} (use_images={use_images})")

    renderer = None
    if use_images:
        import mujoco

        renderer = mujoco.Renderer(env.model, height=128, width=128)

    n_sessions = max(1, int(args.session_num))
    stats = {
        "scripted_complete": 0,
        "grasp_success": 0,
        "frozen": 0,
        "stalled": 0,
        "timeout": 0,
    }

    for si in range(n_sessions):
        seed = int(args.seed) + si * 100_003
        obs = env.reset(seed=seed)
        home_xy = obs.ee_pos[:2].astype(np.float32).copy()
        ox, oy = float(obs.obj_pos[0]), float(obs.obj_pos[1])
        print(
            f"\n[SESSION {si + 1}/{n_sessions}] seed={seed}  cube_xy=({ox:.3f},{oy:.3f})  "
            f"spawn_x={env.spawn_range[0]}  spawn_y={env.spawn_range[1]}"
        )

        last_print = 0.0
        near_count = 0
        auto_seq: Optional[AutoPickPlaceSequence] = None
        step_idx = 0
        path_accum = 0.0
        prev_ee = obs.ee_pos.astype(np.float32).copy()
        stall_count = 0
        scripted_finished = False

        while True:
            if auto_seq is not None:
                act, done = auto_seq.step(env, time.time())
                if done:
                    print("[AUTO] Post-pinch scripted sequence completed.")
                    auto_seq = None
                    scripted_finished = True
            else:
                obs_vec = obs.as_vector().astype(np.float32)
                obs_norm = ((obs_vec - obs_mean) / obs_std).astype(np.float32)
                obs_t = torch.from_numpy(obs_norm).to(device).unsqueeze(0)

                img_t = None
                if use_images:
                    assert renderer is not None
                    renderer.update_scene(env.data, camera=args.image_camera)
                    img = renderer.render().astype(np.float32) / 255.0  # HWC
                    img_t = torch.from_numpy(img).to(device).permute(2, 0, 1).unsqueeze(0)

                with torch.no_grad():
                    act = model(obs_t, img_t).squeeze(0).cpu().numpy().astype(np.float32)

                # For this hybrid controller we only learn approach motion; gripper
                # closure/opening is scripted in the post-trigger routine.
                act[2] = 0.0
                act[:2] *= float(args.act_scale_xy)
                act[:2] = np.clip(act[:2], -float(args.act_max_xy), float(args.act_max_xy))
                dead = float(args.act_deadband_xy)
                act[0] = 0.0 if abs(float(act[0])) < dead else float(act[0])
                act[1] = 0.0 if abs(float(act[1])) < dead else float(act[1])

                ee_xy = obs.ee_pos[:2].astype(np.float32)
                obj_xy = obs.obj_pos[:2].astype(np.float32)
                dist_xy = float(np.linalg.norm(ee_xy - obj_xy))
                if dist_xy <= float(args.trigger_distance):
                    near_count += 1
                else:
                    near_count = 0

                if near_count >= int(args.trigger_dwell_steps):
                    ws_min, ws_max = env.workspace_bounds
                    ee_t = env.ee_target
                    pre_xyz = ee_t.copy()
                    hover_z = float(ee_t[2])
                    if args.snap_to_cube:
                        snap_xy = np.clip(obs.obj_pos[:2].astype(np.float32), ws_min[:2], ws_max[:2])
                        pre_xyz[0] = float(snap_xy[0])
                        pre_xyz[1] = float(snap_xy[1])
                        env.set_ee_target(pre_xyz)
                    table_safe_z = float(args.auto_table_top_z + args.auto_table_clearance_z)
                    grasp_z = float(
                        np.clip(
                            max(obs.obj_pos[2] + float(args.auto_grasp_offset), table_safe_z),
                            ws_min[2] + 0.03,
                            ws_max[2] - 0.05,
                        )
                    )
                    lift_z = float(np.clip(max(hover_z, grasp_z) + 0.12, ws_min[2] + 0.05, ws_max[2] - 0.02))
                    auto_seq = AutoPickPlaceSequence(
                        start_t=time.time(),
                        pre_xyz=pre_xyz,
                        pre_hover_z=hover_z,
                        home_xy=home_xy,
                        grasp_z=grasp_z,
                        lift_z=lift_z,
                        place_z=grasp_z,
                        time_scale=float(args.auto_time_scale),
                        ease_mode=str(args.auto_ease),
                    )
                    print(
                        "[AUTO] Triggered by proximity; running scripted post-pinch sequence "
                        f"(dist_xy={dist_xy:.3f}m, grasp_z={grasp_z:.3f}, lift_z={lift_z:.3f})."
                    )
                    act, _ = auto_seq.step(env, time.time())
                    near_count = 0

            obs = env.step(act)
            step_idx += 1
            step_move = float(np.linalg.norm(obs.ee_pos.astype(np.float32) - prev_ee))
            path_accum += step_move
            if step_move < float(args.stall_step_eps):
                stall_count += 1
            else:
                stall_count = 0
            prev_ee = obs.ee_pos.astype(np.float32).copy()

            if viewer is not None:
                viewer.sync()

            now = time.time()
            if now - last_print > 0.5:
                dist_xy_now = float(np.linalg.norm(obs.ee_pos[:2] - obs.obj_pos[:2]))
                print(f"ctrl={act} | dist_xy={dist_xy_now:.4f}m | near_count={near_count} | step={step_idx}")
                last_print = now
            time.sleep(env.model.opt.timestep)

            if scripted_finished:
                grasped = env.is_object_grasped()
                stats["scripted_complete"] += 1
                if grasped:
                    stats["grasp_success"] += 1
                print(
                    f"[SESSION {si + 1}/{n_sessions}] done  grasp_heuristic={grasped}  "
                    f"path_accum={path_accum:.4f}m  steps={step_idx}"
                )
                break

            if step_idx >= int(args.session_max_steps):
                stats["timeout"] += 1
                print(
                    f"[SESSION {si + 1}/{n_sessions}] TIMEOUT (steps>={args.session_max_steps}); "
                    "starting next session."
                )
                break

            # (1) Frozen-from-start: cumulative path never exceeded threshold (never really moved),
            #     while still far in XY. This misses "moved a bit then stuck" — see (2).
            # (2) Stalled: many consecutive steps with microscopic TCP motion (policy/output stuck,
            #     contact, or joint limit) while still far in XY. Catches the long constant-ctrl
            #     case where path_accum is already > frozen-path-threshold.
            dist_xy_post = float(np.linalg.norm(obs.ee_pos[:2] - obs.obj_pos[:2]))
            if auto_seq is None and dist_xy_post > float(args.frozen_min_dist_xy):
                if (
                    step_idx >= int(args.frozen_min_steps)
                    and path_accum < float(args.frozen_path_threshold)
                ):
                    stats["frozen"] += 1
                    print(
                        f"[SESSION {si + 1}/{n_sessions}] FROZEN (path_accum={path_accum:.5f}m "
                        f"< {args.frozen_path_threshold}, dist_xy={dist_xy_post:.3f}m "
                        f"after {step_idx} steps); starting next session."
                    )
                    break
                if stall_count >= int(args.stall_consecutive):
                    stats["stalled"] += 1
                    print(
                        f"[SESSION {si + 1}/{n_sessions}] STALLED (TCP motion < {args.stall_step_eps}m for "
                        f"{stall_count} consecutive steps, dist_xy={dist_xy_post:.3f}m, "
                        f"path_accum={path_accum:.4f}m); starting next session."
                    )
                    break

    print(
        "\n[EVAL SUMMARY] "
        f"sessions={n_sessions}  scripted_complete={stats['scripted_complete']}  "
        f"grasp_success={stats['grasp_success']}  frozen={stats['frozen']}  stalled={stats['stalled']}  "
        f"timeout={stats['timeout']}"
    )


def main() -> None:
    args = parse_args()
    spawn_x = _parse_range(args.spawn_x_range, "spawn-x-range")
    spawn_y = _parse_range(args.spawn_y_range, "spawn-y-range")
    env = PandaPickPlaceEnv(spawn_x_range=spawn_x, spawn_y_range=spawn_y)
    (sx_lo, sx_hi), (sy_lo, sy_hi) = env.spawn_range
    print(
        f"[ENV] Cube spawn rectangle: x in [{sx_lo:.3f}, {sx_hi:.3f}] m, "
        f"y in [{sy_lo:.3f}, {sy_hi:.3f}] m (clipped to workspace)."
    )

    if not args.viewer:
        run(env, None, args)
        return

    from mujoco import viewer as mj_viewer

    print("Launching viewer. On macOS, run this script with `mjpython`.")
    with mj_viewer.launch_passive(env.model, env.data) as viewer:
        run(env, viewer, args)


if __name__ == "__main__":
    main()
