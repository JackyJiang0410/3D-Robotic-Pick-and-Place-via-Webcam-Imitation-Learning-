from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

from stage2_mujoco.panda_env import PandaPickPlaceEnv


@dataclass
class BCModel:
    W: np.ndarray
    b: np.ndarray
    x_mean: np.ndarray
    x_std: np.ndarray

    def predict(self, obs_vec: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs_vec, dtype=np.float32).reshape(1, -1)
        x = (obs - self.x_mean) / self.x_std
        act = x @ self.W + self.b
        return act.reshape(-1).astype(np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1: evaluate BC policy autonomously in MuJoCo.")
    p.add_argument("--viewer", action="store_true", help="Open MuJoCo passive viewer.")
    p.add_argument("--policy", type=str, default="data/policies/phase1_bc_policy.npz")
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def load_policy(path: Path) -> BCModel:
    data = np.load(str(path))
    if "x_mean" in data and "x_std" in data:
        x_mean = data["x_mean"].astype(np.float32)
        x_std = data["x_std"].astype(np.float32)
    else:
        # Backwards compatibility with earlier policies.
        x_mean = np.zeros((data["W"].shape[0],), dtype=np.float32)
        x_std = np.ones((data["W"].shape[0],), dtype=np.float32)
    return BCModel(W=data["W"], b=data["b"], x_mean=x_mean, x_std=x_std)


def run(env: PandaPickPlaceEnv, policy: BCModel, viewer: Optional[object], seconds: float, seed: int) -> None:
    obs = env.reset(seed=seed)
    end = time.time() + seconds
    last_print = 0.0
    while time.time() < end:
        obs_vec = obs.as_vector()
        act = policy.predict(obs_vec)
        obs = env.step(act)

        if viewer is not None:
            viewer.sync()

        now = time.time()
        if now - last_print > 0.5:
            print(f"ctrl={act}")
            last_print = now

        time.sleep(env.model.opt.timestep)


def main() -> None:
    args = parse_args()
    env = PandaPickPlaceEnv()
    policy = load_policy(Path(args.policy).expanduser().resolve())
    print(f"Loaded policy: {args.policy}")

    if not args.viewer:
        run(env, policy, None, args.seconds, args.seed)
        return

    from mujoco import viewer as mj_viewer

    print("Launching viewer. On macOS, run this script with `mjpython`.")
    with mj_viewer.launch_passive(env.model, env.data) as viewer:
        run(env, policy, viewer, args.seconds, args.seed)


if __name__ == "__main__":
    main()

