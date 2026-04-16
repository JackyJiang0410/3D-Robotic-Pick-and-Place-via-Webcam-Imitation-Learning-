from __future__ import annotations

import time

from stage2_mujoco.panda_env import PandaPickPlaceEnv


def main() -> None:
    env = PandaPickPlaceEnv()
    env.reset(seed=0)

    from mujoco import viewer as mj_viewer

    print("Launching Panda viewer. On macOS, run with `mjpython run_view_panda.py`.")
    with mj_viewer.launch_passive(env.model, env.data) as v:
        end = time.time() + 20.0
        while v.is_running() and time.time() < end:
            # idle step
            env.step([0.0, 0.0, 0.0, 0.0])
            v.sync()
            time.sleep(env.model.opt.timestep)


if __name__ == "__main__":
    main()

