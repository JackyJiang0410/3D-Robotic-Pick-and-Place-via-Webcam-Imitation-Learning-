from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np


DEFAULT_PANDA_SCENE = Path(__file__).resolve().parents[1] / "assets" / "mujoco" / "panda_pick_place_scene.xml"


@dataclass(frozen=True)
class PandaObs:
    qpos: np.ndarray
    qvel: np.ndarray
    ee_pos: np.ndarray
    obj_pos: np.ndarray
    target_pos: np.ndarray
    gripper_open: np.ndarray  # shape (1,)

    def as_vector(self) -> np.ndarray:
        return np.concatenate(
            [
                self.qpos.astype(np.float32),
                self.qvel.astype(np.float32),
                self.ee_pos.astype(np.float32),
                self.obj_pos.astype(np.float32),
                self.target_pos.astype(np.float32),
                self.gripper_open.astype(np.float32),
            ],
            axis=0,
        )


class PandaPickPlaceEnv:
    """
    Phase-1 Panda environment (sim-only).

    - Model: `assets/mujoco/panda_pick_place_scene.xml` (includes Menagerie `panda.xml`)
    - Observation: low-dimensional state vector
    - Action: [dx, dy, g]
        - dx/dy: desired end-effector delta position (meters per step, pre-scaled)
        - g: 1.0 close, 0.0 open

    Control:
      Uses a simple damped-least-squares IK on the Panda arm joints (7 DoF) to
      track an end-effector position target, then sends joint targets to the
      built-in MuJoCo "general" actuators in the Menagerie model.
    """

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        ik_damping: float = 0.05,
        ik_step_size: float = 0.6,
        ik_iters: int = 8,
        workspace_min: tuple[float, float, float] = (0.25, -0.35, 0.15),
        workspace_max: tuple[float, float, float] = (0.85, 0.35, 0.75),
    ):
        self.model_path = Path(model_path) if model_path else DEFAULT_PANDA_SCENE
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)

        self.ik_damping = float(ik_damping)
        self.ik_step_size = float(ik_step_size)
        self.ik_iters = int(ik_iters)
        self.workspace_min = np.array(workspace_min, dtype=np.float32)
        self.workspace_max = np.array(workspace_max, dtype=np.float32)

        self._hand_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self._obj_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        self._target_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target_site")

        if self._hand_body_id < 0:
            raise RuntimeError("Could not find body 'hand' in Panda model.")
        if self._obj_body_id < 0:
            raise RuntimeError("Could not find body 'object' in scene.")
        if self._target_site_id < 0:
            raise RuntimeError("Could not find site 'target_site' in scene.")

        self._arm_joints = [f"joint{i}" for i in range(1, 8)]
        self._arm_jnt_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self._arm_joints
        ]
        if any(j < 0 for j in self._arm_jnt_ids):
            missing = [n for n, j in zip(self._arm_joints, self._arm_jnt_ids) if j < 0]
            raise RuntimeError(f"Missing Panda arm joints in model: {missing}")

        self._arm_qpos_adr = np.array([self.model.jnt_qposadr[j] for j in self._arm_jnt_ids], dtype=np.int32)
        self._arm_dof_adr = np.array([self.model.jnt_dofadr[j] for j in self._arm_jnt_ids], dtype=np.int32)

        self._finger1_jnt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
        self._finger2_jnt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2")
        self._finger1_qpos_adr = int(self.model.jnt_qposadr[self._finger1_jnt]) if self._finger1_jnt >= 0 else None
        self._finger2_qpos_adr = int(self.model.jnt_qposadr[self._finger2_jnt]) if self._finger2_jnt >= 0 else None

        # Gripper actuator is actuator8 in Menagerie panda.xml, ctrlrange [0, 255].
        self._gripper_act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
        if self._gripper_act_id < 0:
            raise RuntimeError("Could not find gripper actuator 'actuator8' in Panda model.")

        # Internal target for end-effector position control.
        self._ee_target = np.zeros(3, dtype=np.float32)

        mujoco.mj_forward(self.model, self.data)

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def workspace_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.workspace_min.copy(), self.workspace_max.copy()

    @property
    def ee_target(self) -> np.ndarray:
        return self._ee_target.copy()

    def set_ee_target(self, target_xyz: np.ndarray) -> None:
        target = np.asarray(target_xyz, dtype=np.float32).reshape(3)
        self._ee_target = np.clip(target, self.workspace_min, self.workspace_max)

    def set_ee_target_z(self, target_z: float) -> None:
        t = self._ee_target.copy()
        t[2] = float(target_z)
        self.set_ee_target(t)

    def reset(self, seed: Optional[int] = None) -> PandaObs:
        if seed is not None:
            np.random.seed(seed)

        mujoco.mj_resetData(self.model, self.data)

        # Try to use the 'home' keyframe from the Menagerie Panda model.
        home_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, home_id)

        # Randomize object pose slightly on the table (sim-only).
        obj_jnt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "object")
        if obj_jnt >= 0:
            adr = int(self.model.jnt_qposadr[obj_jnt])
            self.data.qpos[adr + 0] = 0.55 + 0.10 * (np.random.rand() - 0.5)
            self.data.qpos[adr + 1] = 0.08 + 0.10 * (np.random.rand() - 0.5)
            self.data.qpos[adr + 2] = 0.33
            self.data.qpos[adr + 3 : adr + 7] = np.array([1, 0, 0, 0], dtype=np.float64)

        mujoco.mj_forward(self.model, self.data)

        # Initialize ee target to current.
        self._ee_target = self._ee_pos().astype(np.float32)
        return self.observe()

    def observe(self) -> PandaObs:
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        ee_pos = self._ee_pos().copy()
        obj_pos = self.data.xpos[self._obj_body_id].copy()
        target_pos = self.data.site_xpos[self._target_site_id].copy()

        if self._finger1_qpos_adr is not None and self._finger2_qpos_adr is not None:
            grip = 0.5 * (self.data.qpos[self._finger1_qpos_adr] + self.data.qpos[self._finger2_qpos_adr])
        else:
            grip = 0.0
        gripper_open = np.array([grip], dtype=np.float32)

        return PandaObs(
            qpos=qpos,
            qvel=qvel,
            ee_pos=ee_pos,
            obj_pos=obj_pos,
            target_pos=target_pos,
            gripper_open=gripper_open,
        )

    def step(self, action: np.ndarray) -> PandaObs:
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.shape[0] != 3:
            raise ValueError(f"Expected action shape (3,), got {a.shape}")

        dx, dy, g = float(a[0]), float(a[1]), float(a[2])
        dz = 0.0
        self._ee_target = np.clip(self._ee_target + np.array([dx, dy, dz], dtype=np.float32), self.workspace_min, self.workspace_max)

        # IK: move arm joints toward ee target.
        self._ik_track_target(self._ee_target)

        # Gripper: Menagerie mapping (255=open, 0=closed).
        grip_ctrl = 0.0 if g > 0.5 else 255.0

        # Send joint target controls (actuator1..7) and gripper (actuator8).
        self.data.ctrl[:7] = self.data.qpos[self._arm_qpos_adr].astype(np.float32)
        self.data.ctrl[self._gripper_act_id] = grip_ctrl

        mujoco.mj_step(self.model, self.data)
        return self.observe()

    def _ee_pos(self) -> np.ndarray:
        # Use the hand body position as a simple end-effector point.
        return self.data.xpos[self._hand_body_id]

    def _ik_track_target(self, target_pos: np.ndarray) -> None:
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)

        for _ in range(self.ik_iters):
            cur = self._ee_pos().astype(np.float64)
            err = (target_pos.astype(np.float64) - cur)
            if float(np.linalg.norm(err)) < 1e-4:
                break

            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self._hand_body_id)
            J = jacp[:, self._arm_dof_adr]  # (3,7)

            # Damped least squares: dq = J^T (J J^T + λI)^-1 * err
            JJt = J @ J.T
            lam = (self.ik_damping ** 2) * np.eye(3, dtype=np.float64)
            dq = J.T @ np.linalg.solve(JJt + lam, err)
            dq = np.clip(dq, -0.2, 0.2)

            q = self.data.qpos[self._arm_qpos_adr].astype(np.float64)
            q_new = q + self.ik_step_size * dq

            # Respect joint limits.
            for i, jnt_id in enumerate(self._arm_jnt_ids):
                lo, hi = self.model.jnt_range[jnt_id]
                if lo < hi:
                    q_new[i] = float(np.clip(q_new[i], lo, hi))

            self.data.qpos[self._arm_qpos_adr] = q_new
            mujoco.mj_forward(self.model, self.data)

