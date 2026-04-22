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
        fingertip_local_offset: tuple[float, float, float] = (0.0, 0.0, 0.045),
    ):
        self.model_path = Path(model_path) if model_path else DEFAULT_PANDA_SCENE
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)

        self.ik_damping = float(ik_damping)
        self.ik_step_size = float(ik_step_size)
        self.ik_iters = int(ik_iters)
        self.workspace_min = np.array(workspace_min, dtype=np.float32)
        self.workspace_max = np.array(workspace_max, dtype=np.float32)
        self._fingertip_local_offset = np.array(fingertip_local_offset, dtype=np.float64)

        self._hand_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self._left_finger_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        self._right_finger_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
        self._obj_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        self._target_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target_site")

        # Resting Z = body initial position from XML (XML places it on the table top).
        self._object_rest_z = float(self.model.body_pos[self._obj_body_id][2])
        self._object_initial_z = self._object_rest_z

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

        # Keep the (intentionally soft) Menagerie Panda gripper at near-stock stiffness so the closing
        # force does NOT punch through the cube. Cube grip is provided by friction, not by squeeze force:
        # with cube friction 10.0, even ~2.5 N of closing force gives ~25 N of holding friction per
        # finger, which is ~50x the 0.5 N cube weight. We bump only very slightly (b1: -100 -> -150) to
        # cover any momentary overshoot during the close transient.
        # Equilibrium under affine bias: gain * ctrl == -b1 * length, so we scale gain proportionally to
        # keep ctrl=0..255 -> length=0..0.04.
        self.model.actuator_biasprm[self._gripper_act_id, 1] = -150.0
        self.model.actuator_gainprm[self._gripper_act_id, 0] = 150.0 * 0.04 / 255.0
        self.model.actuator_forcerange[self._gripper_act_id] = np.array([-100.0, 100.0], dtype=np.float64)

        # Cached "rest" arm joint configuration (set at reset time from the 'home' keyframe). The IK uses this
        # as a null-space attractor so the wrist orientation does not drift during long position-only tracks.
        self._arm_rest_qpos: Optional[np.ndarray] = None
        self.ik_nullspace_gain = 0.05

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

        # Cache the home arm pose for null-space pulls in IK (prevents wrist drift during long descend/lift).
        self._arm_rest_qpos = self.data.qpos[self._arm_qpos_adr].astype(np.float64).copy()

        # Randomize object pose on the table top (sim-only). Z is set to (table_top + half_height)
        # so the cube starts exactly resting on the table and does not fall/bounce at t=0.
        obj_jnt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "object")
        if obj_jnt >= 0:
            adr = int(self.model.jnt_qposadr[obj_jnt])
            self.data.qpos[adr + 0] = 0.55 + 0.10 * (np.random.rand() - 0.5)
            self.data.qpos[adr + 1] = 0.08 + 0.10 * (np.random.rand() - 0.5)
            self.data.qpos[adr + 2] = self._object_rest_z
            self.data.qpos[adr + 3 : adr + 7] = np.array([1, 0, 0, 0], dtype=np.float64)

        mujoco.mj_forward(self.model, self.data)

        # Cache the post-reset object Z as the "ground" height for lift-detection.
        self._object_initial_z = float(self.data.xpos[self._obj_body_id][2])

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

        # Compute the IK joint-target in a SCRATCH pass (state saved/restored). We do NOT teleport
        # the joints into that target -- doing so would push the wrist/fingers through any obstacle
        # in the way (e.g. the cube being grasped). Instead we drive the Menagerie arm's stiff
        # position-controlled actuators (actuator1..7, gainprm=4500) toward this target and let
        # mj_step move the joints PHYSICALLY, respecting contacts.
        ik_target_q = self._compute_ik_target(self._ee_target)

        # Gripper: Menagerie mapping (255=open, 0=closed).
        grip_ctrl = 0.0 if g > 0.5 else 255.0

        # Send joint target controls (actuator1..7) and gripper (actuator8).
        self.data.ctrl[:7] = ik_target_q.astype(np.float32)
        self.data.ctrl[self._gripper_act_id] = grip_ctrl

        mujoco.mj_step(self.model, self.data)
        return self.observe()

    @property
    def object_initial_z(self) -> float:
        """Z height the object had right after the last reset (used as 'ground' baseline)."""
        return self._object_initial_z

    def fingers_in_contact_with_object(self) -> tuple[bool, bool]:
        """
        Return (left_in_contact, right_in_contact). A finger counts as 'in contact'
        if any of its geoms appears in any active contact pair with any of the
        object body's geoms.
        """
        left_hit = False
        right_hit = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1 = int(self.model.geom_bodyid[c.geom1])
            b2 = int(self.model.geom_bodyid[c.geom2])
            pair = (b1, b2)
            if self._obj_body_id in pair:
                other = b2 if b1 == self._obj_body_id else b1
                if other == self._left_finger_body_id:
                    left_hit = True
                elif other == self._right_finger_body_id:
                    right_hit = True
                if left_hit and right_hit:
                    break
        return left_hit, right_hit

    def is_object_grasped(self, min_lift_m: float = 0.04) -> bool:
        """
        Heuristic 'success' check: cube has been lifted by at least `min_lift_m` above
        its post-reset rest height AND both fingers are currently in contact with it.
        Use this to decide whether to keep an episode in the dataset.
        """
        left, right = self.fingers_in_contact_with_object()
        lifted = float(self.data.xpos[self._obj_body_id][2]) >= (self._object_initial_z + float(min_lift_m))
        return bool(left and right and lifted)

    def object_lift_height(self) -> float:
        """How far the object's center is above its post-reset rest height (meters)."""
        return float(self.data.xpos[self._obj_body_id][2]) - self._object_initial_z

    def _ee_pos(self) -> np.ndarray:
        # Use midpoint of left/right fingertip proxy points as TCP.
        # This better matches visible gripper center than the hand origin.
        if self._left_finger_body_id >= 0 and self._right_finger_body_id >= 0:
            p_l = self._body_point_world(self._left_finger_body_id, self._fingertip_local_offset)
            p_r = self._body_point_world(self._right_finger_body_id, self._fingertip_local_offset)
            return 0.5 * (p_l + p_r)

        return self.data.xpos[self._hand_body_id]

    def _body_point_world(self, body_id: int, local_offset: np.ndarray) -> np.ndarray:
        xmat = self.data.xmat[body_id].reshape(3, 3)
        return self.data.xpos[body_id] + xmat @ local_offset

    def _tcp_jacobian(self) -> np.ndarray:
        # Jacobian of fingertip midpoint TCP wrt all DoFs.
        if self._left_finger_body_id >= 0 and self._right_finger_body_id >= 0:
            jacp_l = np.zeros((3, self.model.nv), dtype=np.float64)
            jacr_l = np.zeros((3, self.model.nv), dtype=np.float64)
            jacp_r = np.zeros((3, self.model.nv), dtype=np.float64)
            jacr_r = np.zeros((3, self.model.nv), dtype=np.float64)
            p_l = self._body_point_world(self._left_finger_body_id, self._fingertip_local_offset)
            p_r = self._body_point_world(self._right_finger_body_id, self._fingertip_local_offset)
            mujoco.mj_jac(self.model, self.data, jacp_l, jacr_l, p_l, self._left_finger_body_id)
            mujoco.mj_jac(self.model, self.data, jacp_r, jacr_r, p_r, self._right_finger_body_id)
            return 0.5 * (jacp_l + jacp_r)

        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        p_h = self.data.xpos[self._hand_body_id]
        mujoco.mj_jac(self.model, self.data, jacp, jacr, p_h, self._hand_body_id)
        return jacp

    def _compute_ik_target(self, target_pos: np.ndarray) -> np.ndarray:
        """Compute desired arm joint angles to track ``target_pos`` without mutating sim state.

        We need the IK Jacobian computed at intermediate joint configurations, so the routine
        temporarily writes to ``data.qpos`` while iterating, then **restores** the original
        ``qpos``/``qvel`` before returning. The caller is expected to send the returned 7-vector
        to ``data.ctrl[:7]`` (the arm's position-controlled actuators) and run ``mj_step``;
        contacts are then resolved by physics, so the wrist/fingers cannot tunnel through
        objects in their path.
        """
        saved_qpos = self.data.qpos.copy()
        saved_qvel = self.data.qvel.copy()
        try:
            rest = self._arm_rest_qpos if self._arm_rest_qpos is not None else (
                self.data.qpos[self._arm_qpos_adr].astype(np.float64).copy()
            )

            for _ in range(self.ik_iters):
                cur = self._ee_pos().astype(np.float64)
                err = (target_pos.astype(np.float64) - cur)

                J_full = self._tcp_jacobian()
                J = J_full[:, self._arm_dof_adr]  # (3, 7)

                # Damped pseudoinverse: J^+ = J^T (J J^T + λI)^-1
                JJt = J @ J.T
                lam = (self.ik_damping ** 2) * np.eye(3, dtype=np.float64)
                JJt_lam_inv = np.linalg.solve(JJt + lam, np.eye(3, dtype=np.float64))
                J_pinv = J.T @ JJt_lam_inv  # (7, 3)

                dq_task = J_pinv @ err

                # Null-space attractor: pull joints toward home pose without disturbing the EE position.
                # dq_null = (I - J^+ J) * k * (rest - q)
                q = self.data.qpos[self._arm_qpos_adr].astype(np.float64)
                N = np.eye(7, dtype=np.float64) - J_pinv @ J
                dq_null = N @ (self.ik_nullspace_gain * (rest - q))

                dq = dq_task + dq_null
                dq = np.clip(dq, -0.2, 0.2)

                # Stop early once both task error and null-space pull are tiny.
                if float(np.linalg.norm(err)) < 1e-4 and float(np.linalg.norm(dq_null)) < 1e-4:
                    break

                q_new = q + self.ik_step_size * dq

                # Respect joint limits.
                for i, jnt_id in enumerate(self._arm_jnt_ids):
                    lo, hi = self.model.jnt_range[jnt_id]
                    if lo < hi:
                        q_new[i] = float(np.clip(q_new[i], lo, hi))

                self.data.qpos[self._arm_qpos_adr] = q_new
                mujoco.mj_forward(self.model, self.data)

            target_q = self.data.qpos[self._arm_qpos_adr].astype(np.float64).copy()
        finally:
            # Restore the simulator state so the IK was a pure scratch computation.
            self.data.qpos[:] = saved_qpos
            self.data.qvel[:] = saved_qvel
            mujoco.mj_forward(self.model, self.data)

        return target_q

