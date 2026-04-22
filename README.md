# Webcam → MuJoCo Franka Panda → Zarr → Behavior Cloning (sim-only)

A minimal **imitation-learning pipeline** for a 2D pick-and-place task on a simulated Franka Panda:

1. **Stage 1 — Sensory input**: laptop webcam → MediaPipe → low-dim hand signal `[x, y, g]`
2. **Stage 2 — Phase 1 (data collection)**: webcam teleoperation in MuJoCo → record `(obs, act, [img])` per step → Zarr
3. **Training**: behavior cloning from the Zarr dataset
4. **Evaluation**: roll out the learned policy in MuJoCo

The user controls only the end-effector **X/Y** with their hand. A **pinch** triggers a hardcoded auto sequence (descend → close → lift → place → return), and **only successful pick attempts are saved** as episodes.

---

## Requirements

- **Python**: 3.10+ (matches common `mediapipe` wheels)
- **OS**: macOS supported. The MuJoCo passive viewer requires `mjpython` on macOS.
- **Hardware**: any laptop webcam

### Install

```bash
python -m pip install -r requirements.txt
```

Key deps: `mediapipe`, `opencv-python`, `numpy`, `mujoco`, `glfw`, `zarr`.

### Robot assets (vendored)

The MuJoCo Menagerie Franka Panda model is **vendored** at `assets/robots/mujoco_menagerie/franka_emika_panda/` so you do not need a separate download step. The pick-place scene that loads it is at `assets/mujoco/panda_pick_place_scene.xml`.

If you ever refresh the meshes from upstream Menagerie, re-flatten them so MJCF mesh paths still resolve:

```bash
cp -n assets/robots/mujoco_menagerie/franka_emika_panda/assets/* \
      assets/robots/mujoco_menagerie/franka_emika_panda/
```

---

## Stage 1: webcam → `[x, y, g]`

Quick sanity check that MediaPipe + your webcam are working:

```bash
python run_stage1.py --camera-id 0 --width 1280 --height 720
```

Stream gestures to a JSONL file:

```bash
python run_stage1.py --out data/stage1.jsonl --max-seconds 30
```

`data/` is gitignored.

---

## Phase 1: collect demonstrations (the canonical command)

This is the command currently used to collect data, including per-step RGB images for vision-based IL:

```bash
mjpython run_collect_phase1.py \
  --viewer --preview \
  --dataset data/phase1_vision.zarr \
  --save-images --image-camera agent_view --image-size 128x128 \
  --seconds 600
```

> On macOS you **must** launch with `mjpython` (not plain `python`) when `--viewer` is set, otherwise the MuJoCo passive viewer cannot attach to the main thread.

State-only variant (no images, smaller dataset):

```bash
mjpython run_collect_phase1.py --viewer --preview \
  --dataset data/phase1_panda_demos.zarr --seconds 600
```

### How collection works

- One **episode** = one pick attempt = "from `env.reset()` until the auto sequence completes."
- Recording **starts immediately on reset** (so your **teleop approach** — moving the EE above the cube — is captured), and **stops the moment grasp is confirmed** (cube lifted ≥ `--success-lift`). The hardcoded place phase still plays so you can see the cube land, but those frames are intentionally **not** saved (nothing for the policy to learn there).
- Episodes are **only persisted to Zarr if grasp succeeded**. Failed attempts are discarded by default. Use `--save-failures` to keep them.
- After each attempt the env resets with the cube respawned at a slightly randomized XY (so the dataset covers a small distribution rather than one fixed pose).

### What's in the dataset

Each saved episode is a Zarr group `episodes/ep_NNNNNN` with:

| Array | Shape          | Dtype   | Description                                       |
| ----- | -------------- | ------- | ------------------------------------------------- |
| `obs` | `(T, obs_dim)` | float32 | Low-dim state vector (joints, EE pose, cube pose) |
| `act` | `(T, 3)`       | float32 | Action `[dx, dy, g]`                              |
| `t`   | `(T,)`         | float64 | Wall-clock timestamps                             |
| `img` | `(T, H, W, 3)` | uint8   | RGB frames (only when `--save-images` is on)      |

Episode-level attributes: `success` (bool), `peak_lift_m` (float), `image_camera` (when images are recorded).

### Hand → robot mapping

| Hand action     | Robot effect                                                |
| --------------- | ----------------------------------------------------------- |
| Move hand left  | EE moves +X                                                 |
| Move hand right | EE moves −X                                                 |
| Move hand up    | EE moves +Y                                                 |
| Move hand down  | EE moves −Y                                                 |
| Pinch fingers   | Trigger the auto descend→close→lift→place→return sequence   |

Quit the OpenCV preview window with `q`.

### Useful flags

**Recording**

- `--save-images` — record per-step RGB into the `img` dataset (off by default)
- `--image-camera` — MuJoCo camera name (default `agent_view`; `topdown` is also defined)
- `--image-size` — `WxH`, e.g. `128x128`, `256x256`
- `--success-lift` — lift threshold (m) above cube rest height to count as success (default `0.05`)
- `--save-failures` — also keep failed pick attempts
- `--reset-between-attempts` — reset env (and respawn cube) between attempts (on by default)

**Teleop**

- `--delta-scale` — meters per step for full-scale hand X/Y → EE `dx, dy`
- `--camera-id` — webcam index
- `--preview` — show OpenCV webcam window (on macOS this can occasionally conflict with the MuJoCo viewer; omit if it crashes)
- `--print-hz`, `--verbose-numbers` — terminal output controls

**Auto pick-place tuning**

- `--auto-time-scale` — multiplier on phase durations (default `10.0`; **larger = slower**). Bump down to ~5 for faster runs, up to ~15 for very deliberate motion.
- `--auto-ease` — `smoothstep` (default) or `linear`
- `--auto-grasp-offset` — z offset (m) above cube center at the descend target (default `0.0` = TCP aims at cube center for the most secure grip)
- `--snap-to-cube` / `--no-snap-to-cube` — on pinch, snap the descend XY to the cube's current XY so the grip is centered (on by default; turn off if you want the demo to follow your hand exactly)
- `--auto-table-top-z`, `--auto-table-clearance-z` — safety floor that prevents the descend from clipping into the table

The full auto sequence is:

```
descend → close → hold → lift → return_pre → move_home → lower_home → open → raise_home → back_to_pre → settle
```

Recording stops at the first moment grasp is confirmed (typically partway through `lift`).

---

## Train behavior cloning

```bash
python run_train_bc_phase1.py \
  --dataset data/phase1_vision.zarr \
  --out data/phase1_panda_bc_policy.npz
```

---

## Evaluate the policy in MuJoCo

```bash
mjpython run_eval_bc_phase1.py \
  --viewer \
  --policy data/phase1_panda_bc_policy.npz \
  --seconds 30
```

---

## Optional: Panda-only viewer

Useful for visually checking the model loads and the camera framings look right:

```bash
mjpython run_view_panda.py
```

---

## Repository layout

```
robotics_2d_test/
├── run_stage1.py                 # webcam → MediaPipe demo
├── run_collect_phase1.py         # teleop + dataset collection (Phase 1)
├── run_train_bc_phase1.py        # behavior cloning trainer
├── run_eval_bc_phase1.py         # policy evaluation in MuJoCo
├── run_view_panda.py             # standalone Panda viewer
├── stage1_sensory_input/         # MediaPipe extractor
│   └── extractor.py
├── stage2_mujoco/                # MuJoCo env + Zarr logger
│   ├── panda_env.py              # Panda IK env, grasp/lift detection
│   └── zarr_logger.py            # episode group writer
├── assets/
│   ├── mujoco/
│   │   └── panda_pick_place_scene.xml
│   └── robots/mujoco_menagerie/franka_emika_panda/   # vendored MJCF + meshes
├── data/                         # gitignored — datasets and policies live here
├── requirements.txt
└── README.md
```

---

## Troubleshooting

- **`mujoco` viewer error / hang on macOS** — you must launch with `mjpython`, not `python`, whenever `--viewer` is set.
- **Camera not opening** — grant the terminal app camera permission in System Settings → Privacy & Security → Camera.
- **OpenCV preview crashes alongside the MuJoCo viewer** — drop `--preview`. The data collection still works headlessly; you'll just lose the on-screen webcam HUD.
- **Robot looks like it's clipping the cube briefly** — minor visual artifact only; the physical contact is correct (the IK runs in a scratch pass and the arm is driven through the position-controlled actuators, so collisions are respected by the physics engine).
- **Episode says `DISCARDED`** — that attempt did not meet the success criterion (`cube lifted ≥ --success-lift` AND both fingers in contact). Try again, or pass `--save-failures` if you also want the failed traces.
- **Dataset looks too small** — by design, only the **approach + descend + close + initial lift** portion of each episode is saved. The hardcoded place phase is intentionally not recorded.

---

## Notes on data versioning

Zarr datasets can be large, especially with `--save-images`. They live under `data/` which is gitignored. If you intentionally want to version a small reference dataset, use Git LFS rather than committing the raw arrays.
