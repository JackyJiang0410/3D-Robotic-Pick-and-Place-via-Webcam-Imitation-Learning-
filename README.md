# Webcam gesture teleoperation → MuJoCo (Franka Panda) → Zarr demos → behavior cloning (sim-only)

This repository implements a **minimal sim-only imitation-learning pipeline**:

1. **Stage 1**: webcam → MediaPipe → low-dimensional hand signal `[x, y, z, g]`
2. **Stage 2 (Phase 1)**: MuJoCo Franka Panda pick/place scene → teleoperation → dataset logging
3. **Training**: ridge-regression behavior cloning from Zarr
4. **Evaluation**: run the learned policy in MuJoCo

## Requirements

- **Python**: 3.10+ recommended (matches common `mediapipe` wheels)
- **OS**: macOS is supported; some MuJoCo viewer paths require `mjpython`
- **Hardware**: laptop webcam

### Python packages

Install from `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

Key dependencies:

- `mediapipe`, `opencv-python`, `numpy`
- `mujoco`, `glfw`, `zarr`

## Robot assets (Franka Panda / MuJoCo Menagerie)

This repository **vendors** the MuJoCo Menagerie Franka Panda model at:

- `assets/robots/mujoco_menagerie/franka_emika_panda/`

This is intentional so collaborators can run the project **without a separate download step**.

### Mesh files

Menagerie ships meshes under `franka_emika_panda/assets/`. This repository also keeps the same mesh files at the `franka_emika_panda/` top level so MuJoCo can resolve mesh filenames when `panda.xml` is included from `assets/mujoco/panda_pick_place_scene.xml`.

If you refresh Menagerie from upstream, re-copy meshes if needed:

```bash
cp -n assets/robots/mujoco_menagerie/franka_emika_panda/assets/* assets/robots/mujoco_menagerie/franka_emika_panda/
```

### Optional: refresh Menagerie from GitHub

If you prefer to update the Panda files from source instead of using the vendored copy:

```bash
mkdir -p assets/robots && cd assets/robots
git clone --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git mujoco_menagerie
cd mujoco_menagerie
git sparse-checkout set franka_emika_panda
```

### Licensing

Menagerie models have their own license files inside `franka_emika_panda/` (see `LICENSE` there).

## Stage 1: webcam → `[x, y, z, g]`

```bash
python run_stage1.py --camera-id 0 --width 1280 --height 720
```

`z` is configurable in code via `Stage1Config.z_mode` (defaults to `mp_z_spread` in `stage1_sensory_input/extractor.py`).

Outputs JSON lines to stdout; optional JSONL logging:

```bash
python run_stage1.py --out data/stage1.jsonl --max-seconds 30
```

`data/` is ignored by git (see `.gitignore`).

## Phase 1: collect demonstrations (teleop) → Zarr

This records tuples `(observation, action, time)` where:

- **Observation** is a low-dimensional vector from MuJoCo state (joints/velocities + selected poses)
- **Action** is **`[dx, dy, dz, g]`** (end-effector delta command + grip)

Run:

```bash
mjpython run_collect_phase1.py --viewer --preview --camera-id 0 --seconds 120 --dataset data/phase1_panda_demos.zarr
```

Useful flags:

- `--delta-scale`: scales `dx, dy` from hand `x, y`
- `--z-scale`: scales `dz` from hand `z` (defaults to `--delta-scale` if omitted)
- `--z-mode`: `palm_scale | mp_z_spread | mp_z_wrist | reach_2d`
- `--print-hz`: terminal print rate
- `--verbose-numbers`: print numeric `x,y,z` and `dx,dy,dz` in addition to compact summaries
- `--preview`: OpenCV webcam window (may conflict with MuJoCo viewer on some macOS setups)

Quit preview window: press `q` in the OpenCV window.

## Train behavior cloning (ridge regression)

```bash
python run_train_bc_phase1.py --dataset data/phase1_panda_demos.zarr --out data/phase1_panda_bc_policy.npz
```

## Evaluate the policy in MuJoCo

```bash
mjpython run_eval_bc_phase1.py --viewer --policy data/phase1_panda_bc_policy.npz --seconds 30
```

## Optional: Panda-only viewer

```bash
mjpython run_view_panda.py
```

## Repository layout (high level)

- `run_stage1.py`: Stage 1 runner
- `stage1_sensory_input/`: Stage 1 implementation
- `assets/mujoco/panda_pick_place_scene.xml`: Panda pick/place scene
- `assets/robots/mujoco_menagerie/franka_emika_panda/`: Menagerie Panda MJCF + meshes
- `stage2_mujoco/`: MuJoCo environment + Zarr logger
- `run_collect_phase1.py`, `run_train_bc_phase1.py`, `run_eval_bc_phase1.py`: Phase 1 scripts

## Notes

- **Camera permissions**: macOS must allow camera access for the terminal app running Python.
- **Dataset size**: Zarr datasets can be large; keep them under `data/` (gitignored) or use Git LFS if you intentionally version data.
