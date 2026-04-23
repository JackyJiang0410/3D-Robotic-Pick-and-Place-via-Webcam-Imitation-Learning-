# Zarr EDA: `phase1_vision.zarr`

## Location
- Path: `/Users/haojiang/Desktop/CS 395T/CS395T_Project/data/datasets/phase1_vision.zarr`
- Output: `/Users/haojiang/Desktop/CS 395T/CS395T_Project/Results/eda_phase1_vision`

## Corpus overview
- **Episodes (demos / reps)**: 17
- **Total transitions (rows)**: 6,635
- **Obs dim** (`obs`): 41
- **Act dim** (`act`): 3
- **Labeled successes** (`attrs['success']`): 17 / 17 (100.0%)

### Episode length (steps per episode)
- min=22 | max=965 | mean=390.3 | median=361.0 | std=209.1

### Root `meta` attrs
  - **created_at**: `1776869820.0584488`

### Episode-level metadata (counts)
- **`image_camera`**: {'agent_view': 17}
- **`image_shape`**: {(128, 128, 3): 17}

### `peak_lift_m` (where logged)
- n=17 | min=0.2673 | max=0.3077 | mean=0.2978 | median=0.3067

### Wall-clock `t` spacing (within episodes)
- n diffs=6,618 | mean=0.0000s | median=0.0000s | p95=0.0000s

## Observation vector layout
Concatenation order matches `PandaObs.as_vector()` (see `stage2_mujoco/panda_env.py`).

| block | index range | dims | mean | std | p5 … p95 |
|---|---:|---:|---:|---:|---|
| `qpos` `[0:16)` | 16 | 0.08669 | 0.6605 | -1.502 … 1.521 |
| `qvel` `[16:31)` | 15 | 0.01849 | 0.2152 | -0.3002 … 0.4181 |
| `ee_pos` `[31:34)` | 3 | 0.3616 | 0.259 | -0.06597 … 0.6159 |
| `obj_pos` `[34:37)` | 3 | 0.33 | 0.1968 | 0.06935 … 0.6238 |
| `target_pos` `[37:40)` | 3 | 0.2517 | 0.268 | -0.1 … 0.55 |
| `gripper_open` `[40:41)` | 1 | 0.04 | 5.248e-06 | 0.03999 … 0.04001 |

## Actions `[dx, dy, g]`
- `dx`, `dy`: EE delta per control step (meters; scaled at collection).
- `g`: gripper command (>0.5 treated as close in env).

- **Share of steps with |dx,dy| ≈ 0**: 5.5%
- **Share of steps with g > 0.5 (close)**: 0.0%
- **Note:** logged `g` is identically 0 for every step in this corpus; pinch / close is likely from scripted auto-grasp (not written into `act`).

## Geometry in the logged states
- **||ee − obj||_2 (xyz)** — min=0.1852 | p50=0.2159 | p95=0.3318 | max=0.3833 m
- **||ee − obj||_2 (xy)** — min=0.0012 | p50=0.1012 | p95=0.2725 | max=0.3339 m

## Images
- **RGB mean** (0–255, over all pixels): (122.88, 135.33, 147.71)
- **Total image pixels** (T×H×W): 108,707,840
- **Luminance histogram**: 2,000,000 subsampled pixels → `luminance_hist.png`

## Files written
- `SUMMARY.md` (this report)
- `episodes.csv`, `obs_per_dim.csv`, `act_per_dim.csv`
- `episode_lengths.png`, `distributions.png` (+ `luminance_hist.png` if images)
