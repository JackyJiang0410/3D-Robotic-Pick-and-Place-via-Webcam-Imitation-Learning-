# Stage 1: Webcam -> MediaPipe -> `[x, y, z, g]`

This repository currently implements **Stage 1** of your pipeline:

`Laptop Webcam -> MediaPipe Hand Tracking -> Extract low-dimensional control signal [x, y, z, g]`.

## What is implemented

- Real-time webcam capture (`OpenCV`)
- Hand keypoint tracking (`MediaPipe Hands`)
- Action abstraction:
  - `x`: horizontal wrist position (normalized to `[-1, 1]`)
  - `y`: vertical wrist position (normalized to `[-1, 1]`)
  - `z`: depth proxy from palm scale change (normalized and clipped)
  - `g`: gripper command (`1.0` for pinch/close, `0.0` for open)
- Temporal smoothing for stable control
- JSON streaming output for downstream modules
- Optional visualization window with landmarks and live signal
- Works with both MediaPipe APIs:
  - legacy `mp.solutions` (older versions)
  - newer `mediapipe.tasks` (e.g. Python 3.12 environment)

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Stage 1

```bash
python run_stage1.py
```

On first run with the newer `mediapipe.tasks` backend, the hand-landmarker model file is downloaded automatically and cached under `stage1_sensory_input/models/`.

Options:

- `--camera-id`: webcam index (default `0`)
- `--width`, `--height`: capture size
- `--pinch-threshold`: pinch ratio threshold for `g`
- `--print-hz`: max JSON print rate
- `--no-preview`: headless mode (no OpenCV window)

Example:

```bash
python run_stage1.py --camera-id 0 --width 1280 --height 720 --pinch-threshold 0.4
```

## Output format

The script prints JSON records such as:

```json
{"x": -0.12, "y": 0.31, "z": -0.08, "g": 1.0, "t": 1776181275.215}
```

- `x, y, z, g` are your extracted action vector
- `t` is timestamp (seconds)

## Integration point for Stage 2

Use `stage1_sensory_input.Stage1Extractor` in your Data Collection loop:

1. Read frame from webcam.
2. Call `process_frame(frame)`.
3. If signal exists, read `signal.as_vector()` to get `[x, y, z, g]`.
4. Feed that vector into MuJoCo as demonstration action.

This cleanly bridges your **Sensory Input** stage into **Data Collection**.
