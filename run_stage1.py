from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from stage1_sensory_input import Stage1Config, Stage1Extractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 1 webcam -> [x, y, z, g] extraction.")
    parser.add_argument("--camera-id", type=int, default=0, help="OpenCV webcam index.")
    parser.add_argument("--width", type=int, default=1280, help="Capture width.")
    parser.add_argument("--height", type=int, default=720, help="Capture height.")
    parser.add_argument("--pinch-threshold", type=float, default=0.42, help="Pinch ratio threshold for gripper close.")
    parser.add_argument(
        "--print-hz",
        type=float,
        default=10.0,
        help="Maximum rate for JSON console output.",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Run without OpenCV preview window.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output JSONL path for extracted signals.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Optional auto-stop duration in seconds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Stage1Config(
        camera_id=args.camera_id,
        frame_width=args.width,
        frame_height=args.height,
        pinch_threshold=args.pinch_threshold,
    )
    extractor = Stage1Extractor(config)
    cap = extractor.open_camera()
    start_t = time.time()

    out_file = None
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_file = out_path.open("a", encoding="utf-8")

    print("Stage 1 started. Press 'q' in preview window to quit.")
    print("Output schema: {'x': float, 'y': float, 'z': float, 'g': float, 't': float}")
    if out_file is not None:
        print(f"Saving JSONL output to: {out_file.name}")

    last_print_t = 0.0
    min_print_dt = 1.0 / max(args.print_hz, 1e-6)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Failed to read frame from webcam.")

            vis, signal = extractor.process_frame(frame)
            now = time.time()
            if args.max_seconds is not None and (now - start_t) >= args.max_seconds:
                print(f"Reached --max-seconds={args.max_seconds}; exiting.")
                break

            if signal is not None and (now - last_print_t) >= min_print_dt:
                payload = {
                    "x": round(signal.x, 4),
                    "y": round(signal.y, 4),
                    "z": round(signal.z, 4),
                    "g": round(signal.g, 1),
                    "t": round(signal.timestamp, 3),
                }
                payload_json = json.dumps(payload)
                print(payload_json)
                if out_file is not None:
                    out_file.write(payload_json + "\n")
                    out_file.flush()
                last_print_t = now

            if not args.no_preview:
                if signal is not None:
                    status = f"x={signal.x:+.2f} y={signal.y:+.2f} z={signal.z:+.2f} g={signal.g:.0f}"
                else:
                    status = "No hand detected"

                cv2.putText(
                    vis,
                    status,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Stage 1 - Webcam Hand Extractor", vis)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
    finally:
        if out_file is not None:
            out_file.close()
        cap.release()
        extractor.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
