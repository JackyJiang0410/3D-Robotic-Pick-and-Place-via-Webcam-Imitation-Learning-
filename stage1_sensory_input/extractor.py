from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import Optional, Tuple
from urllib.error import URLError
from urllib.request import urlopen

import cv2
import mediapipe as mp
import numpy as np


@dataclasses.dataclass
class Stage1Config:
    camera_id: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    max_num_hands: int = 1
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.6
    pinch_threshold: float = 0.42
    smoothing_alpha: float = 0.35
    max_depth_magnitude: float = 1.0
    # How to derive the teleop "z" channel from a monocular hand track.
    # - palm_scale: legacy depth proxy from palm apparent size (noisy for in/out motion)
    # - mp_z_spread: uses MediaPipe landmark z spread (thumb-index vs wrist), often better for reach
    # - mp_z_wrist: uses wrist landmark z vs a slow reference (works if z is non-degenerate)
    # - reach_2d: 2D-only proxy from wrist-to-fingertip reach in the image plane
    z_mode: str = "mp_z_spread"
    model_asset_path: Optional[str] = None
    model_download_url: str = (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    )


@dataclasses.dataclass
class HandControlSignal:
    x: float
    y: float
    z: float
    g: float
    timestamp: float
    hand_present: bool
    pinch_ratio: float

    def as_vector(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, self.g], dtype=np.float32)


class Stage1Extractor:
    """
    Stage 1: Laptop Webcam -> MediaPipe -> Extract [x, y, z, g].

    - x, y: wrist-centered image position in [-1, 1]
    - z: depth / reach proxy (see Stage1Config.z_mode)
    - g: gripper command from pinch gesture (1 close, 0 open)
    """

    def __init__(self, config: Stage1Config):
        self.config = config
        self._backend = "solutions" if hasattr(mp, "solutions") else "tasks"
        self._task_timestamp_ms = 0

        if self._backend == "solutions":
            self._init_solutions_backend()
        else:
            from mediapipe.tasks import python as mp_tasks_python
            from mediapipe.tasks.python import vision as mp_tasks_vision

            try:
                model_path = self._ensure_task_model()
                base_options = mp_tasks_python.BaseOptions(model_asset_path=str(model_path))
                options = mp_tasks_vision.HandLandmarkerOptions(
                    base_options=base_options,
                    num_hands=config.max_num_hands,
                    min_hand_detection_confidence=config.min_detection_confidence,
                    min_hand_presence_confidence=config.min_tracking_confidence,
                    min_tracking_confidence=config.min_tracking_confidence,
                    running_mode=mp_tasks_vision.RunningMode.VIDEO,
                )
                self._task_hand_landmarker = mp_tasks_vision.HandLandmarker.create_from_options(options)
                self._mp_hands = None
                self._hands = None
                self._drawer = None
            except Exception:
                # Some macOS/OpenGL configurations fail to initialize the task backend.
                # Fall back to the legacy CPU-oriented solution so the teleop loop still runs.
                self._backend = "solutions"
                self._init_solutions_backend()

        self._prev_xyzg: Optional[np.ndarray] = None
        self._depth_reference: Optional[float] = None
        self._z_signal_reference: Optional[float] = None
        self._reach_reference: Optional[float] = None

    def _init_solutions_backend(self) -> None:
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.config.max_num_hands,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        self._drawer = mp.solutions.drawing_utils
        self._task_hand_landmarker = None

    @staticmethod
    def _landmark_xy(landmarks, index: int) -> np.ndarray:
        if hasattr(landmarks, "landmark"):
            p = landmarks.landmark[index]
        else:
            p = landmarks[index]
        return np.array([p.x, p.y], dtype=np.float32)

    @staticmethod
    def _landmark_xyz(landmarks, index: int) -> np.ndarray:
        if hasattr(landmarks, "landmark"):
            p = landmarks.landmark[index]
        else:
            p = landmarks[index]
        z = float(getattr(p, "z", 0.0))
        xy = Stage1Extractor._landmark_xy(landmarks, index)
        return np.array([xy[0], xy[1], z], dtype=np.float32)

    def _ensure_task_model(self) -> Path:
        if self.config.model_asset_path:
            model_path = Path(self.config.model_asset_path).expanduser().resolve()
            if not model_path.exists():
                raise RuntimeError(f"Configured model_asset_path does not exist: {model_path}")
            return model_path

        package_dir = Path(__file__).resolve().parent
        model_dir = package_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "hand_landmarker.task"

        if model_path.exists():
            return model_path

        try:
            with urlopen(self.config.model_download_url, timeout=30) as response:
                data = response.read()
        except URLError as exc:
            raise RuntimeError(
                "Failed to download MediaPipe hand_landmarker model. "
                "Provide a local model path via Stage1Config(model_asset_path='...')."
            ) from exc

        model_path.write_bytes(data)
        return model_path

    @staticmethod
    def _distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    def _compute_signal_from_landmarks(self, landmarks) -> Tuple[np.ndarray, float]:
        wrist = self._landmark_xy(landmarks, 0)
        middle_mcp = self._landmark_xy(landmarks, 9)
        index_mcp = self._landmark_xy(landmarks, 5)
        pinky_mcp = self._landmark_xy(landmarks, 17)
        thumb_tip = self._landmark_xy(landmarks, 4)
        index_tip = self._landmark_xy(landmarks, 8)

        # Center normalized image coordinates to [-1, 1].
        x = float(np.clip((wrist[0] - 0.5) * 2.0, -1.0, 1.0))
        y = float(np.clip((wrist[1] - 0.5) * 2.0, -1.0, 1.0))

        palm_width = max(self._distance(index_mcp, pinky_mcp), 1e-6)
        palm_length = max(self._distance(wrist, middle_mcp), 1e-6)
        palm_scale = 0.5 * (palm_width + palm_length)

        mode = (self.config.z_mode or "palm_scale").lower()
        if mode == "palm_scale":
            if self._depth_reference is None:
                self._depth_reference = palm_scale
            else:
                self._depth_reference = 0.98 * self._depth_reference + 0.02 * palm_scale
            z_raw = (self._depth_reference - palm_scale) / max(self._depth_reference, 1e-6)
        elif mode == "mp_z_spread":
            w = self._landmark_xyz(landmarks, 0)
            tt = self._landmark_xyz(landmarks, 4)
            it = self._landmark_xyz(landmarks, 8)
            # Relative depth cues: fingertip z minus wrist z (hand model coords).
            spread = float((tt[2] - w[2]) + (it[2] - w[2])) * 0.5
            if self._z_signal_reference is None:
                self._z_signal_reference = spread
            else:
                self._z_signal_reference = 0.98 * self._z_signal_reference + 0.02 * spread
            denom = max(abs(self._z_signal_reference), 1e-3)
            z_raw = (spread - self._z_signal_reference) / denom
        elif mode == "mp_z_wrist":
            wz = float(self._landmark_xyz(landmarks, 0)[2])
            if self._z_signal_reference is None:
                self._z_signal_reference = wz
            else:
                self._z_signal_reference = 0.98 * self._z_signal_reference + 0.02 * wz
            denom = max(abs(self._z_signal_reference), 1e-3)
            z_raw = (wz - self._z_signal_reference) / denom
        elif mode == "reach_2d":
            reach = float(self._distance(wrist, index_tip) / max(palm_scale, 1e-6))
            if self._reach_reference is None:
                self._reach_reference = reach
            else:
                self._reach_reference = 0.98 * self._reach_reference + 0.02 * reach
            z_raw = (reach - self._reach_reference) / max(self._reach_reference, 1e-6)
        else:
            raise ValueError(f"Unknown z_mode: {self.config.z_mode!r}")

        z = float(np.clip(z_raw, -self.config.max_depth_magnitude, self.config.max_depth_magnitude))

        pinch_dist = self._distance(thumb_tip, index_tip)
        pinch_ratio = pinch_dist / palm_scale
        g = 1.0 if pinch_ratio < self.config.pinch_threshold else 0.0

        return np.array([x, y, z, g], dtype=np.float32), pinch_ratio

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Optional[HandControlSignal]]:
        signal: Optional[HandControlSignal] = None
        if self._backend == "solutions":
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = self._hands.process(rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                xyzg_raw, pinch_ratio = self._compute_signal_from_landmarks(hand_landmarks)
                signal = self._smooth_and_pack(xyzg_raw, pinch_ratio)
                self._drawer.draw_landmarks(
                    frame_bgr,
                    hand_landmarks,
                    self._mp_hands.HAND_CONNECTIONS,
                )
            else:
                self._prev_xyzg = None
                self._depth_reference = None
                self._z_signal_reference = None
                self._reach_reference = None
        else:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            now_ms = int(time.time() * 1000)
            timestamp_ms = max(now_ms, self._task_timestamp_ms + 1)
            self._task_timestamp_ms = timestamp_ms
            results = self._task_hand_landmarker.detect_for_video(mp_image, timestamp_ms)

            if results.hand_landmarks:
                hand_landmarks = results.hand_landmarks[0]
                xyzg_raw, pinch_ratio = self._compute_signal_from_landmarks(hand_landmarks)
                signal = self._smooth_and_pack(xyzg_raw, pinch_ratio)
                self._draw_task_landmarks(frame_bgr, hand_landmarks)
            else:
                self._prev_xyzg = None
                self._depth_reference = None
                self._z_signal_reference = None
                self._reach_reference = None

        return frame_bgr, signal

    def _smooth_and_pack(self, xyzg_raw: np.ndarray, pinch_ratio: float) -> HandControlSignal:
        if self._prev_xyzg is None:
            xyzg_smoothed = xyzg_raw
        else:
            alpha = self.config.smoothing_alpha
            xyzg_smoothed = alpha * xyzg_raw + (1.0 - alpha) * self._prev_xyzg

        # Keep g discrete after smoothing.
        xyzg_smoothed[3] = 1.0 if xyzg_raw[3] > 0.5 else 0.0
        self._prev_xyzg = xyzg_smoothed

        return HandControlSignal(
            x=float(xyzg_smoothed[0]),
            y=float(xyzg_smoothed[1]),
            z=float(xyzg_smoothed[2]),
            g=float(xyzg_smoothed[3]),
            timestamp=time.time(),
            hand_present=True,
            pinch_ratio=float(pinch_ratio),
        )

    def _draw_task_landmarks(self, frame_bgr: np.ndarray, landmarks) -> None:
        hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17),
        ]

        h, w = frame_bgr.shape[:2]
        points = []
        for lm in landmarks:
            x = int(np.clip(lm.x * w, 0, w - 1))
            y = int(np.clip(lm.y * h, 0, h - 1))
            points.append((x, y))
            cv2.circle(frame_bgr, (x, y), 3, (0, 255, 0), -1)

        for a, b in hand_connections:
            pa = points[a]
            pb = points[b]
            cv2.line(frame_bgr, pa, pb, (0, 200, 255), 1)

    def open_camera(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.config.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        if not cap.isOpened():
            raise RuntimeError(
                f"Unable to open webcam at camera_id={self.config.camera_id}. "
                "Check camera permissions/device index."
            )
        return cap

    def close(self) -> None:
        if self._backend == "solutions":
            self._hands.close()
        elif self._task_hand_landmarker is not None:
            self._task_hand_landmarker.close()
