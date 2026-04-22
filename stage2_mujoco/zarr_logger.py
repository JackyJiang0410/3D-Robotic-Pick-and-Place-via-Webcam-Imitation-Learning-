from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import zarr


@dataclass
class EpisodeHandle:
    group: zarr.Group
    step: int


class ZarrTrajectoryLogger:
    """
    Append-only episode logger.

    Layout:
      root/
        meta/
        episodes/
          ep_000000/
            obs (T, obs_dim) float32
            act (T, act_dim) float32
            t   (T,) float64
            img (T, H, W, 3) uint8        # optional, only present when image_shape was passed
    """

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.store = zarr.DirectoryStore(str(self.root_dir))
        self.root = zarr.group(store=self.store, overwrite=False)
        self.episodes = self.root.require_group("episodes")
        self.meta = self.root.require_group("meta")
        self.meta.attrs.setdefault("created_at", time.time())

    def start_episode(
        self,
        obs_dim: int,
        act_dim: int,
        name: Optional[str] = None,
        image_shape: Optional[Tuple[int, int, int]] = None,
        image_camera: Optional[str] = None,
    ) -> EpisodeHandle:
        """Start a new episode group.

        Pass ``image_shape=(H, W, 3)`` to also create an ``img`` dataset that can be appended to
        per step. ``image_camera`` is stored as an episode attribute for downstream consumers.
        """
        if name is None:
            existing = sorted([k for k in self.episodes.group_keys() if k.startswith("ep_")])
            next_id = int(existing[-1].split("_")[-1]) + 1 if existing else 0
            name = f"ep_{next_id:06d}"

        g = self.episodes.require_group(name)
        g.attrs["obs_dim"] = int(obs_dim)
        g.attrs["act_dim"] = int(act_dim)
        g.attrs["started_at"] = time.time()

        chunks = (1024, obs_dim) if obs_dim > 0 else (1024, 1)
        g.require_dataset("obs", shape=(0, obs_dim), chunks=chunks, dtype=np.float32, overwrite=True)
        g.require_dataset("act", shape=(0, act_dim), chunks=(1024, act_dim), dtype=np.float32, overwrite=True)
        g.require_dataset("t", shape=(0,), chunks=(2048,), dtype=np.float64, overwrite=True)

        if image_shape is not None:
            H, W, C = int(image_shape[0]), int(image_shape[1]), int(image_shape[2])
            # Cap chunk-size in time so large frames don't make multi-MB chunks.
            time_chunk = max(1, min(64, 4 * 1024 * 1024 // max(1, H * W * C)))
            g.require_dataset(
                "img",
                shape=(0, H, W, C),
                chunks=(time_chunk, H, W, C),
                dtype=np.uint8,
                overwrite=True,
            )
            g.attrs["image_shape"] = [H, W, C]
            g.attrs["image_camera"] = str(image_camera) if image_camera is not None else ""

        return EpisodeHandle(group=g, step=0)

    def append(
        self,
        ep: EpisodeHandle,
        obs_vec: np.ndarray,
        act_vec: np.ndarray,
        t: float,
        img: Optional[np.ndarray] = None,
    ) -> None:
        obs = np.asarray(obs_vec, dtype=np.float32).reshape(1, -1)
        act = np.asarray(act_vec, dtype=np.float32).reshape(1, -1)

        ds_obs = ep.group["obs"]
        ds_act = ep.group["act"]
        ds_t = ep.group["t"]

        ds_obs.append(obs)
        ds_act.append(act)
        ds_t.append(np.asarray([t], dtype=np.float64))

        if img is not None:
            if "img" not in ep.group:
                raise RuntimeError(
                    "append() received an image but episode was started without image_shape. "
                    "Pass image_shape=... to start_episode()."
                )
            arr = np.asarray(img, dtype=np.uint8)
            if arr.ndim != 3:
                raise ValueError(f"Expected image with shape (H, W, 3), got {arr.shape}")
            ep.group["img"].append(arr.reshape(1, *arr.shape))

        ep.step += 1

