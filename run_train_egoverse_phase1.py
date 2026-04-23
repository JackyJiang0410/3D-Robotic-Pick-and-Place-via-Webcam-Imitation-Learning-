from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import zarr
from tqdm.auto import tqdm


@dataclass
class TrainBatch:
    obs: torch.Tensor
    act: torch.Tensor
    img: torch.Tensor | None


@dataclass
class EpisodeData:
    obs: np.ndarray
    act: np.ndarray
    img: np.ndarray | None


class EgoVerseStyleBC(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, use_images: bool):
        super().__init__()
        self.use_images = bool(use_images)
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        if self.use_images:
            self.img_encoder = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(64 * 4 * 4, 256),
                nn.ReLU(),
            )
            head_in = 512
        else:
            self.img_encoder = None
            head_in = 256
        self.head = nn.Sequential(
            nn.Linear(head_in, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
        )

    def forward(self, obs: torch.Tensor, img: torch.Tensor | None = None) -> torch.Tensor:
        h_obs = self.obs_encoder(obs)
        if self.use_images:
            if img is None:
                raise RuntimeError("Model was trained with images, but img input is None.")
            h_img = self.img_encoder(img)
            h = torch.cat([h_obs, h_img], dim=-1)
        else:
            h = h_obs
        return self.head(h)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train EgoVerse-style BC policy from Phase1 Zarr.")
    p.add_argument("--dataset", type=str, default="data/datasets/phase1_vision.zarr")
    p.add_argument("--out", type=str, default="data/policies/egoverse_bc_policy.pt")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", help="auto|cpu|mps|cuda")
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--obs-only", action="store_true", help="Ignore img even if available in dataset.")
    p.add_argument("--keep-failures", action="store_true", help="Include episodes with success=false.")
    p.add_argument(
        "--val-split-by-episode",
        action="store_true",
        default=True,
        help="Split train/val by whole episodes (prevents timestep leakage across splits).",
    )
    p.add_argument(
        "--no-val-split-by-episode",
        dest="val_split_by_episode",
        action="store_false",
        help="Use random timestep split instead of episode split.",
    )
    p.add_argument(
        "--history-csv",
        type=str,
        default="data/policies/egoverse_train_history.csv",
        help="Where to save per-epoch train/val metrics CSV.",
    )
    p.add_argument(
        "--history-plot",
        type=str,
        default="data/policies/egoverse_train_history.png",
        help="Where to save train/val metric plot (requires matplotlib).",
    )
    return p.parse_args()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_dataset_episodes(path: Path, obs_only: bool, keep_failures: bool):
    root = zarr.open_group(str(path), mode="r")
    eps = root["episodes"]
    episodes: list[EpisodeData] = []
    use_images = False
    for k in sorted(eps.group_keys()):
        g = eps[k]
        success = bool(g.attrs.get("success", True))
        if (not keep_failures) and (not success):
            continue
        obs = np.asarray(g["obs"], dtype=np.float32)
        act = np.asarray(g["act"], dtype=np.float32)
        n = min(obs.shape[0], act.shape[0])
        if n <= 0:
            continue
        ep_obs = obs[:n]
        ep_act = act[:n]
        ep_img = None
        if (not obs_only) and ("img" in g):
            ep_img = np.asarray(g["img"], dtype=np.uint8)[:n]
            use_images = True

        finite = np.isfinite(ep_obs).all(axis=1) & np.isfinite(ep_act).all(axis=1)
        ep_obs = ep_obs[finite]
        ep_act = ep_act[finite]
        if ep_img is not None:
            ep_img = ep_img[finite]
        if ep_obs.shape[0] <= 0:
            continue

        episodes.append(EpisodeData(obs=ep_obs, act=ep_act, img=ep_img))

    if not episodes:
        raise RuntimeError(f"No usable episodes found in {path}")

    if use_images:
        use_images = all(ep.img is not None for ep in episodes)
        if not use_images:
            for ep in episodes:
                ep.img = None
    return episodes, use_images


def _concat_episodes(episodes: list[EpisodeData], use_images: bool):
    obs = np.concatenate([ep.obs for ep in episodes], axis=0)
    act = np.concatenate([ep.act for ep in episodes], axis=0)
    if use_images:
        img = np.concatenate([ep.img for ep in episodes if ep.img is not None], axis=0)
    else:
        img = None
    return obs, act, img


def make_batches(
    obs: np.ndarray,
    act: np.ndarray,
    img: np.ndarray | None,
    batch_size: int,
    device: torch.device,
):
    n = obs.shape[0]
    idx = np.random.permutation(n)
    for s in range(0, n, batch_size):
        b = idx[s : s + batch_size]
        b_obs = torch.from_numpy(obs[b]).to(device)
        b_act = torch.from_numpy(act[b]).to(device)
        if img is not None:
            # uint8 [0,255] -> float [0,1], NHWC -> NCHW
            b_img = torch.from_numpy(img[b]).to(device=device, dtype=torch.float32) / 255.0
            b_img = b_img.permute(0, 3, 1, 2).contiguous()
        else:
            b_img = None
        yield TrainBatch(obs=b_obs, act=b_act, img=b_img)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_path = Path(args.dataset).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    episodes, use_images = load_dataset_episodes(dataset_path, args.obs_only, args.keep_failures)
    ep_count = len(episodes)
    total_n = int(sum(ep.obs.shape[0] for ep in episodes))
    obs_dim = episodes[0].obs.shape[1]
    act_dim = episodes[0].act.shape[1]
    print(f"Loaded {total_n} samples from {dataset_path} ({ep_count} episodes)")
    print(f"obs_dim={obs_dim} act_dim={act_dim} use_images={use_images}")

    all_obs, _, _ = _concat_episodes(episodes, use_images=False)
    obs_mean = all_obs.mean(axis=0).astype(np.float32)
    obs_std = all_obs.std(axis=0).astype(np.float32)
    obs_std[obs_std < 1e-6] = 1.0

    if args.val_split_by_episode:
        val_ep_n = min(max(1, int(ep_count * float(args.val_split))), max(1, ep_count - 1))
        perm_ep = np.random.permutation(ep_count)
        va_ep_idx = set(perm_ep[:val_ep_n].tolist())
        tr_eps = [ep for i, ep in enumerate(episodes) if i not in va_ep_idx]
        va_eps = [ep for i, ep in enumerate(episodes) if i in va_ep_idx]
    else:
        tr_eps = episodes
        va_eps = episodes

    tr_obs_raw, tr_act, tr_img = _concat_episodes(tr_eps, use_images=use_images)
    va_obs_raw, va_act, va_img = _concat_episodes(va_eps, use_images=use_images)
    tr_obs = ((tr_obs_raw - obs_mean) / obs_std).astype(np.float32)
    va_obs = ((va_obs_raw - obs_mean) / obs_std).astype(np.float32)
    print(f"Split: train_samples={tr_obs.shape[0]} val_samples={va_obs.shape[0]} val_by_episode={args.val_split_by_episode}")

    model = EgoVerseStyleBC(obs_dim=obs_dim, act_dim=act_dim, use_images=use_images).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    train_start = time.time()
    history: list[dict[str, float]] = []

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        train_steps = 0
        train_total = max(1, (tr_obs.shape[0] + int(args.batch_size) - 1) // int(args.batch_size))
        train_bar = tqdm(
            make_batches(tr_obs, tr_act, tr_img, int(args.batch_size), device),
            total=train_total,
            desc=f"Epoch {epoch:03d}/{int(args.epochs):03d} [train]",
            leave=False,
            unit="batch",
        )
        for batch in train_bar:
            pred = model(batch.obs, batch.img)
            loss = F.mse_loss(pred, batch.act)
            mae = F.l1_loss(pred, batch.act)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_loss += float(loss.item())
            train_mae += float(mae.item())
            train_steps += 1
            train_bar.set_postfix(loss=f"{float(loss.item()):.5f}")

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_mae = 0.0
            val_steps = 0
            val_total = max(1, (va_obs.shape[0] + int(args.batch_size) - 1) // int(args.batch_size))
            val_bar = tqdm(
                make_batches(va_obs, va_act, va_img, int(args.batch_size), device),
                total=val_total,
                desc=f"Epoch {epoch:03d}/{int(args.epochs):03d} [val]",
                leave=False,
                unit="batch",
            )
            for batch in val_bar:
                pred = model(batch.obs, batch.img)
                loss = F.mse_loss(pred, batch.act)
                mae = F.l1_loss(pred, batch.act)
                val_loss += float(loss.item())
                val_mae += float(mae.item())
                val_steps += 1
                val_bar.set_postfix(loss=f"{float(loss.item()):.5f}")

        train_loss /= max(train_steps, 1)
        train_mae /= max(train_steps, 1)
        val_loss /= max(val_steps, 1)
        val_mae /= max(val_steps, 1)
        elapsed = time.time() - train_start
        avg_epoch = elapsed / epoch
        eta = avg_epoch * (int(args.epochs) - epoch)
        history.append(
            {
                "epoch": float(epoch),
                "train_mse": train_loss,
                "val_mse": val_loss,
                "train_mae": train_mae,
                "val_mae": val_mae,
            }
        )
        print(
            f"epoch {epoch:03d} | train_mse={train_loss:.6f} | val_mse={val_loss:.6f} | "
            f"train_mae={train_mae:.6f} | val_mae={val_mae:.6f} | "
            f"elapsed={elapsed:.1f}s | eta={eta:.1f}s"
        )

    history_csv = Path(args.history_csv).expanduser().resolve()
    history_csv.parent.mkdir(parents=True, exist_ok=True)
    with history_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_mse", "val_mse", "train_mae", "val_mae"])
        writer.writeheader()
        for row in history:
            writer.writerow(row)
    print(f"Saved training history CSV to: {history_csv}")

    history_plot = Path(args.history_plot).expanduser().resolve()
    history_plot.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt  # type: ignore

        xs = [row["epoch"] for row in history]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(xs, [row["train_mse"] for row in history], label="train_mse")
        axes[0].plot(xs, [row["val_mse"] for row in history], label="val_mse")
        axes[0].set_title("MSE")
        axes[0].set_xlabel("epoch")
        axes[0].legend()
        axes[1].plot(xs, [row["train_mae"] for row in history], label="train_mae")
        axes[1].plot(xs, [row["val_mae"] for row in history], label="val_mae")
        axes[1].set_title("MAE")
        axes[1].set_xlabel("epoch")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(history_plot, dpi=160)
        plt.close(fig)
        print(f"Saved training history plot to: {history_plot}")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Could not write history plot ({exc!r}). Install matplotlib to enable PNG plots.")

    payload = {
        "state_dict": model.state_dict(),
        "obs_mean": obs_mean,
        "obs_std": obs_std,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "use_images": use_images,
        "image_shape": tuple(episodes[0].img.shape[1:]) if (use_images and episodes[0].img is not None) else None,
        "dataset": str(dataset_path),
    }
    torch.save(payload, out_path)
    total = time.time() - train_start
    print(f"Saved EgoVerse-style BC policy to: {out_path}")
    print(f"Total training time: {total:.1f}s")


if __name__ == "__main__":
    main()
