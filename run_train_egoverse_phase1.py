from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path

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
    p.add_argument("--dataset", type=str, default="data/phase1_vision.zarr")
    p.add_argument("--out", type=str, default="data/egoverse_bc_policy.pt")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", help="auto|cpu|mps|cuda")
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--obs-only", action="store_true", help="Ignore img even if available in dataset.")
    p.add_argument("--keep-failures", action="store_true", help="Include episodes with success=false.")
    return p.parse_args()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_dataset(path: Path, obs_only: bool, keep_failures: bool):
    root = zarr.open_group(str(path), mode="r")
    eps = root["episodes"]
    obs_list: list[np.ndarray] = []
    act_list: list[np.ndarray] = []
    img_list: list[np.ndarray] = []
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
        obs_list.append(obs[:n])
        act_list.append(act[:n])
        if (not obs_only) and ("img" in g):
            img = np.asarray(g["img"], dtype=np.uint8)[:n]
            img_list.append(img)
            use_images = True

    if not obs_list:
        raise RuntimeError(f"No usable episodes found in {path}")

    X_obs = np.concatenate(obs_list, axis=0)
    Y_act = np.concatenate(act_list, axis=0)
    finite = np.isfinite(X_obs).all(axis=1) & np.isfinite(Y_act).all(axis=1)
    X_obs = X_obs[finite]
    Y_act = Y_act[finite]

    X_img = None
    if use_images and len(img_list) == len(obs_list):
        X_img = np.concatenate(img_list, axis=0)[finite]
    else:
        use_images = False

    return X_obs, Y_act, X_img, use_images


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

    X_obs, Y_act, X_img, use_images = load_dataset(dataset_path, args.obs_only, args.keep_failures)
    n = X_obs.shape[0]
    obs_dim = X_obs.shape[1]
    act_dim = Y_act.shape[1]
    print(f"Loaded {n} samples from {dataset_path}")
    print(f"obs_dim={obs_dim} act_dim={act_dim} use_images={use_images}")

    obs_mean = X_obs.mean(axis=0).astype(np.float32)
    obs_std = X_obs.std(axis=0).astype(np.float32)
    obs_std[obs_std < 1e-6] = 1.0
    Xn = ((X_obs - obs_mean) / obs_std).astype(np.float32)

    val_n = max(1, int(n * float(args.val_split)))
    perm = np.random.permutation(n)
    val_idx = perm[:val_n]
    tr_idx = perm[val_n:]
    tr_obs, tr_act = Xn[tr_idx], Y_act[tr_idx]
    va_obs, va_act = Xn[val_idx], Y_act[val_idx]
    tr_img = X_img[tr_idx] if X_img is not None else None
    va_img = X_img[val_idx] if X_img is not None else None

    model = EgoVerseStyleBC(obs_dim=obs_dim, act_dim=act_dim, use_images=use_images).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    train_start = time.time()

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_loss = 0.0
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
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_loss += float(loss.item())
            train_steps += 1
            train_bar.set_postfix(loss=f"{float(loss.item()):.5f}")

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
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
                val_loss += float(loss.item())
                val_steps += 1
                val_bar.set_postfix(loss=f"{float(loss.item()):.5f}")

        train_loss /= max(train_steps, 1)
        val_loss /= max(val_steps, 1)
        elapsed = time.time() - train_start
        avg_epoch = elapsed / epoch
        eta = avg_epoch * (int(args.epochs) - epoch)
        print(
            f"epoch {epoch:03d} | train_mse={train_loss:.6f} | val_mse={val_loss:.6f} | "
            f"elapsed={elapsed:.1f}s | eta={eta:.1f}s"
        )

    payload = {
        "state_dict": model.state_dict(),
        "obs_mean": obs_mean,
        "obs_std": obs_std,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "use_images": use_images,
        "image_shape": tuple(X_img.shape[1:]) if X_img is not None else None,
        "dataset": str(dataset_path),
    }
    torch.save(payload, out_path)
    total = time.time() - train_start
    print(f"Saved EgoVerse-style BC policy to: {out_path}")
    print(f"Total training time: {total:.1f}s")


if __name__ == "__main__":
    main()
