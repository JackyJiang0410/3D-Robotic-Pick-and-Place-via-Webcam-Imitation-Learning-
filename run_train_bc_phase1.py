from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr


@dataclass
class BCModel:
    W: np.ndarray  # (obs_dim, act_dim)
    b: np.ndarray  # (act_dim,)
    x_mean: np.ndarray  # (obs_dim,)
    x_std: np.ndarray  # (obs_dim,)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        x = (obs - self.x_mean) / self.x_std
        return (x @ self.W + self.b).astype(np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1: behavior cloning (ridge regression) from Zarr demos.")
    p.add_argument("--dataset", type=str, default="data/phase1_demos.zarr")
    p.add_argument("--out", type=str, default="data/phase1_bc_policy.npz")
    p.add_argument("--lambda", dest="lam", type=float, default=1e-3, help="Ridge regularization strength.")
    return p.parse_args()


def load_all_episodes(dataset_path: Path) -> tuple[np.ndarray, np.ndarray]:
    root = zarr.open_group(str(dataset_path), mode="r")
    eps = root["episodes"]
    Xs: list[np.ndarray] = []
    Ys: list[np.ndarray] = []
    for k in sorted(eps.group_keys()):
        g = eps[k]
        Xs.append(np.asarray(g["obs"], dtype=np.float32))
        Ys.append(np.asarray(g["act"], dtype=np.float32))
    if not Xs:
        raise RuntimeError(f"No episodes found in {dataset_path}")
    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    finite = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    dropped = int((~finite).sum())
    if dropped:
        X = X[finite]
        Y = Y[finite]
        print(f"Dropped {dropped} non-finite samples.")
    return X, Y


def fit_ridge(X: np.ndarray, Y: np.ndarray, lam: float) -> BCModel:
    # Solve: argmin ||XW + b - Y||^2 + lam||W||^2
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    n, d = X.shape

    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)
    x_std[x_std < 1e-6] = 1.0
    Xn = (X - x_mean) / x_std

    X_aug = np.concatenate([Xn, np.ones((n, 1), dtype=np.float64)], axis=1)  # (n, d+1)
    # Numerically stable ridge via augmented least squares:
    # minimize ||X_aug theta - Y||^2 + lam ||theta[:-1]||^2
    lam = float(max(lam, 0.0))
    if lam > 0:
        I = np.eye(d, dtype=np.float64)
        X_reg = np.concatenate([I, np.zeros((d, 1), dtype=np.float64)], axis=1) * np.sqrt(lam)  # (d, d+1)
        Y_reg = np.zeros((d, Y.shape[1]), dtype=np.float64)
        X_tilde = np.concatenate([X_aug, X_reg], axis=0)
        Y_tilde = np.concatenate([Y, Y_reg], axis=0)
    else:
        X_tilde = X_aug
        Y_tilde = Y

    theta, *_ = np.linalg.lstsq(X_tilde, Y_tilde, rcond=None)  # (d+1, act_dim)

    W = theta[:-1, :].astype(np.float32)
    b = theta[-1, :].astype(np.float32)
    return BCModel(W=W, b=b, x_mean=x_mean.astype(np.float32), x_std=x_std.astype(np.float32))


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    X, Y = load_all_episodes(dataset_path)
    model = fit_ridge(X, Y, args.lam)

    Y_hat = model.predict(X)
    mse = float(np.mean((Y_hat - Y) ** 2))
    print(f"Loaded {X.shape[0]} samples. obs_dim={X.shape[1]} act_dim={Y.shape[1]}")
    print(f"Train MSE: {mse:.6f}")

    np.savez_compressed(out_path, W=model.W, b=model.b, x_mean=model.x_mean, x_std=model.x_std)
    print(f"Saved policy to: {out_path}")


if __name__ == "__main__":
    main()

