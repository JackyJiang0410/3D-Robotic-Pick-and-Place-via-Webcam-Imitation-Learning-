#!/usr/bin/env python3
"""
Exploratory stats for Phase-1 Zarr (obs / act / optional img / t).

Example:
  ./.venv/bin/python scripts/eda_phase1_zarr.py \\
    --zarr data/datasets/phase1_vision.zarr \\
    --out Results/eda_phase1_vision
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import zarr

# Observation layout must match stage2_mujoco.panda_env.PandaObs.as_vector() for nq=16, nv=15.
OBS_BLOCKS = (
    ("qpos", 0, 16),
    ("qvel", 16, 31),
    ("ee_pos", 31, 34),
    ("obj_pos", 34, 37),
    ("target_pos", 37, 40),
    ("gripper_open", 40, 41),
)
ACT_NAMES = ("dx", "dy", "g_close_indicator")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EDA summary for phase1 Zarr trajectories.")
    p.add_argument("--zarr", type=str, default="data/datasets/phase1_vision.zarr")
    p.add_argument("--out", type=str, default="Results/eda_zarr")
    p.add_argument("--hist-pixels", type=int, default=2_000_000, help="Max uint8 pixels to sample for image luminance hist.")
    return p.parse_args()


def _fmt_dict(d: dict) -> str:
    lines = []
    for k in sorted(d.keys()):
        lines.append(f"  - **{k}**: `{d[k]}`")
    return "\n".join(lines) if lines else "  - _(none)_"


def main() -> None:
    args = parse_args()
    zpath = Path(args.zarr).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    root = zarr.open_group(str(zpath), mode="r")
    meta_attrs = dict(root["meta"].attrs)
    eps = root["episodes"]
    ep_keys = sorted(eps.group_keys())
    n_eps = len(ep_keys)
    if n_eps == 0:
        raise RuntimeError(f"No episodes under {zpath}/episodes")

    lengths: list[int] = []
    successes: list[bool] = []
    peak_lifts: list[Optional[float]] = []
    cameras: Counter[str] = Counter()
    image_shapes: Counter[tuple[int, ...]] = Counter()

    obs_list: list[np.ndarray] = []
    act_list: list[np.ndarray] = []
    dts: list[float] = []

    for name in ep_keys:
        g = eps[name]
        obs = np.asarray(g["obs"])
        act = np.asarray(g["act"])
        n = min(obs.shape[0], act.shape[0])
        obs = obs[:n].astype(np.float32, copy=False)
        act = act[:n].astype(np.float32, copy=False)
        lengths.append(n)
        successes.append(bool(g.attrs.get("success", True)))
        peak_lifts.append(float(g.attrs["peak_lift_m"]) if "peak_lift_m" in g.attrs else None)
        if "image_camera" in g.attrs:
            cameras[str(g.attrs["image_camera"])] += 1
        if "image_shape" in g.attrs:
            t = tuple(int(x) for x in g.attrs["image_shape"])
            image_shapes[t] += 1

        obs_list.append(obs)
        act_list.append(act)
        if "t" in g and g["t"].shape[0] >= 2:
            t = np.asarray(g["t"][:n], dtype=np.float64)
            if t.shape[0] >= 2:
                dts.extend(np.diff(t).tolist())

    obs_all = np.concatenate(obs_list, axis=0)
    act_all = np.concatenate(act_list, axis=0)
    n_steps = int(obs_all.shape[0])
    obs_dim = int(obs_all.shape[1])
    act_dim = int(act_all.shape[1])

    # --- Scalar-derived features ---
    ee = obs_all[:, 31:34]
    obj = obs_all[:, 34:37]
    dist_xyz = np.linalg.norm(ee - obj, axis=1)
    dist_xy = np.linalg.norm((ee - obj)[:, :2], axis=1)
    act_norm_xy = np.linalg.norm(act_all[:, :2], axis=1)
    idle_xy = act_norm_xy < 1e-6
    pinch = act_all[:, 2] > 0.5

    # --- Per-dimension obs/act tables ---
    pct = (1, 5, 25, 50, 75, 95, 99)
    obs_rows = []
    for j in range(obs_dim):
        col = obs_all[:, j]
        row = {"dim": j, "mean": float(col.mean()), "std": float(col.std())}
        for p in pct:
            row[f"p{p}"] = float(np.percentile(col, p))
        obs_rows.append(row)

    act_rows = []
    for j in range(act_dim):
        col = act_all[:, j]
        row = {"dim": j, "name": ACT_NAMES[j] if j < len(ACT_NAMES) else f"a{j}"}
        row.update({"mean": float(col.mean()), "std": float(col.std())})
        for p in pct:
            row[f"p{p}"] = float(np.percentile(col, p))
        act_rows.append(row)

    # CSVs
    def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    write_csv(
        out_dir / "obs_per_dim.csv",
        ["dim", "mean", "std"] + [f"p{p}" for p in pct],
        [{k: r[k] for k in ["dim", "mean", "std"] + [f"p{p}" for p in pct]} for r in obs_rows],
    )
    write_csv(
        out_dir / "act_per_dim.csv",
        ["dim", "name", "mean", "std"] + [f"p{p}" for p in pct],
        [
            {k: r[k] for k in ["dim", "name", "mean", "std"] + [f"p{p}" for p in pct]}
            for r in act_rows
        ],
    )

    ep_summary = []
    for i, name in enumerate(ep_keys):
        ep_summary.append(
            {
                "episode": name,
                "steps": lengths[i],
                "success": successes[i],
                "peak_lift_m": "" if peak_lifts[i] is None else float(peak_lifts[i]),
            }
        )
    write_csv(out_dir / "episodes.csv", ["episode", "steps", "success", "peak_lift_m"], ep_summary)

    # --- Block-level obs summary ---
    block_lines = []
    for label, a, b in OBS_BLOCKS:
        if b > obs_dim or a > obs_dim:
            continue
        sl = obs_all[:, a:min(b, obs_dim)]
        block_lines.append(
            f"| `{label}` `[{a}:{b})` | {sl.shape[1]} | {float(sl.mean()):.4g} | {float(sl.std()):.4g} | "
            f"{float(np.percentile(sl, 5)):.4g} … {float(np.percentile(sl, 95)):.4g} |"
        )

    # --- Images ---
    img_section = []
    rng = np.random.default_rng(0)
    pix_budget = int(args.hist_pixels)
    pixels_sampled = 0
    rgb_sum = np.zeros(3, dtype=np.float64)
    total_pix = 0
    lum_bins = np.zeros(32, dtype=np.int64)
    for name in ep_keys:
        g = eps[name]
        if "img" not in g:
            continue
        arr = np.asarray(g["img"], dtype=np.uint8).reshape(-1, 3)
        rgb_sum += arr.sum(axis=0).astype(np.float64)
        total_pix += arr.shape[0]
        need = min(max(0, pix_budget - pixels_sampled), arr.shape[0])
        if need > 0:
            idx = rng.choice(arr.shape[0], size=need, replace=False)
            lum = (0.299 * arr[idx, 0] + 0.587 * arr[idx, 1] + 0.114 * arr[idx, 2]) / 255.0
            h, _ = np.histogram(lum, bins=32, range=(0.0, 1.0))
            lum_bins += h.astype(np.int64)
            pixels_sampled += need

    if total_pix > 0:
        rgb_mu = rgb_sum / total_pix
        img_section.append(f"- **RGB mean** (0–255, over all pixels): ({rgb_mu[0]:.2f}, {rgb_mu[1]:.2f}, {rgb_mu[2]:.2f})")
        img_section.append(f"- **Total image pixels** (T×H×W): {total_pix:,}")
        if pixels_sampled > 0:
            img_section.append(f"- **Luminance histogram**: {pixels_sampled:,} subsampled pixels → `luminance_hist.png`")
    else:
        img_section.append("- _(No `img` dataset in episodes.)_")

    # --- Markdown report ---
    succ_n = int(sum(successes))
    lines = [
        f"# Zarr EDA: `{zpath.name}`",
        "",
        "## Location",
        f"- Path: `{zpath}`",
        f"- Output: `{out_dir}`",
        "",
        "## Corpus overview",
        f"- **Episodes (demos / reps)**: {n_eps}",
        f"- **Total transitions (rows)**: {n_steps:,}",
        f"- **Obs dim** (`obs`): {obs_dim}",
        f"- **Act dim** (`act`): {act_dim}",
        f"- **Labeled successes** (`attrs['success']`): {succ_n} / {n_eps} ({100.0 * succ_n / max(n_eps, 1):.1f}%)",
        "",
        "### Episode length (steps per episode)",
        f"- min={min(lengths)} | max={max(lengths)} | mean={float(np.mean(lengths)):.1f} | "
        f"median={float(np.median(lengths)):.1f} | std={float(np.std(lengths)):.1f}",
        "",
        "### Root `meta` attrs",
        _fmt_dict(meta_attrs),
        "",
        "### Episode-level metadata (counts)",
        f"- **`image_camera`**: {dict(cameras)}",
        f"- **`image_shape`**: {dict(image_shapes)}",
        "",
    ]
    pl_arr = np.asarray([x for x in peak_lifts if x is not None], dtype=np.float64)
    if pl_arr.size > 0:
        pl = pl_arr
        lines.extend(
            [
                "### `peak_lift_m` (where logged)",
                f"- n={pl.size} | min={pl.min():.4f} | max={pl.max():.4f} | mean={pl.mean():.4f} | median={float(np.median(pl)):.4f}",
                "",
            ]
        )

    if dts:
        dt = np.asarray(dts, dtype=np.float64)
        lines.extend(
            [
                "### Wall-clock `t` spacing (within episodes)",
                f"- n diffs={len(dt):,} | mean={dt.mean():.4f}s | median={float(np.median(dt)):.4f}s | "
                f"p95={float(np.percentile(dt, 95)):.4f}s",
                "",
            ]
        )

    lines.extend(
        [
            "## Observation vector layout",
            "Concatenation order matches `PandaObs.as_vector()` (see `stage2_mujoco/panda_env.py`).",
            "",
            "| block | index range | dims | mean | std | p5 … p95 |",
            "|---|---:|---:|---:|---:|---|",
            *block_lines,
            "",
            "## Actions `[dx, dy, g]`",
            "- `dx`, `dy`: EE delta per control step (meters; scaled at collection).",
            "- `g`: gripper command (>0.5 treated as close in env).",
            "",
            f"- **Share of steps with |dx,dy| ≈ 0**: {100.0 * float(idle_xy.mean()):.1f}%",
            f"- **Share of steps with g > 0.5 (close)**: {100.0 * float(pinch.mean()):.1f}%",
            *(
                [
                    "- **Note:** logged `g` is identically 0 for every step in this corpus; "
                    "pinch / close is likely from scripted auto-grasp (not written into `act`)."
                ]
                if float(act_all[:, 2].max()) < 0.5
                else []
            ),
            "",
            "## Geometry in the logged states",
            f"- **||ee − obj||_2 (xyz)** — min={dist_xyz.min():.4f} | p50={float(np.percentile(dist_xyz, 50)):.4f} | "
            f"p95={float(np.percentile(dist_xyz, 95)):.4f} | max={dist_xyz.max():.4f} m",
            f"- **||ee − obj||_2 (xy)** — min={dist_xy.min():.4f} | p50={float(np.percentile(dist_xy, 50)):.4f} | "
            f"p95={float(np.percentile(dist_xy, 95)):.4f} | max={dist_xy.max():.4f} m",
            "",
            "## Images",
            *img_section,
            "",
            "## Files written",
            "- `SUMMARY.md` (this report)",
            "- `episodes.csv`, `obs_per_dim.csv`, `act_per_dim.csv`",
            "- `episode_lengths.png`, `distributions.png` (+ `luminance_hist.png` if images)",
            "",
        ]
    )

    (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")

    summary_json = {
        "zarr": str(zpath),
        "episodes": n_eps,
        "total_steps": n_steps,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "successes": succ_n,
        "episode_length": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": float(np.mean(lengths)),
            "median": float(np.median(lengths)),
        },
        "idle_xy_frac": float(idle_xy.mean()),
        "pinch_frac": float(pinch.mean()),
        "dist_xyz_m": {
            "p50": float(np.percentile(dist_xyz, 50)),
            "p95": float(np.percentile(dist_xyz, 95)),
        },
        "image_camera_counts": dict(cameras),
        "image_shape_counts": {str(k): v for k, v in image_shapes.items()},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

    # --- Plots ---
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(n_eps), lengths, color="steelblue")
        ax.set_xlabel("episode index (sorted)")
        ax.set_ylabel("steps")
        ax.set_title("Steps per episode")
        fig.tight_layout()
        fig.savefig(out_dir / "episode_lengths.png", dpi=160)
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes[0, 0].hist(dist_xy, bins=60, color="coral", alpha=0.85)
        axes[0, 0].set_title("||ee − obj|| (xy) [m]")
        axes[0, 1].hist(dist_xyz, bins=60, color="seagreen", alpha=0.85)
        axes[0, 1].set_title("||ee − obj|| (xyz) [m]")
        axes[1, 0].hist(act_all[:, 0], bins=60, color="tab:blue", alpha=0.8, label="dx")
        axes[1, 0].hist(act_all[:, 1], bins=60, color="tab:orange", alpha=0.5, label="dy")
        axes[1, 0].set_title("Action dx, dy")
        axes[1, 0].legend()
        axes[1, 1].hist(act_all[:, 2], bins=40, color="purple", alpha=0.85)
        axes[1, 1].set_title("Action g (gripper)")
        fig.suptitle(f"{zpath.name} — {n_eps} episodes, {n_steps:,} steps", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / "distributions.png", dpi=160)
        plt.close(fig)

        if total_pix > 0 and pixels_sampled > 0:
            fig, ax = plt.subplots(figsize=(7, 3.5))
            centers = np.linspace(0.5 / 32, 1.0 - 0.5 / 32, 32)
            ax.bar(centers, lum_bins, width=1.0 / 32, color="gray", edgecolor="none")
            ax.set_xlim(0, 1)
            ax.set_xlabel("luminance Y (normalized)")
            ax.set_ylabel("count")
            ax.set_title(f"Luminance (subsampled, n={pixels_sampled:,} px)")
            fig.tight_layout()
            fig.savefig(out_dir / "luminance_hist.png", dpi=160)
            plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        (out_dir / "PLOTS_SKIPPED.txt").write_text(f"matplotlib failed: {exc!r}\n", encoding="utf-8")

    print(f"Wrote EDA to: {out_dir}")
    print((out_dir / "SUMMARY.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
