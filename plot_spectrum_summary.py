"""Spectrum readout of SSD runs across seeds: the final readout per dimension, and the spread over training.

    uv run plot_spectrum_summary.py --groups "ant=exp/PaperV2/ant_*" "half_cheetah=exp/PaperV2/half_cheetah_*" \
        --out figures/paper_v2/spectrum

Writes <out>_final.png (per dimension at the last evaluation: spread sigma2_k on a log scale and return R_k
against the 95% permutation floor, one dot per seed) and <out>_training.png (each dimension's share of the total
spread, median over seeds, against environment steps).
"""

import argparse
import glob
import os
from collections import Counter
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from compare_metra import ENV_TITLES, GRID, INK, INK2, SURFACE, read_csv

SSD_BLUE = "#2a78d6"
DIM_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
FLOOR = "#8c8b86"


def load(pattern):
    """Per run: env_steps [E], K_star [E], sigma2 / R / floor [E, K] from eval.csv."""
    runs = []
    for d in sorted(glob.glob(pattern)):
        path = os.path.join(d, "eval.csv")
        if not os.path.exists(path):
            continue
        e = read_csv(path)
        k = sum(1 for key in e if key.startswith("spectrum/sigma2_"))
        stack = lambda name: np.stack([e[f"spectrum/{name}_{i + 1}"] for i in range(k)], axis=1)
        runs.append({"steps": e["env_steps"], "k_star": e["spectrum/K_star"].astype(int),
                     "sigma2": stack("sigma2"), "R": stack("R"), "floor": stack("floor")})
    return runs


def k_star_summary(runs):
    counts = Counter(int(r["k_star"][-1]) for r in runs)
    if len(counts) == 1:
        (k, n), = counts.items()
        return rf"$K^\star$ = {k} in {n} of {len(runs)} seeds"
    ks = sorted(counts)
    return rf"$K^\star$ = {', '.join(map(str, ks))} in {', '.join(str(counts[k]) for k in ks)} of {len(runs)} seeds"


def strip(ax, values, color, rng):
    """One dot per seed for each dimension (columns of `values`), jittered, with the median as a dark tick."""
    n, k = values.shape
    for j in range(k):
        x = j + 1 + rng.uniform(-0.12, 0.12, size=n)
        ax.scatter(x, values[:, j], s=26, color=color, alpha=0.8, edgecolors=SURFACE, linewidths=0.8, zorder=3)
        ax.plot([j + 0.72, j + 1.28], [np.median(values[:, j])] * 2, color=INK, lw=1.6, zorder=4)
    ax.set_xticks(range(1, k + 1))
    ax.set_xlim(0.4, k + 0.6)
    ax.set_xlabel("dimension $k$")
    ax.grid(axis="x", visible=False)


def plot_final(groups, out):
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(len(groups), 2, figsize=(9.2, 3.3 * len(groups)), squeeze=False)
    for row, (env, runs) in enumerate(groups):
        sigma2 = np.stack([r["sigma2"][-1] for r in runs])
        ret = np.stack([r["R"][-1] for r in runs])
        floor = np.stack([r["floor"][-1] for r in runs])
        steps = runs[0]["steps"][-1] / 1e6
        ax1, ax2 = axes[row]
        strip(ax1, sigma2, SSD_BLUE, rng)
        if sigma2.max() / sigma2.min() > 20:
            ax1.set_yscale("log")
            ax1.set_ylabel(r"spread $\sigma_k^2$ (log scale)")
        else:
            ax1.set_ylim(0, 1.15 * sigma2.max())
            ax1.set_ylabel(r"spread $\sigma_k^2$")
        ax1.set_title(f"{ENV_TITLES.get(env, env)}, {steps:.0f}M steps: spread", fontsize=10.5)
        strip(ax2, ret, SSD_BLUE, rng)
        for j in range(ret.shape[1]):
            ax2.plot([j + 0.62, j + 1.38], [np.median(floor[:, j])] * 2, color=FLOOR, lw=1.6, ls=(0, (3, 2)), zorder=2)
        ax2.axhline(0, color=INK2, lw=0.8, zorder=1)
        ax2.set_ylabel(r"return $R_k$")
        ax2.set_title(f"return; {k_star_summary(runs)}", fontsize=10.5)
    handles = [plt.Line2D([], [], ls="", marker="o", ms=5.5, color=SSD_BLUE, mec=SURFACE, label="one seed"),
               plt.Line2D([], [], color=INK, lw=1.6, label="median over seeds"),
               plt.Line2D([], [], color=FLOOR, lw=1.6, ls=(0, (3, 2)), label="95% permutation floor (median)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_training(groups, out):
    fig, axes = plt.subplots(1, len(groups), figsize=(4.9 * len(groups), 3.6), squeeze=False)
    for ax, (env, runs) in zip(axes[0], groups):
        n = min(len(r["steps"]) for r in runs)  # evaluations every seed reached
        steps = runs[0]["steps"][:n] / 1e6
        share = np.stack([r["sigma2"][:n] / r["sigma2"][:n].sum(axis=1, keepdims=True) for r in runs])  # [S, E, K]
        med = 100 * np.median(share, axis=0)
        for j in range(med.shape[1]):
            ax.plot(steps, med[:, j], color=DIM_COLORS[j % len(DIM_COLORS)], lw=2.0, label=f"dimension {j + 1}")
        # end-of-line labels, at least 6 points apart
        order = np.argsort(med[-1])
        ys = med[-1, order].astype(float)
        for i in range(1, len(ys)):
            ys[i] = max(ys[i], ys[i - 1] + 6)
        for j, y in zip(order, ys):
            ax.annotate(f"{j + 1}", (steps[-1], y), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK2, annotation_clip=False)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Environment steps (M)")
        ax.set_ylabel(r"share of total spread $\sigma_k^2$ (%)")
        ax.set_title(f"{ENV_TITLES.get(env, env)}: median over {len(runs)} seeds")
        ax.legend(frameon=False, fontsize=8.5, loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--groups", nargs="+", required=True, help="ENV=GLOB over run directories")
    p.add_argument("--out", required=True, help="output prefix")
    args = p.parse_args()
    groups = []
    for spec in args.groups:
        env, pattern = spec.split("=", 1)
        runs = load(pattern)
        if runs:
            groups.append((env, runs))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plot_final(groups, f"{args.out}_final.png")
    plot_training(groups, f"{args.out}_training.png")
    for env, runs in groups:
        print(f"{env}: {len(runs)} runs, {k_star_summary(runs)}")
    print(f"Wrote {args.out}_final.png and {args.out}_training.png")


if __name__ == "__main__":
    main()
