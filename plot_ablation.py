"""Coverage vs environment steps for groups of SSD runs, with METRA for reference, and a table of final values.

    uv run plot_ablation.py --env ant --max_steps 8e6 --groups "defaults=exp/Benchmark/ant_K2_s[012]" \
        --group_dirs "exp/Tune/ant_*" --legend_outside --out figures/sweep/ant.png

Each --groups entry is LABEL=GLOB over run directories; each directory matched by --group_dirs is one group of
runs, labelled by its name.
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from compare_metra import COLORS, ENV_TITLES, INK2, mean_std, metra_runs, metric_keys, read_csv

PALETTE = ["#2a78d6", "#1baf7a", "#eda100", "#e87ba4", "#7d5bd6", "#16a3b8", "#8a6f4e", "#52514e"]


def group_runs(pattern, env):
    runs = []
    for d in sorted(glob.glob(pattern)):
        path = os.path.join(d, "eval.csv")
        if os.path.exists(path):
            e = read_csv(path)
            runs.append({"dir": d, "steps": e["env_steps"], "value": e[metric_keys(env)[0]]})
    return runs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", default="ant")
    p.add_argument("--groups", nargs="+", default=[], help="LABEL=GLOB")
    p.add_argument("--group_dirs", default=None, help="GLOB over directories that each hold one group's runs")
    p.add_argument("--metra_dir", default=None, help="METRA's experiment group, drawn for reference")
    p.add_argument("--max_steps", type=float, default=None)
    p.add_argument("--title", default=None)
    p.add_argument("--legend_outside", action="store_true", help="legend right of the axes (for crowded plots)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    specs = [spec.rsplit("=", 1) for spec in args.groups]  # labels may contain "=", globs do not
    if args.group_dirs:
        specs += [(os.path.basename(d), os.path.join(d, "*")) for d in sorted(glob.glob(args.group_dirs))]
    curves = []
    for i, (label, pattern) in enumerate(specs):
        runs = group_runs(pattern, args.env)
        if runs:
            curves.append((label, runs, PALETTE[i % len(PALETTE)], "-"))
    metra = metra_runs(args.metra_dir, args.env).get("METRA", []) if args.metra_dir else []
    if metra:
        curves.append(("METRA", metra, COLORS["METRA"], "--"))

    fig, ax = plt.subplots(figsize=(10.4 if args.legend_outside else 7.2, 4.6))
    rows = []
    for label, runs, color, ls in curves:
        if args.max_steps:
            runs = [{**r, "steps": r["steps"][r["steps"] <= args.max_steps * 1.001],
                     "value": r["value"][r["steps"] <= args.max_steps * 1.001]} for r in runs]
        steps, mu, sd, _ = mean_std(runs)
        x = steps / 1e6
        ax.fill_between(x, mu - sd, mu + sd, color=color, alpha=0.13, lw=0)
        ax.plot(x, mu, color=color, lw=2.0, ls=ls, label=f"{label} ({len(runs)} seeds)")
        ax.annotate(f"{mu[-1]:.0f}", (x[-1], mu[-1]), xytext=(5, 0), textcoords="offset points", va="center",
                    fontsize=8.5, color=INK2)
        rows.append(f"{label:40s} {len(runs)} seeds  {steps[-1] / 1e6:5.1f}M  {mu[-1]:7.1f} +- {sd[-1]:6.1f}")
    ax.set_xlabel("Environment steps (M)")
    ax.set_ylabel(metric_keys(args.env)[2])
    ax.set_title(args.title or ENV_TITLES.get(args.env, args.env))
    if args.legend_outside:
        ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(1.07, 1))
    else:
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    Path(args.out).with_suffix(".txt").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
