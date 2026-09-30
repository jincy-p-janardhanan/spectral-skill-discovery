"""Compare SSD runs with METRA and other baselines on state coverage (Kitchen: tasks achieved).

    uv run compare_metra.py --metra_dir <metra_run_group> --envs ant half_cheetah --out figures/metra_comparison
    uv run compare_metra.py --metra_dir <metra_run_group> --envs ant half_cheetah --baselines METRA LSD DIAYN DADS \
        --out figures/all_baselines

Writes <out>/coverage_vs_steps.png, <out>/trajectories.png and <out>/final_coverage.csv.
"""

import argparse
import csv
import glob
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import numpy as np
from matplotlib import pyplot as plt


def parse_args_from_log(run_dir):
    log_path = os.path.join(run_dir, "debug.log")
    if not os.path.exists(log_path):
        return None
    with open(log_path) as f:
        for line in f:
            m = re.search(r"ARGS: Namespace\((.*)\)\s*$", line)
            if m:
                pairs = re.split(r",\s*(?=[A-Za-z_][A-Za-z0-9_]*=)", m.group(1))
                out = {}
                for pair in pairs:
                    key, value = pair.split("=", 1)
                    try:
                        out[key] = eval(value, {"__builtins__": {}}, {"None": None, "True": True, "False": False})
                    except Exception:
                        out[key] = value
                return out
    return None


def method_name(args):
    if args["algo"] == "dads":
        return "DADS"
    inner, dual_reg, sn = args.get("inner", 1), args.get("dual_reg", 1), args.get("spectral_normalization", 0)
    if inner and dual_reg:
        return "METRA"
    if inner and not dual_reg and sn:
        return "LSD"
    if not inner and not dual_reg:
        return "DIAYN"
    return "Other"

COLORS = {"SSD": "#2a78d6", "METRA": "#eb6834", "LSD": "#1baf7a", "DIAYN": "#eda100", "DADS": "#e87ba4"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e5e4e0", "#fcfcfb"
ENV_TITLES = {"ant": "Ant (states)", "half_cheetah": "HalfCheetah (states)", "dmc_quadruped": "Quadruped (pixels)",
              "dmc_humanoid": "Humanoid (pixels)", "kitchen": "Kitchen (pixels)"}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "axes.axisbelow": True, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left",
})


def read_csv(path) -> dict[str, np.ndarray]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    out = {}
    if not rows:
        return out
    for k in rows[0]:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[k]))
            except (TypeError, ValueError):
                vals.append(np.nan)
        out[k] = np.array(vals)
    return out


def metric_keys(env):
    if env == "kitchen":
        return "eval/KitchenOverall", "EvalOp/KitchenOverall", "Tasks achieved (of 6)"
    label = "State coverage (x bins)" if env == "half_cheetah" else "State coverage (x-y bins)"
    return "eval/coverage", "EvalOp/MjNumUniqueCoords", label


def ssd_runs(exp_dir, env):
    runs = []
    for d in sorted(glob.glob(os.path.join(exp_dir, f"{env}_K*_s*"))):
        path = os.path.join(d, "eval.csv")
        if os.path.exists(path):
            e = read_csv(path)
            runs.append({"dir": d, "steps": e["env_steps"], "value": e[metric_keys(env)[0]],
                         "traj": sorted(glob.glob(os.path.join(d, "plots", "trajectories_*.png")))})
    return runs


def metra_runs(exp_dir, env):
    key = metric_keys(env)[1]
    info = {}
    for path in sorted(glob.glob(os.path.join(exp_dir, "*", "progress_eval.csv"))):
        d = os.path.realpath(os.path.dirname(path))
        args = parse_args_from_log(d)
        if args is None or args.get("env") != env:
            continue
        e = read_csv(path)
        if key not in e or len(e[key]) == 0:
            continue
        trajs = sorted(glob.glob(os.path.join(d, "plots", "TrajPlot_RandomZ_*.png")),
                       key=lambda q: int(re.findall(r"_(\d+)\.png$", q)[0]))
        parent = None
        rf = os.path.join(d, "resumed_from.txt")
        if os.path.exists(rf):
            parent = os.path.realpath(open(rf).read().split()[0])
        info[d] = {"dir": d, "method": method_name(args), "seed": int(args["seed"]), "parent": parent,
                   "steps": e["TotalEnvSteps"], "value": e[key], "traj": trajs}

    def chain(d):
        r = info[d]
        if r["parent"] in info:
            p = chain(r["parent"])
            keep = p["steps"] < r["steps"][0]
            return {**r, "steps": np.r_[p["steps"][keep], r["steps"]], "value": np.r_[p["value"][keep], r["value"]],
                    "traj": r["traj"] or p["traj"]}
        return r

    parents = {r["parent"] for r in info.values() if r["parent"]}
    best = {}
    for d in info:
        if d in parents:
            continue  # continued by a resumed run
        c = chain(d)
        k = (c["method"], c["seed"])
        if k not in best or c["steps"][-1] > best[k]["steps"][-1]:
            best[k] = c
    runs = {}
    for (m, _), c in sorted(best.items(), key=lambda kv: kv[0][1]):
        runs.setdefault(m, []).append(c)
    return runs


def mean_std(runs):
    common = set(runs[0]["steps"].tolist())
    for r in runs[1:]:
        common &= set(r["steps"].tolist())
    steps = np.array(sorted(common))
    vals = np.stack([np.array([r["value"][r["steps"] == s][0] for s in steps]) for r in runs])
    return steps, vals.mean(0), vals.std(0), vals


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--envs", nargs="+", default=["ant", "half_cheetah"])
    p.add_argument("--baselines", nargs="+", default=["METRA"], help="METRA, LSD, DIAYN, DADS")
    p.add_argument("--ssd_dir", default="exp/Paper")
    p.add_argument("--metra_dir", required=True, help="METRA's experiment group with the baseline runs")
    p.add_argument("--out", default="figures/metra_comparison")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = {}
    for env in args.envs:
        methods = {"SSD": ssd_runs(args.ssd_dir, env)}
        base = metra_runs(args.metra_dir, env)
        methods.update({m: base.get(m, []) for m in args.baselines})
        data[env] = {m: r for m, r in methods.items() if r}

    # coverage vs environment steps
    fig, axes = plt.subplots(1, len(args.envs), figsize=(6.2 * len(args.envs), 4.3), squeeze=False)
    rows = []
    for ax, env in zip(axes[0], args.envs):
        for m, runs in data[env].items():
            steps, mu, sd, vals = mean_std(runs)
            x = steps / 1e6
            ax.fill_between(x, mu - sd, mu + sd, color=COLORS[m], alpha=0.15, lw=0)
            ax.plot(x, mu, color=COLORS[m], lw=2.2, label=f"{m} ({len(runs)} seeds)")
            ax.annotate(f"{mu[-1]:.0f}", (x[-1], mu[-1]), xytext=(5, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=INK2)
            rows.append({"env": env, "method": m, "seeds": len(runs), "final_env_steps": int(steps[-1]),
                         "final_mean": round(float(mu[-1]), 1), "final_std": round(float(sd[-1]), 1),
                         "per_seed": " ".join(f"{v:.0f}" for v in vals[:, -1])})
        ax.set_title(ENV_TITLES.get(env, env))
        ax.set_xlabel("environment steps (millions)")
        ax.set_ylabel(metric_keys(env)[2])
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0)
    handles, labels = [], []
    for ax in axes[0]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l.split(" (")[0] not in [x.split(" (")[0] for x in labels]:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / "coverage_vs_steps.png", dpi=150)
    plt.close(fig)

    with open(out / "final_coverage.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # final trajectories of the median seed per method
    methods = [m for m in ["SSD", *args.baselines] if any(m in data[e] for e in args.envs)]
    fig, axes = plt.subplots(len(args.envs), len(methods), figsize=(3.6 * len(methods), 3.8 * len(args.envs)),
                             squeeze=False)
    for i, env in enumerate(args.envs):
        for j, m in enumerate(methods):
            ax = axes[i, j]
            ax.axis("off")
            runs = data[env].get(m, [])
            if runs:
                final = np.array([r["value"][-1] for r in runs])
                med = runs[int(np.argsort(final)[len(final) // 2])]
                if med["traj"]:
                    ax.imshow(mpimg.imread(med["traj"][-1]))
                ax.set_title(f"{m}: {ENV_TITLES.get(env, env)}\nmedian seed, coverage {med['value'][-1]:.0f}",
                             fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "trajectories.png", dpi=130)
    plt.close(fig)

    for r in rows:
        print(f"{r['env']:13s} {r['method']:6s} {r['seeds']} seeds @ {r['final_env_steps']/1e6:.1f}M steps: "
              f"{r['final_mean']:7.1f} +- {r['final_std']:6.1f}   per seed: {r['per_seed']}")
    print(f"Wrote {out}/coverage_vs_steps.png, trajectories.png, final_coverage.csv")


if __name__ == "__main__":
    main()
