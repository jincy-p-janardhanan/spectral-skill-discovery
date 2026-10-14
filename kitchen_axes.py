"""Which Kitchen tasks does each skill axis control, and do the axes combine additively?

Cube skills z in {-1, +1}^K / sqrt(K) form a full factorial design over the K axes, so the main effect of axis k on
task t is E[k, t] = P(t | z_k > 0) - P(t | z_k < 0). Per run: every corner (a random subset if there are more than
--max_skills), --repeats deterministic episodes each, and the tasks achieved by the end of each episode.

Reported per run and task: the success rate, the dominant axis and the share of the task's squared main effects on
it, and how much of the variance of the task's success a linear model explains with main effects only (additive
axes), with pairwise interactions added, and with every corner separately (the ceiling).

    uv run kitchen_axes.py --runs "exp/PaperV3/kitchen_K5_s*" --out figures/compose/kitchen_K5
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")  # must be set before mujoco is imported

import argparse
import glob
import itertools

import numpy as np
import torch

from make_videos import load_agent
from ssd.envs import make_env
from ssd.kitchen.env import GOAL_NAMES
from ssd.rollout import rollout_chunked


def r2(y: np.ndarray, x: np.ndarray) -> float:
    """Share of the variance of y explained by least squares on x (with an intercept)."""
    if y.var() == 0:
        return float("nan")
    a = np.c_[x, np.ones(len(y))]
    coef, *_ = np.linalg.lstsq(a, y, rcond=None)
    return float(1 - ((y - a @ coef) ** 2).sum() / ((y - y.mean()) ** 2).sum())


@torch.no_grad()
def evaluate(run_dir: str, max_skills: int, repeats: int, pool: int, device, rng) -> dict:
    agent, run, _ = load_agent(os.path.join(run_dir, "checkpoint.pt"), device)
    k = run["dim_skill"]
    signs = 2 * ((np.arange(2 ** k)[:, None] >> np.arange(k)) & 1) - 1
    if len(signs) > max_skills:
        signs = signs[np.sort(rng.choice(len(signs), max_skills, replace=False))]
    s = np.repeat(signs, repeats, axis=0)
    seed = 666000 + 100 * int(run["seed"])
    envs = [make_env("kitchen", seed=seed + i) for i in range(pool)]
    traj = rollout_chunked(envs, agent.policy, (s / np.sqrt(k)).astype(np.float32), run["horizon"], device,
                           deterministic=True)
    for env in envs:
        if hasattr(env, "close"):
            env.close()
    succ = traj["success"].astype(float)  # [n, tasks], achieved by the end of the episode

    effects = np.array([[succ[s[:, a] > 0, t].mean() - succ[s[:, a] < 0, t].mean() for t in range(succ.shape[1])]
                        for a in range(k)])  # [K, tasks]
    inter = np.array([s[:, i] * s[:, j] for i, j in itertools.combinations(range(k), 2)]).T
    corner = np.unique(s, axis=0, return_inverse=True)[1].reshape(-1)
    onehot = np.eye(corner.max() + 1)[corner][:, 1:]
    fits = np.array([[r2(succ[:, t], s), r2(succ[:, t], np.c_[s, inter]), r2(succ[:, t], onehot)]
                     for t in range(succ.shape[1])])  # [tasks, (main, +pairs, saturated)]
    agree = np.mean(succ.reshape(-1, repeats, succ.shape[1]).std(1) == 0) if repeats > 1 else float("nan")
    return dict(run=os.path.basename(run_dir), k=k, signs=s, success=succ, effects=effects, fits=fits,
                rate=succ.mean(0), repeat_agreement=agree)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", required=True, help="glob over Kitchen run directories with checkpoint.pt")
    p.add_argument("--max_skills", type=int, default=128)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--pool", type=int, default=4, help="parallel environments (each holds a GPU renderer)")
    p.add_argument("--out", required=True, help="output prefix: <out>.txt and <out>_<run>.npz")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    runs = sorted((d for d in glob.glob(args.runs) if os.path.exists(os.path.join(d, "checkpoint.pt"))),
                  key=lambda d: int(d.rsplit("_s", 1)[1]))
    lines = [f"{args.runs}: up to {args.max_skills} corners, {args.repeats} episodes each",
             "per task: success rate | dominant axis (sign of its effect) and its effect | share of squared main "
             "effects on it | R2 main effects / + pairwise / saturated"]
    for d in runs:
        r = evaluate(d, args.max_skills, args.repeats, args.pool, device, rng)
        np.savez(f"{args.out}_{r['run']}.npz", **{k: v for k, v in r.items() if k != "run"})
        lines.append(f"== {r['run']} (K={r['k']}, repeat agreement {r['repeat_agreement']:.2f})")
        for t, name in enumerate(GOAL_NAMES):
            e = r["effects"][:, t]
            a = int(np.argmax(np.abs(e)))
            share = e[a] ** 2 / max((e ** 2).sum(), 1e-12)
            m, pw, sat = r["fits"][t]
            lines.append(f"  {name:13s} {r['rate'][t]:.2f} | axis {a + 1}{'+' if e[a] > 0 else '-'} {e[a]:+.2f} | "
                         f"{share:.2f} | {m:.2f} / {pw:.2f} / {sat:.2f}")
        print("\n".join(lines[-7:]), flush=True)
    with open(f"{args.out}.txt", "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
