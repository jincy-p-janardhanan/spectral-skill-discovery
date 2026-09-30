"""Train Spectral Skill Discovery.

Each iteration collects --rollouts_per_iter trajectories, then runs --updates_per_iter updates.
The spectrum readout runs every --n_eval iterations.

    uv run train.py --env ant --dim_skill 8 --seed 0
"""

import argparse
import csv
import dataclasses
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from ssd.algorithm import SpectralSkillDiscovery, SSDConfig
from ssd.buffer import TrajectoryBuffer
from ssd.envs import ALL_ENVS, encoder_input_idx, make_env
from ssd.plotting import plot_dimension_sweeps, plot_spectrum, plot_trajectories
from ssd.metrics import body_coverage, coverage
from ssd.rollout import SWEEP_ALPHAS, dimension_sweep, rollout, rollout_chunked


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", default="ant", choices=ALL_ENVS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--run_group", default="Debug")
    p.add_argument("--exp_dir", default="exp")
    p.add_argument("--normalize_obs", type=int, default=1, help="METRA's 'preset' observation normaliser")
    p.add_argument("--unit_skills", type=int, default=0,
                   help="normalise z to unit length as METRA does (diagnostic; the SSD algorithm uses z ~ N(0, I_K))")
    p.add_argument("--encoder_hide_pos", type=int, default=0,
                   help="hide the global position (Ant x, y) from phi; the policy and critics still see it")

    p.add_argument("--horizon", type=int, default=200, help="T")
    p.add_argument("--n_iters", type=int, default=100_000)
    p.add_argument("--rollouts_per_iter", type=int, default=8)
    p.add_argument("--updates_per_iter", type=int, default=50)
    p.add_argument("--buffer_size", type=int, default=1_000_000)
    p.add_argument("--min_buffer_size", type=int, default=10_000)

    p.add_argument("--n_log", type=int, default=100, help="log period (iterations)")
    p.add_argument("--n_eval", type=int, default=1000, help="n_eval: spectrum readout / eval period (iterations)")
    p.add_argument("--n_save", type=int, default=10_000, help="checkpoint period (iterations)")
    p.add_argument("--n_eval_rollouts", type=int, default=48, help="deterministic rollouts for plots / coverage")
    p.add_argument("--n_eval_envs", type=int, default=None,
                   help="evaluation env pool, reused in chunks (default: one env per evaluation rollout)")
    p.add_argument("--eval_plot_axis", type=float, default=None, help="half-width of the trajectory plot")
    p.add_argument("--resume", type=int, default=0,
                   help="1: deterministic run directory <env>_K<K>_s<seed>; if it holds resume.pt, continue from it")
    p.add_argument("--resume_every_min", type=float, default=30.0,
                   help="with --resume 1, write resume.pt (agent, optimisers, replay buffer, RNGs) this often")

    # one flag per SSDConfig field
    for f in dataclasses.fields(SSDConfig):
        kind = int if f.type is bool else f.type
        p.add_argument(f"--{f.name}", type=kind, default=f.default)
    return p.parse_args()


class Logger:
    def __init__(self, log_dir: Path, resume_iter: int | None = None, purge_step: int | None = None):
        self.dir = log_dir
        self.tb = SummaryWriter(log_dir / "tb", purge_step=purge_step)
        self._files = {}
        # resume_iter: keep CSV rows up to this iteration and append to them.
        if resume_iter is not None:
            for path in log_dir.glob("*.csv"):
                with path.open(newline="") as old:
                    reader = csv.DictReader(old)
                    fields, rows = reader.fieldnames, [r for r in reader if float(r["iter"]) <= resume_iter]
                if not fields:
                    continue
                f = path.open("w", newline="")
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
                f.flush()
                self._files[path.stem] = (f, writer)

    def scalars(self, name: str, step: int, row: dict[str, float]):
        for k, v in row.items():
            self.tb.add_scalar(k, v, step)
        path = self.dir / f"{name}.csv"
        if name in self._files and not set(row) <= set(self._files[name][1].fieldnames):
            # new columns: rewrite the file with the wider header
            f, writer = self._files.pop(name)
            f.close()
            with path.open(newline="") as old:
                rows = list(csv.DictReader(old))
            fields = list(writer.fieldnames) + [k for k in row if k not in writer.fieldnames]
            f = path.open("w", newline="")
            self._files[name] = (f, csv.DictWriter(f, fieldnames=fields))
            self._files[name][1].writeheader()
            self._files[name][1].writerows(rows)
        if name not in self._files:
            f = path.open("w", newline="")
            self._files[name] = (f, csv.DictWriter(f, fieldnames=list(row)))
            self._files[name][1].writeheader()
        f, writer = self._files[name]
        writer.writerow(row)
        f.flush()

    def figure(self, tag: str, fig, step: int):
        (self.dir / "plots").mkdir(exist_ok=True)
        fig.savefig(self.dir / "plots" / f"{tag}_{step:08d}.png", dpi=110)
        self.tb.add_figure(tag, fig, step)
        plt.close(fig)


def sample_skills(rng, n: int, k: int, unit: bool, discrete: bool = False, even: bool = False,
                  signed: bool = False, cube: bool = False) -> np.ndarray:
    """z ~ N(0, I_K), optionally unit-length, or discrete skills: one-hot, signed (+-e_k) or cube corners.

    even=True cycles through the discrete skills instead of sampling them.
    """
    if discrete and cube:
        idx = np.arange(n) % 2 ** k if even else rng.integers(2 ** k, size=n)
        bits = (idx[:, None] >> np.arange(k)) & 1
        return ((2 * bits - 1) / np.sqrt(k)).astype(np.float32)
    if discrete:
        m = 2 * k if signed else k
        idx = np.arange(n) % m if even else rng.integers(m, size=n)
        sign = np.where(idx < k, 1.0, -1.0).astype(np.float32)
        return np.eye(k, dtype=np.float32)[idx % k] * sign[:, None]
    z = rng.standard_normal((n, k))
    return z / np.linalg.norm(z, axis=1, keepdims=True) if unit else z


def main():
    args = parse_args()
    cfg = SSDConfig(**{f.name: f.type(getattr(args, f.name)) for f in dataclasses.fields(SSDConfig)})
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    run_name = f"{args.env}_K{cfg.dim_skill}_s{args.seed}"
    if not args.resume:
        run_name += f"_{time.strftime('%Y%m%d-%H%M%S')}"
    log_dir = Path(args.exp_dir) / args.run_group / run_name
    resume_path = log_dir / "resume.pt"
    state = None
    if args.resume and resume_path.exists():
        state = torch.load(resume_path, map_location=device, weights_only=False)
    log_dir.mkdir(parents=True, exist_ok=bool(args.resume))
    if state is None:
        (log_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    start_it = state["iter"] if state else 0
    logger = Logger(log_dir, resume_iter=start_it if args.resume else None,
                    purge_step=state["env_steps"] + 1 if state else None)
    print(f"Logging to {log_dir}" + (f" (resuming after iteration {start_it})" if state else ""), flush=True)

    seed_offset = 100_003 * start_it  # fresh env seeds after a resume
    train_envs = [make_env(args.env, seed=args.seed * 10_000 + i + seed_offset, normalize=bool(args.normalize_obs))
                  for i in range(args.rollouts_per_iter)]
    eval_envs = [make_env(args.env, seed=args.seed * 10_000 + 5_000 + i + seed_offset,
                          normalize=bool(args.normalize_obs))
                 for i in range(args.n_eval_envs or args.n_eval_rollouts)]
    obs_dim = train_envs[0].observation_space.shape[0]
    space = train_envs[0].action_space
    k = cfg.dim_skill
    pixel_shape = getattr(train_envs[0], "pixel_shape", None)
    action_repeat = getattr(train_envs[0], "action_repeat", 1)
    has_body = hasattr(eval_envs[0], "joint_range")

    agent = SpectralSkillDiscovery(cfg, obs_dim, space.shape[0], space.low, space.high, device, seed=args.seed,
                                   encoder_input_idx=encoder_input_idx(args.env, obs_dim, bool(args.encoder_hide_pos)),
                                   pixel_shape=pixel_shape)
    buffer = TrajectoryBuffer(args.buffer_size, args.horizon, obs_dim, space.shape[0], k, device,
                              frame_stack=pixel_shape[0] // 3 if pixel_shape else 1,
                              obs_dtype=torch.uint8 if pixel_shape else torch.float32)

    env_steps = 0
    elapsed_before = 0.0
    if state is not None:
        agent.load_state_dict(state["agent"])
        buffer.load_state_dict(state["buffer"])
        rng.bit_generator.state = state["np_rng"]
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        env_steps, elapsed_before = state["env_steps"], state["elapsed_s"]
        del state
    sums: dict[str, torch.Tensor] = defaultdict(lambda: torch.zeros((), device=device))
    n_summed = 0
    t_start = time.time() - elapsed_before
    last_resume_save = time.time()

    for it in range(start_it + 1, args.n_iters + 1):
        # rollouts
        z = sample_skills(rng, args.rollouts_per_iter, k, bool(args.unit_skills), cfg.discrete,
                          signed=cfg.signed_skills, cube=cfg.cube_skills)
        traj = rollout(train_envs, agent.policy, z, args.horizon, device)
        buffer.add(traj["obs"], traj["act"], traj["done"], traj["z"])
        env_steps += args.rollouts_per_iter * args.horizon * action_repeat

        # updates
        if buffer.n_transitions >= args.min_buffer_size:
            for _ in range(args.updates_per_iter):
                for key, val in agent.update(buffer).items():
                    sums[key] += val
                n_summed += 1

        if it % args.n_log == 0:
            row = {"iter": it, "env_steps": env_steps, "updates": agent.n_updates, "barrier_c": agent.c,
                   "train/coverage": coverage(traj["xy"], args.env),
                   "time/elapsed_s": time.time() - t_start}
            row.update({key: (val / max(n_summed, 1)).item() for key, val in sums.items()})
            logger.scalars("progress", env_steps, row)
            print(f"it {it:7d} | steps {env_steps:10d} | J {row.get('ssd/J', float('nan')):8.3f} "
                  f"| h {row.get('ssd/h', float('nan')):.4f} | |C| {row.get('ssd/C_abs_mean', float('nan')):.4f} "
                  f"| nu {row.get('ssd/nu', float('nan')):.3f} | c {agent.c:.3f} "
                  f"| alpha {row.get('sac/alpha', float('nan')):.4f} | {row['time/elapsed_s']:.0f}s", flush=True)
            sums.clear()
            n_summed = 0

        # spectrum readout and evaluation
        if it % args.n_eval == 0 and agent.n_updates > 0:
            spec = agent.spectrum(buffer)
            row = {"iter": it, "env_steps": env_steps, "spectrum/K_star": spec.k_star}
            for i in range(k):
                row[f"spectrum/sigma2_{i + 1}"] = spec.sigma2[i].item()
                row[f"spectrum/R_{i + 1}"] = spec.returns[i].item()
                row[f"spectrum/floor_{i + 1}"] = spec.floor[i].item()

            ez = sample_skills(rng, args.n_eval_rollouts, k, bool(args.unit_skills), cfg.discrete, even=True,
                               signed=cfg.signed_skills, cube=cfg.cube_skills)
            ev = rollout_chunked(eval_envs, agent.policy, ez, args.horizon, device, deterministic=True,
                                 record_raw=has_body)
            row["eval/coverage"] = coverage(ev["xy"], args.env)
            if has_body:
                row.update({f"eval/{key}": val for key, val in body_coverage(ev["raw"], eval_envs[0]).items()})
            if args.env == "kitchen":
                from ssd.kitchen.env import kitchen_metrics

                row.update({f"eval/{key}": val for key, val in kitchen_metrics(ev["success"]).items()})
            logger.scalars("eval", env_steps, row)
            logger.figure("spectrum", plot_spectrum(spec), env_steps)
            logger.figure("trajectories", plot_trajectories(ev["xy"], ez, args.env, args.eval_plot_axis), env_steps)
            if not cfg.discrete:
                sweep = dimension_sweep(eval_envs[:2 * len(SWEEP_ALPHAS)], agent.policy, agent.encoder, k,
                                        args.horizon, device)
                logger.figure("sweeps", plot_dimension_sweeps(sweep, args.env, args.eval_plot_axis), env_steps)
            print(f"[eval] it {it} | K* = {spec.k_star} | coverage {row['eval/coverage']} "
                  f"| orientation {row.get('eval/orientation_coverage', '-')} | posture {row.get('eval/posture_coverage', '-')} "
                  f"| kitchen {row.get('eval/KitchenOverall', '-')} "
                  f"| R = {np.array2string(spec.returns.cpu().numpy(), precision=3)}", flush=True)

        if it % args.n_save == 0 or it == args.n_iters:
            ckpt = {"iter": it, "env_steps": env_steps, "config": vars(args), "agent": agent.state_dict()}
            torch.save(ckpt, log_dir / "checkpoint.pt")
            (log_dir / "checkpoints").mkdir(exist_ok=True)
            torch.save(ckpt, log_dir / "checkpoints" / f"ckpt_{env_steps:09d}.pt")

        if args.resume and (time.time() - last_resume_save >= 60 * args.resume_every_min or it == args.n_iters):
            snapshot = {"iter": it, "env_steps": env_steps, "elapsed_s": time.time() - t_start,
                        "agent": agent.state_dict(), "buffer": buffer.state_dict(),
                        "np_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
            tmp = resume_path.with_suffix(".tmp")
            torch.save(snapshot, tmp)
            os.replace(tmp, resume_path)
            del snapshot
            last_resume_save = time.time()


if __name__ == "__main__":
    main()
