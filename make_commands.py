"""Generate the SSD training commands for the comparison with METRA, one per (env, seed).

    uv run make_commands.py --out commands/commands.txt
    uv run make_commands.py --smoke --out commands/smoke.txt
    uv run make_commands.py --sweep --out commands/sweep.txt
"""

import argparse
from pathlib import Path

COMMON = "--rollouts_per_iter 8 --min_buffer_size 10000 --n_log 100 --n_save 2500 --resume 1"

ENVS = {
    # METRA: --dim_option 2 --discrete 0 --trans_optimization_epochs 50 --sac_max_buffer_size 1000000
    "ant": "--env ant --horizon 200 --dim_skill 2 --discrete 0 --unit_skills 1 --updates_per_iter 50 "
           "--buffer_size 1000000 --eval_plot_axis 50",
    # METRA: --dim_option 16 --discrete 1 --trans_optimization_epochs 50 --sac_max_buffer_size 1000000
    # SSD: 16 cube skills (K = 4) in place of 16 one-hot skills
    "half_cheetah": "--env half_cheetah --horizon 200 --dim_skill 4 --discrete 1 --cube_skills 1 "
                    "--updates_per_iter 50 --buffer_size 1000000",
    # METRA: pixels, --frame_stack 3, action repeat 2, --dim_option 4, --trans_optimization_epochs 200,
    # --sac_max_buffer_size 300000, --n_epochs_per_log 25
    "dmc_quadruped": "--env dmc_quadruped --horizon 200 --dim_skill 4 --discrete 0 --unit_skills 1 "
                     "--temporal_per_dim 1 --updates_per_iter 200 --buffer_size 300000 --eval_plot_axis 15 "
                     "--n_readout_trajs 128 --n_log 25",
    # METRA: as Quadruped with --dim_option 2
    "dmc_humanoid": "--env dmc_humanoid --horizon 200 --dim_skill 2 --discrete 0 --unit_skills 1 --temporal_per_dim 1 "
                    "--updates_per_iter 200 --buffer_size 300000 --eval_plot_axis 15 --n_readout_trajs 128 --n_log 25",
    # METRA: pixels, --max_path_length 50, --dim_option 24 --discrete 1, --trans_optimization_epochs 100,
    # --sac_max_buffer_size 100000, --sac_lr_a -1 (fixed temperature), --n_epochs_per_log 25.
    # SSD: 32 cube skills (K = 5) in place of 24 one-hot skills; 4 shared envs to save GPU memory
    "kitchen": "--env kitchen --horizon 50 --dim_skill 5 --discrete 1 --cube_skills 1 --temporal_per_dim 1 "
               "--updates_per_iter 100 --buffer_size 100000 --lr_alpha 0 --n_readout_trajs 128 --n_log 25 "
               "--n_train_envs 4 --share_envs 1",
}

# pixel envs: one encoder per dimension and scale-aware decorrelation
PIXEL_DECOR = "--encoder_per_dim 1 --decor_scale 10"
for _env in ("dmc_quadruped", "dmc_humanoid", "kitchen"):
    ENVS[_env] += " " + PIXEL_DECOR

# training length and evaluation period of METRA's benchmark, in METRA epochs = SSD iterations
N_ITERS = {"ant": 10001, "half_cheetah": 10001, "dmc_quadruped": 2001, "dmc_humanoid": 3201, "kitchen": 2501}
N_EVAL = {"ant": 500, "half_cheetah": 500, "dmc_quadruped": 100, "dmc_humanoid": 160, "kitchen": 125}

# hyperparameter sweep: one value changed at a time, 8M steps, 3 seeds
SWEEP = {"eta_beta": ["0.001", "0.1"], "c0": ["0.1", "10"], "kappa": ["0", "0.0001"], "n_dual": ["1", "100"],
         "nu0": ["3"]}
SWEEP_ENVS, SWEEP_SEEDS, SWEEP_ITERS = ("ant", "half_cheetah"), (0, 1, 2), 5001


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="commands/commands.txt")
    p.add_argument("--run_group", default="PaperV2")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    p.add_argument("--envs", nargs="+", default=list(ENVS))
    p.add_argument("--smoke", action="store_true", help="one short run per environment")
    p.add_argument("--sweep", action="store_true", help="the hyperparameter sweep on Ant and HalfCheetah")
    args = p.parse_args()

    cmds = []
    if args.sweep:
        for env in SWEEP_ENVS:
            for seed in SWEEP_SEEDS:
                for name, values in SWEEP.items():
                    for value in values:
                        cmds.append(f"python train.py --run_group Tune/{env}_{name}{value} --seed {seed} {COMMON} "
                                    f"{ENVS[env]} --n_iters {SWEEP_ITERS} --n_eval {N_EVAL[env]} --{name} {value}")
    else:
        seeds = [0] if args.smoke else args.seeds
        group = "Smoke" if args.smoke else args.run_group
        for env in args.envs:
            for seed in seeds:
                cmd = f"python train.py --run_group {group} --seed {seed} {COMMON} {ENVS[env]}"  # env flags win
                if args.smoke:
                    cmd += " --n_iters 30 --n_log 5 --n_eval 15 --min_buffer_size 3000 --n_readout_trajs 16"
                else:
                    cmd += f" --n_iters {N_ITERS[env]} --n_eval {N_EVAL[env]}"
                cmds.append(cmd)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(cmds) + "\n")
    print(f"Wrote {len(cmds)} commands to {args.out}")


if __name__ == "__main__":
    main()
