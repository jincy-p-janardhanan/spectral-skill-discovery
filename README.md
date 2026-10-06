# Spectral Skill Discovery (SSD)

Implementation of the SSD algorithm (Spectral Skill Discovery): a METRA-style skill learner whose encoder
dimensions are **ordered** by asymmetric decorrelation constraints, trained with an augmented
Lagrangian (duals β, ν and a growing barrier c). A permutation-tested **spectrum readout** estimates
the intrinsic skill dimension K⋆.

State-based environments, observations, normaliser presets, network sizes and SAC details follow
the METRA reference implementation, so results are directly comparable. The code
itself is new and self-contained: METRA's garage / gym / mujoco-py stack only runs on Python 3.8.

## Requirements

- Python 3.14, PyTorch 2.14 (CUDA 12.6 build), Gymnasium 1.3, MuJoCo 3.14, NumPy 2.5
- [uv](https://docs.astral.sh/uv/) (already at `~/.local/bin/uv`)

```bash
uv sync            # creates .venv with the pinned versions from uv.lock
uv run pytest      # unit tests (no MuJoCo needed)
```

**Cluster notes.**
- The login node's CPU has no AVX, and MuJoCo 3 needs it. Anything that simulates must run on a
  compute node; the unit tests run anywhere.
- torch comes from PyTorch's `cu126` index because the A40 nodes have driver 570 (CUDA 12.8), and
  PyPI's default torch 2.14 wheel needs CUDA 13.
- `cn22-a40` failed every job during setup, so exclude it with `--exclude=cn22-a40`.

## Running

```bash
# State-based Ant, K = 8 (METRA's Ant throughput: 8 rollouts x 200 steps, 50 updates per iteration)
sbatch --exclude=cn22-a40 ssd.slurm --env ant --dim_skill 8 --seed 0

# HalfCheetah
sbatch --exclude=cn22-a40 ssd.slurm --env half_cheetah --dim_skill 8 --seed 0

# One rollout and one update per iteration
uv run train.py --env ant --rollouts_per_iter 1 --updates_per_iter 1 --n_dual 10 --n_eval 20000
```

To push skills beyond walking, `--encoder_hide_pos 1` hides the global position (Ant x, y) from φ; the policy
and critics still see the full state. With position hidden, walking no longer raises the intrinsic reward
directly, so skills have to change the body itself.

Every evaluation logs three coverage metrics over 48 deterministic rollouts:
- `eval/coverage`: unique 1 m x-y cells (METRA's metric).
- `eval/orientation_coverage`: unique 30° cells of torso roll, pitch and yaw.
- `eval/posture_coverage`: unique cells of the 8 joint angles, 5 bins per joint.

Each evaluation also saves a per-dimension sweep figure (skills z = α·eₖ). The same figure, plus videos, can
be made from any checkpoint (on a GPU node; rendering uses EGL):

```bash
uv run visualize.py "exp/<group>/<run>/checkpoint.pt" --out figures/sweeps/run.png --axis 50 --video figures/videos/run
```

`uv run train.py --help` lists every flag. The algorithm's hyperparameters all live in
`SSDConfig` ([ssd/algorithm.py](ssd/algorithm.py)), and each field is exposed as a flag of the same name.

Outputs go to `exp/<run_group>/<env>_K<K>_s<seed>_<time>/`:
- `progress.csv`: training scalars (J, h, |C|, ν, c, SAC stats).
- `eval.csv`: K⋆, σ²ₖ, Rₖ and floorₖ for every k, plus coverage.
- `plots/`: spectrum and x-y trajectory figures.
- `tb/`: TensorBoard logs.
- `checkpoint.pt`: saved checkpoint.

## Comparison with METRA

`make_commands.py` mirrors METRA's benchmark: the same 5 environments, 8 seeds, training lengths
and evaluation cadence. One METRA epoch (8 trajectories, then `trans_optimization_epochs` updates) is one
SSD iteration.

| Env | Obs | Skills | T | Updates / iter | Buffer | Iterations | Eval every |
|---|---|---|---|---|---|---|---|
| `ant` | state (29-D) | 2-D unit-length | 200 | 50 | 1M | 10,001 | 500 |
| `half_cheetah` | state (18-D) | 16 cube (K = 4) | 200 | 50 | 1M | 10,001 | 500 |
| `dmc_quadruped` | 64x64 RGB x3, action repeat 2 | 4-D unit-length | 200 | 200 | 300k | 2,001 | 100 |
| `dmc_humanoid` | 64x64 RGB x3, action repeat 2 | 2-D unit-length | 200 | 200 | 300k | 3,201 | 160 |
| `kitchen` | 64x64 RGB x3 | 32 cube (K = 5), fixed α | 50 | 100 | 100k | 2,501 | 125 |

METRA uses the same settings with 16 (HalfCheetah) and 24 (Kitchen) one-hot skills. The pixel environments also
use METRA's per-dimension temporal bound (`--temporal_per_dim 1`), one encoder network per dimension
(`--encoder_per_dim 1`) and scale-aware decorrelation (`--decor_scale 10`); without the last two, decorrelation
collapses dimension 1 through the shared encoder and learning stalls on pixels. The skill choices, the fixes
behind them and the results are written up in [report/ssd_metra_report.pdf](report/ssd_metra_report.pdf).

```bash
uv run make_commands.py --out commands/commands.txt            # 40 commands, one per (env, seed)
CMDS=commands/commands.txt START=0 COUNT=8 sbatch --account=jincypjanardhanan run_commands.slurm
```

The ports needed to match METRA are listed below.
- **Discrete skills** (`--discrete 1`): one-hot z, and the reward uses METRA's centred mask
  (z − mean z)·K/(K−1). The SSD algorithm samples z ~ N(0, I_K); this mode exists only to match METRA's settings.
  The mask needs correlated dimensions, which the decorrelation constraints forbid (HalfCheetah: 70 x-cells vs
  METRA's 161 at 8M steps). The benchmark runs therefore use **cube skills** (`--discrete 1 --cube_skills 1`): the 2^K
  corners z ∈ {−1, 1}^K/√K, with the reward on z itself. They are unit-length and isotropic, and every skill
  trains every dimension; K = 4 gives METRA's 16 skills and reaches 169. `--signed_skills 1` gives the 2K
  skills ±e_k instead (132).
- **DMC** (`ssd/dmc.py`, `ssd/custom_dmc_tasks/`): METRA's coloured-floor tasks on current `dm_control`.
  The only change in METRA's task code is `np.in1d` → `np.isin` for NumPy 2.
- **Kitchen** (`ssd/kitchen/`): the D4RL / RAPS end-effector kitchen and LEXA wrapper that METRA uses,
  vendored with compatibility fixes only. It already runs on `dm_control` in METRA.
- **Pixels**: every network (φ, policy, both critics) has its own copy of METRA's CNN. The replay buffer
  stores single frames and rebuilds the 3-frame stacks when sampling.

Cluster notes. The `a40` QOS allows 3 jobs of at most 2 GPUs per user, and `l40` / `dgx` allow 4 jobs of
up to 4 GPUs. `debug` is often closed to new submissions. The `cccp` account is shared with the lab, so
SSD jobs run under the personal account `jincypjanardhanan`.

## Mapping to the SSD algorithm

| Part | Where |
|---|---|
| Initialisation (θ, θ̄, ω, ψ, β=0, ν₀, c₀, D) | `SpectralSkillDiscovery.__init__`, `TrajectoryBuffer` |
| Sample z ~ N(0, I_K), roll out T steps | `train.py` loop, `ssd/rollout.py` |
| Intrinsic reward, SAC | `SpectralSkillDiscovery.update`, `ssd/sac.py` |
| Two batches, Cₖⱼ, h | `spectral.decorrelation`, `spectral.temporal_hinge` |
| Ĵ, L, ascent on θ | `spectral.skill_objective`, `spectral.lagrangian` (Adam on −L) |
| Duals every n_dual updates | `SpectralSkillDiscovery.update` |
| Barrier, EMA target | `SpectralSkillDiscovery.update` |
| Spectrum readout, permutation test, K⋆ | `SpectralSkillDiscovery.spectrum`, `spectral.spectrum_readout` |

## Choices not fixed by the algorithm

- **Iteration granularity.** One outer iteration collects `--rollouts_per_iter` trajectories, then
  runs `--updates_per_iter` updates. `n_dual` and the barrier schedule count those updates,
  and `n_eval` counts outer iterations. Updates start once D holds `--min_buffer_size` transitions,
  as in METRA.
- **EMA target θ̄.** It is maintained, but by default neither the reward
  nor the readout uses θ̄. `--reward_use_target 1` and `--readout_use_target 1` switch them over.
- **Gradient through σₖ .** If the whole denominator sg(σₖσⱼ + ε) is stop-gradiented, the gradient
  on φₖ contains an extra C·φ̃ₖ/σₖ² term, so the
  barrier shrinks φₖ, faster and faster as σₖ falls, until the shared ReLU layers die and φ is
  constant. This happened on Ant: σ² reached 0 and K⋆ stayed 0. By default (`--sigma_k_grad 1`)
  the denominator is σₖ·sg(σⱼ) + ε, which makes the gradient scale-invariant, while gradients
  still reach only the later dimension k. `--sigma_k_grad 0` stop-gradients the whole denominator.
- **States in Cₖⱼ.** The decorrelation statistics use s from each batch. ε is added to the
  denominator, and σ uses population std.
- **Effect size in K⋆ .** A dimension also needs R_k ≥ 0.05 · max_j R_j
  (`--readout_min_ratio`, 0 turns it off). The permutation test alone passes real but negligible effects once
  N is large, so K⋆ grew with N and varied between repeated readouts (`readout_sweep.py`, figures/readout/).
- **Readout trajectories.** These are N trajectories sampled from D. Separately, a
  deterministic evaluation rollout with fresh z gives METRA's coverage metric and the trajectory plot.
- **Dual, barrier and encoder rates.** The defaults in `SSDConfig` (ν₀=30 as METRA's λ₀; ηβ=ην=1e-2,
  n_dual=10, c₀=1, κ=1e-5, c_max=100) are reasonable starting points but have **not been tuned**.
