"""Constraints, objective and spectrum readout of Spectral Skill Discovery."""

from dataclasses import dataclass

import torch


def lower_pairs(k: int, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Indices (k, j) of all pairs with j < k."""
    idx = torch.tril_indices(k, k, offset=-1, device=device)
    return idx[0], idx[1]


def decorrelation(phi: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor, eps: float) -> torch.Tensor:
    """Asymmetric correlation estimates C_kj for j < k on one batch.

    C_kj = mean_B[ phi~_k(s) * sg(phi~_j(s)) ] / sg(sigma_k * sigma_j + eps)

    Gradients flow only through the later dimension k, so dimension k is pushed to decorrelate
    from every earlier dimension j < k while earlier dimensions are left untouched. This is what
    orders the representation.

    Args:
        phi: encoder outputs phi(s) on the batch, shape [B, K].
        rows, cols: output of `lower_pairs` (k and j indices).
    Returns:
        Flat tensor of the K(K-1)/2 entries C_kj, ordered like (rows, cols).
    """
    phi_c = phi - phi.mean(dim=0, keepdim=True)
    sigma = phi.detach().std(dim=0, correction=0)
    cov = phi_c.T @ phi_c.detach() / phi.shape[0]
    denom = sigma[:, None] * sigma[None, :] + eps
    return (cov / denom)[rows, cols]


def temporal_hinge(phi_s: torch.Tensor, phi_next: torch.Tensor) -> torch.Tensor:
    """h = mean_(s,s') max(0, ||phi(s) - phi(s')||^2 - 1)."""
    return torch.clamp((phi_s - phi_next).square().sum(dim=-1) - 1.0, min=0.0).mean()


def skill_objective(phi_s: torch.Tensor, phi_next: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Per-sample (phi(s') - phi(s))^T z: the intrinsic reward, and J after averaging."""
    return ((phi_next - phi_s) * z).sum(dim=-1)


def lagrangian(
    j_hat: torch.Tensor,
    c1: torch.Tensor,
    h1: torch.Tensor,
    c2: torch.Tensor,
    h2: torch.Tensor,
    beta: torch.Tensor,
    nu: float,
    c: float,
) -> torch.Tensor:
    """Augmented Lagrangian, maximised w.r.t. theta.

    c * sg(C2) * C1 gives an unbiased gradient of the penalty (c/2) * C^2 (likewise for h).
    """
    penalty = (c2.detach() * c1).sum() + h2.detach() * h1
    return j_hat - nu * h1 - (beta * c1).sum() - c * penalty


@dataclass
class Spectrum:
    sigma2: torch.Tensor  # [K] spread of each dimension, Var_rho[phi_k(s)]
    returns: torch.Tensor  # [K] per-dimension return R_k
    floor: torch.Tensor  # [K] permutation-null quantile of R_k
    k_star: int  # number of leading dimensions with R_k > floor_k


@torch.no_grad()
def spectrum_readout(
    phi_traj: torch.Tensor,
    z: torch.Tensor,
    n_shuffles: int,
    quantile: float = 0.95,
    generator: torch.Generator | None = None,
) -> Spectrum:
    """Spectrum readout with a permutation test.

    Args:
        phi_traj: encoder outputs along N trajectories, shape [N, T+1, K].
        z: the skill each trajectory was collected with, shape [N, K].
        n_shuffles: number M of permutations of z across trajectories.
    """
    n, _, k = phi_traj.shape
    sigma2 = phi_traj.reshape(-1, k).var(dim=0, correction=0)
    delta = phi_traj[:, -1] - phi_traj[:, 0]  # [N, K]
    returns = (delta * z).mean(dim=0)

    # null distribution: shuffle z across trajectories
    perms = torch.argsort(torch.rand(n_shuffles, n, device=z.device, generator=generator), dim=1)
    null = (delta.unsqueeze(0) * z[perms]).mean(dim=1)  # [M, K]
    floor = torch.quantile(null, quantile, dim=0)

    passed = (returns > floor).int()
    k_star = int(passed.cumprod(dim=0).sum().item())
    return Spectrum(sigma2=sigma2, returns=returns, floor=floor, k_star=k_star)
