import pytest
import torch

from ssd import spectral
from ssd.algorithm import SpectralSkillDiscovery, SSDConfig
from ssd.buffer import TrajectoryBuffer


def test_decorrelation_value_is_pearson_correlation():
    torch.manual_seed(0)
    x = torch.randn(4096, 4)
    x[:, 2] += 0.8 * x[:, 0]
    rows, cols = spectral.lower_pairs(4)
    c = spectral.decorrelation(x, rows, cols, eps=0.0)
    ref = torch.corrcoef(x.T)[rows, cols]
    torch.testing.assert_close(c, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("sigma_k_grad", [True, False])
def test_decorrelation_gradient_flows_only_through_later_dimension(sigma_k_grad):
    torch.manual_seed(0)
    phi = torch.randn(512, 3, requires_grad=True)
    rows, cols = spectral.lower_pairs(3)
    spectral.decorrelation(phi, rows, cols, eps=1e-6, sigma_k_grad=sigma_k_grad).sum().backward()
    # Dimension 1 only ever appears as j (the stop-gradient side) -> no gradient.
    assert torch.all(phi.grad[:, 0] == 0)
    assert phi.grad[:, 1].abs().sum() > 0 and phi.grad[:, 2].abs().sum() > 0


@pytest.mark.parametrize("sigma_k_grad", [True, False])
def test_decorrelation_gradient_scale_invariance(sigma_k_grad):
    """With sigma_k in the graph, dC_kj/dphi_k is orthogonal to phi~_k; without it, it is not."""
    torch.manual_seed(0)
    x = torch.randn(1024, 2)
    x[:, 1] += 0.6 * x[:, 0]
    phi = x.clone().requires_grad_()
    rows, cols = spectral.lower_pairs(2)
    spectral.decorrelation(phi, rows, cols, eps=0.0, sigma_k_grad=sigma_k_grad).sum().backward()
    phi_c = x[:, 1] - x[:, 1].mean()
    radial = (phi.grad[:, 1] * phi_c).sum() / (phi.grad[:, 1].norm() * phi_c.norm())
    if sigma_k_grad:
        assert radial.abs() < 1e-5
    else:
        assert radial > 0.1  # gradient ascent on C grows phi_k, so descent on C^2 shrinks it


def test_temporal_hinge():
    a = torch.zeros(2, 2)
    b = torch.tensor([[0.5, 0.0], [2.0, 0.0]])  # squared distances 0.25 and 4
    assert spectral.temporal_hinge(a, b).item() == 1.5  # mean(0, 3)
    # METRA's convention averages over the K = 2 dimensions: mean(max(0, 0.125 - 1), max(0, 2 - 1)) = 0.5
    assert spectral.temporal_hinge(a, b, per_dim=True).item() == 0.5


def test_barrier_gradient_matches_quadratic_penalty():
    """With B1 == B2, grad of c*sg(C2)*C1 equals grad of (c/2)*C^2."""
    torch.manual_seed(0)
    phi = torch.randn(256, 3)
    rows, cols = spectral.lower_pairs(3)
    p1 = phi.clone().requires_grad_()
    c1 = spectral.decorrelation(p1, rows, cols, 1e-6)
    (2.0 * (c1.detach() * c1).sum()).backward()
    p2 = phi.clone().requires_grad_()
    (1.0 * spectral.decorrelation(p2, rows, cols, 1e-6).square().sum()).backward()
    torch.testing.assert_close(p1.grad, p2.grad)


def test_readout_recovers_leading_dimensions():
    torch.manual_seed(0)
    n, t, k = 512, 10, 5
    z = torch.randn(n, k)
    phi = 0.3 * torch.randn(n, t + 1, k)
    # Dims 1 and 2 are controlled by z; dim 3 is not; dim 4 is (but after a gap, so it must not count).
    for d in (0, 1, 3):
        phi[:, -1, d] += z[:, d]
    spec = spectral.spectrum_readout(phi, z, n_shuffles=500, generator=torch.Generator().manual_seed(0))
    assert spec.k_star == 2
    assert spec.returns[3] > spec.floor[3]
    assert spec.sigma2.shape == (k,)


def test_discrete_skill_weights_match_metra_mask():
    """METRA's discrete reward uses (z - mean(z)) * K / (K - 1): +1 on the chosen skill, -1/(K-1) elsewhere."""
    cfg = SSDConfig(dim_skill=4, discrete=True, hidden_dim=8)
    agent = SpectralSkillDiscovery(cfg, obs_dim=3, act_dim=1, act_low=[-1], act_high=[1], device="cpu")
    w = agent.skill_weights(torch.eye(4)[[2]])
    torch.testing.assert_close(w, torch.tensor([[-1 / 3, -1 / 3, 1.0, -1 / 3]]))
    z = torch.randn(5, 4)
    agent.cfg.discrete = False
    assert torch.equal(agent.skill_weights(z), z)


def test_signed_discrete_skills():
    """Signed skills: 2K skills z = +-e_k, spread evenly for evaluation, and the reward projects on z itself."""
    from train import sample_skills

    z = sample_skills(None, 8, 4, unit=False, discrete=True, even=True, signed=True)
    torch.testing.assert_close(torch.as_tensor(z), torch.cat([torch.eye(4), -torch.eye(4)]))
    z = sample_skills(__import__("numpy").random.default_rng(0), 100, 4, unit=False, discrete=True, signed=True)
    assert set(map(tuple, z.tolist())) <= set(map(tuple, torch.cat([torch.eye(4), -torch.eye(4)]).tolist()))
    cfg = SSDConfig(dim_skill=4, discrete=True, signed_skills=True, hidden_dim=8)
    agent = SpectralSkillDiscovery(cfg, obs_dim=3, act_dim=1, act_low=[-1], act_high=[1], device="cpu")
    zt = torch.as_tensor(z)
    assert torch.equal(agent.skill_weights(zt), zt)


def test_agent_update_and_duals():
    torch.manual_seed(0)
    cfg = SSDConfig(dim_skill=3, hidden_dim=32, batch_size=64, n_dual=1, n_readout_trajs=8, n_shuffles=50)
    agent = SpectralSkillDiscovery(cfg, obs_dim=5, act_dim=2, act_low=[-1, -1], act_high=[1, 1], device="cpu")
    buf = TrajectoryBuffer(capacity=200, horizon=10, obs_dim=5, act_dim=2, dim_skill=3, device="cpu")
    buf.add(torch.randn(20, 11, 5), torch.rand(20, 10, 2) * 2 - 1, torch.zeros(20, 10), torch.randn(20, 3))
    before = [p.clone() for p in agent.encoder.parameters()]
    m = agent.update(buf)
    assert all(torch.isfinite(v) for v in m.values())
    assert any(not torch.equal(a, b) for a, b in zip(before, agent.encoder.parameters()))
    assert agent.beta.shape == (3,) and agent.beta.abs().sum() > 0
    assert agent.nu >= 0
    assert agent.c == cfg.c0 * (1 + cfg.kappa)
    spec = agent.spectrum(buf)
    assert 0 <= spec.k_star <= 3


def test_decorrelation_warmup():
    """Before decor_warmup updates, the duals beta and the barrier c stay at their initial values."""
    torch.manual_seed(0)
    cfg = SSDConfig(dim_skill=3, hidden_dim=32, batch_size=64, n_dual=1, decor_warmup=5)
    agent = SpectralSkillDiscovery(cfg, obs_dim=5, act_dim=2, act_low=[-1, -1], act_high=[1, 1], device="cpu")
    buf = TrajectoryBuffer(capacity=200, horizon=10, obs_dim=5, act_dim=2, dim_skill=3, device="cpu")
    buf.add(torch.randn(20, 11, 5), torch.rand(20, 10, 2) * 2 - 1, torch.zeros(20, 10), torch.randn(20, 3))
    for _ in range(5):
        agent.update(buf)
    assert agent.beta.abs().sum() == 0 and agent.c == cfg.c0
    agent.update(buf)
    assert agent.beta.abs().sum() > 0 and agent.c > cfg.c0


def test_decorrelation_ramp():
    """After the warm-up the duals start moving at once, but the decorrelation weight only reaches 1 after the ramp."""
    torch.manual_seed(0)
    cfg = SSDConfig(dim_skill=3, hidden_dim=32, batch_size=64, n_dual=1, decor_warmup=2, decor_ramp=4)
    agent = SpectralSkillDiscovery(cfg, obs_dim=5, act_dim=2, act_low=[-1, -1], act_high=[1, 1], device="cpu")
    buf = TrajectoryBuffer(capacity=200, horizon=10, obs_dim=5, act_dim=2, dim_skill=3, device="cpu")
    buf.add(torch.randn(20, 11, 5), torch.rand(20, 10, 2) * 2 - 1, torch.zeros(20, 10), torch.randn(20, 3))
    for _ in range(2):
        agent.update(buf)
    assert agent.beta.abs().sum() == 0 and agent.c == cfg.c0
    agent.update(buf)  # delta = 1/4
    assert agent.beta.abs().sum() > 0 and agent.c > cfg.c0


def test_encoder_split_and_head_only_decorrelation():
    """phi = head(features); with decor_head_only the update still trains the whole encoder (through J and h)."""
    torch.manual_seed(0)
    cfg = SSDConfig(dim_skill=3, hidden_dim=32, batch_size=64, n_dual=1, decor_head_only=True)
    agent = SpectralSkillDiscovery(cfg, obs_dim=5, act_dim=2, act_low=[-1, -1], act_high=[1, 1], device="cpu")
    obs = torch.randn(7, 5)
    torch.testing.assert_close(agent.encoder(obs), agent.encoder.head(agent.encoder.features(obs)))
    buf = TrajectoryBuffer(capacity=200, horizon=10, obs_dim=5, act_dim=2, dim_skill=3, device="cpu")
    buf.add(torch.randn(20, 11, 5), torch.rand(20, 10, 2) * 2 - 1, torch.zeros(20, 10), torch.randn(20, 3))
    before = [p.clone() for p in agent.encoder.parameters()]
    m = agent.update(buf)
    assert all(torch.isfinite(v) for v in m.values())
    assert all(not torch.equal(a, b) for a, b in zip(before, agent.encoder.parameters()))
    assert agent.beta.abs().sum() > 0


def test_per_dimension_encoder_keeps_dimensions_apart():
    """With one network per dimension, a loss on dimension 3 leaves the networks of dimensions 1 and 2 untouched."""
    from ssd.networks import Encoder

    torch.manual_seed(0)
    enc = Encoder(obs_dim=5, dim_skill=3, hidden_dim=16, n_layers=2, per_dim=True)
    out = enc(torch.randn(8, 5))
    assert out.shape == (8, 3)
    out[:, 2].sum().backward()
    untouched = lambda net: all(p.grad is None or p.grad.abs().sum() == 0 for p in net.parameters())
    assert untouched(enc.dims[0]) and untouched(enc.dims[1])
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in enc.dims[2].parameters() if p.dim() > 1)


def test_scale_aware_decorrelation_weight():
    """decor_scale weights each pair by s_k s_j / (s_k s_j + scale): ~0 for undeveloped dimensions, ~1 once spread."""
    cfg = SSDConfig(dim_skill=3, hidden_dim=8, decor_scale=1.0)
    agent = SpectralSkillDiscovery(cfg, obs_dim=5, act_dim=2, act_low=[-1, -1], act_high=[1, 1], device="cpu")
    torch.manual_seed(0)
    small, large = 0.01 * torch.randn(512, 3), 30 * torch.randn(512, 3)
    assert agent._decor_weight(small).max() < 1e-3
    assert agent._decor_weight(large).min() > 0.99
    c_weighted, _, c_raw = agent._constraints(small, small + 0.001)
    assert c_weighted.abs().max() < 1e-3 * c_raw.abs().max() + 1e-9


def test_cube_discrete_skills():
    """Cube skills: the 2^K unit-length corners of {-1, 1}^K, all distinct, spread evenly for evaluation."""
    import numpy as np
    from train import sample_skills

    z = sample_skills(None, 16, 4, unit=False, discrete=True, even=True, cube=True)
    assert len({tuple(r) for r in z.tolist()}) == 16
    np.testing.assert_allclose(np.abs(z), 0.5)
    np.testing.assert_allclose(z.mean(axis=0), 0.0, atol=1e-7)  # isotropic: E[z] = 0, E[z z^T] = I / K
    np.testing.assert_allclose(z.T @ z / 16, np.eye(4) / 4, atol=1e-7)
    cfg = SSDConfig(dim_skill=4, discrete=True, cube_skills=True, hidden_dim=8)
    agent = SpectralSkillDiscovery(cfg, obs_dim=3, act_dim=1, act_low=[-1], act_high=[1], device="cpu")
    zt = torch.as_tensor(z)
    assert torch.equal(agent.skill_weights(zt), zt)
