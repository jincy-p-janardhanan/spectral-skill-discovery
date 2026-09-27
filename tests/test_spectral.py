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


def test_decorrelation_gradient_flows_only_through_later_dimension():
    torch.manual_seed(0)
    phi = torch.randn(512, 3, requires_grad=True)
    rows, cols = spectral.lower_pairs(3)
    spectral.decorrelation(phi, rows, cols, eps=1e-6).sum().backward()
    # Dimension 1 only ever appears as j (the stop-gradient side) -> no gradient.
    assert torch.all(phi.grad[:, 0] == 0)
    assert phi.grad[:, 1].abs().sum() > 0 and phi.grad[:, 2].abs().sum() > 0


def test_temporal_hinge():
    a = torch.zeros(2, 2)
    b = torch.tensor([[0.5, 0.0], [2.0, 0.0]])  # squared distances 0.25 and 4
    assert spectral.temporal_hinge(a, b).item() == 1.5  # mean(0, 3)


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
