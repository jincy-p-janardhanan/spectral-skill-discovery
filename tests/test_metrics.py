import numpy as np
import torch

from ssd.metrics import body_coverage, coverage, quat_to_euler
from ssd.networks import Encoder


def test_encoder_ignores_hidden_inputs():
    torch.manual_seed(0)
    enc = Encoder(obs_dim=5, dim_skill=3, hidden_dim=16, n_layers=2, input_idx=[2, 3, 4])
    obs = torch.randn(7, 5)
    moved = obs.clone()
    moved[:, :2] += 100.0  # change only the hidden (position) inputs
    torch.testing.assert_close(enc(obs), enc(moved))
    assert enc.net[0].in_features == 3


def test_encoder_default_sees_everything():
    enc = Encoder(obs_dim=5, dim_skill=2, hidden_dim=8, n_layers=1)
    assert enc.net[0].in_features == 5 and enc.input_idx.tolist() == [0, 1, 2, 3, 4]


def test_quat_to_euler():
    half = np.pi / 8  # rotations by pi/4
    identity = np.array([1.0, 0, 0, 0])
    yaw = np.array([np.cos(half), 0, 0, np.sin(half)])
    roll = np.array([np.cos(half), np.sin(half), 0, 0])
    pitch = np.array([np.cos(half), 0, np.sin(half), 0])
    e = quat_to_euler(np.stack([identity, yaw, roll, pitch]))
    np.testing.assert_allclose(e, [[0, 0, 0], [0, 0, np.pi / 4], [np.pi / 4, 0, 0], [0, np.pi / 4, 0]], atol=1e-7)


def test_coverage_counts_unique_cells():
    xy = np.array([[[0.1, 0.1], [0.9, 0.2], [1.5, 0.1], [-0.5, 3.2]]])
    assert coverage(xy, "ant") == 3            # cells (0,0), (1,0), (-1,3)
    assert coverage(xy, "half_cheetah") == 3   # x cells 0, 1, -1


class _FakeAnt:
    joint_range = np.array([[-1.0, 1.0], [0.0, 2.0]])

    @staticmethod
    def orientation(raw):
        return raw[..., 0:1]

    @staticmethod
    def joint_angles(raw):
        return raw[..., 1:3]


def test_body_coverage():
    # orientation 0 and 0.1 share a 30-degree cell, 1.0 does not; joints at the range edges clip into the end bins
    raw = np.array([[[0.0, -1.0, 0.0], [0.1, -1.0, 0.0], [1.0, 1.0, 2.0], [1.0, 0.99, 1.99]]])
    cov = body_coverage(raw, _FakeAnt(), joint_bins=5)
    assert cov == {"orientation_coverage": 2, "posture_coverage": 2}
