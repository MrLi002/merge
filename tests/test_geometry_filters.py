"""Physical/numerical checks independent of implementation's formulas."""
import numpy as np
import pytest
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

from ev6d.geometry import (integrate_pose, interaction_matrix, project, quat_difference,
                           quat_mean, register_depth, skew, unproject)
from ev6d.filters import PoseConfig, PoseUKF, VelocityConfig, VelocityKF


K = np.array([[240., 0, 79.5], [0, 251., 59.5], [0, 0, 1.]])


def test_projection_and_inverse():
    points = np.array([[.3, -.1, 1.], [-.2, .1, 2.]])
    np.testing.assert_allclose(unproject(project(points, K), points[:, 2], K), points)


@pytest.mark.parametrize("axis", range(6))
def test_spatial_velocity_jacobian_against_actual_3d_motion(axis):
    points = np.array([[.3, -.1, 1.], [-.2, .1, 2.], [.2, .25, .8]])
    velocity = np.eye(6)[axis]
    eps = 1e-7
    moved = Rotation.from_rotvec(velocity[3:]*eps).apply(points)+velocity[:3]*eps
    finite_difference = (project(moved, K)-project(points, K))/eps
    predicted = interaction_matrix(project(points, K), points[:, 2], K)@velocity
    np.testing.assert_allclose(predicted, finite_difference, rtol=2e-6, atol=2e-5)


def test_pose_integration_against_homogeneous_matrix_exponential():
    p, q = np.array([.3, -.1, 1.]), Rotation.from_rotvec([.1, .2, -.3]).as_quat()
    twist = np.array([.2, -.3, .1, .5, .3, -.8])
    matrix = np.zeros((4, 4))
    matrix[:3, :3], matrix[:3, 3] = skew(twist[3:]), twist[:3]
    initial = np.eye(4)
    initial[:3, :3], initial[:3, 3] = Rotation.from_quat(q).as_matrix(), p
    expected = expm(matrix*.37)@initial
    actual_p, actual_q = integrate_pose(p, q, twist, .37)
    np.testing.assert_allclose(actual_p, expected[:3, 3], atol=1e-12)
    np.testing.assert_allclose(Rotation.from_quat(actual_q).as_matrix(), expected[:3, :3], atol=1e-12)


def test_rotation_about_object_center_has_no_translation():
    position = np.array([.2, -.1, 1.])
    omega = np.array([.4, .3, .5])
    p, _ = integrate_pose(position, [0, 0, 0, 1], np.r_[-np.cross(omega, position), omega], .8)
    np.testing.assert_allclose(p, position, atol=1e-12)


def test_registration_identity_and_invalid_depth():
    depth = np.array([[1., 2., np.nan], [0., -1., 3.]])
    registered = register_depth(depth, K, K, np.eye(4), depth.shape)
    np.testing.assert_allclose(registered, [[1., 2., np.nan], [np.nan, np.nan, 3.]], equal_nan=True)


def test_depth_extrinsic_and_zbuffer():
    small_k = np.eye(3)
    T = np.eye(4)
    T[0, 3] = 2.
    # Source pixels x=0,Z=1 and x=1,Z=2 both land at event x=2.
    output = register_depth(np.array([[1., 2.]]), small_k, small_k, T, (1, 4))
    assert output[0, 2] == 1. and np.isnan(output[0, 0])
    T[2, 3] = 1.
    result = register_depth(np.array([[1.]]), small_k, small_k, T, (2, 4))
    assert result[0, 1] == 2.  # transformed Z, not input depth


def test_quaternion_double_cover_mean():
    q = Rotation.from_rotvec([.4, -.2, .1]).as_quat()
    mean = quat_mean([q, -q], [.5, .5])
    np.testing.assert_allclose(quat_difference(mean, q), 0, atol=1e-12)


def test_velocity_decay_is_invariant_to_timestamp_partition():
    a, b = VelocityKF(), VelocityKF()
    for kf in (a, b):
        kf.x[:] = 1.
        kf.timestamp = 0.
    a.predict_to(.027)
    for t in (.007, .011, .021, .027):
        b.predict_to(t)
    np.testing.assert_allclose(a.x, b.x, atol=1e-12)
    np.testing.assert_allclose(a.P, b.P, atol=1e-12)
    a.predict_to(1.)
    assert np.linalg.norm(a.x) < 1e-20


def make_velocity_problem(seed=5):
    rng = np.random.default_rng(seed)
    uv = rng.uniform([0., 0.], [159., 119.], (200, 2))
    z = rng.uniform(.5, 2., len(uv))
    twist = np.array([.12, -.18, .1, .1, -.15, .3])
    flow = interaction_matrix(uv, z, K)@twist
    return uv, z, flow, twist


@pytest.mark.parametrize("normal_flow", [True, False])
def test_velocity_recovers_known_rigid_motion(normal_flow):
    uv, z, flow, truth = make_velocity_problem()
    kf = VelocityKF(VelocityConfig(normal_flow=normal_flow, weighting=False, flow_noise_std=.01))
    stats = kf.update(uv, z, flow, K)
    assert stats["observable_rank"] == 6
    np.testing.assert_allclose(kf.x, truth, atol=2e-5)
    assert np.linalg.eigvalsh(kf.P).min() > 0


def test_normal_flow_does_not_constrain_tangential_motion():
    kf = VelocityKF(VelocityConfig(weighting=False, flow_noise_std=.1))
    P = kf.P.copy()
    stats = kf.update(np.tile(K[:2, 2], (20, 1)), np.ones(20), np.tile([20., 0.], (20, 1)), K)
    assert stats["observable_rank"] == 1
    assert kf.P[1, 1] == pytest.approx(P[1, 1])
    assert kf.x[1] == pytest.approx(0.)


def test_laplace_weighting_reduces_large_outlier_influence():
    uv, z, flow, truth = make_velocity_problem()
    corrupt = flow.copy()
    corrupt[:15] += [1600., -1000.]
    weighted = VelocityKF(VelocityConfig(normal_flow=False, weighting=True, flow_noise_std=1.))
    plain = VelocityKF(VelocityConfig(normal_flow=False, weighting=False, flow_noise_std=1.))
    a = weighted.update(uv, z, corrupt, K)
    plain.update(uv, z, corrupt, K)
    assert np.linalg.norm(weighted.x-truth) < np.linalg.norm(plain.x-truth)*.1
    assert a["weight_min"] < a["weight_median"]


def test_pose_prediction_and_quaternion_measurement_sign():
    cfg = PoseConfig(initial_position_std=1e-5, initial_rotation_std=1e-5, process_position_std=0., process_rotation_std=0.)
    a = PoseUKF([0, 0, 1], [0, 0, 0, 1], 0., cfg)
    b = PoseUKF([0, 0, 1], [0, 0, 0, 1], 0., cfg)
    twist = np.array([.1, 0., 0., .1, -.2, .3])
    expected_p, expected_q = integrate_pose(a.position, a.quaternion, twist, .2)
    for filter in (a, b):
        filter.predict_to(.2, twist)
        np.testing.assert_allclose(filter.position, expected_p, atol=1e-9)
        np.testing.assert_allclose(quat_difference(filter.quaternion, expected_q), 0., atol=1e-9)
    a.update(expected_p, expected_q)
    b.update(expected_p, -expected_q)
    np.testing.assert_allclose(a.position, b.position, atol=1e-12)
    np.testing.assert_allclose(a.P, b.P, atol=1e-12)


def test_pose_uncertainty_growth_and_outlier_gate():
    a, b = (PoseUKF([0, 0, 1], [0, 0, 0, 1], 0., PoseConfig(gate_threshold=25.)) for _ in range(2))
    a.predict_to(.1, np.zeros(6), np.eye(6)*.1)
    b.predict_to(.1, np.zeros(6))
    assert np.trace(a.P) > np.trace(b.P)
    before = a.position.copy()
    assert not a.update([10, 10, 10], [0, 0, 0, 1])["accepted"]
    np.testing.assert_array_equal(a.position, before)
    old_trace = np.trace(a.P)
    assert a.update([.001, -.001, 1.001], [0, 0, 0, 1])["accepted"]
    assert np.trace(a.P) < old_trace


def test_covariance_remains_positive_during_repeated_rotating_updates():
    kf = PoseUKF([.1, .2, 1], [0, 0, 0, 1], 0.)
    for i in range(1, 60):
        kf.predict_to(i*.01, np.array([0, 0, 0, .1, -.2, .4]), np.eye(6)*.01)
        if i % 4 == 0:
            assert kf.update(kf.position+[.001, -.002, .001], -kf.quaternion)["accepted"]
        np.testing.assert_allclose(np.linalg.norm(kf.quaternion), 1., atol=1e-12)
        assert np.linalg.eigvalsh(kf.P).min() > 0


def test_invalid_time_and_covariance_are_rejected():
    pose = PoseUKF([0, 0, 1], [0, 0, 0, 1], 0.)
    with pytest.raises(ValueError):
        pose.predict_to(-.1, np.zeros(6))
    with pytest.raises(ValueError):
        pose.predict_to(.1, np.zeros(6), -np.eye(6))
    with pytest.raises(ValueError):
        PoseUKF([0, 0, 1], [0, 0, 0, 0], 0.)


def test_default_pose_update_does_not_reject_fast_valid_motion():
    kf = PoseUKF([0, 0, 1], [0, 0, 0, 1], 0.)
    kf.predict_to(.2, np.zeros(6))
    result = kf.update([.15, 0, 1], Rotation.from_rotvec([0, 0, .3]).as_quat())
    assert result["accepted"]
    assert kf.position[0] > .1
