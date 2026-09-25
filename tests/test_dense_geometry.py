"""Dense backend tests from independent 3D motion and pinhole projection."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import expm

from ev6d.dense_filter import DenseVelocityConfig, DenseVelocityKF, stratified_indices
from ev6d.geometry import interaction_matrix


K = np.array([[410., 0., 159.5], [0., 437., 119.5], [0., 0., 1.]])


def scene():
    # Vary depths independently of image location to exercise all six columns.
    rng = np.random.default_rng(417)
    x, y = np.meshgrid(np.linspace(-.35, .35, 9), np.linspace(-.26, .26, 8))
    return np.c_[x.ravel(), y.ravel(), rng.uniform(.7, 2., x.size)]


def camera_projection(points, calibration=K):
    projected = (calibration@points.T).T
    return projected[:, :2]/projected[:, 2:]


def true_displacement(points, twist, dt):
    # Independently build the se(3) generator, never using interaction_matrix or
    # the project's pose integrator to synthesize the measurement under test.
    wx, wy, wz = twist[3:]
    generator = np.array([[0, -wz, wy, twist[0]], [wz, 0, -wx, twist[1]],
                          [-wy, wx, 0, twist[2]], [0, 0, 0, 0.]])
    transform = expm(generator*dt)
    moved = points@transform[:3, :3].T+transform[:3, 3]
    return camera_projection(moved)-camera_projection(points)


@pytest.mark.parametrize("twist", [
    [.13, -.09, .08, 0, 0, 0], [0, 0, 0, .21, -.17, .14],
    [.13, -.09, .08, .21, -.17, .14],
])
def test_recovers_projected_translation_rotation_and_mixed_motion(twist):
    points, twist, dt = scene(), np.array(twist), 1e-5
    flow = true_displacement(points, twist, dt)
    filt = DenseVelocityKF(DenseVelocityConfig(flow_noise_std=1e-8, noise_inflation=1,
                                               gate_mahalanobis_sq=None))
    diagnostic = filt.update(camera_projection(points), points[:, 2], flow, K, dt)
    assert diagnostic["accepted"] and diagnostic["observable_rank"] == 6
    np.testing.assert_allclose(filt.x, twist, atol=3e-6, rtol=3e-5)
    assert np.linalg.eigvalsh(filt.P).min() > 0


@pytest.mark.parametrize("axis", range(6))
def test_interaction_matrix_against_independent_central_finite_difference(axis):
    points, twist, dt = scene(), np.eye(6)[axis], 1e-6
    derivative = (true_displacement(points, twist, dt)-true_displacement(points, -twist, dt))/(2*dt)
    predicted = interaction_matrix(camera_projection(points), points[:, 2], K)@twist
    np.testing.assert_allclose(predicted, derivative, rtol=1e-8, atol=1e-7)


def test_displacement_noise_and_velocity_noise_are_equivalent_once_scaled():
    points, dt = scene(), .025
    uv = camera_projection(points)
    flow = true_displacement(points, np.array([.1, .2, -.1, .02, -.03, .04]), dt)
    cfg = DenseVelocityConfig(flow_noise_std=2., flow_noise_units="velocity", noise_inflation=1,
                              gate_mahalanobis_sq=None)
    a, b = DenseVelocityKF(cfg), DenseVelocityKF(replace(cfg, flow_noise_std=2*dt,
                                                        flow_noise_units="displacement"))
    a.update(uv, points[:, 2], flow, K, dt)
    b.update(uv, points[:, 2], flow, K, dt)
    np.testing.assert_allclose(a.x, b.x, atol=1e-13)
    np.testing.assert_allclose(a.P, b.P, atol=1e-13)


def test_whitened_small_solve_matches_independent_batch_kf():
    points, dt = scene()[::6], .03
    uv, depths = camera_projection(points), points[:, 2]
    H = dt*interaction_matrix(uv, depths, K).reshape(-1, 6)
    flow = true_displacement(points, np.array([.1, -.04, .2, -.1, .03, .12]), dt)
    filt = DenseVelocityKF(DenseVelocityConfig(flow_noise_std=.7, noise_inflation=1,
                                               gate_mahalanobis_sq=None))
    prior, mean = filt.P.copy(), np.array([.03, -.01, .01, 0, 0, 0.])
    filt.x = mean.copy()
    R = np.eye(len(H))*.7**2
    gain = np.linalg.solve(H@prior@H.T+R, H@prior).T
    expected_mean = mean+gain@(flow.ravel()-H@mean)
    I_KH = np.eye(6)-gain@H
    expected_covariance = I_KH@prior@I_KH.T+gain@R@gain.T
    assert filt.update(uv, depths, flow, K, dt)["accepted"]
    np.testing.assert_allclose(filt.x, expected_mean, atol=1e-12)
    np.testing.assert_allclose(filt.P, expected_covariance, atol=1e-12)


def test_no_observations_and_bad_geometry_retain_prediction():
    filt = DenseVelocityKF()
    filt.x[:] = .2
    filt.predict_to(0.)
    initial = filt.P.copy()
    filt.predict_to(.2)
    prior = filt.P.copy()
    assert np.trace(prior) > np.trace(initial)
    result = filt.update(np.empty((0, 2)), np.empty(0), np.empty((0, 2)), K, .1)
    assert not result["accepted"] and result["reason"] == "no_valid_observations"
    result = filt.update(np.ones((20, 2)), np.ones(20), np.zeros((20, 2)), K, .1)
    assert not result["accepted"] and result["reason"] == "rank_deficient"
    np.testing.assert_array_equal(filt.P, prior)
    np.testing.assert_allclose(filt.x, .2)


def test_condition_rejection_is_explicit():
    points = scene()
    filt = DenseVelocityKF(DenseVelocityConfig(max_condition_number=1.01, gate_mahalanobis_sq=None))
    result = filt.update(camera_projection(points), points[:, 2], np.zeros((len(points), 2)), K, .1)
    assert result["reason"] == "ill_conditioned"
    assert result["observable_rank"] == 6


def test_gate_is_joint_2d_and_invalid_depth_is_counted():
    points, dt = scene(), .02
    flow = np.zeros((len(points), 2))
    flow[5] = [1e6, -1e6]
    depths = points[:, 2].copy()
    depths[:3] = [np.nan, 0, -1]
    filt = DenseVelocityKF()
    result = filt.update(camera_projection(points), depths, flow, K, dt)
    assert result["accepted"] and result["invalid_observations"] == 3
    assert result["gated_observations"] == 1
    assert result["measurements"] == len(points)-4


def test_stratified_sampling_is_bounded_and_covers_the_image():
    xx, yy = np.meshgrid(np.arange(100), np.arange(80))
    uv = np.c_[xx.ravel(), yy.ravel()]
    selected = stratified_indices(uv, 64)
    assert len(selected) == len(set(selected)) == 64
    assert uv[selected, 0].min() < 15 and uv[selected, 0].max() > 85
    assert uv[selected, 1].min() < 12 and uv[selected, 1].max() > 68
    np.testing.assert_array_equal(selected, stratified_indices(uv, 64))
    filt = DenseVelocityKF(DenseVelocityConfig(max_observations=64))
    depth = 1+np.sin(uv[:, 0]/15)*.2
    result = filt.update(uv, depth, np.zeros_like(uv), K, .1)
    assert result["sampled_observations"] == 64 and result["sampling_dropped"] == len(uv)-64


@pytest.mark.parametrize("rate", [0., 1.3])
def test_continuous_process_model_does_not_depend_on_update_frequency(rate):
    a, b = (DenseVelocityKF(DenseVelocityConfig(decay_rate_per_s=rate)) for _ in range(2))
    for filt in (a, b):
        filt.x[:] = .1
        filt.predict_to(0.)
    a.predict_to(.1)
    for timestamp in [.01, .025, .031, .063, .1]:
        b.predict_to(timestamp)
    np.testing.assert_allclose(a.x, b.x, atol=1e-14)
    np.testing.assert_allclose(a.P, b.P, atol=1e-13)


def test_resize_crop_and_depth_unit_scale_consistency():
    # The local model is F~=dt J xi. A tiny interval keeps finite-motion
    # linearization residuals from changing under anisotropic residual weighting.
    points, dt = scene(), 1e-5
    uv = camera_projection(points)
    flow = true_displacement(points, np.array([.1, -.2, .15, .1, .2, -.1]), dt)
    config = DenseVelocityConfig(flow_noise_std=1e-8, noise_inflation=1,
                                 gate_mahalanobis_sq=None)
    base = DenseVelocityKF(config)
    base.update(uv, points[:, 2], flow, K, dt)
    resized_k = K.copy()
    resized_k[0] *= .5
    resized_k[1] *= .75
    resized_k[:2, 2] -= [8, 12]
    resized = DenseVelocityKF(config)
    resized.update(uv*[.5, .75]-[8, 12], points[:, 2], flow*[.5, .75], resized_k, dt)
    np.testing.assert_allclose(base.x, resized.x, atol=2e-7)
    # Converting metres to millimetres changes only translational twist units.
    mm = DenseVelocityKF(replace(config, initial_linear_std=1000))
    mm.update(uv, points[:, 2]*1000, flow, K, dt)
    np.testing.assert_allclose(mm.x[:3]/1000, base.x[:3], atol=2e-7)
    np.testing.assert_allclose(mm.x[3:], base.x[3:], atol=2e-7)


def test_invalid_time_order_and_nonfinite_state_fail_loudly():
    filt = DenseVelocityKF()
    filt.predict_to(1.)
    with pytest.raises(ValueError, match="monotonic"):
        filt.predict_to(.9)
    with pytest.raises(ValueError, match="positive seconds"):
        filt.update(np.empty((0, 2)), np.empty(0), np.empty((0, 2)), K, 0)
    filt.x[0] = np.nan
    with pytest.raises(ValueError, match="state"):
        filt.predict_to(2.)
