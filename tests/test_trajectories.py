"""Validate synthetic trajectory ground truth independently of tracker code."""
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ev6d.trajectories import MotionTrajectory


@pytest.mark.parametrize("speed", ["regular", "fast"])
@pytest.mark.parametrize("base", [(0., 0., 0.), (.1, -.2, .1), (2.1, -.7, .8)])
def test_spatial_twist_matches_finite_difference_pose_and_material_points(speed, base):
    motion = MotionTrajectory(seed=123, speed=speed, base_rotation=base)
    step = 1e-6
    point_object = np.array([.07, -.03, .08])
    for t in (0., .17, .83, 2.1):
        p, q = motion.pose(t)
        p_plus, q_plus = motion.pose(t+step)
        p_minus, q_minus = motion.pose(t-step)
        p_rate = (p_plus-p_minus)/(2*step)
        r, r_plus, r_minus = (Rotation.from_quat(x) for x in (q, q_plus, q_minus))
        omega_numeric = (r_plus*r_minus.inv()).as_rotvec()/(2*step)
        expected = np.r_[p_rate-np.cross(omega_numeric, p), omega_numeric]
        actual = motion.velocity(t)
        np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=2e-9)
        point = r.apply(point_object)+p
        point_rate = ((r_plus.apply(point_object)+p_plus)-
                      (r_minus.apply(point_object)+p_minus))/(2*step)
        np.testing.assert_allclose(actual[:3]+np.cross(actual[3:], point),
                                   point_rate, rtol=2e-7, atol=2e-9)


@pytest.mark.parametrize("factor", [1.5, 3., 5.])
def test_fast_is_exact_same_path_with_faster_clock(factor):
    regular = MotionTrajectory(seed=83, speed="regular", speed_factor=factor)
    fast = MotionTrajectory(seed=83, speed="fast", speed_factor=factor)
    for t in np.linspace(-.1, 2., 27):
        p_regular, q_regular = regular.pose(t*factor)
        p_fast, q_fast = fast.pose(t)
        np.testing.assert_array_equal(p_fast, p_regular)
        np.testing.assert_array_equal(q_fast, q_regular)
        np.testing.assert_allclose(fast.velocity(t), regular.velocity(t*factor)*factor,
                                   rtol=1e-13, atol=1e-13)


def test_initial_pose_and_excursion_bounds_are_explicit():
    base = [.25, -.5, .17]
    motion = MotionTrajectory(center_depth=1.2, translation_amplitude=[.1, .08, .06],
                              rotation_amplitude=[.25, .35, .2], base_rotation=base)
    p0, q0 = motion.pose(0.)
    np.testing.assert_array_equal(p0, [0., 0., 1.2])
    np.testing.assert_allclose(q0, Rotation.from_rotvec(base).as_quat(), atol=1e-14)
    base_inverse = Rotation.from_rotvec(base).inv()
    for t in np.linspace(0, 100, 300):
        p, q = motion.pose(t)
        assert np.all(np.abs(p-p0) <= motion.translation_amplitude+1e-14)
        angle = (Rotation.from_quat(q)*base_inverse).as_euler("xyz")
        assert np.all(np.abs(angle) <= motion.rotation_amplitude+1e-14)
        assert p[2] > 0
        assert np.linalg.norm(q) == pytest.approx(1.)


def test_pure_rotation_has_nonzero_spatial_linear_component():
    motion = MotionTrajectory(translation_amplitude=(0., 0., 0.))
    for t in (0., .5, 1.):
        p, _ = motion.pose(t)
        v = motion.velocity(t)
        assert np.linalg.norm(v[:3]) > 1e-3
        np.testing.assert_allclose(v[:3]+np.cross(v[3:], p), 0., atol=1e-14)


def test_zero_amplitudes_are_a_static_pose():
    motion = MotionTrajectory(translation_amplitude=(0., 0., 0.), rotation_amplitude=(0., 0., 0.))
    for t in (-1., 0., .5, 10.):
        p, q = motion.pose(t)
        np.testing.assert_array_equal(p, motion.pose(0.)[0])
        np.testing.assert_array_equal(q, motion.pose(0.)[1])
        np.testing.assert_array_equal(motion.velocity(t), np.zeros(6))


def test_same_seed_is_reproducible_and_different_seed_changes_motion():
    a, b, c = MotionTrajectory(seed=17), MotionTrajectory(seed=17), MotionTrajectory(seed=18)
    assert a.metadata == b.metadata
    np.testing.assert_array_equal(a.pose(.71)[0], b.pose(.71)[0])
    assert not np.allclose(a.pose(.71)[0], c.pose(.71)[0])
    assert not np.allclose(a.pose(.71)[1], c.pose(.71)[1])


def test_metadata_roundtrip_preserves_full_draws_and_cannot_mutate_motion():
    motion = MotionTrajectory(seed=28, speed="fast", base_rotation=[1.5, -.2, .7])
    metadata = json.loads(json.dumps(motion.metadata))
    restored = MotionTrajectory.from_dict(metadata)
    assert restored.to_dict() == metadata
    for t in (0., .1, .35, 1.2):
        for a, b in zip(restored.pose(t), motion.pose(t)):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(restored.velocity(t), motion.velocity(t))
    before = motion.pose(.3)
    metadata["translation_frequency_hz"][0][0] *= 1.3
    different = MotionTrajectory.from_dict(metadata)
    assert not np.allclose(different.pose(.3)[0], before[0])
    np.testing.assert_array_equal(motion.pose(.3)[0], before[0])


@pytest.mark.parametrize("kwargs", [
    {"seed": -1}, {"seed": 1.2}, {"seed": True}, {"speed": "slow"},
    {"center_depth": 0}, {"center_depth": .03}, {"speed_factor": 1.},
    {"speed_factor": np.inf}, {"translation_amplitude": [1., 2.]},
    {"rotation_amplitude": [1., -2., 3.]}, {"base_rotation": [np.nan, 0., 0.]},
])
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        MotionTrajectory(**kwargs)


@pytest.mark.parametrize("t", [np.nan, np.inf, [0., 1.], [0.]])
def test_only_finite_scalar_times_are_accepted(t):
    motion = MotionTrajectory()
    with pytest.raises(ValueError):
        motion.pose(t)
    with pytest.raises(ValueError):
        motion.velocity(t)
