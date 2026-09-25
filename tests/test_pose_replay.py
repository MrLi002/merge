"""Delayed poses must equal chronological filtering and never double-integrate."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

from ev6d.filters import PoseConfig, PoseUKF
from ev6d.geometry import quat_difference
from ev6d.pose_replay import PoseHistory


POSITION = np.array([.2, -.1, 1.3])
QUATERNION = Rotation.from_rotvec([.1, -.2, .3]).as_quat()
TWISTS = [np.array([.2, .03, -.05, .1, -.2, .3]),
          np.array([-.1, .08, .03, -.2, .1, .2]),
          np.array([.04, -.02, .06, .3, .2, -.1])]
COVARIANCE = np.diag([.01, .02, .015, .02, .03, .025])**2


def history(**kwargs):
    return PoseHistory(PoseUKF(POSITION, QUATERNION, 0.), **kwargs)


def assert_same_pose(a, b, tolerance=2e-11):
    np.testing.assert_allclose(a.position, b.position, atol=tolerance, rtol=0)
    np.testing.assert_allclose(quat_difference(a.quaternion, b.quaternion), 0, atol=tolerance, rtol=0)
    np.testing.assert_allclose(a.P, b.P, atol=tolerance, rtol=0)
    assert a.timestamp == b.timestamp


def test_delayed_interior_pose_matches_chronological_correction():
    delayed, immediate = history(), history()
    observation = POSITION+np.array([.014, -.008, .002])
    q = Rotation.from_rotvec([.13, -.22, .31]).as_quat()
    immediate.predict_to(.04, TWISTS[0], COVARIANCE)
    assert immediate.update(observation, q, .04, .04)["accepted"]
    immediate.predict_to(.1, TWISTS[0], COVARIANCE)
    immediate.predict_to(.2, TWISTS[1], COVARIANCE)
    immediate.predict_to(.3, TWISTS[2], COVARIANCE)
    for endpoint, twist in zip([.1, .2, .3], TWISTS):
        delayed.predict_to(endpoint, twist, COVARIANCE)
    result = delayed.update(observation, q, .04, .28)
    assert result["accepted"] and result["replayed_predictions"] == 4
    assert_same_pose(delayed, immediate)


def test_out_of_order_measurement_times_retain_later_corrections():
    delayed, immediate = history(), history()
    p1, p2 = POSITION+[.006, -.003, .002], POSITION+[-.004, .007, -.009]
    q1 = Rotation.from_rotvec([.11, -.19, .33]).as_quat()
    q2 = Rotation.from_rotvec([.09, -.18, .30]).as_quat()
    immediate.predict_to(.04, TWISTS[0], COVARIANCE)
    immediate.update(p1, q1, .04, .04)
    immediate.predict_to(.1, TWISTS[0], COVARIANCE)
    immediate.predict_to(.15, TWISTS[1], COVARIANCE)
    immediate.update(p2, q2, .15, .15)
    immediate.predict_to(.2, TWISTS[1], COVARIANCE)
    immediate.predict_to(.3, TWISTS[2], COVARIANCE)
    for endpoint, twist in zip([.1, .2, .3], TWISTS):
        delayed.predict_to(endpoint, twist, COVARIANCE)
    delayed.update(p2, q2, .15, .26)
    delayed.update(p1, q1, .04, .29)
    assert_same_pose(delayed, immediate)
    assert delayed.diagnostics()["accepted_in_history"] == 2
    assert delayed.counters["accepted_on_arrival"] == 2


def test_quaternion_antipodes_have_identical_prediction_correction_and_covariance():
    a = PoseHistory(PoseUKF(POSITION, QUATERNION, 0.))
    b = PoseHistory(PoseUKF(POSITION, -QUATERNION, 0.))
    measured_q = Rotation.from_rotvec([.15, -.13, .27]).as_quat()
    for filt in (a, b):
        filt.predict_to(.1, TWISTS[0], COVARIANCE)
    a.update(POSITION+[.01, 0, 0], measured_q, .03, .08)
    b.update(POSITION+[.01, 0, 0], -measured_q, .03, .08)
    assert_same_pose(a, b)
    assert np.isclose(np.linalg.norm(a.quaternion), 1.)


def test_repeat_prediction_and_duplicate_pose_do_not_integrate_twice():
    filt = history()
    filt.predict_to(.1, TWISTS[0], COVARIANCE)
    filt.update(POSITION, QUATERNION, .03, .08, measurement_id="frame-3")
    before = filt.snapshot()
    filt.predict_to(.1, TWISTS[0], COVARIANCE)
    duplicate = filt.update(POSITION, -QUATERNION, .03, .09, measurement_id="frame-3")
    assert duplicate["reason"] == "duplicate_measurement"
    np.testing.assert_array_equal(filt.position, before.position)
    np.testing.assert_array_equal(filt.P, before.covariance)
    assert filt.counters["rejected_duplicate"] == 1


def test_bounded_history_rejects_expired_pose_and_preserves_recent_replay():
    filt = history(max_history_s=.15, max_intervals=2)
    for t in [.1, .2, .3, .4, .5]:
        filt.predict_to(t, TWISTS[0], COVARIANCE)
    before = filt.snapshot()
    result = filt.update(POSITION, QUATERNION, .2, .5)
    assert not result["accepted"] and result["reason"] == "outside_history"
    assert filt.counters["rejected_out_of_history"] == 1
    assert filt.diagnostics()["history_intervals"] <= 2
    np.testing.assert_array_equal(filt.P, before.covariance)
    assert filt.update(POSITION, QUATERNION, .41, .5)["accepted"]
    assert filt.timestamp == .5


def test_measurement_at_initial_timestamp_and_window_boundary():
    a, b = history(), history()
    a.update(POSITION+[.01, 0, 0], QUATERNION, 0., 0.)
    a.predict_to(.1, TWISTS[0], COVARIANCE)
    a.update(POSITION+[.03, 0, 0], QUATERNION, .1, .1)
    a.predict_to(.2, TWISTS[1], COVARIANCE)
    b.predict_to(.1, TWISTS[0], COVARIANCE)
    b.predict_to(.2, TWISTS[1], COVARIANCE)
    b.update(POSITION+[.01, 0, 0], QUATERNION, 0., .15)
    b.update(POSITION+[.03, 0, 0], QUATERNION, .1, .2)
    assert_same_pose(a, b)


def test_future_observation_is_not_consumed_and_can_be_retried():
    filt = history()
    result = filt.update(POSITION, QUATERNION, .1, .2)
    assert not result["accepted"] and result["reason"] == "not_yet_arrived"
    filt.predict_to(.2, TWISTS[0], COVARIANCE)
    assert filt.update(POSITION, QUATERNION, .1, .2)["accepted"]


def test_pose_gate_rejection_does_not_split_uncertain_prediction():
    filt = PoseHistory(PoseUKF(POSITION, QUATERNION, 0., PoseConfig(gate_threshold=10.)))
    filt.predict_to(.2, TWISTS[0], COVARIANCE)
    before = filt.snapshot()
    result = filt.update(POSITION+100, QUATERNION, .05, .2)
    assert not result["accepted"] and result["reason"] == "pose_gate"
    np.testing.assert_allclose(filt.position, before.position, atol=1e-12)
    np.testing.assert_allclose(filt.P, before.covariance, atol=1e-12)


def test_uncertain_twist_increases_pose_covariance():
    uncertain, fixed = history(), history()
    uncertain.predict_to(.2, TWISTS[0], COVARIANCE)
    fixed.predict_to(.2, TWISTS[0], None)
    assert np.trace(uncertain.P) > np.trace(fixed.P)


def test_pure_rotation_moves_object_reference_point_around_camera_origin():
    tiny = replace(PoseConfig(), initial_position_std=1e-6, initial_rotation_std=1e-6,
                   process_position_std=0, process_rotation_std=0)
    filt = PoseHistory(PoseUKF(POSITION, QUATERNION, 0., tiny))
    twist = np.array([0, 0, 0, .4, -.2, .3])
    filt.predict_to(.25, twist)
    R = Rotation.from_rotvec(twist[3:]*.25).as_matrix()
    np.testing.assert_allclose(filt.position, R@POSITION, atol=1e-12)
    assert np.linalg.norm(filt.position-POSITION) > .05


def test_invalid_arrival_order_and_measurement_time_are_rejected():
    filt = history()
    filt.predict_to(.3, TWISTS[0], COVARIANCE)
    filt.update(POSITION, QUATERNION, .1, .2)
    with pytest.raises(ValueError, match="arrival-time order"):
        filt.update(POSITION, QUATERNION, .15, .19)
    with pytest.raises(ValueError, match="arrival>=measurement"):
        filt.update(POSITION, QUATERNION, .3, .2)
    with pytest.raises(ValueError, match="monotonic"):
        filt.predict_to(.1, TWISTS[0])


def test_pose_measurement_capacity_is_bounded_and_snapshot_is_detached():
    filt = history(max_pose_measurements=1)
    filt.update(POSITION+[.01, 0, 0], QUATERNION, 0., 0., "a")
    assert filt.update(POSITION+[.02, 0, 0], QUATERNION, 0., 0., "b")["reason"] == "history_measurement_capacity"
    copy = filt.snapshot()
    copy.position[:] = 99
    copy.covariance[:] = 0
    assert np.linalg.norm(filt.position) < 2
    assert np.trace(filt.P) > 0
