"""Bounded delayed absolute-pose replay around the existing manifold PoseUKF.

The live pose remains at the latest processing time. A delayed measurement is
applied at its own timestamp, with saved twist intervals and later pose updates
replayed exactly once. This changes the current estimate, not emitted past rows.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .filters import PoseUKF, _covariance
from .geometry import quat_multiply


@dataclass
class PoseSnapshot:
    timestamp: float
    position: np.ndarray
    quaternion: np.ndarray
    covariance: np.ndarray

    def copy(self):
        return PoseSnapshot(self.timestamp, self.position.copy(),
                            self.quaternion.copy(), self.covariance.copy())


@dataclass
class _Interval:
    start: float
    end: float
    velocity: np.ndarray
    covariance: np.ndarray | None


@dataclass
class _PoseEvent:
    serial: int
    measurement_time: float
    arrival_time: float
    position: np.ndarray
    quaternion: np.ndarray
    identifier: str | None
    result: dict | None = None


class PoseHistory:
    """Timestamp-aware pose wrapper; do not separately mutate the wrapped filter.

    predict_to(t, xi, Pxi) records the twist used over [current_time,t]. update()
    receives an absolute T_event_object pose (quaternion xyzw). The caller first
    predicts through arrival_time; no observation can be applied before arrival.
    Arrival times must be supplied monotonically, measurement times need not be.
    Duplicate protection uses an optional measurement_id, or equal measurement
    time/pose (including quaternion antipodes) while the event is in the buffer.

    Uncertain twists are re-augmented at each propagation segment by PoseUKF.
    As in the existing two-stage filter, input/state and cross-interval covariance
    are not retained. A measurement splitting an interval uses that approximation
    on both pieces; it matches chronological filtering with the same split.
    """

    def __init__(self, pose_filter, max_history_s=2., max_intervals=1000,
                 max_pose_measurements=1000):
        if not isinstance(pose_filter, PoseUKF):
            raise TypeError("PoseHistory requires an existing PoseUKF")
        if not np.isfinite(max_history_s) or max_history_s <= 0:
            raise ValueError("max_history_s must be positive seconds")
        for value in (max_intervals, max_pose_measurements):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError("History capacities must be positive integers")
        self.filter = pose_filter
        self.max_history_s = float(max_history_s)
        self.max_intervals = int(max_intervals)
        self.max_pose_measurements = int(max_pose_measurements)
        self._anchor = self.snapshot()
        self._intervals = []
        self._snapshots = [self._anchor.copy()]
        self._events = []
        self._serial = 0
        self._last_arrival = -np.inf
        self.counters = {"received": 0, "accepted_on_arrival": 0, "rejected_gate": 0,
                         "rejected_out_of_history": 0, "rejected_future": 0,
                         "rejected_duplicate": 0, "rejected_history_capacity": 0,
                         "replay_calls": 0, "replayed_predictions": 0}

    @property
    def timestamp(self):
        return self.filter.timestamp

    @property
    def position(self):
        return self.filter.position

    @property
    def quaternion(self):
        return self.filter.quaternion

    @property
    def P(self):
        return self.filter.P

    @property
    def earliest_time(self):
        return max(self._anchor.timestamp, self.timestamp-self.max_history_s)

    def snapshot(self):
        return PoseSnapshot(self.filter.timestamp, self.filter.position.copy(),
                            self.filter.quaternion.copy(), self.filter.P.copy())

    def diagnostics(self):
        return {**self.counters, "history_start_s": self.earliest_time,
                "history_end_s": self.timestamp, "history_intervals": len(self._intervals),
                "history_pose_measurements": len(self._events),
                "accepted_in_history": sum(bool(event.result and event.result["accepted"])
                                           for event in self._events)}

    def _restore(self, snapshot):
        self.filter.position = snapshot.position.copy()
        self.filter.quaternion = snapshot.quaternion.copy()
        self.filter.P = snapshot.covariance.copy()
        self.filter.timestamp = snapshot.timestamp

    def predict_to(self, timestamp, velocity, velocity_covariance=None):
        timestamp = float(timestamp)
        velocity = np.asarray(velocity, dtype=float)
        if not np.isfinite(timestamp) or timestamp < self.timestamp or velocity.shape != (6,) or not np.isfinite(velocity).all():
            raise ValueError("Pose history prediction needs monotonic seconds and a finite 6D twist")
        covariance = None if velocity_covariance is None else _covariance(velocity_covariance, 6)
        if timestamp == self.timestamp:
            return
        interval = _Interval(self.timestamp, timestamp, velocity.copy(), covariance)
        self.filter.predict_to(timestamp, velocity, covariance)
        self._intervals.append(interval)
        self._snapshots.append(self.snapshot())
        self._prune()

    def _prune(self):
        cutoff = self.timestamp-self.max_history_s
        count = max(0, len(self._intervals)-self.max_intervals)
        while count < len(self._intervals) and self._intervals[count].end <= cutoff:
            count += 1
        if count:
            self._anchor = self._snapshots[count].copy()
            self._intervals = self._intervals[count:]
            self._snapshots = self._snapshots[count:]
            # The anchor already contains all measurement corrections at its time.
            self._events = [event for event in self._events
                            if event.measurement_time > self._anchor.timestamp]

    def _replay(self):
        self._restore(self._anchor)
        events = sorted(self._events, key=lambda event: (event.measurement_time, event.serial))
        cursor, predictions = 0, 0

        def apply_event(event):
            event.result = self.filter.update(event.position, event.quaternion)

        while cursor < len(events) and events[cursor].measurement_time == self.timestamp:
            apply_event(events[cursor])
            cursor += 1
        snapshots = [self.snapshot()]
        for interval in self._intervals:
            # Each loop resumes from the last timestamp; no duration is integrated twice.
            while cursor < len(events) and events[cursor].measurement_time <= interval.end:
                event = events[cursor]
                before_trial = self.snapshot()
                if event.measurement_time > self.timestamp:
                    self.filter.predict_to(event.measurement_time, interval.velocity, interval.covariance)
                    predictions += 1
                apply_event(event)
                if not event.result["accepted"]:
                    # A rejected interior measurement must not introduce a new
                    # covariance integration split into the accepted trajectory.
                    self._restore(before_trial)
                cursor += 1
            if interval.end > self.timestamp:
                self.filter.predict_to(interval.end, interval.velocity, interval.covariance)
                predictions += 1
            snapshots.append(self.snapshot())
        if cursor != len(events):
            raise RuntimeError("Pose replay event falls outside recorded intervals")
        self._snapshots = snapshots
        self.counters["replay_calls"] += 1
        self.counters["replayed_predictions"] += predictions
        return predictions

    def update(self, position, quaternion, measurement_time, arrival_time, measurement_id=None):
        measurement_time, arrival_time = float(measurement_time), float(arrival_time)
        if not np.isfinite([measurement_time, arrival_time]).all() or arrival_time < measurement_time:
            raise ValueError("Pose measurement and arrival times must be finite, with arrival>=measurement")
        if arrival_time < self._last_arrival:
            raise ValueError("Pose observations must be delivered in arrival-time order")
        position = np.asarray(position, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("Measured position must be a finite three-vector")
        quaternion = quat_multiply(quaternion, [0, 0, 0, 1])
        if measurement_id is not None and not isinstance(measurement_id, str):
            raise ValueError("measurement_id must be a string or None")
        self.counters["received"] += 1
        result = {"accepted": False, "measurement_time": measurement_time,
                  "arrival_time": arrival_time, "delay_s": arrival_time-measurement_time,
                  "processing_time": self.timestamp, "replayed_predictions": 0}

        def reject(reason, counter):
            self.counters[counter] += 1
            return {**result, "reason": reason}

        # A future record may safely be retried after prediction; do not consume it.
        if arrival_time > self.timestamp or measurement_time > self.timestamp:
            return reject("not_yet_arrived", "rejected_future")
        self._last_arrival = arrival_time
        if measurement_time < self.earliest_time:
            return reject("outside_history", "rejected_out_of_history")
        for event in self._events:
            same_id = measurement_id is not None and event.identifier == measurement_id
            same_pose = (event.measurement_time == measurement_time and
                         np.array_equal(event.position, position) and
                         abs(float(event.quaternion@quaternion)) > 1-1e-12)
            if same_id or (measurement_id is None and same_pose):
                return reject("duplicate_measurement", "rejected_duplicate")
        if len(self._events) >= self.max_pose_measurements:
            return reject("history_measurement_capacity", "rejected_history_capacity")
        event = _PoseEvent(self._serial, measurement_time, arrival_time, position.copy(),
                           quaternion.copy(), measurement_id)
        self._serial += 1
        previous = self.snapshot()
        old_results = [None if old.result is None else old.result.copy() for old in self._events]
        self._events.append(event)
        try:
            predictions = self._replay()
        except Exception:
            # Failure is explicit and leaves the live state/history transaction intact.
            self._events.pop()
            for old, old_result in zip(self._events, old_results):
                old.result = old_result
            self._restore(previous)
            raise
        accepted = bool(event.result["accepted"])
        self.counters["accepted_on_arrival" if accepted else "rejected_gate"] += 1
        return {**result, **event.result, "reason": "updated" if accepted else "pose_gate",
                "replayed_predictions": predictions}
