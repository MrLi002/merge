"""Causal, bounded event-triplet optical flow (paper equations 1--3).

This is an independent reconstruction of Li et al., arXiv:2508.14776,
section III-A; the paper does not specify search windows or tolerances.
An incoming event is the *newest* member of a triplet. We search equally
spaced integer locations x, x-d, x-2d, using same-polarity timestamp histories.
Equal spacing reduces the constant-velocity test to equal inter-event times.
It is a bounded search subset, not an exhaustive enumeration of all triplets.
Candidate pruning and finite ROI history are explicit engineering choices.
Within those retained sets, equation 3 is solved exactly (Euclidean norm),
not approximated by a median of individual flows. Minimum-speed tie-breaking
selects normal flow when a straight edge leaves tangential motion ambiguous.

Timestamps are seconds and flow is pixel/second, never displacement per tick.
No frames, depth, poses, or ground-truth data are used in this module.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class FlowConfig:
    search_radius: int = 2
    history_per_pixel: int = 3
    max_age_s: float = .05
    min_dt_s: float = 1e-5
    relative_interval_tolerance: float = .30
    min_speed: float = 2.
    max_speed: float = 2000.
    roi_size: int = 8
    max_candidates: int = 6
    consensus_events: int = 8
    consensus_max_age_s: float = .02
    min_consensus_events: int = 2

    def __post_init__(self):
        for name in ("search_radius", "history_per_pixel", "roi_size", "max_candidates",
                     "consensus_events", "min_consensus_events"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        vals = (self.max_age_s, self.min_dt_s, self.relative_interval_tolerance,
                self.min_speed, self.max_speed, self.consensus_max_age_s)
        if not np.isfinite(vals).all():
            raise ValueError("Flow thresholds must be finite")
        if self.min_dt_s <= 0 or self.max_age_s < 2*self.min_dt_s:
            raise ValueError("max_age_s must allow two positive min_dt_s intervals")
        if not 0 <= self.relative_interval_tolerance < 2:
            raise ValueError("relative_interval_tolerance must lie in [0, 2)")
        if self.min_speed < 0 or self.max_speed <= self.min_speed or self.consensus_max_age_s <= 0:
            raise ValueError("Invalid speed or consensus time limits")
        if self.min_consensus_events > self.consensus_events:
            raise ValueError("min_consensus_events exceeds consensus_events")


@dataclass
class FlowEstimate:
    """t is availability time; quality records the actual event/support times."""

    t: float
    x: int
    y: int
    flow: np.ndarray
    valid: bool
    quality: dict = field(default_factory=dict)


def select_roi_flow(candidate_sets):
    """Return exact equation-3 minimizer and cost for nonempty finite sets.

    Each event contributes its distance to the *nearest member of its own
    candidate set*, irrespective of how many candidates that event has.
    The returned vector is a candidate, not a mean or componentwise median.
    Equal-cost solutions use minimum speed, then lexicographic ordering.
    """
    sets = [np.asarray(s, dtype=float) for s in candidate_sets]
    if not sets or any(s.ndim != 2 or s.shape[1] != 2 or not len(s) or
                       not np.isfinite(s).all() for s in sets):
        raise ValueError("Candidate sets must be nonempty finite N x 2 arrays")
    return _select_roi_flow(sets)


def _select_roi_flow(sets):
    """Internal path: arrays have already passed input and candidate checks."""
    candidates = np.concatenate(sets)
    delta = candidates[:, None, :] - candidates[None, :, :]
    distances = np.sqrt(np.einsum("ijk,ijk->ij", delta, delta))
    starts = np.concatenate(([0], np.cumsum([len(s) for s in sets])[:-1]))
    costs = np.minimum.reduceat(distances, starts, axis=1).sum(axis=1)
    best_cost = costs.min()
    tied = np.flatnonzero(costs <= best_cost+1e-12*(1+abs(best_cost)))
    if len(tied) == 1:
        index = tied[0]
        return candidates[index].copy(), float(costs[index])
    tied_flows = candidates[tied]
    order = np.lexsort((tied_flows[:, 1], tied_flows[:, 0],
                        np.einsum("ij,ij->i", tied_flows, tied_flows)))
    index = tied[order[0]]
    return candidates[index].copy(), float(costs[index])


@dataclass
class _CandidateSet:
    t: float
    oldest_support: float
    flows: np.ndarray


class TripletFlow:
    """Streaming optical flow with O(H W history + ROIs N C) retained state.

    Per-event work is bounded by radius, timestamp-history depth, and N*C
    retained ROI candidates. The exact consensus costs O((N*C)**2); it never
    grows with the total number of input events. Returns one estimate per
    input event; events without triplets are explicitly invalid, not zero-flow
    observations. Call reset() before replaying a stream or changing its clock.
    """

    def __init__(self, width, height, config=None):
        if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v <= 0
               for v in (width, height)):
            raise ValueError("Sensor dimensions must be positive integers")
        self.width, self.height = int(width), int(height)
        self.config = config if config is not None else FlowConfig()
        if not isinstance(self.config, FlowConfig):
            raise TypeError("config must be a FlowConfig")
        r = self.config.search_radius
        # Smaller displacements first make tied pruning deterministic.
        offsets = [(dx, dy) for dy in range(-r, r+1) for dx in range(-r, r+1)
                   if dx or dy]
        offsets.sort(key=lambda d: (d[0]**2+d[1]**2, d[0], d[1]))
        self._offsets = np.array(offsets, dtype=int)
        self._offset_lengths = np.linalg.norm(self._offsets, axis=1)
        self._offset_steps = self._offsets[:, 1]*width+self._offsets[:, 0]
        self._history = np.full((2, height, width, self.config.history_per_pixel), -np.inf)
        self._flat_history = self._history.reshape(2, height*width, self.config.history_per_pixel)
        self._roi = {}
        self._last_event_time = -np.inf
        self._last_emit_time = -np.inf

    def reset(self):
        self._history.fill(-np.inf)
        self._roi.clear()
        self._last_event_time = self._last_emit_time = -np.inf

    def _expire_roi(self, t):
        for key, history in list(self._roi.items()):
            while history and t-history[0].t > self.config.consensus_max_age_s:
                history.popleft()
            if not history:
                del self._roi[key]

    def _candidates(self, t, x, y, polarity):
        cfg = self.config
        offsets = self._offsets
        lengths, steps = self._offset_lengths, self._offset_steps
        # Both older locations must lie inside the sensor.
        margin = 2*cfg.search_radius
        if not (margin <= x < self.width-margin and margin <= y < self.height-margin):
            x2, y2 = x-2*offsets[:, 0], y-2*offsets[:, 1]
            inside = (x2 >= 0) & (x2 < self.width) & (y2 >= 0) & (y2 < self.height)
            offsets, lengths, steps = offsets[inside], lengths[inside], steps[inside]
        if not len(offsets):
            return None
        pixel = y*self.width+x
        middle = self._flat_history[polarity, pixel-steps]
        oldest = self._flat_history[polarity, pixel-2*steps]
        # History combinations allow repeated contrast crossings at a pixel.
        with np.errstate(invalid="ignore", divide="ignore"):
            dt1 = t-middle[:, :, None]
            dt2 = middle[:, :, None]-oldest[:, None, :]
            total = dt1+dt2
            mismatch = 2*np.abs(dt1-dt2)/total
            speed = 2*lengths[:, None, None]/total
            good = (np.isfinite(total) & (dt1 >= cfg.min_dt_s) & (dt2 >= cfg.min_dt_s) &
                    (total <= cfg.max_age_s) & (mismatch <= cfg.relative_interval_tolerance+1e-12) &
                    (speed >= cfg.min_speed) & (speed <= cfg.max_speed))
        d, j, k = np.nonzero(good)
        if not len(d):
            return None
        candidate = 2*offsets[d]/total[d, j, k, None]
        errors = mismatch[d, j, k]
        speeds = speed[d, j, k]
        # Equation 3 still receives several hypotheses per event. Pruning uses
        # interval consistency first, minimum speed only to resolve ties.
        order = np.lexsort((candidate[:, 1], candidate[:, 0], speeds, errors))
        selected, support, seen = [], [], set()
        rounded = np.round(candidate, decimals=8)
        for index in order:
            key = tuple(rounded[index])
            if key in seen:
                continue
            seen.add(key)
            selected.append(candidate[index])
            support.append(oldest[d[index], k[index]])
            if len(selected) == cfg.max_candidates:
                break
        return np.array(selected), float(min(support)), float(errors[order[0]])

    def process(self, events, emit_time=None):
        """Process sorted [t,x,y,p] events causally and emit at emit_time.

        Polarity accepts -1/+1 or 0/1; -1 and 0 both denote OFF. Duplicate
        (t,x,y,polarity) events are invalid and are not inserted twice. Invalid
        input is rejected before any streaming state changes. Empty batches
        can advance the watermark and expire stale ROI evidence. Events older
        than a preceding emission watermark are rejected, never silently sorted.
        """
        data = np.asarray(events, dtype=float)
        if data.ndim != 2 or data.shape[1] != 4 or not np.isfinite(data).all():
            raise ValueError("events must be a finite N x 4 [t,x,y,p] array")
        if len(data):
            if np.any(np.diff(data[:, 0]) < 0) or data[0, 0] < self._last_event_time:
                raise ValueError("Event timestamps must be nondecreasing; call reset() to replay")
            if data[0, 0] < self._last_emit_time-1e-10:
                raise ValueError("Events precede the previous emission watermark")
            xy = data[:, 1:3]
            if (not np.equal(xy, np.floor(xy)).all() or np.any(xy < 0) or
                    np.any(xy[:, 0] >= self.width) or np.any(xy[:, 1] >= self.height)):
                raise ValueError("Event coordinates must be integer pixels inside sensor")
            if not np.isin(data[:, 3], [-1, 0, 1]).all():
                raise ValueError("Event polarity must be -1/+1 or 0/1")
        if emit_time is not None:
            emit_time = float(emit_time)
            if (not np.isfinite(emit_time) or emit_time < self._last_emit_time or
                    (len(data) and emit_time < data[-1, 0]-1e-10)):
                raise ValueError("emit_time must be finite, monotonic, and no earlier than events")
        elif len(data):
            if data[-1, 0] < self._last_emit_time:
                raise ValueError("Event timestamps precede the previous emission watermark")
        results = []
        for t, xf, yf, pf in data:
            x, y, polarity = int(xf), int(yf), int(pf > 0)
            available = float(t if emit_time is None else emit_time)
            quality = {"source_event_time": float(t), "oldest_support_time": float(t),
                       "candidate_count": 0, "consensus_events": 0}
            history = self._history[polarity, y, x]
            duplicate = history[0] == t
            found = None if duplicate else self._candidates(t, x, y, polarity)
            if not duplicate:
                history[1:] = history[:-1]
                history[0] = t
            flow, valid = np.full(2, np.nan), False
            quality["reason"] = "duplicate" if duplicate else "no_triplet"
            if found is not None:
                candidates, oldest_support, error = found
                key = (x//self.config.roi_size, y//self.config.roi_size)
                roi = self._roi.setdefault(key, deque(maxlen=self.config.consensus_events))
                while roi and t-roi[0].t > self.config.consensus_max_age_s:
                    roi.popleft()
                roi.append(_CandidateSet(float(t), oldest_support, candidates))
                quality.update(candidate_count=len(candidates), consensus_events=len(roi),
                               oldest_support_time=min(s.oldest_support for s in roi),
                               interval_error=error)
                if len(roi) >= self.config.min_consensus_events:
                    flow, cost = _select_roi_flow([s.flows for s in roi])
                    quality.update(consensus_cost=cost, reason="valid")
                    valid = True
                else:
                    quality["reason"] = "insufficient_consensus"
            results.append(FlowEstimate(available, x, y, flow, valid, quality))
        if len(data):
            self._last_event_time = float(data[-1, 0])
        if emit_time is not None or len(data):
            self._last_emit_time = float(emit_time if emit_time is not None else data[-1, 0])
            self._expire_roi(self._last_emit_time)
        return results
