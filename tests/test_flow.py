"""Behavioral tests with event-only moving points/edges and exact consensus."""
import numpy as np
import pytest

from ev6d.flow import FlowConfig, TripletFlow, select_roi_flow


def point_events(dx=1, dy=0, speed=100., count=8, polarity=1):
    dt = np.hypot(dx, dy)/speed
    return np.array([[i*dt, 12+i*dx, 12+i*dy, polarity] for i in range(count)])


@pytest.mark.parametrize("dx,dy", [(1, 0), (-1, 0), (0, 1), (1, 1), (2, 1)])
def test_known_triplets_recover_flow_direction_and_si_units(dx, dy):
    events = point_events(dx, dy, count=6)
    engine = TripletFlow(40, 40, FlowConfig(min_consensus_events=1))
    estimates = engine.process(events, emit_time=.2)
    assert not estimates[0].valid and not estimates[1].valid
    expected = 100*np.array([dx, dy])/np.hypot(dx, dy)
    for event, estimate in zip(events[2:], estimates[2:]):
        assert estimate.valid
        np.testing.assert_allclose(estimate.flow, expected, atol=1e-9)
        assert estimate.t == .2
        assert estimate.quality["source_event_time"] == event[0]
        assert estimate.quality["oldest_support_time"] <= event[0]


def test_straight_moving_edge_recovers_normal_flow():
    events = np.array([[x*.005, x, y, 1] for x in range(4, 31) for y in range(5, 20)])
    engine = TripletFlow(40, 30)
    valid = [e for e in engine.process(events) if e.valid]
    assert len(valid) > .8*len(events)
    np.testing.assert_allclose(np.array([e.flow for e in valid]),
                               np.tile([200., 0.], (len(valid), 1)), atol=1e-8)


def test_equation_three_uses_per_event_candidate_sets_not_flattened_medoid():
    sets = [np.array([[0., 0.], [100., 0.]]), np.array([[100., 0.]]),
            np.array([[float(x), 0.] for x in range(5)])]
    flow, cost = select_roi_flow(sets)
    np.testing.assert_array_equal(flow, [100., 0.])
    assert cost == 96.
    all_candidates = np.concatenate(sets)
    brute = [sum(np.linalg.norm(s-f, axis=1).min() for s in sets) for f in all_candidates]
    assert cost == min(brute)
    flattened = np.linalg.norm(all_candidates[:, None]-all_candidates[None], axis=-1).sum(axis=1)
    assert not np.array_equal(all_candidates[flattened.argmin()], flow)


def test_minimum_speed_breaks_unobservable_tangent_ties():
    flow, cost = select_roi_flow([np.array([[100., -100.], [100., 0.], [100., 100.]])]*3)
    np.testing.assert_array_equal(flow, [100., 0.])
    assert cost == 0.


def test_triplets_do_not_mix_polarities():
    events = point_events(count=3)
    events[1, 3] = -1
    assert not any(e.valid for e in TripletFlow(40, 40, FlowConfig(min_consensus_events=1)).process(events))
    # OFF polarity encodings are equivalent; no additional sign inversion.
    events[:, 3] = [0, -1, 0]
    assert TripletFlow(40, 40, FlowConfig(min_consensus_events=1)).process(events)[-1].valid


def test_duplicate_events_do_not_create_triplets_or_overwrite_history():
    engine = TripletFlow(40, 40, FlowConfig(min_consensus_events=1))
    original = point_events(count=3)
    events = np.repeat(original, 6, axis=0)
    result = engine.process(events)
    assert sum(e.valid for e in result) == 1
    assert sum(e.quality["reason"] == "duplicate" for e in result) == 15
    np.testing.assert_allclose(result[12].flow, [100., 0.])


def test_irregular_timing_and_stale_support_are_rejected():
    cfg = FlowConfig(min_consensus_events=1, relative_interval_tolerance=.1)
    irregular = point_events(count=3)
    irregular[-1, 0] = .035
    assert not any(e.valid for e in TripletFlow(40, 40, cfg).process(irregular))
    stale = point_events(speed=10., count=3)
    assert not any(e.valid for e in TripletFlow(40, 40, cfg).process(stale))


def test_pixel_history_recovers_triplet_despite_more_recent_unrelated_crossing():
    events = np.array([[0., 10, 12, 1], [.01, 11, 12, 1],
                       [.016, 11, 12, 1], [.02, 12, 12, 1]])
    cfg = FlowConfig(min_consensus_events=1, relative_interval_tolerance=.1)
    result = TripletFlow(40, 40, cfg).process(events)[-1]
    assert result.valid
    np.testing.assert_allclose(result.flow, [100., 0.])
    assert result.quality["oldest_support_time"] == 0.
    latest_only = FlowConfig(history_per_pixel=1, min_consensus_events=1,
                             relative_interval_tolerance=.1)
    assert not TripletFlow(40, 40, latest_only).process(events)[-1].valid


def test_empty_batch_expires_consensus_and_old_events_cannot_be_reused():
    engine = TripletFlow(40, 40, FlowConfig(min_consensus_events=1))
    events = point_events(count=6)
    assert any(e.valid for e in engine.process(events))
    assert engine.process(np.empty((0, 4)), emit_time=1.) == []
    assert not engine._roi
    next_event = np.array([[1.01, 18, 12, 1.]])
    assert not engine.process(next_event)[0].valid
    with pytest.raises(ValueError, match="watermark"):
        engine.process(np.array([[1.02, 19, 12, 1.]]), emit_time=2.)
        engine.process(np.array([[1.03, 20, 12, 1.]]))


def test_causal_streaming_is_independent_of_batch_partition():
    events = np.array([[x*.005, x, y, 1] for x in range(4, 25) for y in range(5, 12)])
    all_results = TripletFlow(40, 30).process(events)
    engine = TripletFlow(40, 30)
    chunks = np.array_split(events, 11)
    split_results = [e for chunk in chunks for e in engine.process(chunk)]
    assert [e.valid for e in all_results] == [e.valid for e in split_results]
    for a, b in zip(all_results, split_results):
        np.testing.assert_allclose(a.flow, b.flow, equal_nan=True)
        assert a.quality == b.quality


def test_reset_allows_replay_without_leaking_previous_state():
    events = point_events(count=6)
    engine = TripletFlow(40, 40)
    first = engine.process(events)
    with pytest.raises(ValueError):
        engine.process(events)
    engine.reset()
    second = engine.process(events)
    assert [e.valid for e in first] == [e.valid for e in second]
    for a, b in zip(first, second):
        np.testing.assert_allclose(a.flow, b.flow, equal_nan=True)


@pytest.mark.parametrize("bad", [
    [[.01, 1, 1, 1], [0, 2, 1, 1]], [[0, 1.5, 1, 1]], [[0, -1, 1, 1]],
    [[0, 40, 1, 1]], [[0, 1, 1, 2]], [[np.nan, 1, 1, 1]], [[0, 1, 1]],
])
def test_bad_input_rejected_atomically(bad):
    engine = TripletFlow(40, 40)
    with pytest.raises(ValueError):
        engine.process(bad)
    assert np.isneginf(engine._history).all()
    assert not engine._roi


def test_future_events_are_rejected_atomically():
    engine = TripletFlow(40, 40)
    with pytest.raises(ValueError, match="emit_time"):
        engine.process(point_events(count=4), emit_time=.01)
    assert np.isneginf(engine._history).all()


def test_retained_storage_is_bounded_under_dense_stream():
    cfg = FlowConfig(consensus_events=4, max_candidates=3)
    engine = TripletFlow(40, 30, cfg)
    for cycle in range(10):
        events = np.array([[cycle*.15+x*.004, x, y, 1]
                           for x in range(4, 31) for y in range(5, 20)])
        engine.process(events)
        assert all(len(history) <= 4 for history in engine._roi.values())
        assert all(len(s.flows) <= 3 for history in engine._roi.values() for s in history)
    assert engine._history.shape == (2, 30, 40, 3)


@pytest.mark.parametrize("kwargs", [{"search_radius": 0}, {"roi_size": 1.2},
                                    {"min_dt_s": 0}, {"max_age_s": 1e-7},
                                    {"relative_interval_tolerance": 2.},
                                    {"max_speed": np.inf}, {"consensus_events": 1}])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        FlowConfig(**kwargs)
