import numpy as np
import pytest

from ev6d.event_simulator import EventCameraSimulator


def test_analytic_positive_crossings_and_exact_endpoint():
    sim = EventCameraSimulator(np.ones((1, 1)), contrast_threshold=.2)
    events = sim.process(np.exp([[.6]]), .3)
    np.testing.assert_allclose(events, [[.1, 0, 0, 1], [.2, 0, 0, 1], [.3, 0, 0, 1]], atol=1e-14)
    assert events.dtype == np.float64
    assert sim.process(np.exp([[.6]]), .4).shape == (0, 4)


def test_analytic_negative_crossings_and_pixel_coordinates():
    sim = EventCameraSimulator(np.ones((2, 3)), initial_time=4., contrast_threshold=.25)
    image = np.ones((2, 3))
    image[1, 2] = np.exp(-.75)
    events = sim.process(image, 4.3)
    np.testing.assert_allclose(events, [[4.1, 2, 1, -1], [4.2, 2, 1, -1], [4.3, 2, 1, -1]])


def test_reference_accumulation_and_polarity_reversal():
    sim = EventCameraSimulator(np.ones((1, 1)), contrast_threshold=.2)
    assert not len(sim.process(np.exp([[.15]]), .1))
    event = sim.process(np.exp([[.25]]), .2)
    np.testing.assert_allclose(event, [[.15, 0, 0, 1]])
    # Reference is .2, so after reversal the negative threshold is log(I)=0.
    event = sim.process(np.exp([[-.1]]), .3)
    np.testing.assert_allclose(event, [[.2+.1*(.25/.35), 0, 0, -1]])


def test_static_and_subthreshold_frames_are_event_free():
    sim = EventCameraSimulator(np.full((3, 5), .5))
    for i in range(1, 20):
        assert sim.process(np.full((3, 5), .5), i*.001).shape == (0, 4)
    assert not len(sim.process(np.full((3, 5), .5*np.exp(.19)), .1))


def test_global_sort_and_piecewise_linear_partition_equivalence():
    rates = np.array([[.6, -.4], [1., -.8]])
    a = EventCameraSimulator(np.ones((2, 2)))
    b = EventCameraSimulator(np.ones((2, 2)))
    expected = a.process(np.exp(rates), 1.)
    actual = np.concatenate([b.process(np.exp(rates*t), t) for t in [.25, .5, .75, 1.]])
    assert np.all(np.diff(expected[:, 0]) >= 0)
    # Equal-time events can have different stable ordering across partitions.
    sort = lambda x: x[np.lexsort((x[:, 3], x[:, 2], x[:, 1]))]
    np.testing.assert_allclose(sort(actual), sort(expected), atol=1e-13)


def test_refractory_consumes_crossings_without_static_backlog():
    sim = EventCameraSimulator(np.ones((1, 1)), contrast_threshold=.2, refractory_s=.15)
    events = sim.process(np.exp([[1.]]), .5)
    np.testing.assert_allclose(events[:, 0], [.1, .3, .5])
    assert not len(sim.process(np.exp([[1.]]), .6))
    events = sim.process(np.exp([[1.4]]), .8)
    np.testing.assert_allclose(events[:, 0], [.7])


def test_refractory_survives_polarity_change():
    sim = EventCameraSimulator(np.ones((1, 1)), refractory_s=.3)
    assert len(sim.process(np.exp([[.2]]), .1)) == 1
    assert len(sim.process(np.ones((1, 1)), .2)) == 0
    assert len(sim.process(np.exp([[-.2]]), .3)) == 0
    event = sim.process(np.exp([[-.4]]), .4)
    np.testing.assert_allclose(event, [[.4, 0, 0, -1]])


def test_seeded_threshold_mismatch_is_independent_and_reproducible():
    args = dict(threshold_sigma=.05, seed=23)
    a = EventCameraSimulator(np.ones((10, 8)), **args)
    b = EventCameraSimulator(np.ones((10, 8)), **args)
    c = EventCameraSimulator(np.ones((10, 8)), threshold_sigma=.05, seed=24)
    np.testing.assert_array_equal(a.positive_threshold, b.positive_threshold)
    assert not np.array_equal(a.positive_threshold, a.negative_threshold)
    assert not np.array_equal(a.positive_threshold, c.positive_threshold)
    np.testing.assert_array_equal(a.process(np.exp(np.ones((10, 8))), 1), b.process(np.exp(np.ones((10, 8))), 1))


@pytest.mark.parametrize("image", [[], [1, 2], [[0]], [[-1]], [[np.nan]], [[np.inf]]])
def test_invalid_initial_images(image):
    with pytest.raises(ValueError):
        EventCameraSimulator(image)


@pytest.mark.parametrize("params", [dict(contrast_threshold=0), dict(contrast_threshold=-1),
                                      dict(threshold_sigma=-1), dict(refractory_s=-1), dict(initial_time=np.nan)])
def test_invalid_parameters(params):
    with pytest.raises(ValueError):
        EventCameraSimulator([[1]], **params)


def test_invalid_updates_and_crossing_budget_are_atomic():
    sim = EventCameraSimulator([[1.]])
    for image, t in [([[1]], 0), ([[1]], -1), ([[1]], np.inf), ([[0]], 1), ([[1, 1]], 1)]:
        with pytest.raises(ValueError):
            sim.process(image, t)
        assert sim.timestamp == 0
        np.testing.assert_array_equal(sim.reference_log, [[0.]])
    sim.MAX_CROSSINGS = 2
    with pytest.raises(ValueError, match="Too many"):
        sim.process(np.exp([[.8]]), 1)
    assert sim.timestamp == 0
    np.testing.assert_array_equal(sim.previous_log, [[0.]])
    assert len(sim.process(np.exp([[.2]]), .1)) == 1
