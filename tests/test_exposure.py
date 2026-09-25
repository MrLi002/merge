"""Analytic exposure integration checks independent of a renderer."""

import numpy as np
import pytest

from ev6d.exposure import ExposureIntegrator


def test_constant_intensity_at_irregular_times():
    initial = np.array([[[0.1, 0.5, 2.0], [0.3, 0.2, 0.8]]], dtype=np.float32)
    exposure = ExposureIntegrator(initial, frame_hz=60)
    frames = []
    for t in (0.002, 0.0107, 0.023, 0.080, 0.100):
        frames.extend(exposure.process(initial, t))
    assert len(frames) == 6
    for _, rgb in frames:
        np.testing.assert_array_equal(rgb, initial)
        assert rgb.dtype == np.float32
    assert exposure.metadata["duty_cycle"] == 1.0
    assert exposure.metadata["timestamp_convention"] == "exposure_end"


def test_linear_ramp_has_analytic_exposure_average():
    base = np.array([[[0.2, 0.3, 0.4]], [[0.7, 0.1, 0.0]]])
    slope = np.array([[[0.5, -0.1, 1.5]], [[0.2, 0.7, 0.0]]])
    exposure = ExposureIntegrator(base, frame_hz=60)
    frames = exposure.process(base + slope * 0.1, 0.1)
    assert len(frames) == 6
    for i, (t, rgb) in enumerate(frames, 1):
        assert t == i / 60
        midpoint = (i - 0.5) / 60
        np.testing.assert_allclose(rgb, base + slope * midpoint, atol=5e-8)


def test_500_hz_source_produces_exactly_60_frames_per_second():
    exposure = ExposureIntegrator(np.zeros((2, 3, 3), dtype=np.float32))
    frames = []
    for k in range(1, 501):
        t = k / 500
        frames.extend(exposure.process(np.full((2, 3, 3), t, dtype=np.float32), t))
    assert len(frames) == 60
    np.testing.assert_array_equal([t for t, _ in frames], np.arange(1, 61) / 60)
    np.testing.assert_allclose([rgb[0, 0, 0] for _, rgb in frames], (np.arange(60) + 0.5) / 60, atol=1e-7)


def test_linear_interpolation_is_invariant_to_source_interval_splitting():
    shape = (2, 2, 3)
    sample = lambda t: np.full(shape, 0.4 + 0.75 * t, dtype=np.float64)
    single = ExposureIntegrator(sample(0), frame_hz=60)
    split = ExposureIntegrator(sample(0), frame_hz=60)
    whole = single.process(sample(0.15), 0.15)
    parts = []
    for t in (0.001, 0.007, 0.023, 0.081, 0.102, 0.15):
        parts.extend(split.process(sample(t), t))
    np.testing.assert_array_equal([t for t, _ in parts], [t for t, _ in whole])
    for (_, a), (_, b) in zip(whole, parts):
        np.testing.assert_allclose(a, b, atol=1e-7)


def test_partial_exposure_is_retained_and_nonzero_origin_supported():
    image = np.ones((1, 1, 3), dtype=np.float32)
    exposure = ExposureIntegrator(image, initial_time=3.0, frame_hz=20)
    assert exposure.process(image, 3.02) == []
    frames = exposure.process(image, 3.075)
    assert len(frames) == 1 and frames[0][0] == 3.05
    np.testing.assert_array_equal(frames[0][1], image)
    frames = exposure.process(image, 3.10)
    assert len(frames) == 1 and frames[0][0] == 3.10
    np.testing.assert_array_equal(frames[0][1], image)


@pytest.mark.parametrize("kwargs", [
    {"initial_time": np.nan}, {"initial_time": np.inf},
    {"frame_hz": 0}, {"frame_hz": -1}, {"frame_hz": np.nan},
    {"frame_hz": np.inf}, {"frame_hz": 1e30, "initial_time": 1.0},
    {"exposure_s": 0.001}, {"exposure_s": np.nan},
])
def test_invalid_initial_timing(kwargs):
    with pytest.raises(ValueError):
        ExposureIntegrator(np.ones((1, 1, 3), dtype=np.float32), **kwargs)


@pytest.mark.parametrize("bad", [np.zeros((2, 3)), np.zeros((2, 2, 4)),
                                 np.zeros((0, 2, 3)), np.full((2, 2, 3), np.nan)])
def test_invalid_initial_image(bad):
    with pytest.raises(ValueError):
        ExposureIntegrator(bad)


@pytest.mark.parametrize("bad_time", [0.0, -0.01, np.nan, np.inf])
def test_invalid_process_timestamp_does_not_change_state(bad_time):
    image = np.ones((2, 2, 3), dtype=np.float32)
    exposure = ExposureIntegrator(image)
    with pytest.raises(ValueError):
        exposure.process(image, bad_time)
    frames = exposure.process(image, 1 / 60)
    assert len(frames) == 1
    np.testing.assert_array_equal(frames[0][1], image)


@pytest.mark.parametrize("bad", [np.ones((3, 2, 3)), np.full((2, 2, 3), np.inf)])
def test_invalid_process_image_does_not_change_state(bad):
    image = np.ones((2, 2, 3), dtype=np.float32)
    exposure = ExposureIntegrator(image)
    with pytest.raises(ValueError):
        exposure.process(bad, 0.01)
    frames = exposure.process(image, 1 / 60)
    assert len(frames) == 1
    np.testing.assert_array_equal(frames[0][1], image)


def test_input_images_are_copied_and_equal_timestamps_rejected():
    initial = np.ones((1, 1, 3), dtype=np.float32)
    exposure = ExposureIntegrator(initial, exposure_s=1 / 60)
    initial.fill(5)
    sample = np.ones((1, 1, 3), dtype=np.float32)
    assert exposure.process(sample, 0.01) == []
    sample.fill(8)
    with pytest.raises(ValueError):
        exposure.process(sample, 0.01)
    frames = exposure.process(np.ones_like(initial), 1 / 60)
    np.testing.assert_array_equal(frames[0][1], np.ones_like(initial))
