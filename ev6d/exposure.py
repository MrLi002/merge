"""Full-duty RGB exposure from irregular samples of linear-light intensity.

Each source interval is interpolated linearly and integrated across exact
``initial_time + k / frame_hz`` exposure boundaries. This supports 500 Hz
source images and 60 Hz output without rounding their noninteger rate ratio.
There is no output at initialization, and an unfinished exposure is retained.
"""
from __future__ import annotations

import numpy as np


class ExposureIntegrator:
    """Integrate linear RGB samples; emit ``(exposure_end, average_rgb)``.

    Inputs and emitted images have shape H x W x 3. Inputs are interpreted as
    linear-light intensity: any sRGB decoding belongs before this class, and
    display encoding belongs after it. Values are not clipped, allowing HDR.
    Arithmetic and accumulation use float64; emitted images use float32.

    Only full-duty exposure is supported. ``exposure_s`` may be omitted or
    explicitly equal ``1 / frame_hz``; other durations are rejected.
    """

    def __init__(self, initial_rgb, initial_time=0.0, frame_hz=60.0, exposure_s=None):
        initial_time, frame_hz = float(initial_time), float(frame_hz)
        if not np.isfinite(initial_time) or not np.isfinite(frame_hz) or frame_hz <= 0:
            raise ValueError("Initial time must be finite and frame_hz positive and finite")
        period = 1.0 / frame_hz
        if not np.isfinite(period) or initial_time + period <= initial_time:
            raise ValueError("Frame period is not representable at the initial timestamp")
        if exposure_s is not None:
            exposure_s = float(exposure_s)
            if not np.isfinite(exposure_s) or not np.isclose(exposure_s, period, rtol=1e-12, atol=0):
                raise ValueError("Only full-duty exposure_s = 1 / frame_hz is supported")
        initial = self._image(initial_rgb)
        self.initial_time = initial_time
        self.frame_hz = frame_hz
        self.exposure_s = period
        self.metadata = {
            "mode": "piecewise_linear_temporal_integration",
            "duty_cycle": 1.0,
            "frame_hz": frame_hz,
            "exposure_s": period,
            "timestamp_convention": "exposure_end",
            "color_space": "linear_rgb",
            "accumulator_dtype": "float64",
            "output_dtype": "float32",
        }
        self._shape = initial.shape
        self._previous_rgb = initial.copy()
        self._previous_time = initial_time
        self._frame_start = initial_time
        self._frame_index = 1
        self._next_boundary = initial_time + 1.0 / frame_hz
        self._integral = np.zeros(initial.shape, dtype=np.float64)

    @staticmethod
    def _image(rgb):
        image = np.asarray(rgb, dtype=np.float64)
        if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
            raise ValueError("RGB image must have nonempty H x W x 3 shape")
        if not np.isfinite(image).all():
            raise ValueError("RGB image must contain finite linear-light values")
        return image

    def process(self, rgb, t):
        """Add a strictly later sample and return all newly finished exposures.

        Images and timestamps are validated before any state is modified.
        No frame extends beyond ``t``; partial final exposures are not emitted.
        """
        t = float(t)
        image = self._image(rgb)
        if image.shape != self._shape:
            raise ValueError("RGB image shape must remain unchanged")
        if not np.isfinite(t) or t <= self._previous_time:
            raise ValueError("Source timestamps must be finite and strictly increasing")
        interval = t - self._previous_time
        if not np.isfinite(interval):
            raise ValueError("Source timestamp interval must be finite")
        change = image - self._previous_rgb
        cursor = self._previous_time
        frames = []
        while cursor < t:
            end = min(t, self._next_boundary)
            duration = end - cursor
            # Exact integral of the line joining the two source samples.
            mid_fraction = ((cursor - self._previous_time) + (end - self._previous_time)) / (2 * interval)
            self._integral += (self._previous_rgb + mid_fraction * change) * duration
            cursor = end
            if end == self._next_boundary:
                # Normalize by actual represented duration to preserve constant
                # intensity even when timestamps use a large nonzero origin.
                average = self._integral / (end - self._frame_start)
                frames.append((float(end), average.astype(np.float32)))
                self._integral.fill(0.0)
                self._frame_start = end
                self._frame_index += 1
                self._next_boundary = self.initial_time + self._frame_index / self.frame_hz
                if self._next_boundary <= self._frame_start:
                    raise ValueError("Frame boundaries are not representable at this timestamp")
        self._previous_rgb = image.copy()
        self._previous_time = t
        return frames
