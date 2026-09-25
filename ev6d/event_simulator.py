"""Contrast-threshold events from sampled positive linear luminance.

This is a deterministic, piecewise-linear log-intensity camera model, not an
implementation of a specific physical sensor or of the unpublished paper.
No pose, optical flow, or other ground truth enters event generation.
"""
from __future__ import annotations

import numpy as np


class EventCameraSimulator:
    """Generate [time_seconds, x, y, polarity] events from image samples.

    Each pixel retains a log-intensity reference and independent positive and
    negative thresholds. Threshold mismatch is sampled once at construction
    from N(contrast_threshold, threshold_sigma**2), independently for the two
    polarities, then clipped to 10% of the nominal threshold. This clipping
    avoids zero/negative thresholds; sigma is in log-intensity units.

    All threshold crossings advance the reference. During the refractory
    interval, crossings are consumed but their output is suppressed. Thus a
    later static frame cannot emit an artificial backlog. This explicitly
    chosen reset model is simpler than physical sensor refractory dynamics.

    Log intensity is linearly interpolated between frames. Motion that occurs
    between samples, photoreceptor bandwidth, leak, and shot noise are not
    simulated. process() rejects an interval exceeding MAX_CROSSINGS to bound
    temporary/output arrays; feed smaller time intervals in that case.
    """

    MAX_CROSSINGS = 2_000_000

    def __init__(self, initial_intensity, initial_time=0, contrast_threshold=.2,
                 threshold_sigma=0., refractory_s=0., seed=7):
        image = self._image(initial_intensity)
        values = np.asarray([initial_time, contrast_threshold, threshold_sigma, refractory_s], dtype=float)
        if not np.isfinite(values).all() or contrast_threshold <= 0 or threshold_sigma < 0 or refractory_s < 0:
            raise ValueError("Time and camera parameters must be finite; threshold > 0, sigma/refractory >= 0")
        self.timestamp = float(initial_time)
        self.contrast_threshold = float(contrast_threshold)
        self.threshold_sigma = float(threshold_sigma)
        self.refractory_s = float(refractory_s)
        self.shape = image.shape
        self.previous_log = np.log(image)
        self.reference_log = self.previous_log.copy()
        self.last_event_time = np.full(self.shape, -np.inf)
        rng = np.random.default_rng(seed)
        if threshold_sigma:
            floor = max(.1*contrast_threshold, np.finfo(float).tiny)
            self.positive_threshold = np.maximum(rng.normal(contrast_threshold, threshold_sigma, self.shape), floor)
            self.negative_threshold = np.maximum(rng.normal(contrast_threshold, threshold_sigma, self.shape), floor)
        else:
            self.positive_threshold = np.full(self.shape, contrast_threshold, dtype=float)
            self.negative_threshold = np.full(self.shape, contrast_threshold, dtype=float)

    @staticmethod
    def _image(value):
        image = np.asarray(value, dtype=float)
        if image.ndim != 2 or not image.size or not np.isfinite(image).all() or np.any(image <= 0):
            raise ValueError("Intensity must be a nonempty finite positive H x W image in linear luminance")
        return image

    def process(self, intensity, t):
        """Consume one later image; return a time-sorted float64 N x 4 array.

        Invalid inputs or excessive crossing counts leave simulator state
        unchanged. Input images may be reused by the caller after this call.
        """
        image = self._image(intensity)
        if image.shape != self.shape:
            raise ValueError("Image shape changed during event simulation")
        if not np.isscalar(t) or not np.isfinite(t) or t <= self.timestamp:
            raise ValueError("Image timestamps must be finite and strictly increasing")
        t = float(t)
        dt = t-self.timestamp
        current = np.log(image)
        delta = current-self.reference_log
        polarity = np.where(delta >= 0, 1., -1.)
        threshold = np.where(delta >= 0, self.positive_threshold, self.negative_threshold)
        # Accommodate log(exp(k*C)) roundoff at exact threshold boundaries.
        tolerance = 8*np.finfo(float).eps*np.maximum(1., np.maximum(np.abs(current), np.abs(self.reference_log)))
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            count_float = np.floor((np.abs(delta)+tolerance)/threshold)
        count_float[current == self.previous_log] = 0
        if not np.isfinite(count_float).all() or count_float.sum() > self.MAX_CROSSINGS:
            raise ValueError("Too many threshold crossings in one interval; use smaller image time intervals")
        counts = count_float.astype(np.int64)
        active = np.flatnonzero(counts)
        result = np.empty((0, 4), dtype=np.float64)
        if len(active):
            n = counts.ravel()[active]
            s = polarity.ravel()[active]
            c = threshold.ravel()[active]
            ref = self.reference_log.ravel()[active]
            old = self.previous_log.ravel()[active]
            change = current.ravel()[active]-old
            # With constant slope, all crossings in a pixel form a time lattice.
            # This also permits vectorized refractory suppression without an
            # event-by-event Python loop or a dense event volume.
            spacing = dt*c/np.abs(change)
            first = self.timestamp + dt*(ref+s*c-old)/change
            first = np.clip(first, self.timestamp, t)
            start = np.zeros(len(active), dtype=np.int64)
            stride = np.ones(len(active), dtype=np.int64)
            if self.refractory_s:
                eps_t = 8*np.finfo(float).eps*max(abs(t), abs(self.timestamp), dt, 1e-12)
                last = self.last_event_time.ravel()[active]
                # Clip before integer conversion: a huge refractory interval
                # can suppress all crossings without overflowing int64.
                skip = np.ceil((last+self.refractory_s-first-eps_t)/spacing)
                start = np.clip(skip, 0, n).astype(np.int64)
                every = np.ceil((self.refractory_s-eps_t)/spacing)
                stride = np.clip(every, 1, n+1).astype(np.int64)
            emitted = np.maximum(0, (n-1-start)//stride+1)
            total = int(emitted.sum())
            if total:
                owner = np.repeat(np.arange(len(active)), emitted)
                offsets = np.cumsum(emitted)-emitted
                ordinal = np.arange(total)-np.repeat(offsets, emitted)
                crossing = start[owner]+ordinal*stride[owner]
                times = np.clip(first[owner]+crossing*spacing[owner], self.timestamp, t)
                pixel = active[owner]
                result = np.column_stack([times, pixel % self.shape[1], pixel // self.shape[1], s[owner]])
                order = np.argsort(result[:, 0], kind="stable")
                result = result[order]
                emitted_pixels = emitted > 0
                final_crossing = start[emitted_pixels]+(emitted[emitted_pixels]-1)*stride[emitted_pixels]
                self.last_event_time.ravel()[active[emitted_pixels]] = np.clip(
                    first[emitted_pixels]+final_crossing*spacing[emitted_pixels], self.timestamp, t)
        self.reference_log += polarity*threshold*counts
        self.previous_log = current
        self.timestamp = t
        return result
