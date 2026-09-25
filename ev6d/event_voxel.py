"""E-RAFT trilinear voxels for rectified [t_s,x,y,p] events.

Official time normalization uses the first/last event, not window endpoints.
We preserve that convention, with explicit handling of empty/degenerate input.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch


@dataclass(frozen=True)
class VoxelInfo:
    event_count: int
    accepted_count: int
    rejected_coordinates: int
    usable: bool
    reason: str


def voxelize(events, width: int, height: int, num_bins: int = 15,
             normalize: bool = True, device: str = 'cpu',
             t_start: float | None = None, t_end: float | None = None):
    """Return [C,H,W] float32 tensor and diagnostics; never sort bad input.

    Coordinates may be fractional after rectification. Points outside the image
    are dropped and counted. Polarity -1/+1 and 0/1 both map to signed votes.
    Window bounds, when supplied, are enforced as [start,end).
    """
    if any(isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1
           for n in (width, height, num_bins)):
        raise ValueError('width, height and num_bins must be positive integers')
    arr = np.asarray(events, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 4 or not np.isfinite(arr).all():
        raise ValueError('events must be finite N x 4 [t_seconds,x,y,p]')
    if len(arr) and np.any(np.diff(arr[:, 0]) < 0):
        raise ValueError('Event timestamps must be nondecreasing')
    if not np.isin(arr[:, 3], [-1, 0, 1]).all():
        raise ValueError('Polarity must be -1/+1 or 0/1')
    if (t_start is None) != (t_end is None):
        raise ValueError('Supply both window endpoints')
    if t_start is not None:
        if not np.isfinite([t_start, t_end]).all() or t_end <= t_start:
            raise ValueError('Invalid event interval')
        if len(arr) and (arr[0, 0] < t_start or arr[-1, 0] >= t_end):
            raise ValueError('Events must be inside half-open [start,end) interval')
    grid = torch.zeros((num_bins, height, width), dtype=torch.float32, device=device)
    n = len(arr)
    if not n:
        return grid, VoxelInfo(0, 0, 0, False, 'empty_events')
    # Time normalization precedes coordinate filtering to match official voxels.
    span = arr[-1, 0] - arr[0, 0]
    inside = ((arr[:, 1] >= 0) & (arr[:, 1] < width) &
              (arr[:, 2] >= 0) & (arr[:, 2] < height))
    accepted = int(inside.sum())
    if span <= 0:
        return grid, VoxelInfo(n, accepted, n-accepted, False, 'degenerate_time_span')
    if not accepted:
        return grid, VoxelInfo(n, 0, n, False, 'no_in_bounds_events')
    tn = torch.as_tensor(((arr[:, 0]-arr[0, 0])/span*(num_bins-1))[inside],
                         dtype=torch.float32, device=device)
    x = torch.as_tensor(arr[inside, 1], dtype=torch.float32, device=device)
    y = torch.as_tensor(arr[inside, 2], dtype=torch.float32, device=device)
    p = torch.as_tensor(np.where(arr[inside, 3] > 0, 1., -1.),
                        dtype=torch.float32, device=device)
    x0, y0, t0 = x.long(), y.long(), tn.long()
    with torch.no_grad():
        for xx in (x0, x0+1):
            for yy in (y0, y0+1):
                for tt in (t0, t0+1):
                    good = (xx < width) & (yy < height) & (tt < num_bins)
                    weights = p*(1-(xx-x).abs())*(1-(yy-y).abs())*(1-(tt-tn).abs())
                    indices = tt*height*width + yy*width + xx
                    grid.put_(indices[good], weights[good], accumulate=True)
        if normalize:
            nonzero = grid != 0
            vals = grid[nonzero]
            if vals.numel():
                centered = vals-vals.mean()
                # Unbiased std is the official convention; define singleton std=0.
                std = vals.std(unbiased=True) if vals.numel() > 1 else vals.new_tensor(0.)
                grid[nonzero] = centered/std if std > 0 else centered
    usable = bool(torch.any(grid != 0).item())
    return grid, VoxelInfo(n, accepted, n-accepted, usable,
                           'valid' if usable else 'zero_voxel_support')
