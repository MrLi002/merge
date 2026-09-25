"""Supervised DSEC-Flow reader; filenames and timestamp rows are paired in order.

Data-format source: https://dsec.ifi.uzh.ch/data-format/
Training tree: https://github.com/uzh-rpg/bflow/blob/master/data/dsec/subsequence/base.py
This is a new supervised loader; official E-RAFT's released loader is test-only.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import hdf5plugin  # noqa: F401 -- registers official DSEC Blosc/Zstd filters
except ImportError:
    hdf5plugin = None

from .event_voxel import voxelize


def read_flow_png(path):
    encoded = np.fromfile(path, dtype=np.uint8)
    raw = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED) if encoded.size else None
    if raw is None or raw.dtype != np.uint16 or raw.ndim != 3 or raw.shape[2] != 3:
        raise ValueError(f'DSEC flow must be a 3-channel uint16 PNG: {path}')
    rgb = raw[..., ::-1]  # OpenCV BGR -> DSEC RGB; mask is blue, not red.
    if not np.isin(rgb[..., 2], [0, 1]).all():
        raise ValueError(f'DSEC flow valid channel must contain 0/1: {path}')
    valid = rgb[..., 2] == 1
    flow = (rgb[..., :2].astype(np.float32)-32768.)/128.
    flow[~valid] = 0
    return np.moveaxis(flow, -1, 0).copy(), valid


def read_flow_timestamps(path):
    # Released files are comma separated; whitespace exports are also explicit.
    content = Path(path).read_text(encoding='utf-8-sig')
    lines = [line.split('#', 1)[0].strip() for line in content.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        raise ValueError(f'Empty DSEC flow timestamp file: {path}')
    rows = []
    for line in lines:
        fields = [value.strip() for value in line.split(',')] if ',' in line else line.split()
        if len(fields) != 2 or any(re.fullmatch(r'[0-9]+', value) is None for value in fields):
            raise ValueError(f'Expected two nonnegative integer microsecond timestamp columns: {path}')
        rows.append([int(value) for value in fields])
    try:
        values = np.asarray(rows, dtype=np.int64)
    except OverflowError as error:
        raise ValueError(f'DSEC timestamps exceed int64 microsecond range: {path}') from error
    if np.any(values[:, 1] <= values[:, 0]) or np.any(np.diff(values[:, 0]) <= 0):
        raise ValueError('DSEC forward timestamps must be ordered, positive-duration intervals')
    return values


def _lower_bound(h5_times, target):
    """Fallback HDF5 binary search; reads O(log N) timestamps, not whole data."""
    lo, hi = 0, len(h5_times)
    while lo < hi:
        mid = (lo+hi)//2
        if int(h5_times[mid]) < target:
            lo = mid+1
        else:
            hi = mid
    return lo


def slice_events(h5, t_start_us, t_end_us, rectify_map):
    offset = int(h5['t_offset'][()])
    a, b = int(t_start_us)-offset, int(t_end_us)-offset
    if a < 0 or b <= a:
        raise ValueError('Requested event interval precedes recording offset')
    times = h5['events/t']
    if len(times) and b > int(times[-1])+1:
        raise ValueError('Requested event interval exceeds recording coverage')
    if 'ms_to_idx' in h5:
        mapping = h5['ms_to_idx']
        ma, mb = a//1000, (b+999)//1000
        i0 = int(mapping[ma]) if ma < len(mapping) else _lower_bound(times, a)
        i1 = int(mapping[mb]) if mb < len(mapping) else len(times)
        if not 0 <= i0 <= i1 <= len(times):
            raise ValueError('DSEC ms_to_idx contains invalid event indices')
        for millisecond, index in ((ma, i0), (mb, i1)):
            if millisecond < len(mapping):
                boundary = millisecond*1000
                if ((index < len(times) and int(times[index]) < boundary) or
                        (index > 0 and int(times[index-1]) >= boundary)):
                    raise ValueError('DSEC ms_to_idx is inconsistent with event timestamps')
        ts = np.asarray(times[i0:i1], dtype=np.int64)
        if np.any(np.diff(ts) < 0):
            raise ValueError('DSEC event timestamps are not sorted')
        ia, ib = np.searchsorted(ts, [a, b], side='left')
        begin, end = i0+int(ia), i0+int(ib)
        ts = ts[ia:ib]
    else:
        raise ValueError('DSEC events.h5 is missing ms_to_idx')
    x = np.asarray(h5['events/x'][begin:end], dtype=np.int64)
    y = np.asarray(h5['events/y'][begin:end], dtype=np.int64)
    p = np.asarray(h5['events/p'][begin:end])
    if not np.isin(p, [0, 1]).all():
        raise ValueError('Raw DSEC polarity must be 0/1')
    h, w = rectify_map.shape[:2]
    if np.any(x < 0) or np.any(y < 0) or np.any(x >= w) or np.any(y >= h):
        raise ValueError('Raw DSEC coordinates exceed rectify_map')
    xy = rectify_map[y, x]
    # Preserve fractional rectification and convert microseconds exactly once.
    return np.column_stack(((ts+offset).astype(np.float64)*1e-6, xy, p))


class DSECFlowDataset(Dataset):
    def __init__(self, root, sequences, num_bins=15, normalize=True, window_s=.1,
                 crop_size=None, augment=False, seed=1234):
        self.root = Path(root)
        if isinstance(sequences, (str, bytes)) or not sequences or len(set(sequences)) != len(sequences):
            raise ValueError('Provide a nonempty list of unique sequence names')
        self.sequences = list(sequences)
        if not isinstance(num_bins, int) or isinstance(num_bins, bool) or num_bins < 1:
            raise ValueError('num_bins must be a positive integer')
        if not np.isfinite(window_s) or window_s <= 0:
            raise ValueError('window_s must be positive seconds')
        self.num_bins, self.normalize, self.window_s = num_bins, normalize, float(window_s)
        self.crop_size = tuple(crop_size) if crop_size is not None else None
        if self.crop_size is not None and (len(self.crop_size) != 2 or any(
                not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in self.crop_size)):
            raise ValueError('crop_size must be positive [height,width]')
        self.augment, self.seed, self.epoch = augment, int(seed), 0
        self.records, self.maps, self.event_paths = [], {}, {}
        self.skipped_coverage = 0
        for name in self.sequences:
            if not isinstance(name, str) or Path(name).name != name or name in ('.', '..'):
                raise ValueError('Sequence names must be single directory names')
            seq = self.root/name
            pairs = read_flow_timestamps(seq/'flow'/'forward_timestamps.txt')
            files = sorted((seq/'flow'/'forward').glob('*.png'))
            if len(files) != len(pairs):
                raise ValueError(f'{name}: {len(files)} PNG files != {len(pairs)} timestamp rows')
            if any(not p.stem.isdigit() for p in files):
                raise ValueError('DSEC flow filenames must be numeric, zero-padded indices')
            files.sort(key=lambda file: int(file.stem))
            if len({int(file.stem) for file in files}) != len(files):
                raise ValueError('DSEC flow filenames have duplicate numeric indices')
            ev = seq/'events'/'left'
            with h5py.File(ev/'rectify_map.h5', 'r') as handle:
                mapping = np.asarray(handle['rectify_map'], dtype=np.float32)
            if (mapping.ndim != 3 or mapping.shape[2] != 2 or min(mapping.shape[:2]) < 1
                    or not np.isfinite(mapping).all()):
                raise ValueError('rectify_map must be finite [H,W,2]')
            self.maps[name] = mapping
            self.event_paths[name] = ev/'events.h5'
            try:
                with h5py.File(self.event_paths[name], 'r') as handle:
                    for key in ('events/x', 'events/y', 'events/p', 'events/t', 't_offset', 'ms_to_idx'):
                        if key not in handle:
                            raise ValueError(f'{name}: missing HDF5 dataset {key}')
                        dataset = handle[key]
                        expected_rank = 0 if key == 't_offset' else 1
                        if (dataset.ndim != expected_rank or
                                not np.issubdtype(dataset.dtype, np.integer)):
                            raise ValueError(f'{name}: {key} must be rank {expected_rank} integer data')
                    lengths = [len(handle['events/'+key]) for key in ('x', 'y', 'p', 't')]
                    if len(set(lengths)) != 1 or lengths[0] == 0:
                        raise ValueError('DSEC events arrays must have equal nonzero length')
                    offset = int(handle['t_offset'][()])
                    if offset < 0 or int(handle['events/t'][0]) < 0:
                        raise ValueError('DSEC event timestamps and t_offset must be nonnegative')
                    ms_mapping = np.asarray(handle['ms_to_idx'], dtype=np.int64)
                    if (not ms_mapping.size or np.any(ms_mapping < 0) or
                            np.any(ms_mapping > lengths[0]) or np.any(np.diff(ms_mapping) < 0)):
                        raise ValueError('DSEC ms_to_idx must be ordered valid event indices')
                    end_us = int(handle['events/t'][-1])+offset+1
            except OSError as error:
                raise RuntimeError('Cannot read DSEC HDF5; install hdf5plugin for Blosc/Zstd data') from error
            for (a, b), file in zip(pairs, files):
                dt = int(b)-int(a)
                if not np.isclose(dt*1e-6, self.window_s, rtol=0, atol=1e-3):
                    raise ValueError(f'{name}: GT interval {dt*1e-6}s differs from configured {self.window_s}s')
                if a-dt < offset or b > end_us:
                    self.skipped_coverage += 1
                    continue
                self.records.append((name, file, int(a), int(b)))
        if not self.records:
            raise ValueError('No supervised samples have complete history/current event windows')

    def __len__(self):
        return len(self.records)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def summary(self):
        return {'root': str(self.root.resolve()), 'sequences': self.sequences,
                'samples': len(self), 'skipped_incomplete_coverage': self.skipped_coverage,
                'num_bins': self.num_bins, 'window_s': self.window_s,
                'source_frame': 'rectified_left_event_camera',
                'flow_unit': 'pixel/interval', 'crop_size': self.crop_size}

    def __getitem__(self, index):
        name, file, a, b = self.records[index]
        mapping = self.maps[name]
        h, w = mapping.shape[:2]
        with h5py.File(self.event_paths[name], 'r') as handle:
            old_ev = slice_events(handle, a-(b-a), a, mapping)
            new_ev = slice_events(handle, a, b, mapping)
        old, old_info = voxelize(old_ev, w, h, self.num_bins, self.normalize)
        new, new_info = voxelize(new_ev, w, h, self.num_bins, self.normalize)
        flow, valid = read_flow_png(file)
        if flow.shape != (2, h, w):
            raise ValueError(f'{file}: flow image and event geometry differ')
        valid &= old_info.usable and new_info.usable
        flow, valid = torch.from_numpy(flow), torch.from_numpy(valid)
        rng = np.random.default_rng(self.seed + self.epoch*len(self) + index)
        if self.crop_size is not None:
            ch, cw = self.crop_size
            if ch > h or cw > w:
                raise ValueError('Training crop exceeds image size')
            y = int(rng.integers(h-ch+1)) if self.augment else (h-ch)//2
            x = int(rng.integers(w-cw+1)) if self.augment else (w-cw)//2
            old, new = old[:, y:y+ch, x:x+cw], new[:, y:y+ch, x:x+cw]
            flow, valid = flow[:, y:y+ch, x:x+cw], valid[y:y+ch, x:x+cw]
        if self.augment and rng.random() < .5:
            old, new, flow, valid = [v.flip(-1) for v in (old, new, flow, valid)]
            flow[0] *= -1
        usable = old_info.usable and new_info.usable and bool((old != 0).any() and (new != 0).any())
        if not usable:
            valid = torch.zeros_like(valid)
        return {'old': old, 'new': new, 'flow': flow, 'valid': valid,
                't_start': a*1e-6, 't_end': b*1e-6, 'sequence': name,
                'file_index': int(file.stem),
                'usable_events': usable}


def validate_split(train_sequences, val_sequences):
    for sequences in (train_sequences, val_sequences):
        if (isinstance(sequences, (str, bytes)) or not sequences or
                len(sequences) != len(set(sequences))):
            raise ValueError('Train and validation require lists of unique sequence names')
    overlap = sorted(set(train_sequences) & set(val_sequences))
    if not train_sequences or not val_sequences or overlap:
        raise ValueError(f'Train and validation require disjoint nonempty sequences; overlap={overlap}')


def create_synthetic_dsec_fixture(root, size=128, samples=2, seed=1234):
    """Write clearly labelled format fixture; it is NOT a real DSEC sequence.

    Constant +1-pixel flow supervises a translating synthetic event pattern.
    Used for loader/backprop/resume plumbing, never dataset performance claims.
    """
    if not isinstance(size, int) or not isinstance(samples, int) or samples < 1 or size < samples+11:
        raise ValueError('Fixture needs samples >= 1 and size >= samples + 11')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    (root/'SYNTHETIC_FIXTURE.json').write_text(json.dumps({
        'synthetic': True, 'purpose': 'DSEC format, training and checkpoint smoke test',
        'seed': seed, 'size': size, 'samples_per_sequence': samples}, indent=2), encoding='utf-8')
    rng = np.random.default_rng(seed)
    for name in ('synthetic_train', 'synthetic_val'):
        seq = root/name
        ev_dir, flow_dir = seq/'events'/'left', seq/'flow'/'forward'
        ev_dir.mkdir(parents=True)
        flow_dir.mkdir(parents=True)
        offset = 1000000
        # Same base events repeat every 100 ms shifted one pixel right.
        n = 2000
        base_t = np.sort(rng.integers(0, 100000, n))
        base_x = rng.integers(4, size-samples-5, n)
        base_y = rng.integers(4, size-4, n)
        base_p = rng.integers(0, 2, n)
        ts = np.concatenate([base_t+i*100000 for i in range(samples+2)])
        xx = np.concatenate([base_x+i for i in range(samples+2)])
        yy = np.tile(base_y, samples+2)
        pp = np.tile(base_p, samples+2)
        with h5py.File(ev_dir/'events.h5', 'w') as handle:
            for key, val in dict(t=ts, x=xx, y=yy, p=pp).items():
                handle.create_dataset('events/'+key, data=val)
            handle['t_offset'] = offset
            handle['ms_to_idx'] = np.searchsorted(ts, np.arange((samples+2)*100+1)*1000)
        y, x = np.mgrid[:size, :size]
        with h5py.File(ev_dir/'rectify_map.h5', 'w') as handle:
            handle['rectify_map'] = np.stack([x, y], -1).astype(np.float32)
        intervals = []
        for i in range(samples):
            a, b = offset+(i+1)*100000, offset+(i+2)*100000
            intervals.append([a, b])
            rgb = np.full((size, size, 3), 32768, dtype=np.uint16)
            rgb[..., 0] += 128
            rgb[..., 2] = 1
            rgb[:, -2:, 2] = 0
            ok, encoded = cv2.imencode('.png', rgb[..., ::-1])
            if not ok:
                raise OSError('Failed to write synthetic uint16 flow PNG')
            encoded.tofile(flow_dir/f'{i:06d}.png')
        np.savetxt(seq/'flow'/'forward_timestamps.txt', intervals, fmt='%d', delimiter=',')
    return root
