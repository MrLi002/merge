"""Network-independent displacement contract and source-time geometry adapter.

The legacy schema-1 event/depth/pose files are retained. Its pixel/s declaration
describes triplet output only; FlowResult has its own pixel/interval contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
from pathlib import Path

import cv2
import numpy as np

from .data import Dataset
from .geometry import register_depth
from .target import model_target_mask


@dataclass
class FlowResult:
    flow: np.ndarray
    t_start: float
    t_end: float
    source_frame: str
    valid_mask: np.ndarray
    available_at: float | None = None
    confidence: np.ndarray | None = None
    preprocessing: dict = field(default_factory=dict)

    def __post_init__(self):
        self.flow = np.asarray(self.flow, dtype=np.float32)
        self.valid_mask = np.asarray(self.valid_mask)
        if self.flow.ndim != 4 or self.flow.shape[1] != 2 or min(self.flow.shape) < 1:
            raise ValueError('Flow must have shape [B,2,H,W]')
        if self.valid_mask.shape != self.flow.shape[:1]+self.flow.shape[2:]:
            raise ValueError('Flow valid_mask must have shape [B,H,W]')
        if self.valid_mask.dtype != bool:
            raise ValueError('Flow valid_mask must be boolean')
        if not np.isfinite(self.flow).all():
            raise ValueError('Flow contains nonfinite values (invalid pixels must have finite placeholders)')
        self.t_start, self.t_end = float(self.t_start), float(self.t_end)
        self.available_at = float(self.t_end if self.available_at is None else self.available_at)
        if (not np.isfinite([self.t_start, self.t_end, self.available_at]).all()
                or self.t_end <= self.t_start or self.available_at < self.t_end):
            raise ValueError('Flow needs increasing seconds and available_at >= t_end')
        if not isinstance(self.source_frame, str) or not self.source_frame:
            raise ValueError('Flow source_frame must be explicit')
        if not isinstance(self.preprocessing, dict):
            raise ValueError('Flow preprocessing must be a dictionary')
        if self.confidence is not None:
            self.confidence = np.asarray(self.confidence, dtype=np.float32)
            if (self.confidence.shape != self.valid_mask.shape
                    or not np.isfinite(self.confidence).all() or np.any(self.confidence < 0)):
                raise ValueError('Confidence must be finite nonnegative [B,H,W]')
            if not self.preprocessing.get('confidence_source'):
                raise ValueError('A confidence map needs an explicit confidence_source')

    @property
    def dt(self):
        return self.t_end-self.t_start

    def save(self, path):
        metadata = {'format_version': 1, 't_start': self.t_start, 't_end': self.t_end,
                    'available_at': self.available_at, 'source_frame': self.source_frame,
                    'flow_unit': 'pixel/interval', 'preprocessing': self.preprocessing,
                    'has_confidence': self.confidence is not None}
        path = Path(path)
        if path.suffix != '.npz':
            raise ValueError('Flow cache paths must explicitly end in .npz')
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f'Refusing to replace flow cache: {path}')
        np.savez_compressed(path, flow=self.flow, valid_mask=self.valid_mask,
                            confidence=self.confidence if self.confidence is not None else np.empty(0),
                            metadata=np.array(json.dumps(metadata, allow_nan=False)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive['metadata'].item()))
            if meta.get('format_version') != 1 or meta.get('flow_unit') != 'pixel/interval':
                raise ValueError('Unsupported flow cache version/units')
            return cls(archive['flow'], meta['t_start'], meta['t_end'], meta['source_frame'],
                       archive['valid_mask'], meta['available_at'],
                       archive['confidence'] if meta['has_confidence'] else None,
                       meta['preprocessing'])


def rectify_dense_events(events, camera):
    """Retain fractional undistorted pixels for trilinear event voting."""
    result = np.asarray(events, dtype=np.float64).copy()
    if len(result) and np.any(camera.get('distortion', [])):
        K = np.asarray(camera['K'], dtype=float)
        result[:, 1:3] = cv2.undistortPoints(result[:, 1:3].reshape(-1, 1, 2), K,
            np.asarray(camera['distortion'], dtype=float), P=K).reshape(-1, 2)
    return result


def _binary_mask(path, shape):
    mask = np.load(path, allow_pickle=False)
    if mask.shape != shape or not np.isin(mask, [0, 1]).all():
        raise ValueError(f'Target mask must be binary {shape}: {path}')
    return mask.astype(bool)


def is_oracle(source):
    words = str(source).lower().replace('-', '_').replace(' ', '_')
    return ('oracle' in words or 'ground_truth' in words or 'gt' in re.split(r'[^a-z0-9]+', words)
            or ('known_synthetic' in words and 'initial_pose' in words))


class DenseSequence:
    """Adapter for actually inspected ev6d dataset.json schema 1.

    Depth is held from a preceding capture only with an explicit maximum age;
    there is no temporal warping or endpoint-depth substitution. Native RGB/depth
    masks are rejected: importers must supply a calibrated event-plane mask.
    """
    source_frame = 'event_rectified'

    def __init__(self, root):
        self.dataset = Dataset(root)
        ds = self.dataset
        self.root, self.meta = ds.root, ds.meta
        if 'poses' in self.meta and not (self.root/self.meta['poses']).is_file():
            raise FileNotFoundError(f"Declared pose observations are missing: {self.root/self.meta['poses']}")
        self.start, self.end, self.poses = ds.start, ds.end, ds.poses
        self.calibration = ds.calibration
        self.camera = ds.calibration['event']
        self.K = np.asarray(self.camera['K'], dtype=float)
        self.shape = (self.camera['height'], self.camera['width'])
        self.events = rectify_dense_events(ds.events, self.camera)
        self._registered = {}
        self.mask_source = self.meta.get('mask_source',
            'synthetic_label' if 'simulation' in self.meta else 'unspecified_event_plane_mask')
        for frame in ds.frames:
            mt = float(frame.get('mask_t', frame['depth_t']))
            ma = float(frame.get('mask_available_at', frame['depth_available_at']))
            if not np.isfinite([mt, ma]).all() or ma < mt:
                raise ValueError('Mask timestamps need finite seconds and arrival>=capture')
            if frame.get('target_mask_event') and frame.get('mask_frame', self.source_frame) != self.source_frame:
                raise ValueError('Masks must be reprojected to event_rectified using calibration')
            if frame.get('target_mask_rgb') or frame.get('target_mask_depth'):
                raise ValueError('Native-camera masks require explicit calibrated reprojection before import')

    def fingerprint(self):
        """Bind caches to exact event bytes, calibration and sequence metadata."""
        digest = hashlib.sha256((self.root/'dataset.json').read_bytes())
        with (self.root/self.meta['events']).open('rb') as stream:
            for chunk in iter(lambda: stream.read(2**20), b''):
                digest.update(chunk)
        return digest.hexdigest()

    def intervals(self, window_s):
        if not np.isfinite(window_s) or window_s <= 0:
            raise ValueError('window_s must be positive seconds')
        count = int(np.floor((self.end-self.start)/window_s+1e-9))
        for index in range(1, count):
            yield self.start+index*window_s, self.start+(index+1)*window_s

    def event_pair(self, t_start, t_end):
        if not np.isfinite([t_start, t_end]).all() or t_end <= t_start:
            raise ValueError('Event pair needs increasing finite timestamps in seconds')
        dt = t_end-t_start
        if t_start-dt < self.start-1e-9 or t_end > self.end+1e-9:
            raise ValueError('Event pair extends outside sequence support')
        indices = np.searchsorted(self.events[:, 0], [t_start-dt, t_start, t_end], side='left')
        return self.events[indices[0]:indices[1]], self.events[indices[1]:indices[2]]

    def source_geometry(self, t_start, position, quaternion, max_age_s=.025,
                        mask_policy='auto', roi_margin_px=3, roi_depth_margin_m=.08):
        if not np.isfinite([t_start, max_age_s]).all() or max_age_s < 0:
            raise ValueError('Geometry time must be finite and maximum age nonnegative')
        if mask_policy not in ('auto', 'supplied', 'projected_cuboid'):
            raise ValueError('mask_policy must be auto, supplied, or projected_cuboid')
        info = {'source_time_s': float(t_start), 'depth_time_s': None, 'depth_age_s': None,
                'mask_time_s': None, 'mask_age_s': None, 'mask_source': None,
                'depth_invalid_pixels': 0, 'reason': 'missing_source_depth',
                'depth_method': 'latest_prior_capture_zbuffer_no_temporal_warp'}
        depth = np.full(self.shape, np.nan)
        mask = np.zeros(self.shape, dtype=bool)
        candidates = [(i, f) for i, f in enumerate(self.dataset.frames)
                      if f['depth_t'] <= t_start+1e-10 and f['depth_available_at'] <= t_start+1e-10]
        if not candidates:
            return depth, mask, info
        index, frame = max(candidates, key=lambda item: item[1]['depth_t'])
        age = max(0., float(t_start-frame['depth_t']))
        info.update(depth_time_s=frame['depth_t'], depth_age_s=age)
        if age > max_age_s+1e-10:
            info['reason'] = 'stale_source_depth'
            return depth, mask, info
        if index not in self._registered:
            c = self.calibration
            self._registered = {index: register_depth(self.dataset.load_depth(frame), c['depth']['K'],
                self.K, c['T_event_depth'], self.shape, dist_depth=c['depth'].get('distortion'))}
        depth = self._registered[index]
        info['depth_invalid_pixels'] = int((~np.isfinite(depth) | (depth <= 0)).sum())
        model = self.meta.get('model', {})
        supplied = mask_policy == 'supplied' or (mask_policy == 'auto' and 'target_mask_event' in frame)
        if supplied:
            if 'target_mask_event' not in frame:
                info['reason'] = 'missing_source_mask'
                return depth, mask, info
            mt = float(frame.get('mask_t', frame['depth_t']))
            ma = float(frame.get('mask_available_at', frame['depth_available_at']))
            info.update(mask_time_s=mt, mask_age_s=float(t_start-mt), mask_source=self.mask_source)
            if mt > t_start+1e-10 or ma > t_start+1e-10:
                info['reason'] = 'unavailable_source_mask'
                return depth, mask, info
            if t_start-mt > max_age_s+1e-10 or abs(mt-frame['depth_t']) > 1e-6:
                info['reason'] = 'stale_or_unsynchronized_source_mask'
                return depth, mask, info
            mask = _binary_mask(self.root/frame['target_mask_event'], self.shape)
        elif model.get('type') == 'cuboid':
            mask = model_target_mask(position, quaternion, model['size'], self.K, self.shape,
                                     depth, roi_margin_px, roi_depth_margin_m)
            info.update(mask_time_s=float(t_start), mask_age_s=0.,
                        mask_source='estimated_source_pose_projected_cuboid_depth_gate')
        else:
            info['reason'] = 'missing_source_mask'
            return depth, mask, info
        mask &= np.isfinite(depth) & (depth > 0)
        info['reason'] = 'valid' if mask.any() else 'empty_target_or_invalid_depth'
        return depth, mask, info

    def inspect(self, full=True):
        invalid = 0
        if full:
            for frame in self.dataset.frames:
                depth = self.dataset.load_depth(frame)
                invalid += int((~np.isfinite(depth) | (depth <= 0)).sum())
                if 'target_mask_event' in frame:
                    _binary_mask(self.root/frame['target_mask_event'], self.shape)
        return {'path': str(self.root.resolve()), 'schema_version': 1, 'events': len(self.events),
                'shape': list(self.shape), 'frames': len(self.dataset.frames), 'poses': len(self.poses),
                'start_s': self.start, 'end_s': self.end, 'source_frame': self.source_frame,
                'purpose': self.meta.get('purpose', 'unspecified'), 'mask_source': self.mask_source,
                'pose_sources': sorted({obs.source for obs in self.poses}),
                'oracle_pose_input': any(is_oracle(obs.source) for obs in self.poses),
                'initial_pose_source': self.meta['initial_pose'].get('source', 'unspecified'),
                'invalid_depth_pixels_all_frames': invalid if full else None,
                'calibration': 'explicit_event_depth_rgb_extrinsics_and_intrinsics',
                'dense_flow_unit': 'pixel/interval', 'raw_events_time_unit': 's'}


def prepare_output(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f'Output is not empty; choose a new directory to preserve experiments: {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path
