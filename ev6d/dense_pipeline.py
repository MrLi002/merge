"""Causal dense-flow tracking and immutable flow caches.

Flow estimates arrive after both half-open event windows. The last available
twist drives forward pose prediction; an arriving interval estimate affects
future predictions. Source geometry is frozen before any future observations.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

from .dense_data import DenseSequence, FlowResult, is_oracle, prepare_output
from .dense_filter import DenseVelocityConfig, DenseVelocityKF, stratified_indices
from .filters import PoseConfig, PoseUKF
from .pose_replay import PoseHistory


def load_dense_config(path=None):
    config = {'checkpoint': None,
              'eraft': {'num_bins': 15, 'iterations': 12, 'window_s': .1,
                        'normalize': True, 'device': 'cpu'},
              'velocity': asdict(DenseVelocityConfig()), 'pose': asdict(PoseConfig()),
              'max_depth_age_s': .025, 'mask_policy': 'auto', 'roi_margin_px': 3,
              'roi_depth_margin_m': .08, 'flow_latency_s': 0.,
              'allow_oracle_pose': False, 'use_pose_observations': True,
              'history': {'max_history_s': 2., 'max_intervals': 1000,
                          'max_pose_measurements': 1000}}
    if path is not None:
        user = dict(path) if isinstance(path, Mapping) else json.loads(Path(path).read_text(encoding='utf-8'))
        if not isinstance(user, dict):
            raise ValueError('Dense config must be a JSON object')
        for key, value in user.items():
            if key not in config:
                raise ValueError(f'Unknown dense config key: {key}')
            if isinstance(config[key], dict):
                if not isinstance(value, dict) or set(value)-set(config[key]):
                    raise ValueError(f'Unknown or invalid options in {key}')
                config[key].update(value)
            else:
                config[key] = value
    for key in ('max_depth_age_s', 'roi_margin_px', 'roi_depth_margin_m', 'flow_latency_s'):
        value = config[key]
        if isinstance(value, bool) or not np.isscalar(value) or not np.isfinite(value) or value < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
    if int(config['roi_margin_px']) != config['roi_margin_px']:
        raise ValueError('roi_margin_px must be an integer')
    if config['mask_policy'] not in ('auto', 'supplied', 'projected_cuboid'):
        raise ValueError('Invalid mask_policy')
    for key in ('allow_oracle_pose', 'use_pose_observations'):
        if not isinstance(config[key], bool):
            raise ValueError(f'{key} must be boolean')
    eraft = config['eraft']
    for key in ('num_bins', 'iterations'):
        if isinstance(eraft[key], bool) or not isinstance(eraft[key], int) or eraft[key] < 1:
            raise ValueError(f'eraft.{key} must be a positive integer')
    if not np.isfinite(eraft['window_s']) or eraft['window_s'] <= 0:
        raise ValueError('eraft.window_s must be positive seconds')
    if not isinstance(eraft['normalize'], bool) or not isinstance(eraft['device'], str):
        raise ValueError('Invalid eraft normalization/device')
    history = config['history']
    if not np.isfinite(history['max_history_s']) or history['max_history_s'] <= 0:
        raise ValueError('history.max_history_s must be positive seconds')
    for key in ('max_intervals', 'max_pose_measurements'):
        if isinstance(history[key], bool) or not isinstance(history[key], int) or history[key] < 1:
            raise ValueError(f'history.{key} must be a positive integer')
    DenseVelocityConfig(**config['velocity'])
    PoseConfig(**config['pose'])
    return config


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _intervals(sequence, config, maximum):
    if maximum is not None and (isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1):
        raise ValueError('max_intervals must be a positive integer or None')
    result = list(sequence.intervals(config['eraft']['window_s']))
    return result if maximum is None else result[:maximum]


def _geometry_contract(sequence):
    return {'shape': list(sequence.shape), 'K': sequence.K.tolist(),
            'source_frame': sequence.source_frame, 'coordinates': 'rectified_zero_distortion'}


def _flow_provenance(flow):
    fields = ('checkpoint', 'checkpoint_sha256', 'training_provenance',
              'checkpoint_metadata_verified', 'upstream_commit', 'num_bins',
              'window_s', 'normalize', 'voxel_convention', 'context_input', 'iterations')
    return {key: flow.preprocessing[key] for key in fields if key in flow.preprocessing}


def _validate_flow(flow, sequence, interval, fingerprint):
    if not isinstance(flow, FlowResult):
        raise TypeError('Frontend/cache must return FlowResult')
    if flow.flow.shape != (1, 2, *sequence.shape) or flow.source_frame != sequence.source_frame:
        raise ValueError('Flow source frame or image geometry differs from the sequence')
    if not np.allclose([flow.t_start, flow.t_end], interval, rtol=0, atol=1e-10):
        raise ValueError('Flow timestamps differ from the scheduled event windows')
    existing = flow.preprocessing.get('sequence_fingerprint')
    if existing is not None and existing != fingerprint:
        raise ValueError('Flow belongs to a different sequence')
    return flow


def _make_frontend(checkpoint, config, frontend):
    if frontend is not None:
        if checkpoint is not None:
            raise ValueError('Pass either a frontend or a checkpoint, not both')
        return frontend
    from .eraft import ERAFTFrontend
    weights = checkpoint or config['checkpoint']
    if weights is None:
        raise ValueError('Specify an E-RAFT checkpoint or a verified flow cache; random inference is forbidden')
    return ERAFTFrontend(weights, config['eraft'])


class FlowCache:
    """Validate identity, geometry, timestamps and per-file integrity on read."""

    def __init__(self, path, sequence, config):
        self.root = Path(path)
        self.sequence = sequence
        self.manifest = json.loads((self.root/'manifest.json').read_text(encoding='utf-8'))
        meta = self.manifest
        if meta.get('format_version') != 1 or meta.get('kind') != 'ev6d_dense_flow_cache':
            raise ValueError('Unsupported dense flow cache manifest')
        if meta.get('complete') is not True:
            raise ValueError('Flow cache precomputation did not complete')
        self.fingerprint = sequence.fingerprint()
        if meta.get('sequence_fingerprint') != self.fingerprint:
            raise ValueError('Flow cache sequence fingerprint mismatch')
        if meta.get('geometry') != _geometry_contract(sequence):
            raise ValueError('Flow cache source geometry mismatch')
        if meta.get('window_s') != config['eraft']['window_s']:
            raise ValueError('Flow cache window_s differs from tracking configuration')
        self.entries = meta.get('flows')
        if not isinstance(self.entries, list):
            raise ValueError('Flow cache needs an ordered flows list')
        expected = _intervals(sequence, config, meta.get('requested_max_intervals'))
        if len(self.entries) != len(expected) or meta.get('interval_count') != len(expected):
            raise ValueError('Flow cache interval count does not match its declared precomputation range')
        seen = set()
        last_arrival = -np.inf
        for entry, interval in zip(self.entries, expected):
            name = entry.get('file', '')
            if not name or Path(name).name != name or name in seen or not name.endswith('.npz'):
                raise ValueError('Cache filenames must be unique local .npz basenames')
            seen.add(name)
            stamps = [entry['t_start'], entry['t_end'], entry['available_at']]
            if (not np.isfinite(stamps).all() or stamps[2] < stamps[1]
                    or not np.allclose(stamps[:2], interval, rtol=0, atol=1e-10)):
                raise ValueError('Flow cache timestamps are not the expected ordered intervals')
            if stamps[2] < last_arrival:
                raise ValueError('Out-of-order flow arrivals are unsupported; use ordered fixed-window inference')
            last_arrival = stamps[2]
            if not (self.root/name).is_file():
                raise FileNotFoundError(self.root/name)

    def load(self, index):
        entry = self.entries[index]
        path = self.root/entry['file']
        if _sha256(path) != entry.get('sha256'):
            raise ValueError(f'Flow cache file integrity mismatch: {path.name}')
        flow = FlowResult.load(path)
        _validate_flow(flow, self.sequence, (entry['t_start'], entry['t_end']), self.fingerprint)
        if flow.available_at != entry['available_at']:
            raise ValueError('Flow cache arrival time differs from manifest')
        if flow.preprocessing.get('sequence_fingerprint') != self.fingerprint:
            raise ValueError('Flow cache item lacks the matching sequence fingerprint')
        if _flow_provenance(flow) != self.manifest.get('frontend_provenance'):
            raise ValueError('Flow cache model provenance differs between item and manifest')
        return flow


def precompute_flow(dataset, output, checkpoint=None, config=None, frontend=None, max_intervals=None):
    """Save immutable full-resolution displacement fields and their provenance.

    ``frontend`` is a callable adapter injection for other flow networks or
    explicit synthetic tests; its actual class name is recorded in the cache.
    """
    start = time.perf_counter()
    config = load_dense_config(config)
    sequence = DenseSequence(dataset)
    intervals = _intervals(sequence, config, max_intervals)
    fingerprint = sequence.fingerprint()
    engine = _make_frontend(checkpoint, config, frontend) if intervals else None
    output = prepare_output(output)
    manifest = {'format_version': 1, 'kind': 'ev6d_dense_flow_cache',
                'sequence_fingerprint': fingerprint, 'geometry': _geometry_contract(sequence),
                'sequence_path': str(sequence.root.resolve()),
                'dataset_purpose': sequence.meta.get('purpose', 'unspecified'),
                'window_s': config['eraft']['window_s'], 'flow_unit': 'pixel/interval',
                'requested_max_intervals': max_intervals, 'interval_count': len(intervals),
                'frontend': type(engine).__name__ if engine is not None else 'no_complete_pairs',
                'frontend_provenance': None,
                'config': config, 'flows': [], 'complete': False}
    if engine is not None and hasattr(engine, 'reset'):
        engine.reset()
    timings = {'voxel': 0., 'network': 0., 'cache_write': 0.}
    for index, (t_start, t_end) in enumerate(intervals):
        history, current = sequence.event_pair(t_start, t_end)
        flow = engine.infer(history, current, t_start, t_end, sequence.shape[1], sequence.shape[0],
                            source_frame=sequence.source_frame,
                            available_at=t_end+config['flow_latency_s'])
        _validate_flow(flow, sequence, (t_start, t_end), fingerprint)
        provenance = _flow_provenance(flow)
        if index == 0:
            manifest['frontend_provenance'] = provenance
        elif provenance != manifest['frontend_provenance']:
            raise ValueError('Precomputation changed model/checkpoint provenance between intervals')
        flow.preprocessing['sequence_fingerprint'] = fingerprint
        name = f'flow_{index:06d}.npz'
        tick = time.perf_counter()
        flow.save(output/name)
        digest = _sha256(output/name)
        timings['cache_write'] += time.perf_counter()-tick
        for key in ('voxel', 'network'):
            timings[key] += float(flow.preprocessing.get('timing_s', {}).get(key, 0.))
        manifest['flows'].append({'file': name, 'sha256': digest, 't_start': flow.t_start,
                                 't_end': flow.t_end, 'available_at': flow.available_at,
                                 'valid_pixels': int(flow.valid_mask.sum())})
        _json(output/'manifest.json', manifest)
    manifest.update(complete=True, processing_s=time.perf_counter()-start, timings_s=timings)
    _json(output/'manifest.json', manifest)
    return {'output': str(output.resolve()), 'intervals': len(intervals),
            'sequence_fingerprint': fingerprint, 'processing_s': manifest['processing_s'],
            'timings_s': timings, 'frontend': manifest['frontend']}


def run_dense_tracking(dataset, output, checkpoint=None, config=None, flow_cache=None,
                       frontend=None, max_intervals=None):
    """Track from live adapter inference or precomputed, identity-checked flow.

    This offline replay obeys acquisition/arrival timestamps. Measured processing
    times are reported separately, not silently folded into sensor timestamps.
    Source depth must already be available at the flow source instant. The
    endpoint velocity is a held estimate of interval-average motion, so there
    is one-window estimator lag during changes in velocity.
    """
    started = time.perf_counter()
    config = load_dense_config(config)
    sequence = DenseSequence(dataset)
    initial = sequence.meta['initial_pose']
    initial_source = initial.get('source', 'unspecified')
    oracle_initial = is_oracle(initial_source)
    poses = sequence.poses if config['use_pose_observations'] else []
    if oracle_initial and not config['allow_oracle_pose']:
        raise ValueError('GT/oracle pose input or known synthetic initialization requires allow_oracle_pose=True')
    if flow_cache is not None and (checkpoint is not None or frontend is not None):
        raise ValueError('Pass flow_cache or an inference frontend/checkpoint, not both')
    cache = FlowCache(flow_cache, sequence, config) if flow_cache is not None else None
    intervals = _intervals(sequence, config, max_intervals)
    if cache is not None:
        intervals = intervals[:len(cache.entries)]
    fingerprint = cache.fingerprint if cache is not None else sequence.fingerprint()
    arrivals = ([entry['available_at'] for entry in cache.entries[:len(intervals)]] if cache is not None
                else [end+config['flow_latency_s'] for _, end in intervals])
    full_count = len(list(sequence.intervals(config['eraft']['window_s'])))
    limited = len(intervals) < full_count
    acquisition_end = intervals[-1][1] if intervals and limited else sequence.end
    future_pose_count = sum(obs.t > acquisition_end for obs in poses)
    poses = [obs for obs in poses if obs.t <= acquisition_end]
    # Drain delayed poses measured inside the selected acquisition range, even
    # when their delivery follows its last flow/event. History bounds still apply.
    horizon = max([acquisition_end, *arrivals, *(obs.available_at for obs in poses)])
    oracle_pose = any(is_oracle(obs.source) for obs in poses)
    if oracle_pose and not config['allow_oracle_pose']:
        raise ValueError('GT/oracle pose input requires allow_oracle_pose=True')
    engine = _make_frontend(checkpoint, config, frontend) if intervals and cache is None else None
    if engine is not None and hasattr(engine, 'reset'):
        engine.reset()
    output = prepare_output(output)
    velocity = DenseVelocityKF(DenseVelocityConfig(**config['velocity']))
    velocity.predict_to(sequence.start)
    pose = PoseHistory(PoseUKF(initial['position'], initial['quaternion'], sequence.start,
                               PoseConfig(**config['pose'])), **config['history'])
    mode = 'pose_fusion' if poses else 'velocity_integration_only'
    if oracle_initial or oracle_pose:
        mode = 'oracle_'+mode
    elif any('simulat' in obs.source.lower() or 'synthetic' in obs.source.lower() for obs in poses):
        mode = 'synthetic_'+mode
    source_schedule, arrival_schedule, pose_schedule = {}, {}, {}
    for index, ((start, _), arrival) in enumerate(zip(intervals, arrivals)):
        source_schedule.setdefault(start, []).append(index)
        arrival_schedule.setdefault(arrival, []).append(index)
    for index, obs in enumerate(poses):
        pose_schedule.setdefault(max(sequence.start, obs.available_at), []).append((index, obs))
    timeline = sorted({sequence.start, horizon, *source_schedule, *arrival_schedule, *pose_schedule})
    source_geometry = {}
    records, diagnostics, pose_diagnostics, old_flows, dense_flows, batch_times = [], [], [], [], [], []
    timings = {'voxel': 0., 'network': 0., 'cache_read': 0., 'observation': 0., 'velocity_kf': 0., 'pose_ukf': 0.}
    cached_timings = {'voxel': 0., 'network': 0.}
    frontend_provenance = cache.manifest.get('frontend_provenance') if cache is not None else None
    counters = {'flow_intervals': 0, 'velocity_updates': 0, 'flow_measurements': 0,
                'invalid_flow_intervals': 0, 'pose_updates': 0, 'pose_rejected': 0}
    try:
        for timestamp in timeline:
            tick_batch = time.perf_counter()
            tick = time.perf_counter()
            # Integrate each physical duration once using the previously known input.
            pose.predict_to(timestamp, velocity.x.copy(), velocity.P.copy())
            timings['pose_ukf'] += time.perf_counter()-tick
            tick = time.perf_counter()
            velocity.predict_to(timestamp)
            timings['velocity_kf'] += time.perf_counter()-tick
            state = 'prediction_only'
            for index in arrival_schedule.get(timestamp, []):
                start, end = intervals[index]
                tick = time.perf_counter()
                if cache is not None:
                    flow = cache.load(index)
                    timings['cache_read'] += time.perf_counter()-tick
                else:
                    history, current = sequence.event_pair(start, end)
                    flow = engine.infer(history, current, start, end, sequence.shape[1], sequence.shape[0],
                                        source_frame=sequence.source_frame, available_at=timestamp)
                _validate_flow(flow, sequence, (start, end), fingerprint)
                provenance = _flow_provenance(flow)
                if frontend_provenance is None:
                    frontend_provenance = provenance
                elif provenance != frontend_provenance:
                    raise ValueError('Frontend changed model/checkpoint provenance between intervals')
                if flow.available_at != timestamp:
                    raise ValueError('Frontend available_at differs from the scheduled arrival time')
                timing_target = cached_timings if cache is not None else timings
                for key in ('voxel', 'network'):
                    timing_target[key] += float(flow.preprocessing.get('timing_s', {}).get(key, 0.))
                depth, mask, info = source_geometry.pop(index)
                tick = time.perf_counter()
                selected = mask & flow.valid_mask[0] & np.isfinite(depth) & (depth > 0)
                rows, cols = np.nonzero(selected)
                uv = np.column_stack([cols, rows]).astype(float)
                z = depth[rows, cols]
                displacement = flow.flow[0, :, rows, cols]
                timings['observation'] += time.perf_counter()-tick
                tick = time.perf_counter()
                detail = velocity.update(uv, z, displacement, sequence.K, flow.dt)
                timings['velocity_kf'] += time.perf_counter()-tick
                counters['flow_intervals'] += 1
                counters['velocity_updates'] += int(detail['accepted'])
                counters['flow_measurements'] += int(detail['measurements'])
                counters['invalid_flow_intervals'] += int(not flow.valid_mask.any())
                state = 'flow_updated' if detail['accepted'] else 'flow_rejected'
                diagnostics.append({'t': timestamp, 't_start': start, 't_end': end,
                                    'available_at': flow.available_at, 'flow_latency_s': timestamp-end,
                                    'flow_status': flow.preprocessing.get('status', 'adapter_supplied'),
                                    'valid_flow_pixels': int(flow.valid_mask.sum()),
                                    'confidence_used': False, 'source_geometry': info, **detail})
                sampled = stratified_indices(uv, config['velocity']['max_observations'])
                for pixel, d, f in zip(uv[sampled], z[sampled], displacement[sampled]):
                    old_flows.append([timestamp, *pixel, *(f/flow.dt), d, 1., start, start-flow.dt])
                    dense_flows.append([start, end, timestamp, *pixel, *f, d])
            # At equal timestamps, arrived measurements precede the new source
            # snapshot; arrivals at later times can never change that snapshot.
            for index, obs in pose_schedule.get(timestamp, []):
                tick = time.perf_counter()
                detail = pose.update(obs.position, obs.quaternion, obs.t, obs.available_at,
                                     measurement_id=f'pose_{index}')
                timings['pose_ukf'] += time.perf_counter()-tick
                counters['pose_updates' if detail['accepted'] else 'pose_rejected'] += 1
                pose_diagnostics.append({'source': obs.source, **detail})
                if detail['accepted']:
                    state += '+pose_corrected'
            for index in source_schedule.get(timestamp, []):
                tick = time.perf_counter()
                depth, mask, info = sequence.source_geometry(
                    timestamp, pose.position.copy(), pose.quaternion.copy(),
                    max_age_s=config['max_depth_age_s'], mask_policy=config['mask_policy'],
                    roi_margin_px=config['roi_margin_px'], roi_depth_margin_m=config['roi_depth_margin_m'])
                info.update(source_position=pose.position.tolist(), source_quaternion=pose.quaternion.tolist())
                source_geometry[index] = (depth.copy(), mask.copy(), info)
                timings['observation'] += time.perf_counter()-tick
            if not np.isfinite(np.r_[pose.position, pose.quaternion, velocity.x, pose.P.ravel(), velocity.P.ravel()]).all():
                raise FloatingPointError('Non-finite pose/velocity state or covariance')
            min_eigen = min(np.linalg.eigvalsh(pose.P)[0], np.linalg.eigvalsh(velocity.P)[0])
            if min_eigen < -1e-9:
                raise FloatingPointError('Pose/velocity covariance is not positive semidefinite')
            records.append({'t': timestamp, 'position': pose.position.copy(),
                            'quaternion': pose.quaternion.copy(), 'velocity': velocity.x.copy(),
                            'v_reference': velocity.x[:3]+np.cross(velocity.x[3:], pose.position),
                            'pose_variance': np.diag(pose.P).copy(),
                            'velocity_variance': np.diag(velocity.P).copy(), 'state': state})
            batch_times.append(time.perf_counter()-tick_batch)
    except Exception as error:
        _json(output/'failure.json', {'status': 'failed', 'type': type(error).__name__,
                                     'error': str(error), 'processing_time_s': locals().get('timestamp'),
                                     'completed_flow_intervals': counters['flow_intervals']})
        raise
    arrays = {key: np.asarray([row[key] for row in records]) for key in records[0]}
    arrays.update(v_O=arrays['velocity'][:, :3], omega=arrays['velocity'][:, 3:], mode=np.array(mode))
    np.savez_compressed(output/'trajectory.npz', **arrays)
    with (output/'trajectory.csv').open('w', encoding='utf-8', newline='') as stream:
        import csv
        writer = csv.writer(stream)
        writer.writerow(['t', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw',
                         'vox', 'voy', 'voz', 'wx', 'wy', 'wz', 'vref_x', 'vref_y', 'vref_z', 'state'])
        for row in records:
            writer.writerow([row['t'], *row['position'], *row['quaternion'], *row['velocity'],
                             *row['v_reference'], row['state']])
    np.save(output/'flows.npy', np.asarray(old_flows, dtype=float).reshape(-1, 9))
    samples = np.asarray(dense_flows, dtype=float).reshape(-1, 8)
    np.savez_compressed(output/'sampled_dense_flows.npz', t_start=samples[:, 0], t_end=samples[:, 1],
                        available_at=samples[:, 2], uv=samples[:, 3:5], flow=samples[:, 5:7],
                        depth=samples[:, 7], flow_unit=np.array('pixel/interval'))
    elapsed = time.perf_counter()-started
    statistics = {'variant': 'dense_full_2d', 'mode': mode, **counters,
                  'dataset_purpose': sequence.meta.get('purpose', 'unspecified'),
                  'sequence_fingerprint': fingerprint, 'input_events': len(sequence.events),
                  'source_frame': sequence.source_frame, 'mask_source': sequence.mask_source,
                  'pose_sources': sorted({obs.source for obs in poses}),
                  'initial_pose_source': initial_source, 'oracle_initialization': oracle_initial,
                  'oracle_pose_observations': oracle_pose,
                  'pose_observations_after_acquisition_excluded': int(future_pose_count),
                  'pose_arrival_policy': 'drain all poses measured no later than acquisition_end; reject outside bounded replay history',
                  'frontend': cache.manifest['frontend'] if cache is not None else (
                      type(engine).__name__ if engine is not None else 'no_complete_pairs'),
                  'flow_cache': str(Path(flow_cache).resolve()) if cache is not None else None,
                  'frontend_provenance': frontend_provenance,
                  'checkpoint': (frontend_provenance or {}).get('checkpoint'),
                  'checkpoint_sha256': (frontend_provenance or {}).get('checkpoint_sha256'),
                  'training_provenance': (frontend_provenance or {}).get('training_provenance'),
                  'requested_checkpoint': str(checkpoint or config['checkpoint']) if checkpoint or config['checkpoint'] else None,
                  'processing_s': elapsed, 'timings_s': timings,
                  'cached_precompute_timings_s': cached_timings if cache is not None else None,
                  'batch_compute_ms_p50': float(np.percentile(batch_times, 50)*1000),
                  'batch_compute_ms_p95': float(np.percentile(batch_times, 95)*1000),
                  'acquisition_start_s': sequence.start, 'acquisition_end_s': acquisition_end,
                  'output_end_s': horizon, 'window_s': config['eraft']['window_s'],
                  'nominal_flow_update_hz': 1/config['eraft']['window_s'],
                  'flow_processing_throughput_hz': len(intervals)/elapsed,
                  'event_collection_wait_s': config['eraft']['window_s'],
                  'flow_latency_s': [float(a-e) for a, (_, e) in zip(arrivals, intervals)],
                  'pose_history': pose.diagnostics(), 'python': platform.python_version(),
                  'velocity_semantics': 'camera-relative spatial twist [v_O,omega]; v_reference=v_O+omega cross object-origin position',
                  'prediction_policy': 'hold last available interval-average twist; new flow affects future pose intervals only',
                  'timing_note': 'Offline causal replay with declared availability. Measured processing time is separate; no live realtime guarantee.',
                  'confidence_note': 'No native calibrated E-RAFT confidence; optional adapter maps are not used by this KF.',
                  'flows_npy_units': 'legacy pixel/s view of sampled displacements divided by dt once for export only'}
    _json(output/'runtime.json', statistics)
    _json(output/'config.json', config)
    _json(output/'diagnostics.json', diagnostics)
    _json(output/'pose_diagnostics.json', pose_diagnostics)
    return statistics
