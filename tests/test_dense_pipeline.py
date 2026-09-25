"""Integrated causality/cache tests with independent projected 3D motion.

The adapter below is explicitly synthetic; these tests do not claim E-RAFT
checkpoint accuracy or real-dataset benchmark performance.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from ev6d.data import PoseObservation, write_poses
from ev6d.dense_data import DenseSequence, FlowResult
from ev6d.dense_pipeline import FlowCache, load_dense_config, precompute_flow, run_dense_tracking


@pytest.fixture
def sequence(tmp_path):
    root = tmp_path/'sequence'
    root.mkdir()
    width, height = 16, 12
    K = [[18., 0., 7.5], [0., 19., 5.5], [0., 0., 1.]]
    camera = {'K': K, 'width': width, 'height': height, 'distortion': []}
    y, x = np.mgrid[:height, :width]
    depth = 1.+.02*x+.006*y+.15*np.sin(.7*x)
    np.save(root/'depth.npy', depth)
    np.save(root/'mask.npy', np.ones((height, width), dtype=bool))
    times = np.unique(np.r_[np.arange(0., .4, .005), .1, .2, .3, .4])
    np.save(root/'events.npy', np.column_stack([times, np.arange(len(times))%width,
            np.arange(len(times))%height, np.arange(len(times))%2]))
    write_poses(root/'poses.csv', [])
    metadata = {'schema_version': 1, 'units': {'time': 's', 'length': 'm', 'angle': 'rad', 'flow': 'pixel/s'},
                'pose_convention': 'T_event_object; quaternion_xyzw; omega_event',
                'events': 'events.npy', 'poses': 'poses.csv', 'start_time': 0., 'end_time': .4,
                'initial_pose': {'position': [0., 0., 1.], 'quaternion': [0., 0., 0., 1.],
                                 'source': 'manual_initialization'},
                'calibration': {'event': camera, 'rgb': camera, 'depth': camera,
                                'T_event_depth': np.eye(4).tolist(), 'T_event_rgb': np.eye(4).tolist()},
                'mask_source': 'synthetic_projected_geometry_test',
                'frames': [{'depth_t': t, 'depth': 'depth.npy', 'target_mask_event': 'mask.npy'}
                           for t in (0., .1, .2, .3, .4)]}
    (root/'dataset.json').write_text(json.dumps(metadata))
    # Invalid bytes would fail immediately if the tracker accessed evaluation GT.
    (root/'ground_truth.npz').write_bytes(b'TRACKING MUST NOT READ ME')
    return root


class ProjectedTranslation:
    def __init__(self, sequence, speed=.03, valid=True):
        self.depth = np.load(Path(sequence)/'depth.npy')
        self.speed, self.valid, self.calls = speed, valid, []

    def infer(self, old, new, start, end, width, height, source_frame, available_at):
        self.calls.append((old.copy(), new.copy(), start, end))
        # Generate pixels by projecting translated 3D points, without using J.
        yy, xx = np.mgrid[:height, :width]
        X = (xx-7.5)*self.depth/18.
        Y = (yy-5.5)*self.depth/19.
        moved_u = 18.*(X+self.speed*(end-start))/self.depth+7.5
        moved_v = 19.*Y/self.depth+5.5
        flow = np.stack([moved_u-xx, moved_v-yy])[None].astype(np.float32)
        return FlowResult(flow, start, end, source_frame, np.full((1, height, width), self.valid),
                          available_at, preprocessing={'status': 'independent_projection_test'})


def config(**extra):
    return {'velocity': {'flow_noise_std': .0001, 'noise_inflation': 1.,
                         'gate_mahalanobis_sq': None, 'max_observations': 100}, **extra}


def test_online_and_verified_cache_are_identical_and_never_read_gt(sequence, tmp_path):
    engine = ProjectedTranslation(sequence)
    summary = precompute_flow(sequence, tmp_path/'cache', config=config(), frontend=engine)
    assert summary['intervals'] == 3
    live = run_dense_tracking(sequence, tmp_path/'live', config=config(), frontend=ProjectedTranslation(sequence))
    replay = run_dense_tracking(sequence, tmp_path/'cached', config=config(), flow_cache=tmp_path/'cache')
    assert live['velocity_updates'] == replay['velocity_updates'] == 3
    assert live['mode'] == 'velocity_integration_only'
    with np.load(tmp_path/'live'/'trajectory.npz') as a, np.load(tmp_path/'cached'/'trajectory.npz') as b:
        for name in ('t', 'position', 'quaternion', 'velocity', 'v_reference', 'pose_variance'):
            np.testing.assert_allclose(a[name], b[name], atol=0, rtol=0)
        np.testing.assert_allclose(a['velocity'][a['t'] < .2], 0., atol=0)
        np.testing.assert_allclose(a['velocity'][-1], [.03, 0, 0, 0, 0, 0], atol=2e-6)
        np.testing.assert_allclose(a['v_reference'], a['v_O']+np.cross(a['omega'], a['position']))
    rows = np.load(tmp_path/'cached'/'flows.npy')
    assert rows.shape[1] == 9 and np.all(rows[:, 7] < rows[:, 0])
    assert replay['timings_s']['network'] == 0.


def test_half_open_windows_include_boundary_only_once(sequence):
    sequence = DenseSequence(sequence)
    old, new = sequence.event_pair(.1, .2)
    assert np.all((old[:, 0] >= 0.) & (old[:, 0] < .1))
    assert np.all((new[:, 0] >= .1) & (new[:, 0] < .2))
    assert np.count_nonzero(new[:, 0] == .1) == 1
    assert not np.any(new[:, 0] == .2)
    for start, end in ((.1, .1), (.1, np.nan), (.05, .2), (.3, .5)):
        with pytest.raises(ValueError):
            sequence.event_pair(start, end)


def test_source_snapshot_precedes_future_pose_and_delayed_update_replays(sequence, tmp_path):
    write_poses(sequence/'poses.csv', [PoseObservation(.15, np.array([.4, 0., 1.]),
                np.array([0., 0., 0., 1.]), 'external_estimate', .25)])
    stats = run_dense_tracking(sequence, tmp_path/'result', config=config(), frontend=ProjectedTranslation(sequence))
    detail = json.loads((tmp_path/'result'/'diagnostics.json').read_text())
    assert abs(detail[0]['source_geometry']['source_position'][0]) < 1e-12
    assert abs(detail[1]['source_geometry']['source_position'][0]) < 1e-10
    assert detail[2]['source_geometry']['source_position'][0] > .1
    assert stats['pose_updates'] == 1 and stats['pose_history']['replay_calls'] == 1
    poses = json.loads((tmp_path/'result'/'pose_diagnostics.json').read_text())
    assert poses[0]['measurement_time'] == .15 and poses[0]['processing_time'] == .25
    assert poses[0]['replayed_predictions'] > 0


def test_arrival_latency_prevents_premature_velocity_updates(sequence, tmp_path):
    stats = run_dense_tracking(sequence, tmp_path/'delayed',
                              config=config(flow_latency_s=.03), frontend=ProjectedTranslation(sequence))
    with np.load(tmp_path/'delayed'/'trajectory.npz') as data:
        assert np.all(data['velocity'][data['t'] < .23] == 0)
        assert data['t'][-1] == pytest.approx(.43)
    assert stats['flow_latency_s'] == pytest.approx([.03]*3)


def test_empty_invalid_flow_predicts_without_zero_measurement(sequence, tmp_path):
    stats = run_dense_tracking(sequence, tmp_path/'invalid', frontend=ProjectedTranslation(sequence, valid=False))
    assert stats['velocity_updates'] == 0 and stats['invalid_flow_intervals'] == 3
    with np.load(tmp_path/'invalid'/'trajectory.npz') as track:
        assert np.all(track['velocity'] == 0)
        assert track['velocity_variance'][-1, 0] > track['velocity_variance'][0, 0]
    assert np.load(tmp_path/'invalid'/'flows.npy').shape == (0, 9)


@pytest.mark.parametrize('problem', ['missing', 'stale', 'future_capture', 'future_arrival', 'invalid_depth'])
def test_unusable_source_depth_skips_updates(sequence, tmp_path, problem):
    engine = ProjectedTranslation(sequence)
    path = sequence/'dataset.json'
    meta = json.loads(path.read_text())
    if problem == 'missing':
        meta['frames'] = []
    elif problem == 'stale':
        meta['frames'] = meta['frames'][:1]
    elif problem == 'future_capture':
        meta['frames'] = [{**meta['frames'][0], 'depth_t': .4}]
    elif problem == 'future_arrival':
        for frame in meta['frames']:
            frame['depth_available_at'] = frame['depth_t']+1.
    else:
        np.save(sequence/'depth.npy', np.full((12, 16), np.nan))
    path.write_text(json.dumps(meta))
    stats = run_dense_tracking(sequence, tmp_path/'result', frontend=engine)
    assert stats['velocity_updates'] == 0
    details = json.loads((tmp_path/'result'/'diagnostics.json').read_text())
    assert all(row['source_geometry']['reason'] != 'valid' for row in details)


def test_oracle_pose_requires_explicit_flag_even_for_initialization(sequence, tmp_path):
    path = sequence/'dataset.json'
    meta = json.loads(path.read_text())
    meta['initial_pose']['source'] = 'known synthetic initial pose'
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='allow_oracle_pose'):
        run_dense_tracking(sequence, tmp_path/'blocked', frontend=ProjectedTranslation(sequence))
    assert not (tmp_path/'blocked').exists()
    stats = run_dense_tracking(sequence, tmp_path/'allowed', config={'allow_oracle_pose': True},
                               frontend=ProjectedTranslation(sequence))
    assert stats['oracle_initialization'] and stats['mode'] == 'oracle_velocity_integration_only'


def test_cache_rejects_sequence_geometry_time_and_integrity_changes(sequence, tmp_path):
    root = tmp_path/'cache'
    precompute_flow(sequence, root, frontend=ProjectedTranslation(sequence))
    path = root/'manifest.json'
    original = json.loads(path.read_text())
    for field, value, message in (
        ('sequence_fingerprint', 'wrong', 'fingerprint'),
        ('geometry', {}, 'geometry'), ('window_s', .2, 'window_s'),
        ('complete', False, 'complete'), ('interval_count', 1, 'count')):
        path.write_text(json.dumps({**original, field: value}))
        with pytest.raises(ValueError, match=message):
            FlowCache(root, DenseSequence(sequence), load_dense_config())
    bad = json.loads(json.dumps(original))
    bad['flows'][0]['t_start'] += .01
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='timestamps'):
        FlowCache(root, DenseSequence(sequence), load_dense_config())
    path.write_text(json.dumps(original))
    with (root/'flow_000000.npz').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='integrity'):
        run_dense_tracking(sequence, tmp_path/'bad_run', flow_cache=root)
    failure = json.loads((tmp_path/'bad_run'/'failure.json').read_text())
    assert failure['type'] == 'ValueError' and failure['completed_flow_intervals'] == 0


def test_cache_binds_to_actual_event_bytes(sequence, tmp_path):
    precompute_flow(sequence, tmp_path/'cache', frontend=ProjectedTranslation(sequence))
    events = np.load(sequence/'events.npy')
    events[0, 1] += 1
    np.save(sequence/'events.npy', events)
    with pytest.raises(ValueError, match='fingerprint'):
        run_dense_tracking(sequence, tmp_path/'wrong_events', flow_cache=tmp_path/'cache')


def test_preserves_output_and_requires_explicit_npz_suffix(sequence, tmp_path):
    output = tmp_path/'existing'
    output.mkdir()
    (output/'experiment.txt').write_text('preserve')
    with pytest.raises(FileExistsError):
        precompute_flow(sequence, output, frontend=ProjectedTranslation(sequence))
    assert (output/'experiment.txt').read_text() == 'preserve'
    flow = ProjectedTranslation(sequence).infer([], [], .1, .2, 16, 12, 'event_rectified', .2)
    with pytest.raises(ValueError, match='.npz'):
        flow.save(tmp_path/'no_suffix')
    with pytest.raises(ValueError, match='.npz'):
        flow.save(tmp_path/'uppercase.NPZ')
    flow.save(tmp_path/'flow.npz')
    before = (tmp_path/'flow.npz').read_bytes()
    with pytest.raises(FileExistsError):
        flow.save(tmp_path/'flow.npz')
    assert (tmp_path/'flow.npz').read_bytes() == before


def test_missing_weights_fail_instead_of_random_inference(sequence, tmp_path):
    with pytest.raises(ValueError, match='checkpoint'):
        run_dense_tracking(sequence, tmp_path/'missing')
    assert not (tmp_path/'missing').exists()
    with pytest.raises(FileNotFoundError):
        precompute_flow(sequence, tmp_path/'missing2', checkpoint=tmp_path/'missing.pth')


def test_partial_config_limit_and_no_complete_pairs(sequence, tmp_path):
    partial = config(flow_latency_s=.02)
    original = json.dumps(partial)
    merged = load_dense_config(partial)
    assert merged['eraft']['window_s'] == .1 and json.dumps(partial) == original
    for bad in ({'typo': 1}, {'velocity': {'typo': 1}}, {'flow_latency_s': -1},
                {'eraft': {'window_s': 0}}, {'history': {'max_intervals': 0}}):
        with pytest.raises(ValueError):
            load_dense_config(bad)
    summary = precompute_flow(sequence, tmp_path/'short_cache', frontend=ProjectedTranslation(sequence), max_intervals=1)
    assert summary['intervals'] == 1
    stats = run_dense_tracking(sequence, tmp_path/'short_run', flow_cache=tmp_path/'short_cache')
    assert stats['acquisition_end_s'] == .2 and stats['flow_intervals'] == 1
    stats = run_dense_tracking(sequence, tmp_path/'no_pairs', config={'eraft': {'window_s': .5}})
    assert stats['flow_intervals'] == 0 and stats['frontend'] == 'no_complete_pairs'


def test_declared_missing_pose_file_and_unordered_events_reject(sequence):
    events = np.load(sequence/'events.npy')
    events[[0, 1]] = events[[1, 0]]
    np.save(sequence/'events.npy', events)
    with pytest.raises(ValueError, match='nondecreasing'):
        DenseSequence(sequence)
    events[[0, 1]] = events[[1, 0]]
    np.save(sequence/'events.npy', events)
    (sequence/'poses.csv').unlink()
    with pytest.raises(FileNotFoundError, match='pose observations'):
        DenseSequence(sequence)


def test_delayed_tail_pose_is_drained_and_future_measurement_is_counted(sequence, tmp_path):
    write_poses(sequence/'poses.csv', [
        PoseObservation(.35, np.array([.2, 0., 1.]), np.array([0., 0., 0., 1.]), 'external', .55),
        PoseObservation(.7, np.array([.4, 0., 1.]), np.array([0., 0., 0., 1.]), 'oracle_unused_future', .8)])
    stats = run_dense_tracking(sequence, tmp_path/'tail', frontend=ProjectedTranslation(sequence))
    assert stats['output_end_s'] == .55 and stats['pose_updates'] == 1
    assert stats['pose_observations_after_acquisition_excluded'] == 1
    assert stats['oracle_pose_observations'] is False
    pose_rows = json.loads((tmp_path/'tail'/'pose_diagnostics.json').read_text())
    assert pose_rows[0]['arrival_time'] == .55 and pose_rows[0]['measurement_time'] == .35


def test_cached_runtime_reports_producer_checkpoint_not_unused_config(sequence, tmp_path):
    class TaggedProjection(ProjectedTranslation):
        def infer(self, *args, **kwargs):
            flow = super().infer(*args, **kwargs)
            flow.preprocessing.update(checkpoint='producer.pth', checkpoint_sha256='actual_test_hash',
                                      training_provenance={'kind': 'synthetic_geometry_test'})
            return flow
    precompute_flow(sequence, tmp_path/'tagged_cache', frontend=TaggedProjection(sequence))
    stats = run_dense_tracking(sequence, tmp_path/'tagged_run', flow_cache=tmp_path/'tagged_cache',
                              config={'checkpoint': 'not_used.pth'})
    assert stats['checkpoint'] == 'producer.pth'
    assert stats['checkpoint_sha256'] == 'actual_test_hash'
    assert stats['requested_checkpoint'] == 'not_used.pth'
    assert stats['training_provenance']['kind'] == 'synthetic_geometry_test'
