"""Offline reports and calibrated RGB pose projection; no fabricated ground truth."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def qualitative_report(dataset, result, make_plot=True):
    from .evaluation import _validate_trajectory
    dataset, result = Path(dataset), Path(result)
    with np.load(result / 'trajectory.npz', allow_pickle=False) as data:
        t, p, q = _validate_trajectory(data, 'estimate')
    report = {'status': 'qualitative_only', 'ground_truth_available': False,
              'samples': len(t), 'start_s': float(t[0]), 'end_s': float(t[-1]),
              'note': 'No error metrics: ground_truth.npz is absent.'}
    (result / 'qualitative_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if make_plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
        ax.plot(t, p, label=['x', 'y', 'z'])
        ax.set(xlabel='Time (s)', ylabel='Position (m)', title='Estimated position; no ground truth')
        ax.legend()
        ax.grid(alpha=.2)
        fig.savefig(result / 'trajectory_qualitative.png', dpi=150)
        plt.close(fig)
    return report


def reproject_tracking(dataset, result, output=None, max_frames=12):
    """Project object coordinate axes / centered box into calibrated raw RGB.

    Interpolates emitted poses for offline visualization only, not filter input.
    A size-only YCB model produces a labelled bounding box, never a mesh overlay.
    """
    from .evaluation import _validate_trajectory
    from .dense_data import prepare_output
    dataset, result = Path(dataset), Path(result)
    if isinstance(max_frames, bool) or int(max_frames) != max_frames or max_frames < 1:
        raise ValueError('max_frames must be a positive integer')
    metadata = json.loads((dataset / 'dataset.json').read_text(encoding='utf-8'))
    calibration = metadata['calibration']
    camera = calibration['rgb']
    K = np.asarray(camera['K'], dtype=float)
    from .geometry import _intrinsics
    _intrinsics(K)
    if any(isinstance(camera[k], bool) or not isinstance(camera[k], int) or camera[k] < 1
           for k in ('width', 'height')):
        raise ValueError('RGB image dimensions must be positive integers')
    T = np.asarray(calibration['T_event_rgb'], dtype=float)
    if (T.shape != (4, 4) or not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1])
            or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-6)
            or np.linalg.det(T[:3, :3]) <= 0):
        raise ValueError('Expected rigid calibrated T_event_rgb in SE(3)')
    rgb_from_event = np.linalg.inv(T)
    distortion = np.asarray(camera.get('distortion', []), dtype=float)
    if distortion.ndim != 1 or distortion.size not in (0, 4, 5, 8, 12, 14) or not np.isfinite(distortion).all():
        raise ValueError('Unsupported RGB pinhole distortion')
    with np.load(result / 'trajectory.npz', allow_pickle=False) as data:
        t, p, q = _validate_trajectory(data, 'estimate')
    def frame_time(frame):
        timestamp = frame.get('rgb_t', frame.get('depth_t'))
        if timestamp is None or not np.isfinite(float(timestamp)):
            raise ValueError('RGB frame requires finite rgb_t (or depth_t fallback)')
        return float(timestamp)

    frames = [f for f in metadata.get('frames', []) if 'rgb' in f and
              t[0] <= frame_time(f) <= t[-1]]
    if not frames:
        raise ValueError('No RGB frames overlap the estimated trajectory')
    indices = np.unique(np.linspace(0, len(frames)-1, min(max_frames, len(frames))).astype(int))
    destination = prepare_output(output or result / 'reprojection')
    size = np.asarray(metadata.get('model', {}).get('size', []), dtype=float)
    has_box = size.shape == (3,) and np.isfinite(size).all() and (size > 0).all()
    scale = float(size.max())*.6 if has_box else .05
    axis_points = np.vstack([np.zeros(3), np.eye(3)*scale])
    vertices = np.asarray(list(itertools.product([-1., 1.], repeat=3)))*size/2 if has_box else None
    edges = [(a, b) for a in range(8) for b in range(a+1, 8)
             if bin(a ^ b).count('1') == 1]
    rotations = Slerp(t, Rotation.from_quat(q))
    records = []
    for idx in indices:
        frame = frames[idx]
        timestamp = frame_time(frame)
        image = cv2.imdecode(np.fromfile(dataset / frame['rgb'], dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (camera['height'], camera['width']):
            raise ValueError(f'RGB size/read failure: {frame["rgb"]}')
        translation = np.array([np.interp(timestamp, t, p[:, j]) for j in range(3)])
        orientation = rotations(timestamp)

        def project(points):
            event_points = orientation.apply(points)+translation
            rgb_points = event_points @ rgb_from_event[:3, :3].T+rgb_from_event[:3, 3]
            uv, _ = cv2.projectPoints(rgb_points, np.zeros(3), np.zeros(3), K, distortion)
            return uv.reshape(-1, 2), rgb_points[:, 2] > 1e-6

        def draw(points, pairs, color, thickness):
            uv, visible = project(points)
            for a, b in pairs:
                if visible[a] and visible[b] and np.isfinite(uv[[a, b]]).all():
                    # Bound conversion before OpenCV's signed 32-bit coordinates.
                    xy = np.clip(uv[[a, b]], -1e6, 1e6).astype(int)
                    cv2.line(image, tuple(xy[0]), tuple(xy[1]), color, thickness, cv2.LINE_AA)
        if has_box:
            draw(vertices, edges, (0, 210, 255), 1)
        for axis, color in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)], 1):
            draw(axis_points, [(0, axis)], color, 2)
        label = f't={timestamp:.3f}s estimate axes' + (' + model bounding box' if has_box else '')
        cv2.putText(image, label, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
        filename = f'pose_{len(records):04d}.png'
        encoded_ok, encoded = cv2.imencode('.png', image)
        if not encoded_ok:
            raise OSError('Failed to write projected RGB image')
        encoded.tofile(destination / filename)
        records.append({'t': timestamp, 'rgb': frame['rgb'], 'overlay': filename})
    report = {'coordinate_frame': 'raw_rgb_with_declared_distortion',
              'pose_frame': 'T_event_object', 'transform': 'inverse(T_event_rgb)',
              'visualization': 'estimated_axes_and_centered_model_box' if has_box else 'estimated_axes',
              'pose_sampling': 'offline_position_interpolation_quaternion_slerp',
              'records': records, 'output': str(destination.resolve())}
    (destination / 'manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report
