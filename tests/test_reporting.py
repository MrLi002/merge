import json

import cv2
import numpy as np
import pytest

from ev6d.evaluation import evaluate_tracking
from ev6d.reporting import reproject_tracking


def test_reference_point_velocity_is_not_spatial_linear_velocity(tmp_path):
    dataset, result = tmp_path / 'data', tmp_path / 'result'
    dataset.mkdir()
    result.mkdir()
    t = np.array([0., 1.])
    q = np.tile([0., 0., 0., 1.], (2, 1))
    p = np.tile([0., 0., 1.], (2, 1))
    vg = np.tile([0., 0., 0., 0., 1., 0.], (2, 1))
    ve = vg.copy()
    ve[:, 3:] = 0
    np.savez(dataset / 'ground_truth.npz', t=t, position=p, quaternion=q, velocity=vg)
    np.savez(result / 'trajectory.npz', t=t, position=p, quaternion=q, velocity=ve)
    metrics = evaluate_tracking(dataset, result, False)
    assert metrics['spatial_linear_velocity_rmse_m_s'] == 0
    assert metrics['angular_velocity_rmse_rad_s'] == 1
    assert metrics['reference_point_linear_velocity_rmse_m_s'] == 1


def test_no_ground_truth_reports_no_error_metric(tmp_path):
    np.savez(tmp_path / 'trajectory.npz', t=[0., 1.], position=np.zeros((2, 3)),
             quaternion=[[0, 0, 0, 1]]*2)
    result = evaluate_tracking(tmp_path, tmp_path, False)
    assert result['status'] == 'qualitative_only'
    assert not any('rmse' in key for key in result)
    assert not (tmp_path / 'metrics.json').exists()


def test_add_requires_explicit_object_frame_model(tmp_path):
    t, q = [0., 1.], [[0., 0., 0., 1.]]*2
    np.savez(tmp_path / 'ground_truth.npz', t=t, position=np.zeros((2, 3)), quaternion=q)
    np.savez(tmp_path / 'trajectory.npz', t=t, position=np.tile([.1, 0, 0], (2, 1)), quaternion=q)
    np.save(tmp_path / 'points.npy', np.array([[0., 0., 0.], [0., 1., 0.]]))
    meta = {'evaluation_model': {'points': 'points.npy', 'units': 'm', 'metric': 'ADD'}}
    (tmp_path / 'dataset.json').write_text(json.dumps(meta))
    assert evaluate_tracking(tmp_path, tmp_path, False)['add_mean_m'] == pytest.approx(.1)
    meta['evaluation_model']['units'] = 'mm'
    (tmp_path / 'dataset.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='units'):
        evaluate_tracking(tmp_path, tmp_path, False)


def test_rgb_overlay_uses_inverse_event_rgb_extrinsic(tmp_path):
    # Event object is at x=.2. RGB camera is itself at event x=.2, so object
    # projects to RGB image center. Ignoring/inverting extrinsic incorrectly fails.
    K = [[100., 0., 50.], [0., 100., 40.], [0., 0., 1.]]
    T = np.eye(4)
    T[0, 3] = .2
    meta = {'calibration': {'rgb': {'K': K, 'width': 100, 'height': 80},
                            'T_event_rgb': T.tolist()},
            'frames': [{'rgb': 'rgb.png', 'rgb_t': .5}]}
    (tmp_path / 'dataset.json').write_text(json.dumps(meta))
    cv2.imencode('.png', np.zeros((80, 100, 3), np.uint8))[1].tofile(tmp_path / 'rgb.png')
    np.savez(tmp_path / 'trajectory.npz', t=[0., 1.],
             position=[[.2, 0., 1.]]*2, quaternion=[[0., 0., 0., 1.]]*2)
    report = reproject_tracking(tmp_path, tmp_path, max_frames=1)
    image = cv2.imdecode(np.fromfile(tmp_path / 'reprojection' / 'pose_0000.png', dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image[40, 50].any()
    assert not image[40, 70].any()
    assert report['transform'] == 'inverse(T_event_rgb)'
    with pytest.raises(FileExistsError):
        reproject_tracking(tmp_path, tmp_path)
    meta['calibration']['T_event_rgb'][0][0] = 2.
    (tmp_path / 'dataset.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError, match=r'SE\(3\)'):
        reproject_tracking(tmp_path, tmp_path, tmp_path / 'invalid_output')
