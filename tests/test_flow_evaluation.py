import json

import numpy as np
import pytest

from ev6d.dense_data import FlowResult
from ev6d.flow_evaluation import evaluate_flow_cache


def test_roi_epe_does_not_include_background_or_missing_predictions(tmp_path):
    predictions = tmp_path / 'predictions'
    predictions.mkdir()
    valid = np.ones((1, 2, 3), dtype=bool)
    flow = np.zeros((1, 2, 2, 3))
    FlowResult(flow, .1, .2, 'event_rectified', valid).save(tmp_path / 'gt.npz')
    predicted = flow.copy()
    predicted[0, 0] = [[3., 20., 0.], [3., 20., 0.]]
    valid[:, :, 2] = False
    FlowResult(predicted, .1, .2, 'event_rectified', valid).save(predictions / 'flow_000000.npz')
    np.save(tmp_path / 'mask.npy', [[1, 0, 0], [1, 0, 0]])
    manifest = {'kind': 'ground_truth_flow', 'flow_unit': 'pixel/interval',
                'source': 'analytic_test_only', 'source_frame': 'event_rectified',
                'records': [{'file': 'gt.npz', 'target_mask': 'mask.npy'}]}
    (tmp_path / 'gt.json').write_text(json.dumps(manifest))
    report = evaluate_flow_cache(predictions, tmp_path / 'gt.json')
    assert report['epe_px'] == pytest.approx(11.5)
    assert report['target_epe_px'] == 3
    assert report['intervals'][0]['missing_prediction_pixels'] == 2
    assert report['status'] == 'evaluated_with_missing_predictions'
    assert report['total_gt_pixels'] == 6 and report['missing_prediction_pixels'] == 2
    assert report['prediction_coverage'] == pytest.approx(2/3)
    assert report['target_prediction_coverage'] == 1
    manifest['records'].append(manifest['records'][0])
    (tmp_path / 'gt.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='unique'):
        evaluate_flow_cache(predictions, tmp_path / 'gt.json')


def test_coverage_aggregates_pixels_and_identifies_partial_roi_labels(tmp_path):
    predictions = tmp_path / 'predictions'
    predictions.mkdir()
    records = []
    for i, width in enumerate((2, 4)):
        valid_gt = np.ones((1, 2, width), dtype=bool)
        flow = np.zeros((1, 2, 2, width))
        valid_prediction = valid_gt.copy()
        if i == 0:
            valid_prediction[:] = False
            valid_prediction[0, 0, 0] = True
        FlowResult(flow, float(i), float(i+1), 'event_rectified', valid_gt).save(tmp_path / f'gt{i}.npz')
        FlowResult(flow, float(i), float(i+1), 'event_rectified', valid_prediction).save(predictions / f'flow_{i}.npz')
        records.append({'file': f'gt{i}.npz'})
    np.save(tmp_path / 'target.npy', [[1, 0], [1, 0]])
    records[0]['target_mask'] = 'target.npy'
    manifest = {'kind': 'ground_truth_flow', 'flow_unit': 'pixel/interval',
                'source': 'coverage_test_only', 'source_frame': 'event_rectified', 'records': records}
    (tmp_path / 'gt.json').write_text(json.dumps(manifest))
    report = evaluate_flow_cache(predictions, tmp_path / 'gt.json')
    assert report['total_gt_pixels'] == 12 and report['valid_pixels'] == 9
    assert report['missing_prediction_pixels'] == 3
    assert report['prediction_coverage'] == .75  # Aggregate pixels, not mean of interval fractions.
    assert report['target_gt_pixels'] == 2 and report['target_missing_prediction_pixels'] == 1
    assert report['target_prediction_coverage'] == .5
    assert report['target_mask_intervals'] == 1 and report['intervals_without_target_mask'] == 1
    assert report['epe_px'] == 0 and report['target_epe_px'] == 0
    assert report['status'] == 'evaluated_with_missing_predictions'


@pytest.mark.parametrize('has_gt', [True, False])
def test_empty_prediction_or_gt_never_reports_zero_error(tmp_path, has_gt):
    prediction = tmp_path / 'pred'
    prediction.mkdir()
    flow = np.zeros((1, 2, 2, 2))
    gt_valid = np.full((1, 2, 2), has_gt)
    FlowResult(flow, 0., 1., 'event_rectified', gt_valid).save(tmp_path / 'gt.npz')
    FlowResult(flow, 0., 1., 'event_rectified', np.zeros_like(gt_valid)).save(prediction / 'flow_0.npz')
    manifest = {'kind': 'ground_truth_flow', 'flow_unit': 'pixel/interval', 'source': 'test_only',
                'source_frame': 'event_rectified', 'records': [{'file': 'gt.npz'}]}
    (tmp_path / 'gt.json').write_text(json.dumps(manifest))
    report = evaluate_flow_cache(prediction, tmp_path / 'gt.json', tmp_path / 'metrics.json')
    assert report['epe_px'] is None and report['target_epe_px'] is None
    assert report['prediction_coverage'] == (0. if has_gt else None)
    assert report['status'] == ('no_valid_predictions' if has_gt else 'no_valid_ground_truth')
