"""Offline full-image/target-ROI EPE against explicitly supplied interval GT."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .dense_data import FlowResult


def evaluate_flow_cache(prediction, ground_truth_manifest, output=None):
    prediction, manifest_path = Path(prediction), Path(ground_truth_manifest)
    truth = json.loads(manifest_path.read_text(encoding='utf-8'))
    if truth.get('kind') != 'ground_truth_flow' or truth.get('flow_unit') != 'pixel/interval':
        raise ValueError('GT manifest needs kind=ground_truth_flow and flow_unit=pixel/interval')
    if not truth.get('source') or not truth.get('source_frame') or not truth.get('records'):
        raise ValueError('GT must declare source, source_frame and nonempty records')
    predictions = [FlowResult.load(path) for path in sorted(prediction.glob('flow_*.npz'))]
    if not predictions:
        raise ValueError('No flow_*.npz predictions found')
    total, count, roi_total, roi_count = 0., 0, 0., 0
    gt_count, roi_gt_count, roi_intervals = 0, 0, 0
    records, used = [], set()
    for record in truth['records']:
        gt = FlowResult.load(manifest_path.parent / record['file'])
        matches = [i for i, pred in enumerate(predictions) if
                   abs(pred.t_start-gt.t_start) <= 1e-8 and abs(pred.t_end-gt.t_end) <= 1e-8]
        if len(matches) != 1 or matches[0] in used:
            raise ValueError('GT and predictions require unique matching start/end intervals')
        used.add(matches[0])
        pred = predictions[matches[0]]
        if pred.source_frame != gt.source_frame or gt.source_frame != truth['source_frame']:
            raise ValueError('GT and prediction source coordinate frames differ')
        if pred.flow.shape != gt.flow.shape or pred.flow.shape[0] != 1:
            raise ValueError('Expected matching single-sequence [1,2,H,W] GT/prediction shapes')
        valid = pred.valid_mask[0] & gt.valid_mask[0]
        # Do not let a missing prediction count as a successful zero-error pixel.
        valid_gt = int(gt.valid_mask.sum())
        gt_count += valid_gt
        errors = np.linalg.norm(pred.flow[0]-gt.flow[0], axis=0)
        n = int(valid.sum())
        total += float(errors[valid].sum(dtype=np.float64))
        count += n
        item = {'t_start': gt.t_start, 't_end': gt.t_end, 'valid_gt': valid_gt,
                'evaluated_pixels': n, 'missing_prediction_pixels': valid_gt-n,
                'prediction_coverage': n/valid_gt if valid_gt else None,
                'epe_px': float(errors[valid].mean()) if n else None}
        if record.get('target_mask'):
            mask = np.load(manifest_path.parent / record['target_mask'], allow_pickle=False)
            if mask.shape != valid.shape or not np.isin(mask, [0, 1]).all():
                raise ValueError('Target mask must be binary source-plane HxW')
            roi = valid & mask.astype(bool)
            nr = int(roi.sum())
            roi_total += float(errors[roi].sum(dtype=np.float64))
            roi_count += nr
            nr_gt = int((gt.valid_mask[0] & mask.astype(bool)).sum())
            roi_gt_count += nr_gt
            roi_intervals += 1
            item.update(target_pixels=nr, target_gt_pixels=nr_gt,
                        target_missing_prediction_pixels=nr_gt-nr,
                        target_prediction_coverage=nr/nr_gt if nr_gt else None,
                        target_epe_px=float(errors[roi].mean()) if nr else None)
        records.append(item)
    status = ('no_valid_ground_truth' if not gt_count else
              'no_valid_predictions' if not count else
              'evaluated_with_missing_predictions' if count < gt_count else 'evaluated')
    report = {'status': status,
              'gt_source': truth['source'], 'flow_unit': 'pixel/interval',
              'epe_px': total/count if count else None, 'valid_pixels': count,
              'total_gt_pixels': gt_count, 'missing_prediction_pixels': gt_count-count,
              'prediction_coverage': count/gt_count if gt_count else None,
              'epe_support': 'intersection_of_valid_gt_and_valid_predictions',
              'target_epe_px': roi_total/roi_count if roi_count else None,
              'target_valid_pixels': roi_count, 'target_gt_pixels': roi_gt_count,
              'target_missing_prediction_pixels': roi_gt_count-roi_count,
              'target_prediction_coverage': roi_count/roi_gt_count if roi_gt_count else None,
              'target_mask_intervals': roi_intervals,
              'intervals_without_target_mask': len(records)-roi_intervals,
              'intervals': records,
              'unused_prediction_intervals': len(predictions)-len(used)}
    if output:
        destination = Path(output)
        if destination.exists():
            raise FileExistsError(f'Refusing to replace flow metrics: {destination}')
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return report
