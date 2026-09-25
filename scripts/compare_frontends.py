"""Run frozen/fine-tuned E-RAFT and legacy triplet under the same input contract.

This script requires user-supplied checkpoints; it never downloads or trains.
Names describe the user's intended experiment, not a performance conclusion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from ev6d.dense_data import DenseSequence, prepare_output
from ev6d.dense_pipeline import load_dense_config, precompute_flow, run_dense_tracking
from ev6d.evaluation import evaluate_tracking
from ev6d.pipeline import run_tracking


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frozen-checkpoint', type=Path, required=True)
    parser.add_argument('--finetuned-checkpoint', type=Path, required=True)
    parser.add_argument('--dense-config', type=Path, default=Path('configs/dense_eraft.json'))
    parser.add_argument('--triplet-config', type=Path)
    parser.add_argument('--allow-oracle-pose', action='store_true')
    args = parser.parse_args()
    for checkpoint in (args.frozen_checkpoint, args.finetuned_checkpoint):
        if not checkpoint.is_file():
            parser.error(f'Missing checkpoint: {checkpoint}')
    config = load_dense_config(args.dense_config)
    if args.allow_oracle_pose:
        config['allow_oracle_pose'] = True
    seq = DenseSequence(args.dataset)
    output = prepare_output(args.output)
    summary = {'dataset': seq.inspect(), 'dataset_fingerprint': seq.fingerprint(),
               'note': 'Common sequence/depth/masks/pose files. Legacy triplet rejects delayed poses; '
                       'dense backend replays them. Treat any delayed-pose comparison as a backend confound.',
               'results': {}}
    for name, checkpoint in [('frozen_eraft', args.frozen_checkpoint),
                             ('finetuned_eraft', args.finetuned_checkpoint)]:
        cache = output / (name+'_flow')
        precompute_flow(args.dataset, cache, checkpoint=checkpoint, config=config)
        result = output / name
        runtime = run_dense_tracking(args.dataset, result, flow_cache=cache, config=config)
        summary['results'][name] = {'checkpoint': str(checkpoint.resolve()),
            'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'runtime': runtime, 'metrics': evaluate_tracking(args.dataset, result)}
    result = output / 'triplet'
    summary['results']['triplet'] = {'runtime': run_tracking(args.dataset, result, args.triplet_config),
                                   'metrics': evaluate_tracking(args.dataset, result)}
    # Report a second set on exactly the same output timestamps. Raw per-backend
    # cadence differs (triplet ticks versus dense windows); those RMSEs alone
    # are not a fair numerical comparison.
    trajectories = {}
    for name in summary['results']:
        with np.load(output/name/'trajectory.npz', allow_pickle=False) as archive:
            trajectories[name] = dict(archive)
    candidates = trajectories['frozen_eraft']['t']
    lower = max(track['t'][0] for track in trajectories.values())
    upper = min(track['t'][-1] for track in trajectories.values())
    common = candidates[(candidates >= lower) & (candidates <= upper)]
    if len(common) < 2:
        raise ValueError('Insufficient overlapping output timestamps for a frontend comparison')
    for name, track in trajectories.items():
        sampled = {'t': common,
                   'position': np.column_stack([np.interp(common, track['t'], track['position'][:, i]) for i in range(3)]),
                   'quaternion': Slerp(track['t'], Rotation.from_quat(track['quaternion']))(common).as_quat(),
                   'velocity': np.column_stack([np.interp(common, track['t'], track['velocity'][:, i]) for i in range(6)])}
        metrics_dir = prepare_output(output/'common_time_evaluation'/name)
        np.savez_compressed(metrics_dir/'trajectory.npz', **sampled)
        summary['results'][name]['common_time_metrics'] = evaluate_tracking(args.dataset, metrics_dir)
    summary['common_evaluation_times_s'] = common.tolist()
    summary['common_evaluation_note'] = 'Offline interpolation to frozen-network output timestamps; original emitted trajectories are retained.'
    (output / 'comparison.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
