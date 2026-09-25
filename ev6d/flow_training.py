"""Independent, supervised E-RAFT training and exact batch-boundary resume.

No tracking filter is imported. Each sample starts with zero initial flow.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dsec import DSECFlowDataset, validate_split
from .eraft import ERAFTConfig, load_checkpoint, make_model, synchronize


DEFAULTS = {
    'model': {'num_bins': 15, 'iterations': 12, 'window_s': .1, 'normalize': True, 'device': 'cpu'},
    'data': {'root': 'data/dsec/train', 'train_sequences': [], 'val_sequences': [],
             'crop_size': None, 'augment': True},
    'training': {'output_dir': 'output/flow_training', 'epochs': 100, 'max_steps': 100000,
                 'batch_size': 1, 'num_workers': 0, 'seed': 1234, 'lr': 2e-5,
                 'weight_decay': 1e-4, 'epsilon': 1e-8, 'clip_grad': 1., 'gamma': .8,
                 'max_flow': 400., 'checkpoint_every': 1000, 'validate_every': 1000,
                 'freeze_batch_norm': True, 'deterministic': False, 'num_threads': 4},
    'initialization': {'checkpoint': None},
}


def load_training_config(config):
    if isinstance(config, (str, Path)):
        config = json.loads(Path(config).read_text(encoding='utf-8'))
    if not isinstance(config, dict):
        raise ValueError('Training config must be a mapping or JSON path')
    out = copy.deepcopy(DEFAULTS)
    for section, values in config.items():
        if section not in out or not isinstance(values, dict):
            raise ValueError(f'Unknown/non-mapping training section: {section}')
        unknown = set(values)-set(out[section])
        if unknown:
            raise ValueError(f'Unknown {section} settings: {sorted(unknown)}')
        out[section].update(copy.deepcopy(values))
    ERAFTConfig(**out['model'])
    validate_split(out['data']['train_sequences'], out['data']['val_sequences'])
    tr = out['training']
    for key in ('epochs', 'max_steps', 'batch_size', 'checkpoint_every', 'validate_every', 'num_threads'):
        if isinstance(tr[key], bool) or not isinstance(tr[key], int) or tr[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('num_workers', 'seed'):
        if isinstance(tr[key], bool) or not isinstance(tr[key], int) or tr[key] < 0:
            raise ValueError(f'{key} must be a nonnegative integer')
    for key in ('lr', 'epsilon', 'clip_grad', 'max_flow'):
        if not np.isfinite(tr[key]) or tr[key] <= 0:
            raise ValueError(f'{key} must be positive and finite')
    if not np.isfinite(tr['weight_decay']) or tr['weight_decay'] < 0 or not 0 < tr['gamma'] <= 1:
        raise ValueError('Require nonnegative weight_decay and 0<gamma<=1')
    for section, key in [('training', 'freeze_batch_norm'), ('training', 'deterministic'), ('data', 'augment')]:
        if not isinstance(out[section][key], bool):
            raise ValueError(f'{section}.{key} must be boolean')
    return out


def sequence_loss(predictions, target, valid, gamma=.8, max_flow=400.):
    """Masked iteration-weighted L1; empty support returns differentiable zero."""
    if not predictions or target.ndim != 4 or target.shape[1] != 2 or valid.shape != target.shape[:1]+target.shape[2:]:
        raise ValueError('Need iterative Bx2xHxW predictions, target and BxHxW mask')
    if not 0 < gamma <= 1 or not np.isfinite(max_flow) or max_flow <= 0:
        raise ValueError('Invalid loss gamma/max_flow')
    mask = valid.bool() & torch.isfinite(target).all(dim=1) & (torch.linalg.vector_norm(target, dim=1) < max_flow)
    n = int(mask.sum().item())
    safe_target = torch.nan_to_num(target)
    loss = predictions[0].sum()*0.
    for index, prediction in enumerate(predictions):
        if prediction.shape != target.shape or not torch.isfinite(prediction).all():
            raise FloatingPointError('Nonfinite or incorrectly shaped flow prediction')
        if n:
            error = (prediction-safe_target).abs().permute(0, 2, 3, 1)[mask]
            loss = loss+gamma**(len(predictions)-index-1)*error.mean()
    return loss, n


def _dataset(config, split):
    data, model, tr = config['data'], config['model'], config['training']
    return DSECFlowDataset(data['root'], data[split+'_sequences'], num_bins=model['num_bins'],
        normalize=model['normalize'], window_s=model['window_s'], crop_size=data['crop_size'],
        augment=data['augment'] and split == 'train', seed=tr['seed'])


def _loader(dataset, config, epoch=0, train=False):
    tr = config['training']
    dataset.set_epoch(epoch)
    generator = torch.Generator().manual_seed(tr['seed']+epoch)
    return DataLoader(dataset, batch_size=tr['batch_size'], shuffle=train,
        num_workers=tr['num_workers'], generator=generator, drop_last=False)


def _seed(config):
    tr = config['training']
    random.seed(tr['seed'])
    np.random.seed(tr['seed'])
    torch.manual_seed(tr['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(tr['seed'])
    torch.set_num_threads(tr['num_threads'])
    if tr['deterministic']:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(tr['deterministic'])
    torch.backends.cudnn.benchmark = False


def _rng_state():
    state = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [state[0], state[1].tolist(), *state[2:]],
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state):
    random.setstate(state['python'])
    ns = state['numpy']
    np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), *ns[2:]))
    torch.set_rng_state(state['torch'])
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def _training_signature(config):
    signature = copy.deepcopy(config)
    signature['data']['root'] = str(Path(signature['data']['root']).resolve())
    signature.pop('initialization')
    signature['model'].pop('device')
    for key in ('output_dir', 'num_workers', 'checkpoint_every', 'validate_every'):
        signature['training'].pop(key)
    return signature


def _supervised_batch(batch, device, max_flow):
    target, valid = batch['flow'].to(device), batch['valid'].to(device)
    valid &= torch.isfinite(target).all(dim=1) & (torch.linalg.vector_norm(target, dim=1) < max_flow)
    keep = valid.flatten(1).any(dim=1)
    # Invalid samples must not affect BatchNorm statistics of supervised samples.
    return (batch['old'].to(device)[keep], batch['new'].to(device)[keep], target[keep], valid[keep])


def _validate(model, dataset, config, max_batches=None):
    device, tr = config['model']['device'], config['training']
    if max_batches is not None and (isinstance(max_batches, bool) or max_batches < 1):
        raise ValueError('max_batches must be positive')
    was_training = model.training
    model.eval()
    count, skipped, batches, epe_sum, loss_sum, above3 = 0, 0, 0, 0., 0., 0
    synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in _loader(dataset, config):
            if max_batches is not None and batches >= max_batches:
                break
            batches += 1
            old, new, target, valid = _supervised_batch(batch, device, tr['max_flow'])
            n = int(valid.sum())
            if not n:
                skipped += 1
                continue
            _, predictions = model(old, new, iters=config['model']['iterations'])
            loss, n = sequence_loss(predictions, target, valid, tr['gamma'], tr['max_flow'])
            errors = torch.linalg.vector_norm(predictions[-1]-torch.nan_to_num(target), dim=1)[valid]
            epe_sum += float(errors.double().sum())
            loss_sum += float(loss)*n
            above3 += int((errors > 3.).sum())
            count += n
    synchronize(device)
    duration = time.perf_counter()-started
    model.train(was_training)
    if was_training and tr['freeze_batch_norm']:
        model.freeze_bn()
    return {'status': 'evaluated' if count else 'no_valid_supervision', 'valid_pixels': count,
            'epe_px': epe_sum/count if count else None,
            'sequence_loss': loss_sum/count if count else None,
            'fraction_epe_above_3px': above3/count if count else None,
            'batches': batches, 'skipped_batches': skipped, 'processing_s': duration,
            'dataset': dataset.summary(), 'target_roi': 'not_supplied_all_valid_gt_pixels'}


def evaluate_flow(config, *, checkpoint=None, split='val', max_batches=None, output=None):
    config = load_training_config(config)
    if split not in ('train', 'val'):
        raise ValueError('split must be train or val')
    checkpoint = checkpoint or config['initialization']['checkpoint']
    if not checkpoint:
        raise ValueError('Evaluation requires an explicit trained checkpoint')
    if output and Path(output).exists():
        raise FileExistsError(f'Refusing to replace flow metrics: {output}')
    _seed(config)
    model_config = ERAFTConfig(**config['model'])
    model = make_model(model_config)
    payload = load_checkpoint(model, checkpoint, model_config)
    model.to(model_config.device).eval().requires_grad_(False)
    dataset = _dataset(config, split)
    dataset.augment = False
    report = _validate(model, dataset, config, max_batches)
    report.update(checkpoint=str(Path(checkpoint).resolve()), split=split,
                  training_provenance=payload.get('training_provenance', 'external_checkpoint_unverified'),
                  synthetic_fixture=(Path(config['data']['root'])/'SYNTHETIC_FIXTURE.json').exists())
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        _write_json(output, report)
    return report


def _save(path, model, optimizer, scheduler, progress, config, provenance):
    payload = {'format_version': 1, 'model': model.state_dict(),
               'model_metadata': ERAFTConfig(**config['model']).metadata(),
               'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
               'progress': copy.deepcopy(progress), 'global_step': progress['global_step'],
               'config': copy.deepcopy(config), 'training_signature': _training_signature(config),
               'resume_contract': 'Requires unchanged dataset bytes; path/config/split are checked, raw content is not hashed.',
               'training_provenance': provenance, 'rng_state': _rng_state()}
    temporary = Path(str(path)+'.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def train_flow(config, *, resume=None, output_dir=None, max_steps=None):
    """Train up to runtime target; schedule horizon remains config.max_steps.

    Resume never overwrites the source experiment directory. Deterministic
    sample order + epoch-indexed augmentation allow skipping processed batches.
    """
    config = load_training_config(config)
    tr = config['training']
    target_steps = tr['max_steps'] if max_steps is None else max_steps
    if isinstance(target_steps, bool) or not isinstance(target_steps, int) or not 1 <= target_steps <= tr['max_steps']:
        raise ValueError('max_steps stop target must be positive and <= configured scheduler horizon')
    if output_dir is not None:
        tr['output_dir'] = str(output_dir)
    output = Path(tr['output_dir'])
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'Training output must be new/empty to preserve experiments: {output}')
    _seed(config)
    train, validation = _dataset(config, 'train'), _dataset(config, 'val')
    model_config = ERAFTConfig(**config['model'])
    model = make_model(model_config)
    payload = None
    initial_checkpoint = config['initialization']['checkpoint']
    if resume:
        payload = load_checkpoint(model, resume, model_config)
        required = {'optimizer', 'scheduler', 'progress', 'rng_state', 'training_signature'}
        if not required <= set(payload):
            raise ValueError('Resume needs a full training checkpoint; use initialization.checkpoint for fine-tuning')
        if payload['training_signature'] != _training_signature(config):
            raise ValueError('Resume configuration/data split/scheduler horizon differs from saved training signature')
    elif initial_checkpoint:
        load_checkpoint(model, initial_checkpoint, model_config)
    device = model_config.device
    model.to(device).train()
    if tr['freeze_batch_norm']:
        model.freeze_bn()
    optimizer = torch.optim.AdamW(model.parameters(), lr=tr['lr'], weight_decay=tr['weight_decay'], eps=tr['epsilon'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tr['max_steps'])
    progress = {'epoch': 0, 'next_batch': 0, 'global_step': 0, 'attempted_batches': 0, 'skipped_batches': 0}
    synthetic = (Path(config['data']['root'])/'SYNTHETIC_FIXTURE.json').exists()
    provenance = {'synthetic_fixture': synthetic, 'initialization': 'checkpoint' if initial_checkpoint else 'random',
                  'initial_checkpoint': str(initial_checkpoint) if initial_checkpoint else None,
                  'train_sequences': list(config['data']['train_sequences']),
                  'val_sequences': list(config['data']['val_sequences']),
                  'temporal_training': 'independent_pairs_no_warm_start'}
    if payload is not None:
        optimizer.load_state_dict(payload['optimizer'])
        scheduler.load_state_dict(payload['scheduler'])
        progress.update(payload['progress'])
        provenance = payload.get('training_provenance', provenance)
        _restore_rng(payload['rng_state'])
        if progress['global_step'] >= target_steps:
            raise ValueError('Resume target must exceed checkpoint global_step')
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output/'config.json', config)
    _write_json(output/'data_summary.json', {'train': train.summary(), 'val': validation.summary(),
        'synthetic_fixture': synthetic, 'resume': str(resume) if resume else None})
    synchronize(device)
    started = time.perf_counter()
    initial_step = progress['global_step']
    history, latest_validation = [], None
    try:
        for epoch in range(progress['epoch'], tr['epochs']):
            loader = _loader(train, config, epoch, train=True)
            offset = progress['next_batch'] if epoch == progress['epoch'] else 0
            progress['epoch'] = epoch
            for batch_index, batch in enumerate(loader):
                if batch_index < offset:
                    continue
                progress.update(next_batch=batch_index+1, attempted_batches=progress['attempted_batches']+1)
                old, new, target, valid = _supervised_batch(batch, device, tr['max_flow'])
                if not bool(valid.any()):
                    progress['skipped_batches'] += 1
                    continue
                optimizer.zero_grad(set_to_none=True)
                _, predictions = model(old, new, iters=model_config.iterations)
                loss, valid_count = sequence_loss(predictions, target, valid, tr['gamma'], tr['max_flow'])
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite supervised loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tr['clip_grad'], error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                progress['global_step'] += 1
                step = progress['global_step']
                row = {'step': step, 'epoch': epoch, 'batch': batch_index, 'loss': float(loss.detach()),
                       'valid_pixels': valid_count, 'gradient_norm_before_clip': float(norm),
                       'lr': optimizer.param_groups[0]['lr']}
                if step % tr['validate_every'] == 0:
                    latest_validation = _validate(model, validation, config)
                    row['validation'] = latest_validation
                history.append(row)
                with (output/'train.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                if step % tr['checkpoint_every'] == 0 or step >= target_steps:
                    _save(output/'last.pt', model, optimizer, scheduler, progress, config, provenance)
                if step >= target_steps:
                    break
            if progress['global_step'] >= target_steps:
                break
            progress.update(epoch=epoch+1, next_batch=0)
        if progress['global_step'] == initial_step:
            raise ValueError('No optimization steps completed; inspect event support/GT masks/epoch limit')
        latest_validation = _validate(model, validation, config)
        _save(output/'last.pt', model, optimizer, scheduler, progress, config, provenance)
        synchronize(device)
        report = {'status': 'target_reached' if progress['global_step'] >= target_steps else 'epochs_exhausted',
                  'global_step': progress['global_step'], 'initial_step': initial_step,
                  'steps_this_run': progress['global_step']-initial_step, 'progress': progress,
                  'checkpoint': str((output/'last.pt').resolve()), 'validation': latest_validation,
                  'training_provenance': provenance, 'history': history,
                  'processing_s': time.perf_counter()-started,
                  'scheduler': 'CosineAnnealingLR', 'scheduler_horizon_steps': tr['max_steps']}
        _write_json(output/'training_report.json', report)
        return report
    except Exception as error:
        _write_json(output/'error.json', {'type': type(error).__name__, 'message': str(error), 'progress': progress})
        raise
