"""Short real-network optimization checks, using synthetic DSEC-format data.

These tests establish executable training/checkpoint plumbing, not real-data
accuracy or equivalence to the paper's temporal training schedule.
"""
import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from ev6d.dsec import create_synthetic_dsec_fixture
from ev6d.eraft import ERAFTConfig, ERAFTFrontend, read_checkpoint
from ev6d.flow_training import (evaluate_flow, load_training_config,
                                sequence_loss, train_flow)


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)
    torch.use_deterministic_algorithms(deterministic)


def test_sequence_loss_supervises_each_iteration_on_valid_pixels():
    target = torch.zeros(1, 2, 2, 2)
    valid = torch.tensor([[[True, False], [False, False]]])
    first = torch.full_like(target, 2., requires_grad=True)
    final = torch.full_like(target, 1., requires_grad=True)
    loss, count = sequence_loss([first, final], target, valid, gamma=.8)
    assert count == 1
    assert loss.item() == pytest.approx(2.6)
    loss.backward()
    torch.testing.assert_close(first.grad[0, :, 0, 0], torch.tensor([.4, .4]))
    assert first.grad[0, :, 1, 1].count_nonzero() == 0
    torch.testing.assert_close(final.grad[0, :, 0, 0], torch.tensor([.5, .5]))


def test_sequence_loss_excludes_nonfinite_gt_and_excessive_flow():
    target = torch.zeros(1, 2, 2, 2)
    target[0, :, 0, 1] = float('nan')
    target[0, :, 1, 0] = 500
    target[0, :, 1, 1] = float('inf')
    prediction = torch.ones_like(target, requires_grad=True)
    loss, count = sequence_loss([prediction], target, torch.ones(1, 2, 2, dtype=torch.bool), max_flow=400)
    assert count == 1 and loss.item() == pytest.approx(1.)
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.count_nonzero() == 2


def test_sequence_loss_identifies_absent_supervision():
    target = torch.zeros(1, 2, 2, 2)
    loss, count = sequence_loss([target.clone().requires_grad_()], target,
                                torch.zeros(1, 2, 2, dtype=torch.bool))
    assert count == 0
    assert loss is None or float(loss) == 0


@pytest.fixture
def training_config(tmp_path):
    data = create_synthetic_dsec_fixture(tmp_path/'fixture', size=128, samples=2)
    return {
        'model': {'num_bins': 3, 'iterations': 1, 'window_s': .1,
                  'normalize': True, 'device': 'cpu'},
        'data': {'root': str(data), 'train_sequences': ['synthetic_train'],
                 'val_sequences': ['synthetic_val'], 'crop_size': [128, 128],
                 'augment': False},
        'training': {'output_dir': str(tmp_path/'training'), 'epochs': 3,
                     'max_steps': 3, 'batch_size': 1, 'num_workers': 0,
                     'seed': 731, 'lr': .00002, 'weight_decay': .0001,
                     'epsilon': 1e-8, 'clip_grad': 1., 'gamma': .8,
                     'max_flow': 400, 'checkpoint_every': 1,
                     'validate_every': 1, 'freeze_batch_norm': True,
                     'deterministic': True, 'num_threads': 2},
        'initialization': {'checkpoint': None},
    }


def test_training_rejects_train_validation_sequence_leakage(training_config):
    training_config['data']['val_sequences'] = ['synthetic_train']
    with pytest.raises(ValueError, match='disjoint|overlap|split'):
        train_flow(training_config, max_steps=1)


def test_config_path_defaults_and_input_isolation(training_config, tmp_path):
    original = copy.deepcopy(training_config)
    path = tmp_path/'config.json'
    path.write_text(json.dumps(training_config), encoding='utf-8')
    loaded = load_training_config(path)
    assert loaded['training']['max_steps'] == 3
    loaded['data']['train_sequences'].append('another_sequence')
    assert training_config == original


@pytest.mark.parametrize('section,key,value', [
    ('training', 'typo', 1), ('training', 'max_steps', 0),
    ('training', 'lr', float('nan')), ('training', 'gamma', 1.1),
    ('data', 'augment', 'yes'), ('model', 'num_bins', 0),
])
def test_malformed_training_configuration_fails_early(training_config, section, key, value):
    training_config[section][key] = value
    with pytest.raises(ValueError):
        load_training_config(training_config)


def test_nonfinite_prediction_is_explicit_failure():
    target = torch.zeros(1, 2, 2, 2)
    prediction = target.clone()
    prediction[0, 0, 0, 0] = float('nan')
    with pytest.raises(FloatingPointError):
        sequence_loss([prediction], target, torch.ones(1, 2, 2, dtype=torch.bool))


def test_all_invalid_training_data_skips_optimizer_and_reports_reason(training_config):
    folder = Path(training_config['data']['root'])/'synthetic_train'/'flow'/'forward'
    for path in folder.glob('*.png'):
        raw = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        raw[..., 0] = 0  # OpenCV blue channel is DSEC validity.
        cv2.imencode('.png', raw)[1].tofile(path)
    with pytest.raises(ValueError, match='No optimization steps'):
        train_flow(training_config, max_steps=1)
    output = Path(training_config['training']['output_dir'])
    error = json.loads((output/'error.json').read_text(encoding='utf-8'))
    assert error['progress']['global_step'] == 0
    assert error['progress']['skipped_batches'] == 6
    assert not (output/'last.pt').exists()


def test_real_eraft_optimization_resume_matches_uninterrupted_and_frontend_load(training_config, tmp_path):
    first_report = train_flow(training_config, max_steps=1)
    first_path = first_report['checkpoint']
    first = read_checkpoint(first_path)
    assert first_report['global_step'] == first_report['steps_this_run'] == 1
    assert first['progress']['next_batch'] == 1
    assert first['global_step'] == first['scheduler']['last_epoch'] == 1
    assert first['scheduler']['T_max'] == 3
    assert first['model_metadata']['num_bins'] == 3
    assert first['training_provenance']['synthetic_fixture'] is True
    assert set(first['rng_state']) >= {'python', 'numpy', 'torch'}
    first_weight = first['model']['fnet.conv1.weight'].clone()
    del first

    resumed_report = train_flow(training_config, resume=first_path,
                                output_dir=tmp_path/'resumed', max_steps=2)
    assert resumed_report['global_step'] == 2
    assert resumed_report['initial_step'] == resumed_report['steps_this_run'] == 1
    resumed = read_checkpoint(resumed_report['checkpoint'])
    assert resumed['progress']['attempted_batches'] == 2
    assert resumed['scheduler']['last_epoch'] == 2
    assert resumed['scheduler']['T_max'] == 3
    assert {int(state['step']) for state in resumed['optimizer']['state'].values()} == {2}
    assert not torch.equal(first_weight, resumed['model']['fnet.conv1.weight'])
    assert read_checkpoint(first_path)['global_step'] == 1  # source run preserved

    direct_report = train_flow(training_config, output_dir=tmp_path/'uninterrupted', max_steps=2)
    direct = read_checkpoint(direct_report['checkpoint'])
    for key, value in resumed['model'].items():
        torch.testing.assert_close(value, direct['model'][key], rtol=0, atol=0,
                                   msg=lambda message: f'{key}: {message}')
    assert direct['scheduler'] == resumed['scheduler']
    del resumed, direct

    frontend = ERAFTFrontend(resumed_report['checkpoint'], ERAFTConfig(**training_config['model']))
    assert not frontend.model.training
    assert all(not parameter.requires_grad for parameter in frontend.model.parameters())
    assert frontend.checkpoint_metadata['flow_unit'] == 'pixel/interval'
    metrics_path = tmp_path/'evaluation.json'
    metrics = evaluate_flow(training_config, checkpoint=resumed_report['checkpoint'],
                            max_batches=1, output=metrics_path)
    assert metrics['status'] == 'evaluated'
    assert metrics['valid_pixels'] == 128*126
    assert metrics['epe_px'] >= 0 and metrics['synthetic_fixture']
    assert metrics['batches'] == 1 and metrics_path.is_file()

    altered = copy.deepcopy(training_config)
    altered['training']['max_steps'] = 4
    with pytest.raises(ValueError, match='signature|horizon'):
        train_flow(altered, resume=first_path, output_dir=tmp_path/'bad_resume', max_steps=2)

    fine_tune = copy.deepcopy(training_config)
    fine_tune['initialization']['checkpoint'] = first_path
    fine_tune_report = train_flow(fine_tune, output_dir=tmp_path/'fine_tune', max_steps=1)
    assert fine_tune_report['initial_step'] == 0
    assert fine_tune_report['global_step'] == 1
    assert fine_tune_report['training_provenance']['initialization'] == 'checkpoint'
