"""Checkpoint compatibility and event/model contract regression tests.

Network weights in these tests are random fixtures, never pretrained accuracy
evidence. Official-code equivalence is checked separately when its archive exists.
"""
import importlib.util
from pathlib import Path
import zipfile

import numpy as np
import pytest
import torch

from ev6d.eraft import ERAFTConfig, ERAFTFrontend, load_checkpoint, make_model, read_checkpoint
from ev6d.event_voxel import voxelize


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('legacy_name', [False, True])
def test_numpy_scalar_checkpoint_safe_under_numpy_versions(tmp_path, legacy_name):
    model = torch.nn.Linear(2, 1)
    path = tmp_path / 'scalar.tar'
    torch.save({'model': model.state_dict(), 'metric': np.float64(1.25),
                'epoch': np.int64(4), 'loss': np.float32(.5)}, path)
    # The supplied official data.pkl names NumPy 1.x's module, while current
    # NumPy writes the NumPy 2.x spelling. Exercise both on either runtime.
    with zipfile.ZipFile(path) as src:
        entries = [(item, src.read(item.filename)) for item in src.infolist()]
    with zipfile.ZipFile(path, 'w') as dst:
        for item, data in entries:
            if item.filename.endswith('/data.pkl'):
                data = data.replace(b'numpy._core.multiarray', b'numpy.core.multiarray') if legacy_name else data.replace(b'numpy.core.multiarray', b'numpy._core.multiarray')
            dst.writestr(item, data)
    allowed_before = list(torch.serialization.get_safe_globals())
    payload = load_checkpoint(model, path)
    assert payload['metric'] == 1.25 and payload['epoch'] == 4 and payload['loss'] == .5
    assert torch.serialization.get_safe_globals() == allowed_before


def test_bad_checkpoint_reports_corruption_and_never_loads_unknown_objects(tmp_path):
    path = tmp_path / 'bad.tar'
    path.write_text('<html>download failed</html>')
    with pytest.raises(ValueError, match='incomplete download, HTML error page'):
        read_checkpoint(path)
    torch.save({'unsupported': Path('not-a-tensor')}, path)
    with pytest.raises(ValueError, match='No unsafe pickle fallback'):
        read_checkpoint(path)
    with pytest.raises(FileNotFoundError, match='random inference is forbidden'):
        read_checkpoint(tmp_path / 'missing.tar')


def test_strict_parameter_and_metadata_diagnostics(tmp_path):
    model = torch.nn.Linear(2, 1)
    path = tmp_path / 'model.tar'
    state = model.state_dict()
    torch.save({'model': {'module.' + k: v for k, v in state.items()}}, path)
    load_checkpoint(model, path)
    torch.save({'model': {'weight': state['weight'], 'module.bias': state['bias']}}, path)
    with pytest.raises(ValueError, match='mixes prefixed'):
        load_checkpoint(model, path)
    torch.save({'model': {'weight': torch.ones(1, 3), 'extra': torch.ones(1)}}, path)
    with pytest.raises(ValueError, match=r"missing=\['bias'\].*unexpected=\['extra'\].*shape_mismatch"):
        load_checkpoint(model, path)
    torch.save({'model': {**state, 'weight': torch.full_like(state['weight'], float('nan'))}}, path)
    with pytest.raises(ValueError, match='Non-finite'):
        load_checkpoint(model, path)
    torch.save({'model': state, 'model_metadata': {}}, path)
    with pytest.raises(ValueError, match='metadata mismatch for num_bins'):
        load_checkpoint(model, path, ERAFTConfig())
    torch.save({'model': state, 'model_metadata': 'unvalidated'}, path)
    with pytest.raises(ValueError, match='must be a dictionary'):
        load_checkpoint(model, path, ERAFTConfig())


def test_voxel_fractional_coordinates_normalization_and_polarities():
    events = np.array([[0., .5, .5, 1], [.1, 2., 2., 0]])
    grid, info = voxelize(events, 3, 3, num_bins=3, normalize=False)
    expected = np.zeros((3, 3, 3), np.float32)
    expected[0, :2, :2] = .25
    expected[2, 2, 2] = -1
    np.testing.assert_array_equal(grid.numpy(), expected)
    assert info.usable and info.accepted_count == 2
    signed = events.copy()
    signed[-1, -1] = -1
    normalized, _ = voxelize(signed, 3, 3, num_bins=3)
    values = expected[expected != 0]
    expected[expected != 0] = (values - values.mean()) / values.std(ddof=1)
    np.testing.assert_allclose(normalized.numpy(), expected, atol=1e-7)


def test_voxel_matches_official_in_bounds_implementation():
    upstream = Path(__file__).resolve().parents[1] / 'research/eraft_archive/E-RAFT-c58ce0524ea0ebfa9849991caafb547f44fe9bfd/utils/dsec_utils.py'
    if not upstream.is_file():
        pytest.skip('Optional archived upstream source unavailable')
    spec = importlib.util.spec_from_file_location('upstream_dsec_utils', upstream)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    rng = np.random.default_rng(52)
    events = np.column_stack([np.sort(rng.uniform(.001, .099, 100)),
                              rng.uniform(0, 8, 100), rng.uniform(0, 6, 100),
                              rng.integers(0, 2, 100)])
    time = (events[:, 0] - events[0, 0]).astype(np.float32)
    time /= time[-1]
    values = {'t': torch.from_numpy(time),
              **{key: torch.tensor(events[:, index], dtype=torch.float32)
                 for key, index in [('x', 1), ('y', 2), ('p', 3)]}}
    expected = official.VoxelGrid((15, 6, 8), normalize=True).convert(values)
    actual, _ = voxelize(events, 8, 6, t_start=0, t_end=.1)
    torch.testing.assert_close(actual, expected, atol=6e-6, rtol=6e-6)


def test_voxel_empty_degenerate_bounds_and_time_order():
    for events, reason in [(np.empty((0, 4)), 'empty_events'),
                           (np.array([[0, 1, 1, 1]]), 'degenerate_time_span')]:
        result, info = voxelize(events, 4, 4)
        assert not info.usable and info.reason == reason and not result.any()
    events = np.array([[0, 1, 1, 1], [.05, -1, 2, 0], [.09, 3, 3, 0]])
    _, info = voxelize(events, 4, 4, t_start=0, t_end=.1)
    assert info.rejected_coordinates == 1 and info.accepted_count == 2
    with pytest.raises(ValueError, match='nondecreasing'):
        voxelize(events[::-1], 4, 4)
    with pytest.raises(ValueError, match='half-open'):
        voxelize(events, 4, 4, t_start=0, t_end=.09)


def test_real_model_context_final_resolution_and_reset(tmp_path):
    torch.manual_seed(3)
    cfg = ERAFTConfig(iterations=1)
    model = make_model(cfg).eval()
    with torch.inference_mode():
        context_inputs = []
        hook = model.cnet.register_forward_pre_hook(lambda module, args: context_inputs.append(args[0]))
        old, new = torch.zeros(1, 15, 32, 40), torch.ones(1, 15, 32, 40)
        low, outputs = model(old, new, iters=2)
        hook.remove()
        assert low.shape == (1, 2, 16, 16) and len(outputs) == 2
        assert outputs[-1].shape == (1, 2, 32, 40)
        assert torch.isfinite(outputs[-1]).all()
        assert torch.equal(context_inputs[0][..., -32:, -40:], new)
        assert not context_inputs[0][..., :-32, :].any()
        _, changed = model(torch.zeros(1, 15, 129, 131), torch.ones(1, 15, 129, 131), iters=1)
        assert changed[-1].shape == (1, 2, 129, 131)
    path = tmp_path / 'random_fixture.tar'
    torch.save({'model': model.state_dict(), 'model_metadata': cfg.metadata()}, path)
    frontend = ERAFTFrontend(path, cfg)
    assert not frontend.model.training and not any(p.requires_grad for p in frontend.model.parameters())
    old_events = np.array([[.01, 5, 5, 1], [.08, 10, 10, 0]])
    new_events = old_events + np.array([.1, 1, 0, 0])
    result = frontend.infer(old_events, new_events, .1, .2, 40, 32)
    assert result.flow.shape == (1, 2, 32, 40) and np.isfinite(result.flow).all()
    assert result.confidence is None and result.valid_mask.all()
    assert result.preprocessing['checkpoint_metadata_verified']
    invalid = frontend.infer(np.empty((0, 4)), new_events, .1, .2, 40, 32)
    assert not invalid.valid_mask.any() and invalid.preprocessing['timing_s']['network'] == 0
    with pytest.raises(ValueError, match='window_s'):
        frontend.infer(old_events, new_events, .1, .15, 40, 32)
    with pytest.raises(ValueError, match='available before'):
        frontend.infer(old_events, new_events, .1, .2, 40, 32, available_at=.15)
