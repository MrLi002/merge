"""Strict E-RAFT checkpoint adapter, independent of tracking filters."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import pickle
import time
import numpy as np
import torch

from .event_voxel import voxelize
from .vendor.eraft.eraft import ERAFT

UPSTREAM_COMMIT = 'c58ce0524ea0ebfa9849991caafb547f44fe9bfd'
VOXEL_CONVENTION = 'signed_trilinear_event_endpoints_nonzero_unbiased_std_v1'


@dataclass(frozen=True)
class ERAFTConfig:
    num_bins: int = 15
    iterations: int = 12
    window_s: float = .1
    normalize: bool = True
    device: str = 'cpu'

    def __post_init__(self):
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 1
               for n in (self.num_bins, self.iterations)):
            raise ValueError('num_bins and iterations must be positive integers')
        if not np.isfinite(self.window_s) or self.window_s <= 0:
            raise ValueError('window_s must be positive seconds')
        if not isinstance(self.normalize, bool):
            raise ValueError('normalize must be boolean')

    def metadata(self):
        return {**asdict(self), 'voxel_convention': VOXEL_CONVENTION,
                'warm_start': False, 'upstream_commit': UPSTREAM_COMMIT,
                'context_input': 'current_event_volume', 'flow_unit': 'pixel/interval'}


def make_model(config: ERAFTConfig):
    return ERAFT({'subtype': 'standard'}, n_first_channels=config.num_bins)


def _numeric_numpy_scalar(dtype, raw):
    """Restore legacy checkpoint metric scalars without general pickle loading."""
    if (not isinstance(dtype, np.dtype) or dtype.kind not in 'biufc'
            or dtype.fields is not None or dtype.subdtype is not None
            or not isinstance(raw, bytes) or len(raw) != dtype.itemsize):
        raise ValueError('Checkpoint NumPy metadata must contain numeric scalars')
    return np.frombuffer(raw, dtype=dtype, count=1)[0].item()


def _scalar_alias(module):
    # PyTorch 2.5 safe_globals does not support (callable, legacy_name) tuples.
    # Local wrappers give NumPy 1.x/2.x pickle names the same restricted reader.
    def scalar(dtype, raw):
        return _numeric_numpy_scalar(dtype, raw)
    scalar.__module__ = module
    scalar.__name__ = 'scalar'
    scalar.__qualname__ = 'scalar'  # PyTorch >=2.6 keys allowed globals by qualname.
    return scalar


_NUMPY_SCALAR_GLOBALS = [
    _scalar_alias('numpy.core.multiarray'),
    _scalar_alias('numpy._core.multiarray'),
    np.dtype,
    *{type(np.dtype(name)) for name in ('bool', 'int8', 'uint8', 'int16',
       'uint16', 'int32', 'uint32', 'int64', 'uint64', 'float16', 'float32',
       'float64', 'complex64', 'complex128')},
]


def read_checkpoint(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'E-RAFT weights do not exist: {path}; random inference is forbidden')
    # Official dsec.tar also contains NumPy scalar metrics. Allow only their
    # numeric reconstruction, scoped to this load; never use weights_only=False.
    try:
        with torch.serialization.safe_globals(_NUMPY_SCALAR_GLOBALS):
            return torch.load(path, map_location='cpu', weights_only=True)
    except (RuntimeError, EOFError, pickle.UnpicklingError, ValueError) as exc:
        raise ValueError(
            f'Cannot safely read E-RAFT checkpoint {path} ({path.stat().st_size} bytes). '
            'Check for an incomplete download, HTML error page, or unsupported '
            f'checkpoint metadata. No unsafe pickle fallback was attempted. {exc}'
        ) from exc


def load_checkpoint(model, path, config: ERAFTConfig | None = None):
    """Accept upstream {'model':...}, state_dict, model_state_dict, or raw dict.

    All keys/shapes must match. A module. prefix is stripped only consistently.
    Returns the original payload for optional optimizer/scheduler restoration.
    """
    checkpoint = read_checkpoint(path)
    if not isinstance(checkpoint, dict):
        raise ValueError('Expected dictionary checkpoint')
    state = checkpoint
    for key in ('model', 'state_dict', 'model_state_dict'):
        if key in checkpoint:
            state = checkpoint[key]
            break
    if (not isinstance(state, dict) or not state
            or not all(isinstance(k, str) and isinstance(v, torch.Tensor)
                       for k, v in state.items())):
        raise ValueError('Checkpoint model state must map names to tensors')
    keys = list(state)
    prefixed = [key.startswith('module.') for key in keys]
    if any(prefixed) and not all(prefixed):
        raise ValueError('Checkpoint mixes prefixed and unprefixed parameter keys')
    if all(prefixed):
        state = {k[7:]: v for k, v in state.items()}
    expected = model.state_dict()
    missing = sorted(set(expected)-set(state))
    unexpected = sorted(set(state)-set(expected))
    mismatched = {k: {'checkpoint': list(state[k].shape), 'model': list(expected[k].shape)}
                  for k in set(state) & set(expected) if state[k].shape != expected[k].shape}
    if missing or unexpected or mismatched:
        raise ValueError(f'Incompatible E-RAFT checkpoint: missing={missing}; '
                         f'unexpected={unexpected}; shape_mismatch={mismatched}')
    nonfinite = sorted(k for k, v in state.items() if not torch.isfinite(v).all())
    if nonfinite:
        raise ValueError(f'Non-finite E-RAFT checkpoint tensors: {nonfinite}')
    metadata = checkpoint.get('model_metadata')
    if 'model_metadata' in checkpoint and not isinstance(metadata, dict):
        raise ValueError('Checkpoint model_metadata must be a dictionary')
    if config is not None and metadata is not None:
        want = config.metadata()
        for name in ('num_bins', 'window_s', 'normalize', 'voxel_convention', 'context_input'):
            if name not in metadata or metadata[name] != want[name]:
                raise ValueError(f'Checkpoint metadata mismatch for {name}: '
                                 f'{metadata.get(name)!r} != {want[name]!r}')
    model.load_state_dict(state, strict=True)
    return checkpoint


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


class ERAFTFrontend:
    def __init__(self, checkpoint, config: ERAFTConfig | dict | None = None):
        self.config = config if isinstance(config, ERAFTConfig) else ERAFTConfig(**(config or {}))
        self.model = make_model(self.config)
        self.checkpoint_path = str(Path(checkpoint).resolve())
        payload = load_checkpoint(self.model, checkpoint, self.config)
        self.checkpoint_metadata = payload.get('model_metadata')
        self.training_provenance = payload.get('training_provenance', {'kind': 'external_checkpoint_unverified'})
        digest = hashlib.sha256()
        with Path(checkpoint).open('rb') as stream:
            for chunk in iter(lambda: stream.read(2**20), b''):
                digest.update(chunk)
        self.checkpoint_sha256 = digest.hexdigest()
        self.model.to(self.config.device).eval()
        self.model.requires_grad_(False)

    def reset(self):
        """No temporal state is retained by this no-warm-start baseline."""
        self.model.image_padder.pad_height = None
        self.model.image_padder.pad_width = None

    def infer(self, history_events, current_events, t_start, t_end, width, height,
              source_frame='event_rectified', available_at=None):
        from .dense_data import FlowResult
        cfg = self.config
        dt = float(t_end)-float(t_start)
        if not np.isfinite([t_start, t_end]).all() or dt <= 0:
            raise ValueError('Flow interval must have positive duration')
        if not np.isclose(dt, cfg.window_s, rtol=0, atol=1e-6):
            raise ValueError('Flow interval differs from checkpoint/config window_s; do not rescale trained flow')
        available_at = float(t_end if available_at is None else available_at)
        if not np.isfinite(available_at) or available_at < t_end:
            raise ValueError('Flow cannot be available before the second event window ends')
        synchronize(cfg.device)
        tick = time.perf_counter()
        old, old_info = voxelize(history_events, width, height, cfg.num_bins, cfg.normalize,
                                 cfg.device, t_start-dt, t_start)
        new, new_info = voxelize(current_events, width, height, cfg.num_bins, cfg.normalize,
                                 cfg.device, t_start, t_end)
        synchronize(cfg.device)
        voxel_s = time.perf_counter()-tick
        preprocess = {**cfg.metadata(), 'checkpoint': self.checkpoint_path,
                      'checkpoint_sha256': self.checkpoint_sha256,
                      'training_provenance': self.training_provenance,
                      'checkpoint_metadata_verified': self.checkpoint_metadata is not None,
                      'history_interval_s': [float(t_start-dt), float(t_start)],
                      'old_voxel': asdict(old_info), 'new_voxel': asdict(new_info),
                      'timing_s': {'voxel': voxel_s, 'network': 0.},
                      'padding': 'top_left_zero_multiple32_min128_then_unpad',
                      'confidence_source': None}
        usable = old_info.usable and new_info.usable
        if usable:
            tick = time.perf_counter()
            with torch.inference_mode():
                _, predictions = self.model(old.unsqueeze(0), new.unsqueeze(0), iters=cfg.iterations)
                flow = predictions[-1].float().cpu().numpy()
            synchronize(cfg.device)
            preprocess['timing_s']['network'] = time.perf_counter()-tick
            if not np.isfinite(flow).all():
                raise FloatingPointError('Non-finite E-RAFT output')
            valid = np.ones((1, height, width), dtype=bool)
            preprocess['status'] = 'valid'
        else:
            flow = np.zeros((1, 2, height, width), dtype=np.float32)
            valid = np.zeros((1, height, width), dtype=bool)
            preprocess['status'] = 'invalid_event_support'
        return FlowResult(flow=flow, t_start=float(t_start), t_end=float(t_end),
                          source_frame=source_frame, valid_mask=valid, available_at=available_at,
                          confidence=None, preprocessing=preprocess)
