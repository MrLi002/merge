"""Validated dense E-RAFT flow contract (pixel displacement per interval)."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np

@dataclass
class FlowResult:
    flow: np.ndarray
    t_start: float
    t_end: float
    source_frame: str
    valid_mask: np.ndarray
    available_at: float | None = None
    confidence: np.ndarray | None = None
    preprocessing: dict = field(default_factory=dict)

    def __post_init__(self):
        self.flow = np.asarray(self.flow, dtype=np.float32)
        self.valid_mask = np.asarray(self.valid_mask)
        if self.flow.ndim != 4 or self.flow.shape[1] != 2 or min(self.flow.shape) < 1:
            raise ValueError('Flow must have shape [B,2,H,W]')
        if self.valid_mask.shape != self.flow.shape[:1]+self.flow.shape[2:]:
            raise ValueError('Flow valid_mask must have shape [B,H,W]')
        if self.valid_mask.dtype != bool:
            raise ValueError('Flow valid_mask must be boolean')
        if not np.isfinite(self.flow).all():
            raise ValueError('Flow contains nonfinite values (invalid pixels must have finite placeholders)')
        self.t_start, self.t_end = float(self.t_start), float(self.t_end)
        self.available_at = float(self.t_end if self.available_at is None else self.available_at)
        if (not np.isfinite([self.t_start, self.t_end, self.available_at]).all()
                or self.t_end <= self.t_start or self.available_at < self.t_end):
            raise ValueError('Flow needs increasing seconds and available_at >= t_end')
        if not isinstance(self.source_frame, str) or not self.source_frame:
            raise ValueError('Flow source_frame must be explicit')
        if not isinstance(self.preprocessing, dict):
            raise ValueError('Flow preprocessing must be a dictionary')
        if self.confidence is not None:
            self.confidence = np.asarray(self.confidence, dtype=np.float32)
            if (self.confidence.shape != self.valid_mask.shape
                    or not np.isfinite(self.confidence).all() or np.any(self.confidence < 0)):
                raise ValueError('Confidence must be finite nonnegative [B,H,W]')
            if not self.preprocessing.get('confidence_source'):
                raise ValueError('A confidence map needs an explicit confidence_source')

    @property
    def dt(self):
        return self.t_end-self.t_start

    def save(self, path):
        metadata = {'format_version': 1, 't_start': self.t_start, 't_end': self.t_end,
                    'available_at': self.available_at, 'source_frame': self.source_frame,
                    'flow_unit': 'pixel/interval', 'preprocessing': self.preprocessing,
                    'has_confidence': self.confidence is not None}
        path = Path(path)
        if path.suffix != '.npz':
            raise ValueError('Flow cache paths must explicitly end in .npz')
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f'Refusing to replace flow cache: {path}')
        np.savez_compressed(path, flow=self.flow, valid_mask=self.valid_mask,
                            confidence=self.confidence if self.confidence is not None else np.empty(0),
                            metadata=np.array(json.dumps(metadata, allow_nan=False)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive['metadata'].item()))
            if meta.get('format_version') != 1 or meta.get('flow_unit') != 'pixel/interval':
                raise ValueError('Unsupported flow cache version/units')
            return cls(archive['flow'], meta['t_start'], meta['t_end'], meta['source_frame'],
                       archive['valid_mask'], meta['available_at'],
                       archive['confidence'] if meta['has_confidence'] else None,
                       meta['preprocessing'])


