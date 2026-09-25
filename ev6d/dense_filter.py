"""Complete 2D displacement observations for a six-dimensional spatial-twist KF.

This is a separate backend from the legacy triplet/normal-flow VelocityKF.
State: [v_O (m/s), omega (rad/s)] in event camera axes; Xdot=v_O+omega x X.
Observations: forward pixel displacement, F=dt*J*state, using source-time depth.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .filters import _covariance
from .geometry import interaction_matrix


@dataclass(frozen=True)
class DenseVelocityConfig:
    # Continuous diffusion standard deviation: (m/s)/sqrt(s), (rad/s)/sqrt(s).
    process_linear_std: float = .5
    process_angular_std: float = 1.
    initial_linear_std: float = 1.
    initial_angular_std: float = 2.
    # lambda=0 means constant velocity, lambda>0 continuous OU decay per second.
    decay_rate_per_s: float = 0.
    flow_noise_std: float = 1.
    flow_noise_units: str = "displacement"  # pixel/interval or pixel/s ("velocity")
    noise_inflation: float = 4.  # heuristic correlated-flow protection, not calibrated
    gate_mahalanobis_sq: float | None = 25.
    min_observations: int = 6
    max_observations: int = 512
    rank_rtol: float = 1e-8
    max_condition_number: float = 1e6

    def __post_init__(self):
        for name in ("process_linear_std", "process_angular_std", "decay_rate_per_s"):
            value = getattr(self, name)
            if not np.isscalar(value) or not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("initial_linear_std", "initial_angular_std", "flow_noise_std",
                     "noise_inflation", "rank_rtol", "max_condition_number"):
            value = getattr(self, name)
            if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.rank_rtol >= 1 or self.max_condition_number < 1 or self.noise_inflation < 1:
            raise ValueError("Need rank_rtol<1, max_condition_number>=1, noise_inflation>=1")
        if self.flow_noise_units not in ("displacement", "velocity"):
            raise ValueError("flow_noise_units must be displacement or velocity")
        if self.gate_mahalanobis_sq is not None and (
            not np.isfinite(self.gate_mahalanobis_sq) or self.gate_mahalanobis_sq <= 0
        ):
            raise ValueError("gate_mahalanobis_sq must be positive or None")
        for name in ("min_observations", "max_observations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 3:
                raise ValueError(f"{name} must be an integer >=3")
        if self.max_observations < self.min_observations:
            raise ValueError("max_observations must be >=min_observations")


def stratified_indices(uv, maximum):
    """Deterministic round-robin image strata sampling, with no duplicates.

    Uniform bins cover the candidate bounding box. Their centres are preferred;
    every occupied bin contributes once before any contributes a second point.
    A bounded sample reduces, but does not eliminate, correlated network errors.
    """
    uv = np.asarray(uv, dtype=float)
    if len(uv) <= maximum:
        return np.arange(len(uv))
    side = max(1, int(np.sqrt(maximum)))
    extent = np.maximum(np.ptp(uv, axis=0), 1.)
    relative = (uv-uv.min(axis=0))/extent
    cells = np.minimum((relative*side).astype(int), side-1)
    labels = cells[:, 0]+side*cells[:, 1]
    distance = np.linalg.norm(relative*side-(cells+.5), axis=1)
    ordered = np.lexsort((np.arange(len(uv)), distance, labels))
    sorted_labels = labels[ordered]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_labels))+1]
    lengths = np.diff(np.r_[starts, len(uv)])
    within_cell = np.arange(len(uv))-np.repeat(starts, lengths)
    return ordered[np.lexsort((sorted_labels, within_cell))[:maximum]]


class DenseVelocityKF:
    """A bounded, gated, full optical-flow linear KF with only small solves.

    predict_to() advances the prior clock. update() never advances that clock:
    the caller explicitly schedules an interval estimate once its flow arrives.
    The estimated twist approximates motion across that interval, not necessarily
    the instantaneous endpoint velocity. Rejected observations leave x/P intact.
    """

    def __init__(self, config=None):
        self.config = config or DenseVelocityConfig()
        self.x = np.zeros(6)
        self.P = np.diag([self.config.initial_linear_std**2]*3 +
                         [self.config.initial_angular_std**2]*3)
        self.timestamp = None

    def predict_to(self, timestamp):
        timestamp = float(timestamp)
        if not np.isfinite(timestamp) or (self.timestamp is not None and timestamp < self.timestamp):
            raise ValueError("Velocity timestamps must be finite and monotonic seconds")
        if not np.isfinite(self.x).all() or self.x.shape != (6,):
            raise ValueError("Invalid velocity state")
        prior = _covariance(self.P, 6)
        dt = 0. if self.timestamp is None else timestamp-self.timestamp
        if dt > 0:
            rate = self.config.decay_rate_per_s
            decay = np.exp(-rate*dt)
            duration = dt if rate == 0 else -np.expm1(-2*rate*dt)/(2*rate)
            diffusion = np.diag([self.config.process_linear_std**2]*3 +
                                [self.config.process_angular_std**2]*3)
            self.x = decay*self.x
            self.P = _covariance(decay**2*prior+duration*diffusion, 6)
        self.timestamp = timestamp

    def update(self, uv, depth, flow_displacement, K, dt):
        """Assimilate N source pixels/depths and Nx2 interval displacements.

        Invalid sample values are counted and removed. Invalid interface shapes,
        intrinsics, time units, state or covariance raise instead of hiding errors.
        Noise configured as pixel/s is multiplied by dt once to displacement std.
        """
        uv = np.asarray(uv, dtype=float)
        depth = np.asarray(depth, dtype=float)
        flow = np.asarray(flow_displacement, dtype=float)
        dt = float(dt)
        if uv.ndim != 2 or uv.shape[1:] != (2,) or depth.shape != (len(uv),) or flow.shape != uv.shape:
            raise ValueError("Expected Nx2 pixels, N depths and Nx2 displacements")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Flow interval dt must be positive seconds")
        if self.x.shape != (6,) or not np.isfinite(self.x).all():
            raise ValueError("Invalid velocity state")
        prior = _covariance(self.P, 6)
        # Validate calibration even when all observations are absent.
        interaction_matrix(np.empty((0, 2)), np.empty(0), K)
        cfg = self.config
        valid = np.isfinite(uv).all(axis=1) & np.isfinite(depth) & (depth > 0) & np.isfinite(flow).all(axis=1)
        result = {"accepted": False, "reason": "no_valid_observations", "input_observations": len(uv),
                  "invalid_observations": int((~valid).sum()), "sampled_observations": 0,
                  "sampling_dropped": 0, "gated_observations": 0, "measurements": 0,
                  "constraints": 0, "observable_rank": 0, "condition_number": None,
                  "dt_s": dt, "flow_units": "pixel/interval"}
        uv, depth, flow = uv[valid], depth[valid], flow[valid]
        if not len(uv):
            return result
        indices = stratified_indices(uv, cfg.max_observations)
        result["sampling_dropped"] = len(uv)-len(indices)
        uv, depth, flow = uv[indices], depth[indices], flow[indices]
        result["sampled_observations"] = len(uv)
        H = dt*interaction_matrix(uv, depth, K)
        std = cfg.flow_noise_std*(dt if cfg.flow_noise_units == "velocity" else 1.)
        variance = std**2*cfg.noise_inflation
        if not np.isfinite(variance) or variance <= 0:
            raise ValueError("Invalid effective displacement variance")
        result["effective_flow_std_px"] = float(np.sqrt(variance))
        residual = flow-np.einsum("nij,j->ni", H, self.x)
        innovation_cov = np.einsum("nij,jk,nlk->nil", H, prior, H)+variance*np.eye(2)
        whitened = np.linalg.solve(innovation_cov, residual[..., None])[..., 0]
        distances = np.einsum("ni,ni->n", residual, whitened)
        keep = np.ones(len(H), dtype=bool) if cfg.gate_mahalanobis_sq is None else distances <= cfg.gate_mahalanobis_sq
        result["gated_observations"] = int((~keep).sum())
        result["mahalanobis_sq_max"] = float(distances.max())
        H, flow, residual = H[keep], flow[keep], residual[keep]
        result["measurements"], result["constraints"] = len(H), 2*len(H)
        if len(H) < cfg.min_observations:
            result["reason"] = "insufficient_observations_after_gating"
            return result
        matrix = H.reshape(-1, 6)
        singular = np.linalg.svd(matrix, compute_uv=False)
        rank = int(np.count_nonzero(singular > cfg.rank_rtol*singular[0]))
        result["observable_rank"] = rank
        if rank < 6:
            result["reason"] = "rank_deficient"
            return result
        condition = float(singular[0]/singular[-1])
        result["condition_number"] = condition
        if condition > cfg.max_condition_number:
            result["reason"] = "ill_conditioned"
            return result
        # Whiten the prior: P=L L'. Posterior is L(I+A'A)^-1 L' for
        # A=H L/sigma, avoiding both P^-1 and any 2N x 2N matrix.
        root = np.linalg.cholesky(prior)
        A = matrix@root/np.sqrt(variance)
        information = np.eye(6)+A.T@A
        factor = np.linalg.cholesky(information)
        rhs = A.T@residual.ravel()/np.sqrt(variance)
        correction = np.linalg.solve(factor.T, np.linalg.solve(factor, rhs))
        new_x = self.x+root@correction
        covariance_root = np.linalg.solve(factor, root.T)
        new_P = _covariance(covariance_root.T@covariance_root, 6)
        if not np.isfinite(new_x).all():
            raise FloatingPointError("Non-finite velocity posterior")
        self.x, self.P = new_x, new_P
        result.update(accepted=True, reason="updated",
                      residual_rms_before=float(np.sqrt(np.mean(residual**2))),
                      residual_rms_after=float(np.sqrt(np.mean((flow.ravel()-matrix@new_x)**2))),
                      covariance_min_eigenvalue=float(np.linalg.eigvalsh(new_P)[0]))
        return result
