"""Velocity KF and pose manifold UKF for the paper's two-stage estimator.

All defaults except the paper's alpha=.5 are explicit engineering choices.
Velocity decay/noise is tied to a reference period, so observation scheduling
does not change the underlying velocity prior. Pose sigma points live on
R3 x SO(3); velocity uncertainty is augmented in each prediction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import interaction_matrix, integrate_pose, quat_difference, quat_exp, quat_mean, quat_multiply


def _covariance(matrix, size):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (size, size) or not np.isfinite(matrix).all():
        raise ValueError(f"Expected finite {size}x{size} covariance")
    if not np.allclose(matrix, matrix.T, atol=1e-9):
        raise ValueError("Covariance must be symmetric")
    eigenvalues, vectors = np.linalg.eigh((matrix+matrix.T)/2)
    if eigenvalues.min() < -1e-8:
        raise ValueError("Covariance must be positive semidefinite")
    return (vectors*np.maximum(eigenvalues, 1e-12))@vectors.T


def _positive_parameters(config, names, allow_zero=False):
    for name in names:
        value = getattr(config, name)
        if not np.isscalar(value) or not np.isfinite(value) or (value < 0 if allow_zero else value <= 0):
            raise ValueError(f"Invalid {name}")


@dataclass(frozen=True)
class VelocityConfig:
    alpha: float = .5
    decay_reference_s: float = .01
    process_linear_std: float = .5
    process_angular_std: float = 1.
    initial_linear_std: float = 1.
    initial_angular_std: float = 2.
    flow_noise_std: float = 20.
    laplace_scale: float = 20.
    min_weight: float = 1e-6
    normal_flow: bool = True
    weighting: bool = True

    def __post_init__(self):
        _positive_parameters(self, ("alpha", "decay_reference_s", "initial_linear_std", "initial_angular_std", "flow_noise_std", "laplace_scale", "min_weight"))
        _positive_parameters(self, ("process_linear_std", "process_angular_std"), allow_zero=True)
        if self.alpha > 1 or self.min_weight > 1:
            raise ValueError("alpha and min_weight must be in (0,1]")
        if not isinstance(self.normal_flow, bool) or not isinstance(self.weighting, bool):
            raise ValueError("normal_flow and weighting must be booleans")


class VelocityKF:
    def __init__(self, config=None):
        self.config = config or VelocityConfig()
        self.x = np.zeros(6)
        self.P = np.diag([self.config.initial_linear_std**2]*3 + [self.config.initial_angular_std**2]*3)
        self.timestamp = None

    def predict_to(self, timestamp):
        timestamp = float(timestamp)
        if not np.isfinite(timestamp) or (self.timestamp is not None and timestamp < self.timestamp):
            raise ValueError("Velocity prediction timestamps must be finite and monotonic")
        if self.timestamp is None:
            self.timestamp = timestamp
            return
        dt = timestamp-self.timestamp
        if dt:
            cfg = self.config
            periods = dt/cfg.decay_reference_s
            decay = cfg.alpha**periods
            scale = periods if cfg.alpha == 1 else -np.expm1(2*periods*np.log(cfg.alpha))/(1-cfg.alpha**2)
            Q = np.diag([cfg.process_linear_std**2]*3+[cfg.process_angular_std**2]*3)
            self.x *= decay
            self.P = _covariance(decay*decay*self.P+scale*Q, 6)
        self.timestamp = timestamp

    def update(self, uv, depth, flow, K):
        """Stack independent normal constraints, or two components in ablation.

        Equation 9's rank-one 2D residual is represented by its single nonzero
        scalar constraint n.T J V = ||F||. Equation 10 weights use the distance
        from median residual magnitude, R_eff=flow_noise_std**2/L. This is a
        one-pass robust weighting choice, not the full ERL reference algorithm.
        A 6x6 information solve avoids a quadratic number of event covariances.
        """
        uv, z, flow = np.asarray(uv, dtype=float), np.asarray(depth, dtype=float), np.asarray(flow, dtype=float)
        if uv.ndim != 2 or uv.shape[1] != 2 or z.shape != (len(uv),) or flow.shape != (len(uv), 2):
            raise ValueError("Expected Nx2 pixels, N depth, Nx2 optical flow")
        valid = np.isfinite(uv).all(axis=1) & np.isfinite(z) & (z > 0) & np.isfinite(flow).all(axis=1)
        if self.config.normal_flow:
            valid &= np.linalg.norm(flow, axis=1) > 1e-10
        uv, z, flow = uv[valid], z[valid], flow[valid]
        if not len(uv):
            return {"accepted": False, "measurements": 0, "observable_rank": 0}
        J = interaction_matrix(uv, z, K)
        cfg = self.config
        if cfg.normal_flow:
            magnitude = np.linalg.norm(flow, axis=1)
            normal = flow/magnitude[:, None]
            H = np.einsum("ni,nij->nj", normal, J)
            observation = magnitude
            residual = observation-H@self.x
            residual_magnitude = np.abs(residual)
        else:
            H = J.reshape(-1, 6)
            observation = flow.ravel()
            residual = observation-H@self.x
            residual_magnitude = np.linalg.norm(residual.reshape(-1, 2), axis=1)
        if cfg.weighting:
            median = np.median(residual_magnitude)
            weights = np.maximum(np.exp(-np.abs(residual_magnitude-median)/cfg.laplace_scale)/(2*cfg.laplace_scale), cfg.min_weight)
        else:
            weights = np.ones(len(uv))
        row_weights = weights if cfg.normal_flow else np.repeat(weights, 2)
        precision = row_weights/cfg.flow_noise_std**2
        prior = _covariance(self.P, 6)
        information = np.linalg.solve(prior, np.eye(6)) + H.T@(precision[:, None]*H)
        self.x = self.x + np.linalg.solve(information, H.T@(precision*residual))
        self.P = _covariance(np.linalg.solve(information, np.eye(6)), 6)
        singular = np.linalg.svd(np.sqrt(precision)[:, None]*H, compute_uv=False)
        threshold = singular[0]*1e-8 if len(singular) else 0
        rank = int(np.count_nonzero(singular > threshold))
        condition = float(singular[0]/singular[-1]) if rank == 6 else None
        return {"accepted": True, "measurements": len(uv), "constraints": len(H),
                "observable_rank": rank, "condition_number": condition,
                "weight_min": float(weights.min()), "weight_median": float(np.median(weights)), "weight_max": float(weights.max()),
                "residual_rms_before": float(np.sqrt(np.mean(residual**2))),
                "residual_rms_after": float(np.sqrt(np.mean((observation-H@self.x)**2)))}


@dataclass(frozen=True)
class PoseConfig:
    initial_position_std: float = .02
    initial_rotation_std: float = .08
    process_position_std: float = .02
    process_rotation_std: float = .10
    measurement_position_std: float = .01
    measurement_rotation_std: float = .05
    gate_threshold: float | None = None
    sigma_alpha: float = 1.
    sigma_beta: float = 2.
    sigma_kappa: float = 0.

    def __post_init__(self):
        _positive_parameters(self, ("initial_position_std", "initial_rotation_std", "measurement_position_std", "measurement_rotation_std", "sigma_alpha"))
        if self.gate_threshold is not None:
            _positive_parameters(self, ("gate_threshold",))
        _positive_parameters(self, ("process_position_std", "process_rotation_std", "sigma_beta"), allow_zero=True)
        if not np.isfinite(self.sigma_kappa) or self.sigma_kappa <= -6:
            raise ValueError("sigma_kappa must be finite and greater than -6")


class PoseUKF:
    def __init__(self, position, quaternion, timestamp, config=None):
        self.config = config or PoseConfig()
        self.position = np.asarray(position, dtype=float).copy()
        if self.position.shape != (3,) or not np.isfinite(self.position).all() or not np.isfinite(timestamp):
            raise ValueError("Invalid initial pose or timestamp")
        self.quaternion = quat_multiply(quaternion, [0, 0, 0, 1])
        self.timestamp = float(timestamp)
        self.P = np.diag([self.config.initial_position_std**2]*3+[self.config.initial_rotation_std**2]*3)

    def _sigma(self, covariance):
        n = len(covariance)
        cfg = self.config
        scale = cfg.sigma_alpha**2*(n+cfg.sigma_kappa)
        if not np.isfinite(scale) or scale < 1e-10:
            raise ValueError("Sigma scaling is numerically degenerate")
        root = np.linalg.cholesky(_covariance(covariance, n))*np.sqrt(scale)
        points = np.vstack([np.zeros(n), root.T, -root.T])
        wm = np.full(2*n+1, 1/(2*scale))
        wm[0] = (scale-n)/scale
        wc = wm.copy()
        wc[0] += 1-cfg.sigma_alpha**2+cfg.sigma_beta
        return points, wm, wc

    @staticmethod
    def _pose_statistics(positions, quaternions, wm, wc, initial):
        position = wm@positions
        quaternion = quat_mean(quaternions, wm, initial)
        rotations = (Rotation.from_quat(quaternions)*Rotation.from_quat(quaternion).inv()).as_rotvec()
        errors = np.column_stack([positions-position, rotations])
        covariance = np.einsum("n,ni,nj->ij", wc, errors, errors)
        return position, quaternion, covariance, errors

    def predict_to(self, timestamp, velocity, velocity_covariance=None):
        timestamp, velocity = float(timestamp), np.asarray(velocity, dtype=float)
        if not np.isfinite(timestamp) or timestamp < self.timestamp or velocity.shape != (6,) or not np.isfinite(velocity).all():
            raise ValueError("Invalid prediction timestamp or spatial velocity")
        dt = timestamp-self.timestamp
        if not dt:
            return
        if velocity_covariance is None:
            covariance = self.P
        else:
            covariance = np.zeros((12, 12))
            covariance[:6, :6] = self.P
            covariance[6:, 6:] = _covariance(velocity_covariance, 6)
        points, wm, wc = self._sigma(covariance)
        positions, quaternions = [], []
        for point in points:
            p = self.position+point[:3]
            q = quat_multiply(quat_exp(point[3:6]), self.quaternion)
            v = velocity+(point[6:] if len(point) == 12 else 0)
            p, q = integrate_pose(p, q, v, dt)
            positions.append(p)
            quaternions.append(q)
        p, q, P, _ = self._pose_statistics(np.asarray(positions), np.asarray(quaternions), wm, wc, self.quaternion)
        cfg = self.config
        Q = np.diag([cfg.process_position_std**2]*3+[cfg.process_rotation_std**2]*3)*dt
        self.position, self.quaternion, self.P = p, q, _covariance(P+Q, 6)
        self.timestamp = timestamp

    def update(self, position, quaternion):
        position = np.asarray(position, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("Invalid measured position")
        measured_q = quat_multiply(quaternion, [0, 0, 0, 1])
        points, wm, wc = self._sigma(self.P)
        positions = self.position+points[:, :3]
        quaternions = (Rotation.from_rotvec(points[:, 3:])*Rotation.from_quat(self.quaternion)).as_quat()
        mean_p, mean_q, obs_cov, errors = self._pose_statistics(positions, quaternions, wm, wc, self.quaternion)
        innovation = np.r_[position-mean_p, quat_difference(measured_q, mean_q)]
        cfg = self.config
        R = np.diag([cfg.measurement_position_std**2]*3+[cfg.measurement_rotation_std**2]*3)
        S = _covariance(obs_cov+R, 6)
        mahalanobis = float(innovation@np.linalg.solve(S, innovation))
        if cfg.gate_threshold is not None and mahalanobis > cfg.gate_threshold:
            return {"accepted": False, "mahalanobis_squared": mahalanobis}
        cross = np.einsum("n,ni,nj->ij", wc, points, errors)
        gain = np.linalg.solve(S, cross.T).T
        correction = gain@innovation
        new_position = self.position+correction[:3]
        new_quaternion = quat_multiply(quat_exp(correction[3:]), self.quaternion)
        posterior = _covariance(self.P-gain@S@gain.T, 6)
        # Recenter the posterior distribution onto the updated SO(3) chart.
        # Simply retaining a Euclidean covariance after a quaternion correction
        # ignores this reset. A second sigma transform transports it directly.
        residual_points, rw, rc = self._sigma(posterior)
        total = residual_points+correction
        recentered_q = (Rotation.from_rotvec(total[:, 3:])*Rotation.from_quat(self.quaternion)).as_quat()
        rotations = (Rotation.from_quat(recentered_q)*Rotation.from_quat(new_quaternion).inv()).as_rotvec()
        recentered = np.column_stack([residual_points[:, :3], rotations])
        mean_error = rw@recentered
        recentered -= mean_error
        P = np.einsum("n,ni,nj->ij", rc, recentered, recentered)
        self.position, self.quaternion, self.P = new_position, new_quaternion, _covariance(P, 6)
        return {"accepted": True, "mahalanobis_squared": mahalanobis}
