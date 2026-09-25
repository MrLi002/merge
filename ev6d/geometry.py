"""Geometry for T_event_object and spatial twist Xdot = vo + omega cross X.

Quaternions are xyzw. Rotation perturbations are expressed in camera axes and
left-multiply the nominal rotation. Optical flow is pixel/s, not displacement.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def skew(vector):
    x, y, z = np.asarray(vector, dtype=float)
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def _quaternion(q):
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError("Expected finite nonzero xyzw quaternion")
    return q / np.linalg.norm(q)


def quat_matrix(q):
    return Rotation.from_quat(_quaternion(q)).as_matrix()


def quat_multiply(a, b):
    return (Rotation.from_quat(_quaternion(a)) * Rotation.from_quat(_quaternion(b))).as_quat()


def quat_inverse(q):
    q = _quaternion(q).copy()
    q[:3] *= -1
    return q


def quat_exp(rotvec):
    vector = np.asarray(rotvec, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("Rotation vector must be finite and have length 3")
    return Rotation.from_rotvec(vector).as_quat()


def quat_log(q):
    return Rotation.from_quat(_quaternion(q)).as_rotvec()


def quat_difference(q, reference):
    return quat_log(quat_multiply(q, quat_inverse(reference)))


def quat_mean(quaternions, weights, initial=None):
    """Local manifold mean using left rotation errors (small UKF uncertainty)."""
    qs, weights = np.asarray(quaternions, dtype=float), np.asarray(weights, dtype=float)
    if qs.ndim != 2 or qs.shape[1] != 4 or weights.shape != (len(qs),) or not len(qs):
        raise ValueError("Quaternion mean requires Nx4 samples and N weights")
    if not np.isfinite(weights).all() or not np.isclose(weights.sum(), 1.):
        raise ValueError("Mean weights must be finite and sum to one")
    # SciPy operates on all samples at once, avoiding per-sigma-point calls.
    if not np.isfinite(qs).all() or np.any(np.linalg.norm(qs, axis=1) < 1e-12):
        raise ValueError("Invalid quaternion samples")
    rotations = Rotation.from_quat(qs)
    mean = Rotation.from_quat(_quaternion(qs[0] if initial is None else initial))
    for _ in range(60):
        delta = weights @ (rotations * mean.inv()).as_rotvec()
        if np.linalg.norm(delta) < 1e-11:
            return mean.as_quat()
        mean = Rotation.from_rotvec(delta) * mean
    raise ValueError("Quaternion mean did not converge; uncertainty exceeds local chart")


def _intrinsics(K):
    K = np.asarray(K, dtype=float)
    if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("Invalid pinhole intrinsics")
    if not np.allclose(K[2], [0, 0, 1]) or abs(K[0, 1]) > 1e-12 or abs(K[1, 0]) > 1e-12:
        raise ValueError("Only zero-skew pinhole intrinsics supported")
    return K


def project(points, K):
    points, K = np.asarray(points, dtype=float), _intrinsics(K)
    if points.shape[-1:] != (3,) or not np.isfinite(points).all() or np.any(points[..., 2] <= 0):
        raise ValueError("Project requires finite 3D points in front of camera")
    return points[..., :2]/points[..., 2, None] * [K[0, 0], K[1, 1]] + K[:2, 2]


def unproject(uv, depth, K):
    uv, z, K = np.asarray(uv, dtype=float), np.asarray(depth, dtype=float), _intrinsics(K)
    if uv.shape[-1:] != (2,) or not np.isfinite(uv).all() or not np.isfinite(z).all() or np.any(z <= 0):
        raise ValueError("Unproject requires finite pixels and positive Z depth")
    z = np.broadcast_to(z, uv.shape[:-1])
    xy = (uv - K[:2, 2]) / [K[0, 0], K[1, 1]]
    return np.concatenate([xy*z[..., None], z[..., None]], axis=-1)


def interaction_matrix(uv, depth, K):
    """Nx2x6 rigid-object projection Jacobian, derived from Xdot=vo+omega×X.

    The omega_x coefficient in the second row is negative. The printed positive
    coefficient in the target paper's Eq8 is incompatible with that convention.
    """
    uv, z, K = np.asarray(uv, dtype=float), np.asarray(depth, dtype=float), _intrinsics(K)
    if uv.ndim != 2 or uv.shape[1] != 2 or z.shape != (len(uv),):
        raise ValueError("Expected Nx2 pixels and N depths")
    if not np.isfinite(uv).all() or not np.isfinite(z).all() or np.any(z <= 0):
        raise ValueError("Interaction matrix requires finite pixels and positive depth")
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    a, b = uv[:, 0]-cx, uv[:, 1]-cy
    J = np.zeros((len(uv), 2, 6))
    J[:, 0, 0], J[:, 0, 2] = fx/z, -a/z
    J[:, 1, 1], J[:, 1, 2] = fy/z, -b/z
    J[:, 0, 3], J[:, 0, 4], J[:, 0, 5] = -a*b/fy, (fx*fx+a*a)/fx, -b*fx/fy
    J[:, 1, 3], J[:, 1, 4], J[:, 1, 5] = -(fy*fy+b*b)/fy, a*b/fx, a*fy/fx
    return J


def integrate_pose(position, quaternion, velocity, dt):
    """Exact constant spatial-twist integration, a refinement of paper Eq12."""
    p, v = np.asarray(position, dtype=float), np.asarray(velocity, dtype=float)
    if p.shape != (3,) or v.shape != (6,) or not np.isfinite(np.r_[p, v, dt]).all() or dt < 0:
        raise ValueError("Invalid pose/twist/time interval")
    q = _quaternion(quaternion)
    phi = v[3:]*dt
    angle = np.linalg.norm(phi)
    W = skew(phi)
    if angle < 1e-5:
        left_J = np.eye(3) + (.5-angle**2/24)*W + (1/6-angle**2/120)*(W@W)
    else:
        left_J = np.eye(3) + (1-np.cos(angle))/angle**2*W + (angle-np.sin(angle))/angle**3*(W@W)
    increment = Rotation.from_rotvec(phi)
    return increment.apply(p) + left_J@v[:3]*dt, (increment*Rotation.from_quat(q)).as_quat()


def register_depth(depth, K_depth, K_event, T_event_depth, shape, dist_depth=None, dist_event=None):
    """Backproject, transform, then project with nearest-pixel Z buffering.

    Depth is camera Z in metres. Occluded points lose to the nearest positive
    transformed Z; unobserved pixels remain NaN. This does not fill holes.
    """
    depth = np.asarray(depth, dtype=float)
    kd, ke = _intrinsics(K_depth), _intrinsics(K_event)
    T = np.asarray(T_event_depth, dtype=float)
    if depth.ndim != 2 or len(shape) != 2 or any(int(v) != v or v <= 0 for v in shape):
        raise ValueError("Invalid depth/output image dimensions")
    if T.shape != (4, 4) or not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1]):
        raise ValueError("Invalid depth-to-event transform")
    if not np.allclose(T[:3, :3].T@T[:3, :3], np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(T[:3, :3]), 1.):
        raise ValueError("Depth extrinsic rotation must belong to SO(3)")
    h, w = map(int, shape)
    output = np.full(h*w, np.inf)
    y, x = np.nonzero(np.isfinite(depth) & (depth > 0))
    if not len(x):
        return np.full((h, w), np.nan)
    uv = np.column_stack([x, y]).astype(float)
    if dist_depth is not None and np.any(dist_depth):
        uv = cv2.undistortPoints(uv.reshape(-1, 1, 2), kd, np.asarray(dist_depth, dtype=float), P=kd).reshape(-1, 2)
    points = unproject(uv, depth[y, x], kd) @ T[:3, :3].T + T[:3, 3]
    points = points[points[:, 2] > 1e-8]
    if len(points):
        if dist_event is not None and np.any(dist_event):
            pixels = cv2.projectPoints(points, np.zeros(3), np.zeros(3), ke, np.asarray(dist_event, dtype=float))[0].reshape(-1, 2)
        else:
            pixels = project(points, ke)
        inside = np.isfinite(pixels).all(axis=1) & (pixels[:, 0] >= -.5) & (pixels[:, 0] < w-.5) & (pixels[:, 1] >= -.5) & (pixels[:, 1] < h-.5)
        pixels = np.rint(pixels[inside]).astype(int)
        np.minimum.at(output, pixels[:, 1]*w+pixels[:, 0], points[inside, 2])
    output[~np.isfinite(output)] = np.nan
    return output.reshape(h, w)
