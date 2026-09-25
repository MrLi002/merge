"""Deterministic ray-cast RGB-D + log-contrast event simulation.

This is a method validation dataset, NOT the paper's Unreal/YCB dataset.
No ground-truth flow or velocity is passed to the event tracker.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .data import PoseObservation, write_poses


SCENARIOS = ("static", "translation", "rotation", "mixed", "event_gap", "pose_gap", "outliers")
SIZE = np.array([.58, .44, .30])


def trajectory(t, scenario="mixed"):
    """T_event_object; only generator and evaluator may call this function."""
    p = np.array([0., 0., 1.15])
    r = np.array([.13, -.25, .08])
    if scenario not in ("static", "rotation"):
        p += [.15 * np.sin(2*np.pi*.8*t), .065 * np.sin(2*np.pi*.6*t), .055 * np.sin(2*np.pi*.45*t)]
    if scenario not in ("static", "translation"):
        r += [.20 * np.sin(2*np.pi*.7*t), .24 * np.sin(2*np.pi*.6*t), .30 * np.sin(2*np.pi*.8*t)]
    return p, Rotation.from_rotvec(r).as_quat()


def true_velocity(t, scenario):
    """Diagnostic oracle isolated from all estimation code."""
    eps = 1e-5
    p, q = trajectory(t, scenario)
    p1, q1 = trajectory(t + eps, scenario)
    p0, q0 = trajectory(t - eps, scenario)
    omega = (Rotation.from_quat(q1) * Rotation.from_quat(q0).inv()).as_rotvec() / (2*eps)
    return np.r_[(p1-p0)/(2*eps) - np.cross(omega, p), omega]


def render(t, scenario, K, width, height, T_event_camera=None):
    """Ray/box intersections, analytic continuous texture, true surface z depth."""
    p, q = trajectory(t, scenario)
    R = Rotation.from_quat(q).as_matrix()
    T = np.eye(4) if T_event_camera is None else T_event_camera
    yy, xx = np.indices((height, width))
    rays_camera = np.stack([(xx-K[0, 2])/K[0, 0], (yy-K[1, 2])/K[1, 1], np.ones_like(xx)], -1)
    rays = rays_camera @ T[:3, :3].T @ R
    origin = (T[:3, 3]-p) @ R
    with np.errstate(divide="ignore", invalid="ignore"):
        ta = (-SIZE/2 - origin) / rays
        tb = (SIZE/2 - origin) / rays
    lo = np.minimum(ta, tb)
    hi = np.maximum(ta, tb)
    entry = lo.max(axis=-1)
    leave = hi.min(axis=-1)
    valid = (entry > 0) & (entry <= leave)
    points = origin + np.where(valid, entry, 0)[..., None] * rays
    face = np.argmax(lo, axis=-1)
    # Spatially varying edge orientations; smooth texture limits aliasing.
    a = np.where(face == 0, points[..., 2], points[..., 0])
    b = np.where(face == 1, points[..., 2], points[..., 1])
    texture = .48 + .18*np.sin(67*a+15*b) + .14*np.sin(13*a-79*b) + .08*np.cos(105*a+67*b)
    background = .25 + .025*np.sin(xx*.13)*np.sin(yy*.16)
    intensity = np.where(valid, np.clip(texture, .06, .95), background)
    depth = np.where(valid, entry, 3.0)
    # Colors only for readable overlays; event generation uses scalar intensity.
    rgb = np.stack([intensity*.86, intensity, intensity*.70], -1)
    return intensity, depth.astype(np.float32), (np.clip(rgb, 0, 1)*255).astype(np.uint8)


def generate_dataset(root, scenario="mixed", duration=1.2, seed=7, render_hz=1000, width=160, height=120):
    if scenario not in SCENARIOS or not np.isfinite([duration, render_hz]).all() or duration <= 0 or render_hz <= 0:
        raise ValueError("Invalid synthetic scenario/duration/render_hz")
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 16 for v in (width, height)):
        raise ValueError("Synthetic width and height must be integers >= 16")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "frames").mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    K = np.array([[180.*width/160, 0., (width-1)/2], [0., 184.*height/120, (height-1)/2], [0., 0., 1.]])
    T_event_rgb = np.eye(4)
    T_event_rgb[:3, 3] = [.018, -.004, .002]
    cam = {"width": width, "height": height, "K": K.tolist(), "distortion": [0., 0., 0., 0., 0.]}
    calibration = {"event": cam, "rgb": dict(cam), "depth": dict(cam),
                   "T_event_rgb": T_event_rgb.tolist(), "T_event_depth": T_event_rgb.tolist()}
    contrast = .20
    image, _, _ = render(0, scenario, K, width, height)
    previous = np.log(image)
    reference = previous.copy()
    chunks = []
    times = np.linspace(0., duration, int(np.ceil(duration*render_hz))+1)
    for prev_t, t in zip(times[:-1], times[1:]):
        image, _, _ = render(t, scenario, K, width, height)
        current = np.log(image)
        delta = current - reference
        counts = np.floor(np.abs(delta)/contrast).astype(int)
        ys, xs = np.nonzero(counts)
        if len(xs):
            polarity = np.sign(delta[ys, xs])
            max_count = int(counts.max())
            for crossing in range(1, max_count+1):
                keep = counts[ys, xs] >= crossing
                y, x, sign = ys[keep], xs[keep], polarity[keep]
                threshold = reference[y, x] + sign*contrast*crossing
                difference = current[y, x]-previous[y, x]
                frac = np.divide(threshold-previous[y, x], difference, out=np.ones_like(threshold), where=np.abs(difference)>1e-12)
                event_t = prev_t + np.clip(frac, 0, 1)*(t-prev_t)
                chunks.append(np.column_stack([event_t, x, y, sign]))
            reference[ys, xs] += polarity*contrast*counts[ys, xs]
        previous = current
    events = np.concatenate(chunks) if chunks else np.empty((0, 4))
    events = events[np.argsort(events[:, 0], kind="stable")]
    gap = [duration*.32, duration*.66]
    if scenario == "event_gap":
        events = events[(events[:, 0] < gap[0]) | (events[:, 0] > gap[1])]
    if scenario == "outliers":
        n = max(500, int(len(events)*.10))
        noise = np.column_stack([rng.uniform(0, duration, n), rng.integers(0, width, n),
                                 rng.integers(0, height, n), rng.choice([-1, 1], n)])
        events = np.concatenate([events, noise])
        events = events[np.argsort(events[:, 0], kind="stable")]
    np.save(root / "events.npy", events)
    frames = []
    for i, t in enumerate(np.arange(0, duration+1e-9, 1/60)):
        _, depth, rgb = render(t, scenario, K, width, height, T_event_rgb)
        # Invalid depth pixels explicitly test geometric registration gaps.
        depth[rng.random(depth.shape) < .006] = np.nan
        dn, rn = f"frames/depth_{i:04d}.npy", f"frames/rgb_{i:04d}.png"
        np.save(root / dn, depth)
        cv2.imwrite(str(root / rn), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        frames.append({"depth_t": float(t), "rgb_t": float(t), "depth": dn, "rgb": rn})
    poses = []
    for i, t in enumerate(np.arange(0, duration+1e-9, .2)):
        if scenario == "pose_gap" and gap[0] <= t <= duration*.85:
            continue
        p, q = trajectory(t, scenario)
        p = p + rng.normal(0, .008, 3)
        q = (Rotation.from_rotvec(rng.normal(0, .035, 3)) * Rotation.from_quat(q)).as_quat()
        if scenario == "outliers" and i == 3:
            p += [.35, -.2, .15]
            q = (Rotation.from_rotvec([.6, -.4, .2]) * Rotation.from_quat(q)).as_quat()
        poses.append(PoseObservation(float(t), p, q, "simulated_noisy_pose_NOT_DOPE", float(t)))
    write_poses(root / "poses.csv", poses)
    p0, q0 = trajectory(0, scenario)
    meta = {"schema_version": 1, "scenario": scenario, "seed": seed,
            "purpose": "Synthetic method verification, not reproduction of the paper's reported numbers",
            "units": {"time": "s", "length": "m", "angle": "rad", "flow": "pixel/s"},
            "pose_convention": "T_event_object; quaternion_xyzw; omega_event",
            "events": "events.npy", "poses": "poses.csv", "start_time": 0., "end_time": duration,
            "initial_pose": {"position": p0.tolist(), "quaternion": q0.tolist(), "source": "known synthetic initial pose"},
            "model": {"type": "cuboid", "size": SIZE.tolist()}, "frames": frames, "calibration": calibration,
            "simulation": {"render_hz": render_hz, "contrast_threshold": contrast, "depth_hz": 60, "pose_hz": 5,
                           "event_gap": gap if scenario == "event_gap" else None,
                           "pose_noise_std_m": .008, "pose_noise_std_rad": .035,
                           "camera_baseline_nonidentity": True}}
    (root / "dataset.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    gt_t = np.unique(np.r_[np.arange(0, duration, .005), duration])
    gt = [trajectory(t, scenario) for t in gt_t]
    np.savez(root / "ground_truth.npz", t=gt_t, position=np.array([a for a, _ in gt]),
             quaternion=np.array([b for _, b in gt]), velocity=np.array([true_velocity(t, scenario) for t in gt_t]))
    return {"scenario": scenario, "events": len(events), "duration_s": duration, "path": str(root)}
