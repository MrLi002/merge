"""Offline evaluation only. The tracker must never import this module.

Ground truth is interpolated only within its recorded time support. Quaternion
errors use the SO(3) logarithm, so q and -q represent the same rotation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def _validate_trajectory(data, name):
    t, p, q = (np.asarray(data[k], dtype=float) for k in ("t", "position", "quaternion"))
    if t.ndim != 1 or len(t) < 2 or p.shape != (len(t), 3) or q.shape != (len(t), 4):
        raise ValueError(f"{name}: expected >=2 timestamps, Nx3 position, Nx4 quaternion")
    if not all(np.isfinite(x).all() for x in (t, p, q)) or np.any(np.diff(t) <= 0):
        raise ValueError(f"{name}: samples must be finite and timestamps strictly increasing")
    if np.any(np.linalg.norm(q, axis=1) < 1e-10):
        raise ValueError(f"{name}: invalid zero quaternion")
    return t, p, q


def evaluate_tracking(dataset, result, make_plot=True):
    dataset, result = Path(dataset), Path(result)
    if not (dataset / "ground_truth.npz").is_file():
        from .reporting import qualitative_report
        return qualitative_report(dataset, result, make_plot=make_plot)
    with np.load(dataset / "ground_truth.npz", allow_pickle=False) as archive:
        gt = dict(archive)
    with np.load(result / "trajectory.npz", allow_pickle=False) as archive:
        estimate = dict(archive)
    tg, pg, qg = _validate_trajectory(gt, "ground truth")
    te, pe, qe = _validate_trajectory(estimate, "estimate")
    valid = (te >= tg[0]) & (te <= tg[-1])
    if not np.any(valid):
        raise ValueError("No overlapping ground-truth and estimate timestamps")
    t, p, q = te[valid], pe[valid], qe[valid]
    reference_p = np.column_stack([np.interp(t, tg, pg[:, i]) for i in range(3)])
    reference_q = Slerp(tg, Rotation.from_quat(qg))(t)
    dp = p - reference_p
    dr = (Rotation.from_quat(q) * reference_q.inv()).as_rotvec()
    ep = np.linalg.norm(dp, axis=1)
    er = np.rad2deg(np.linalg.norm(dr, axis=1))
    metrics = {
        "samples": len(t), "excluded_outside_gt_support": int((~valid).sum()),
        "position_rmse_m": float(np.sqrt(np.mean(ep**2))),
        "position_axis_rmse_m": np.sqrt(np.mean(dp**2, axis=0)).tolist(),
        "rotation_rmse_deg": float(np.sqrt(np.mean(er**2))),
        "rotation_axis_rmse_deg": np.rad2deg(np.sqrt(np.mean(dr**2, axis=0))).tolist(),
        "position_median_m": float(np.median(ep)), "position_p95_m": float(np.percentile(ep, 95)),
        "rotation_median_deg": float(np.median(er)), "rotation_p95_deg": float(np.percentile(er, 95)),
        "note": "Offline comparison to supplied ground truth. Spatial twist v_O differs from object reference-point velocity; no world-motion claim.",
    }
    meta_path = dataset / "dataset.json"
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        metrics["dataset_purpose"] = metadata.get("purpose", "unspecified")
    if (result / "runtime.json").exists():
        runtime = json.loads((result / "runtime.json").read_text(encoding="utf-8"))
        metrics["experiment_mode"] = runtime.get("mode", runtime.get("variant", "unspecified"))
        metrics["pose_sources"] = runtime.get("pose_sources", [])
    if "velocity" in gt and "velocity" in estimate:
        metrics["velocity_evaluation_time_basis"] = "instantaneous_gt_interpolated_at_emitted_trajectory_timestamps"
        metrics["velocity_evaluation_note"] = (
            "Compares each emitted velocity with instantaneous GT at its trajectory timestamp. "
            "Interval optical flow and delayed availability can introduce lag; this metric "
            "includes that effect and does not use window-averaged GT velocities."
        )
        vg, ve = np.asarray(gt["velocity"]), np.asarray(estimate["velocity"])
        if vg.shape != (len(tg), 6) or ve.shape != (len(te), 6) or not np.isfinite(vg).all() or not np.isfinite(ve).all():
            raise ValueError("Velocity arrays must be finite Nx6 spatial twists")
        reference_v = np.column_stack([np.interp(t, tg, vg[:, i]) for i in range(6)])
        dv = ve[valid] - reference_v
        metrics["velocity_axis_rmse"] = np.sqrt(np.mean(dv**2, axis=0)).tolist()
        metrics["spatial_linear_velocity_rmse_m_s"] = float(np.sqrt(np.mean(np.sum(dv[:, :3]**2, axis=1))))
        metrics["angular_velocity_rmse_rad_s"] = float(np.sqrt(np.mean(np.sum(dv[:, 3:]**2, axis=1))))
        # Reference point is the object-coordinate origin; not necessarily its centroid.
        reference_linear = reference_v[:, :3] + np.cross(reference_v[:, 3:], reference_p)
        estimated_linear = ve[valid, :3] + np.cross(ve[valid, 3:], p)
        metrics["reference_point_linear_velocity_rmse_m_s"] = float(np.sqrt(
            np.mean(np.sum((estimated_linear-reference_linear)**2, axis=1))))
    metrics.update(_model_metrics(dataset, p, q, reference_p, reference_q.as_quat()))
    (result / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    np.savez(result / "errors.npz", t=t, position_error_m=dp, rotation_error_rad=dr)
    if make_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
        colors = ("#2878b5", "#d95f02", "#278441")
        for i, (label, color) in enumerate(zip(("x", "y", "z"), colors)):
            axes[0].plot(t, p[:, i], color=color, label=f"tracked {label}")
            axes[0].plot(t, reference_p[:, i], color=color, linestyle="--", alpha=.6, label=f"reference {label}")
        axes[0].set_ylabel("Position (m)")
        axes[0].legend(ncol=3, fontsize=8)
        axes[1].plot(t, ep*1000, color=colors[0])
        axes[1].set_ylabel("Position error (mm)")
        axes[2].plot(t, er, color=colors[1])
        axes[2].set_ylabel("Rotation error (deg)")
        axes[2].set_xlabel("Time (s)")
        for ax in axes:
            ax.grid(True, alpha=.2)
        fig.suptitle("Estimated trajectory and supplied ground truth")
        fig.savefig(result / "tracking.png", dpi=160)
        plt.close(fig)
    return metrics


def _model_metrics(dataset, p, q, reference_p, reference_q):
    """ADD only with explicit metre-valued model points and symmetry declaration."""
    path = dataset / "dataset.json"
    if not path.exists():
        return {}
    model = json.loads(path.read_text(encoding="utf-8")).get("evaluation_model", {})
    if not model:
        return {}
    if model.get("units") != "m" or model.get("metric") not in ("ADD", "ADD-S"):
        raise ValueError("evaluation_model requires units=m and metric=ADD or ADD-S")
    vertices = np.load(dataset / model["points"], allow_pickle=False)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError("evaluation_model.points must contain finite Nx3 object-frame vertices")
    from scipy.spatial import cKDTree
    distances = []
    for pp, qq, pg, qg in zip(p, q, reference_p, reference_q):
        a = Rotation.from_quat(qq).apply(vertices)+pp
        b = Rotation.from_quat(qg).apply(vertices)+pg
        error = cKDTree(b).query(a)[0] if model["metric"] == "ADD-S" else np.linalg.norm(a-b, axis=1)
        distances.append(float(error.mean()))
    return {model["metric"].lower().replace("-", "_")+"_mean_m": float(np.mean(distances)),
            "model_metric_point_count": len(vertices), "model_metric_convention": model}
