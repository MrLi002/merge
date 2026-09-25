"""Causal asynchronous runner. No imports from synthetic or evaluation modules."""
from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict
from pathlib import Path
from collections.abc import Mapping

import cv2
import numpy as np

from .data import Dataset, rectify_events
from .filters import PoseConfig, PoseUKF, VelocityConfig, VelocityKF
from .flow import FlowConfig, TripletFlow
from .geometry import register_depth
from .target import filter_target_events, model_target_mask


VARIANTS = ("full", "no_normal", "no_weight", "pose_only", "velocity_only")


def load_config(path=None):
    config = {"tick_s": .01, "max_depth_age_s": .05, "flow_warmup_s": .05, "roi_margin_px": 3, "roi_depth_margin_m": .08,
              "flow": asdict(FlowConfig()), "velocity": asdict(VelocityConfig()), "pose": asdict(PoseConfig())}
    if path is not None:
        user = dict(path) if isinstance(path, Mapping) else json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(user, dict):
            raise ValueError("Config must be a JSON object")
        for key, value in user.items():
            if key not in config:
                raise ValueError(f"Unknown config key: {key}")
            if isinstance(config[key], dict):
                if not isinstance(value, dict) or set(value) - set(config[key]):
                    raise ValueError(f"Unknown or invalid options in {key}")
                config[key].update(value)
            else:
                config[key] = value
    if not np.isfinite([config["tick_s"], config["max_depth_age_s"], config["flow_warmup_s"], config["roi_depth_margin_m"], config["roi_margin_px"]]).all():
        raise ValueError("Configuration values must be finite")
    if config["tick_s"] <= 0 or config["max_depth_age_s"] < 0 or config["flow_warmup_s"] < 0:
        raise ValueError("Invalid timing configuration")
    if config["roi_depth_margin_m"] < 0 or config["roi_margin_px"] < 0 or int(config["roi_margin_px"]) != config["roi_margin_px"]:
        raise ValueError("Invalid target ROI margin")
    FlowConfig(**config["flow"])
    VelocityConfig(**config["velocity"])
    PoseConfig(**config["pose"])
    return config


def run_tracking(dataset, output, config=None, variant="full"):
    if variant not in VARIANTS:
        raise ValueError(variant)
    cfg = load_config(config)
    if variant == "no_normal":
        cfg["velocity"]["normal_flow"] = False
    if variant == "no_weight":
        cfg["velocity"]["weighting"] = False
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    start_wall = time.perf_counter()
    ds = Dataset(dataset)
    c = ds.calibration
    cam = c["event"]
    K = np.array(cam["K"], dtype=float)
    h, w = cam["height"], cam["width"]
    events = rectify_events(ds.events, cam)
    flow_engine = TripletFlow(w, h, FlowConfig(**cfg["flow"]))
    velocity = VelocityKF(VelocityConfig(**cfg["velocity"]))
    velocity.timestamp = ds.start
    initial = ds.meta["initial_pose"]
    pose = PoseUKF(np.array(initial["position"]), np.array(initial["quaternion"]), ds.start, PoseConfig(**cfg["pose"]))
    # Actual observation arrival times are scheduled independently of KF ticks.
    pose_arrivals = [p.available_at for p in ds.poses if ds.start <= p.available_at <= ds.end]
    regular = np.arange(ds.start, ds.end, cfg["tick_s"])
    times = np.unique(np.r_[regular, ds.end, pose_arrivals])
    # Collapse floating-point duplicates (e.g. 0.6 vs 3*0.2) onto the later
    # instant, so no observation is consumed ahead of its actual timestamp.
    merged = []
    for timestamp in times:
        tolerance = 8*np.spacing(max(1., abs(float(timestamp))))
        if merged and timestamp-merged[-1] <= tolerance:
            merged[-1] = timestamp
        else:
            merged.append(timestamp)
    timeline = np.asarray(merged)
    i_event = i_pose = 0
    frame_index = -1
    depth = np.full((h, w), np.nan)
    depth_t = -np.inf
    flow_rows, records, diagnostics, latency = [], [], [], []
    counters = {"input_events": len(ds.events), "rectified_events": len(events), "target_events": 0,
                "flow_measurements": 0, "depth_valid_flows": 0, "stale_depth_batches": 0,
                "pose_updates": 0, "late_pose_rejected": 0, "pose_gate_rejected": 0, "warmup_flow_measurements": 0}
    timer = {"registration_s": 0., "flow_s": 0., "velocity_kf_s": 0., "pose_ukf_s": 0., "target_filter_s": 0.}
    previous_t = ds.start
    for t in timeline:
        batch_wall = time.perf_counter()
        # Use only depth captured no later than START of event collection interval.
        old_frame = frame_index
        while frame_index+1 < len(ds.frames) and ds.frames[frame_index+1]["depth_available_at"] <= previous_t:
            frame_index += 1
        if frame_index != old_frame:
            tr = time.perf_counter()
            f = ds.frames[frame_index]
            depth = register_depth(ds.load_depth(f), np.array(c["depth"]["K"]), K,
                                   np.array(c["T_event_depth"]), (h, w),
                                   dist_depth=np.array(c["depth"].get("distortion", [])), dist_event=None)
            depth_t = f["depth_t"]
            timer["registration_s"] += time.perf_counter()-tr
        begin_p, begin_q = pose.position.copy(), pose.quaternion.copy()
        tr = time.perf_counter()
        used_velocity = np.zeros(6) if variant == "pose_only" else velocity.x.copy()
        used_cov = None if variant == "pose_only" else velocity.P.copy()
        pose.predict_to(float(t), used_velocity, used_cov)
        timer["pose_ukf_s"] += time.perf_counter()-tr
        tr = time.perf_counter()
        velocity.predict_to(float(t))
        timer["velocity_kf_s"] += time.perf_counter()-tr
        j_event = int(np.searchsorted(events[:, 0], t, side="right"))
        batch = events[i_event:j_event]
        i_event = j_event
        if variant != "pose_only":
            tr = time.perf_counter()
            fresh = np.isfinite(depth_t) and t-depth_t <= cfg["max_depth_age_s"]+1e-10
            if not fresh:
                target = np.empty((0, 4))
                counters["stale_depth_batches"] += 1
            else:
                model = ds.meta.get("model")
                if model and model.get("type") == "cuboid":
                    mask = model_target_mask(begin_p, begin_q, model["size"], K, (h, w), depth,
                                             cfg["roi_margin_px"], cfg["roi_depth_margin_m"])
                elif frame_index >= 0 and "target_mask_event" in ds.frames[frame_index]:
                    # Supplied masks MUST be event-camera rectified coordinates.
                    mask = np.load(ds.root / ds.frames[frame_index]["target_mask_event"], allow_pickle=False).astype(bool)
                    if mask.shape != (h, w):
                        raise ValueError("Target mask size mismatch")
                    mask &= np.isfinite(depth) & (depth > 0)
                else:
                    raise ValueError("A known cuboid model or timestamped target masks are required")
                target = filter_target_events(batch, mask)
            timer["target_filter_s"] += time.perf_counter()-tr
            counters["target_events"] += len(target)
            tr = time.perf_counter()
            estimates = flow_engine.process(target, emit_time=float(t))
            timer["flow_s"] += time.perf_counter()-tr
            # Build history first: with an empty timestamp surface the slow
            # true motion may not yet have enough support while false fast
            # triplets already do. This configurable startup guard is an
            # engineering addition, not a parameter disclosed by the paper.
            candidates = [e for e in estimates if e.valid]
            valid = [e for e in candidates if e.quality.get("source_event_time", e.t)-ds.start >= cfg["flow_warmup_s"]]
            counters["warmup_flow_measurements"] += len(candidates)-len(valid)
            counters["flow_measurements"] += len(valid)
            if valid and fresh:
                uv = np.array([[e.x, e.y] for e in valid], dtype=float)
                z = depth[uv[:, 1].astype(int), uv[:, 0].astype(int)]
                f = np.array([e.flow for e in valid])
                keep = np.isfinite(z) & (z > 0)
                counters["depth_valid_flows"] += int(keep.sum())
                tr = time.perf_counter()
                detail = velocity.update(uv[keep], z[keep], f[keep], K)
                timer["velocity_kf_s"] += time.perf_counter()-tr
                detail = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in detail.items()}
                diagnostics.append({"t": float(t), "depth_age_s": float(t-depth_t), **detail})
                for e, d, ok in zip(valid, z, keep):
                    source_t = e.quality.get("source_event_time", e.t)
                    support_t = e.quality.get("oldest_support_time", source_t)
                    flow_rows.append([t, e.x, e.y, *e.flow, d, float(ok), source_t, support_t])
                    latency.append([t-source_t, t-support_t])
        while i_pose < len(ds.poses) and ds.poses[i_pose].available_at <= t+1e-10:
            obs = ds.poses[i_pose]
            i_pose += 1
            if variant == "velocity_only":
                continue
            # Explicitly reject delayed/OOS observations; no disguised time shift.
            if abs(obs.t-t) > 1e-8:
                counters["late_pose_rejected"] += 1
                continue
            tr = time.perf_counter()
            detail = pose.update(obs.position, obs.quaternion)
            timer["pose_ukf_s"] += time.perf_counter()-tr
            accepted = detail.get("accepted", True) if isinstance(detail, dict) else True
            counters["pose_updates" if accepted else "pose_gate_rejected"] += 1
        records.append(np.r_[t, pose.position, pose.quaternion, velocity.x, np.diag(pose.P)])
        counters.setdefault("batch_compute_s", []).append(time.perf_counter()-batch_wall)
        previous_t = float(t)
    processing_s = time.perf_counter()-start_wall
    batches = counters.pop("batch_compute_s")
    flows = np.asarray(flow_rows, dtype=float).reshape(-1, 9)
    arr = np.array(records)
    np.savez(output / "trajectory.npz", t=arr[:, 0], position=arr[:, 1:4], quaternion=arr[:, 4:8],
             velocity=arr[:, 8:14], pose_variance=arr[:, 14:20])
    np.save(output / "flows.npy", flows)
    with (output / "trajectory.csv").open("w", encoding="utf-8") as f:
        np.savetxt(f, arr[:, :14], delimiter=",", header="t,tx,ty,tz,qx,qy,qz,qw,vox,voy,voz,wx,wy,wz", comments="")
    statistics = {"variant": variant, **counters, "processing_s": processing_s,
                  "event_throughput_per_s": len(ds.events)/processing_s,
                  "target_event_throughput_per_s": counters["target_events"]/processing_s,
                  "sequence_duration_s": ds.end-ds.start, "processing_to_sequence_ratio": processing_s/(ds.end-ds.start),
                  "batch_compute_ms_p50": float(np.percentile(batches, 50)*1000),
                  "batch_compute_ms_p95": float(np.percentile(batches, 95)*1000),
                  "timings": timer,
                  "event_source_age_ms_p50": float(np.median(np.array(latency)[:, 0])*1000) if latency else None,
                  "event_source_age_ms_p95": float(np.percentile(np.array(latency)[:, 0], 95)*1000) if latency else None,
                  "triplet_support_age_ms_p95": float(np.percentile(np.array(latency)[:, 1], 95)*1000) if latency else None,
                  "pose_sources": sorted({p.source for p in ds.poses}), "python": platform.python_version(),
                  "platform": platform.platform(), "processor": platform.processor(), "numpy": np.__version__,
                  "latency_note": "Algorithmic source age and measured offline batch CPU time; not live sensor latency or realtime guarantee"}
    (output / "runtime.json").write_text(json.dumps(statistics, indent=2), encoding="utf-8")
    (output / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return statistics
