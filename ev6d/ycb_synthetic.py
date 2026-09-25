"""Independent YCB RGB-D/event benchmark inspired by the paper's data recipe.

This does not reproduce the authors' unpublished Unreal scenes, trajectories,
camera settings, DOPE detections, or measured real-sensor data. Ground truth is
written for offline evaluation and never supplied to the event/pose tracker.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .assets import YCB_OBJECTS, fetch_ycb
from .data import PoseObservation, write_poses
from .event_simulator import EventCameraSimulator
from .exposure import ExposureIntegrator
from .mesh_renderer import MeshRenderer
from .trajectories import MotionTrajectory


def _linear_to_srgb(rgb):
    value = np.clip(rgb, 0, 1)
    return np.where(value <= .0031308, 12.92 * value, 1.055 * value ** (1 / 2.4) - .055)


def _save_rgb(path, linear_rgb):
    srgb = np.rint(_linear_to_srgb(linear_rgb) * 255).astype(np.uint8)
    okay, encoded = cv2.imencode(".png", cv2.cvtColor(srgb, cv2.COLOR_RGB2BGR))
    if not okay:
        raise OSError(f"Could not write image: {path}")
    Path(path).write_bytes(encoded.tobytes())


def _luminance(rgb):
    return np.clip(rgb @ np.array([.2126, .7152, .0722], dtype=np.float32), 1e-4, None)


def _sample_times(duration, hz):
    count = int(np.floor(duration * hz + 1e-10))
    times = np.arange(1, count + 1, dtype=np.float64) / hz
    if not len(times) or duration - times[-1] > 1e-10:
        times = np.r_[times, duration]
    else:
        times[-1] = duration
    return times


def _camera(K, width, height):
    return {"width": width, "height": height, "K": K.tolist(), "distortion": [0., 0., 0., 0., 0.]}


def generate_ycb_sequence(root, obj_path, object_name, speed="regular", duration=1., seed=7,
                          width=640, height=480, render_hz=500., frame_hz=60., pose_hz=5.,
                          contrast_threshold=.2, threshold_sigma=0., refractory_s=0.,
                          supersample=1, pose_noise_m=.008, pose_noise_rad=.035):
    """Write a stand-alone ev6d Dataset with synchronized RGB-D, events, and GT.

    A 500-Hz clear-image stream drives both the event simulator and piecewise-
    linear full-duty 60-Hz exposure integration. RGB/depth use a known camera
    offset; depth and target masks are sampled at exposure end. Synthetic
    5-Hz noisy pose observations stand in for DOPE, and are labelled as such.
    """
    numbers = [duration, render_hz, frame_hz, pose_hz, contrast_threshold,
               threshold_sigma, refractory_s, pose_noise_m, pose_noise_rad]
    if (not np.isfinite(numbers).all() or min(numbers[:5]) <= 0 or
            min(numbers[5:]) < 0 or frame_hz > render_hz):
        raise ValueError("Invalid duration, rates, contrast, refractory, or pose noise")
    if speed not in ("regular", "fast") or object_name not in YCB_OBJECTS:
        raise ValueError("Unsupported YCB object or trajectory speed")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 16 for v in (width, height)):
        raise ValueError("width and height must be integers >= 16")
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty dataset directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "frames").mkdir()
    # Engineering intrinsics: ~56-degree horizontal view at any resolution.
    K = np.array([[.9 * width, 0., (width - 1) / 2],
                  [0., .9 * width, (height - 1) / 2], [0., 0., 1.]])
    T_event_rgb = np.eye(4)
    T_event_rgb[:3, 3] = [.018, -.004, .002]
    cam = _camera(K, width, height)
    calibration = {"event": cam, "rgb": dict(cam), "depth": dict(cam),
                   "T_event_rgb": T_event_rgb.tolist(), "T_event_depth": T_event_rgb.tolist()}
    # The raw YCB scans are Z-up. Choose a visible labelled side for each
    # object; keep this fixed transform inside the recorded trajectory.
    base_rotation = {"003_cracker_box": [0., np.pi / 2, 0.],
                     "005_tomato_soup_can": [-np.pi / 2, 0., 0.],
                     "006_mustard_bottle": [-np.pi / 2, 0., 0.],
                     "010_potted_meat_can": [-np.pi / 2, 0., 0.]}[object_name]
    trajectory = MotionTrajectory(seed=seed, speed=speed, base_rotation=base_rotation)
    rng = np.random.default_rng(seed + 11)
    frames, chunks = [], []
    # Keep all high-rate GT samples, including the exact end of the interval.
    sample_times = np.r_[0., _sample_times(duration, render_hz)]
    with MeshRenderer(obj_path, K, width, height, supersample) as renderer:
        def render_pair(t):
            p, q = trajectory.pose(float(t))
            event_rgb, _, event_mask = renderer.render(p, q)
            rgb, depth, _ = renderer.render(p, q, T_event_rgb)
            return event_rgb, rgb, depth, event_mask

        def write_frame(index, t, rgb, depth, event_mask, start):
            depth_name = f"frames/depth_{index:04d}.npy"
            rgb_name = f"frames/rgb_{index:04d}.png"
            mask_name = f"frames/mask_event_{index:04d}.npy"
            np.save(root / depth_name, depth)
            np.save(root / mask_name, event_mask)
            _save_rgb(root / rgb_name, rgb)
            frames.append({"depth_t": float(t), "depth": depth_name,
                           "rgb_t": float(t), "rgb": rgb_name,
                           "target_mask_event": mask_name,
                           "rgb_exposure_start": float(start),
                           "rgb_exposure_end": float(t)})

        first_event, first_rgb, first_depth, first_mask = render_pair(0.)
        write_frame(0, 0., first_rgb, first_depth, first_mask, 0.)
        event_camera = EventCameraSimulator(_luminance(first_event), 0.,
                                            contrast_threshold, threshold_sigma,
                                            refractory_s, seed + 23)
        exposure = ExposureIntegrator(first_rgb, 0., frame_hz)
        for index, t in enumerate(sample_times[1:], 1):
            event_rgb, clear_rgb, _, _ = render_pair(t)
            chunk = event_camera.process(_luminance(event_rgb), t)
            if len(chunk):
                chunks.append(chunk)
            for end, average in exposure.process(clear_rgb, t):
                # Exposure is temporally averaged; depth/mask are instantaneous
                # at its end and therefore never use a rounded 500-Hz sample.
                p, q = trajectory.pose(end)
                _, depth, _ = renderer.render(p, q, T_event_rgb)
                _, _, mask = renderer.render(p, q)
                write_frame(len(frames), end, average, depth, mask, end - 1 / frame_hz)
        model_info = renderer.model_info

    events = np.concatenate(chunks) if chunks else np.empty((0, 4), dtype=np.float64)
    np.save(root / "events.npy", events)
    # Observation cadence stays exactly at pose_hz; do not insert a final
    # off-grid measurement when a short sequence ends between two 5-Hz ticks.
    pose_times = np.arange(int(np.floor(duration * pose_hz + 1e-10)) + 1,
                           dtype=np.float64) / pose_hz
    observations = []
    for t in pose_times:
        p, q = trajectory.pose(t)
        noisy_p = p + rng.normal(0., pose_noise_m, 3)
        noisy_q = (Rotation.from_rotvec(rng.normal(0., pose_noise_rad, 3)) *
                   Rotation.from_quat(q)).as_quat()
        observations.append(PoseObservation(float(t), noisy_p, noisy_q,
                                            "simulated_noisy_pose_NOT_DOPE", float(t)))
    write_poses(root / "poses.csv", observations)
    gt = [trajectory.pose(t) for t in sample_times]
    np.savez(root / "ground_truth.npz", t=sample_times,
             position=np.array([p for p, _ in gt]),
             quaternion=np.array([q for _, q in gt]),
             velocity=np.array([trajectory.velocity(t) for t in sample_times]))
    p0, q0 = trajectory.pose(0.)
    meta = {
        "schema_version": 1,
        "purpose": "Independent YCB synthetic method verification; not the authors' Unreal/real benchmark",
        "object_name": object_name, "scenario": speed, "seed": seed,
        "units": {"time": "s", "length": "m", "angle": "rad", "flow": "pixel/s"},
        "pose_convention": "T_event_object; quaternion_xyzw; omega_event",
        "events": "events.npy", "poses": "poses.csv", "frames": frames,
        "start_time": 0., "end_time": float(duration),
        "initial_pose": {"position": p0.tolist(), "quaternion": q0.tolist(),
                         "source": "known synthetic initial pose"},
        "calibration": calibration,
        "model": {"type": "ycb_mesh", "name": object_name, "size": model_info["size"],
                  "raw_model_sha256": model_info["obj_sha256"],
                  "centering": model_info["centering_convention"]},
        "simulation": {
            "clear_render_hz": float(render_hz), "rgb_frame_hz": float(frame_hz),
            "depth_frame_hz": float(frame_hz), "pose_observation_hz": float(pose_hz),
            "event_model": "per_pixel_log_contrast_linear_time_crossings",
            "contrast_threshold_log": float(contrast_threshold),
            "threshold_sigma_log": float(threshold_sigma), "refractory_s": float(refractory_s),
            "luminance": "linear RGB Rec.709 weights; floor 1e-4",
            "exposure": exposure.metadata,
            "depth": "metric Z sampled at exposure end; background 3 m",
            "target_mask": "perfect synthetic instance mask in rectified event coordinates; oracle segmentation",
            "pose_observations": "GT plus independent Gaussian position/rotvec noise, zero latency; NOT DOPE",
            "pose_noise_std_m": float(pose_noise_m), "pose_noise_std_rad": float(pose_noise_rad),
            "trajectory": trajectory.metadata, "mesh_renderer": model_info,
            "unavailable_from_paper": ["Unreal scenes", "authors' trajectories", "original simulator parameters",
                                       "DOPE detections", "measured RealSense/event data"]}}
    (root / "dataset.json").write_text(json.dumps(meta, indent=2, allow_nan=False), encoding="utf-8")
    return {"object": object_name, "speed": speed, "events": int(len(events)),
            "frames": len(frames), "poses": len(observations), "duration_s": float(duration),
            "path": str(root.resolve())}


def generate_ycb_suite(output, assets, objects=YCB_OBJECTS, speeds=("regular", "fast"),
                       fetch=False, **kwargs):
    """Generate all requested object/speed combinations under output."""
    objects, speeds = tuple(objects), tuple(speeds)
    if not objects or not speeds or any(o not in YCB_OBJECTS for o in objects) or any(s not in ("regular", "fast") for s in speeds):
        raise ValueError("Select supported YCB object names and regular/fast speeds")
    assets, output = Path(assets), Path(output)
    if fetch:
        sources = fetch_ycb(assets, objects)
    else:
        source_file = assets / "sources.json"
        if not source_file.exists():
            raise FileNotFoundError(f"YCB assets missing; run fetch-ycb first: {source_file}")
        sources = json.loads(source_file.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for obj in objects:
        source = sources["objects"][obj]
        model = assets / source["model"]
        if not model.exists():
            raise FileNotFoundError(model)
        for speed in speeds:
            path = output / obj / speed
            print(f"Generating {obj}/{speed}...", flush=True)
            result = generate_ycb_sequence(path, model, obj, speed=speed, **kwargs)
            # Persist suite-relative paths so the generated directory can be
            # copied from Windows to a Linux server without rewriting JSON.
            result["path"] = (Path(obj) / speed).as_posix()
            result["source_archive_sha256"] = source["archive_sha256"]
            results.append(result)
            (output / "suite.json").write_text(json.dumps({
                "kind": "independent_ycb_synthetic_suite", "sequences": results,
                "source_index": sources["source_index"], "license": sources["license"],
                "disclaimer": "Paper-inspired independent synthetic dataset, not authors' original benchmark"
            }, indent=2), encoding="utf-8")
            print(f"Completed {obj}/{speed}: {result['events']} events, {result['frames']} frames", flush=True)
    return {"sequences": len(results), "events": sum(r["events"] for r in results),
            "path": str(output.resolve())}
