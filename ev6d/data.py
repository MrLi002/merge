"""Explicit SI-unit data contract. The tracker never reads ground_truth.npz."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class PoseObservation:
    t: float
    position: np.ndarray
    quaternion: np.ndarray
    source: str
    available_at: float


def read_poses(path: Path) -> list[PoseObservation]:
    if not path.exists():
        return []
    result = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            t = float(row["t"])
            p = np.array([float(row[k]) for k in ("tx", "ty", "tz")])
            q = np.array([float(row[k]) for k in ("qx", "qy", "qz", "qw")])
            if not np.isfinite(np.r_[t, p, q]).all() or np.linalg.norm(q) < 1e-10:
                raise ValueError("Invalid pose observation")
            arrival = float(row.get("available_at", t))
            if not np.isfinite(arrival) or arrival < t - 1e-9:
                raise ValueError("available_at must be finite and no earlier than t")
            result.append(PoseObservation(t, p, q / np.linalg.norm(q), row["source"], arrival))
    return sorted(result, key=lambda x: x.available_at)


def write_poses(path: Path, rows: list[PoseObservation]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "tx", "ty", "tz", "qx", "qy", "qz", "qw", "source", "available_at"])
        for p in rows:
            writer.writerow([p.t, *p.position, *p.quaternion, p.source, p.available_at])


class Dataset:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.meta = json.loads((self.root / "dataset.json").read_text(encoding="utf-8"))
        if self.meta.get("schema_version") != 1:
            raise ValueError("Unsupported dataset schema_version (expected 1)")
        if self.meta["units"] != {"time": "s", "length": "m", "angle": "rad", "flow": "pixel/s"}:
            raise ValueError("Convert data to SI units before tracking")
        if self.meta.get("pose_convention") != "T_event_object; quaternion_xyzw; omega_event":
            raise ValueError("Unsupported pose convention")
        self.calibration = self.meta["calibration"]
        for name in ("event", "depth", "rgb"):
            cam = self.calibration[name]
            K = np.array(cam["K"], dtype=float)
            if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
                raise ValueError("Invalid camera intrinsics")
            if not np.allclose(K[2], [0, 0, 1]) or abs(K[0, 1]) > 1e-12 or abs(K[1, 0]) > 1e-12:
                raise ValueError("Only standard zero-skew pinhole intrinsics are supported")
            if any(not isinstance(cam[k], int) or isinstance(cam[k], bool) or cam[k] <= 0 for k in ("width", "height")):
                raise ValueError("Camera image dimensions must be positive integers")
            distortion = np.asarray(cam.get("distortion", []), dtype=float)
            if distortion.ndim != 1 or distortion.size not in (0, 4, 5, 8, 12, 14) or not np.isfinite(distortion).all():
                raise ValueError("Unsupported OpenCV pinhole distortion coefficients")
        for name in ("T_event_depth", "T_event_rgb"):
            T = np.array(self.calibration[name], dtype=float)
            if T.shape != (4, 4) or not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1]):
                raise ValueError("Invalid extrinsic transform")
            if not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-6) or np.linalg.det(T[:3, :3]) < 0:
                raise ValueError("Extrinsic rotation must be in SO(3)")
        self.events = np.load(self.root / self.meta["events"], allow_pickle=False)
        if self.events.ndim != 2 or self.events.shape[1] != 4 or not np.isfinite(self.events).all():
            raise ValueError("events must be finite N x 4 [t,x,y,p]")
        if len(self.events) and np.any(np.diff(self.events[:, 0]) < 0):
            raise ValueError("Event timestamps must be nondecreasing; sort explicitly upstream")
        if len(self.events):
            c = self.calibration["event"]
            xy = self.events[:, 1:3]
            if not np.all(xy == np.floor(xy)) or np.any(xy < 0) or np.any(xy[:, 0] >= c["width"]) or np.any(xy[:, 1] >= c["height"]):
                raise ValueError("Raw event coordinates must be integer pixels inside sensor")
            if not np.isin(self.events[:, 3], [-1, 0, 1]).all():
                raise ValueError("Polarity must be -1/+1 or 0/1")
        self.frames = []
        for original in self.meta.get("frames", []):
            frame = dict(original)
            frame["depth_t"] = float(frame["depth_t"])
            frame["depth_available_at"] = float(frame.get("depth_available_at", frame["depth_t"]))
            if not np.isfinite([frame["depth_t"], frame["depth_available_at"]]).all() or frame["depth_available_at"] < frame["depth_t"]:
                raise ValueError("Depth arrival must be finite and no earlier than capture")
            self.frames.append(frame)
        self.frames.sort(key=lambda f: f["depth_available_at"])
        if any(b["depth_t"] < a["depth_t"] for a, b in zip(self.frames, self.frames[1:])):
            raise ValueError("Out-of-order depth capture times are not supported")
        self.poses = read_poses(self.root / self.meta.get("poses", "poses.csv"))
        self.start = float(self.meta["start_time"])
        self.end = float(self.meta["end_time"])
        if not np.isfinite([self.start, self.end]).all() or self.end <= self.start:
            raise ValueError("Invalid dataset interval")
        initial = self.meta["initial_pose"]
        p, q = np.asarray(initial["position"], dtype=float), np.asarray(initial["quaternion"], dtype=float)
        if p.shape != (3,) or q.shape != (4,) or not np.isfinite(np.r_[p, q]).all() or np.linalg.norm(q) < 1e-10:
            raise ValueError("Invalid initial pose")
        model = self.meta.get("model")
        if model and model.get("type") == "cuboid":
            size = np.asarray(model["size"], dtype=float)
            if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
                raise ValueError("Cuboid size must contain three positive lengths in metres")
        if len(self.events) and (self.events[0, 0] < self.start - 1e-9 or self.events[-1, 0] > self.end + 1e-9):
            raise ValueError("Events outside dataset time interval")

    def load_depth(self, frame):
        depth = np.load(self.root / frame["depth"], allow_pickle=False).astype(float)
        cam = self.calibration["depth"]
        if depth.shape != (cam["height"], cam["width"]):
            raise ValueError("Depth image size differs from calibration")
        return depth


def rectify_events(events: np.ndarray, camera: dict) -> np.ndarray:
    """Rectification before triplet lattice matching; rounding is documented."""
    if not len(events) or not np.any(camera.get("distortion", [])):
        return events
    K = np.asarray(camera["K"], dtype=float)
    uv = cv2.undistortPoints(events[:, 1:3].astype(float).reshape(-1, 1, 2), K,
                             np.asarray(camera["distortion"], dtype=float), P=K).reshape(-1, 2)
    out = events.copy()
    out[:, 1:3] = np.rint(uv)
    valid = (out[:, 1] >= 0) & (out[:, 2] >= 0) & (out[:, 1] < camera["width"]) & (out[:, 2] < camera["height"])
    return out[valid]


def convert_dope(input_dir, timestamp_csv, output, object_name, length_unit, pose_frame, calibration):
    """Adapt official DOPE inference JSON objects/location/quaternion_xyzw.

    timestamp_csv explicitly maps file,t[,available_at]. No filename timestamp guesses.
    Units and camera frame are mandatory choices; this is not an inference backend.
    """
    if length_unit not in ("m", "cm", "mm") or pose_frame not in ("rgb", "event"):
        raise ValueError("Specify valid DOPE length units and pose frame")
    scale = {"m": 1.0, "cm": .01, "mm": .001}[length_unit]
    T = np.eye(4) if pose_frame == "event" else np.array(calibration["T_event_rgb"], dtype=float)
    observations = []
    with Path(timestamp_csv).open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            content = json.loads((Path(input_dir) / row["file"]).read_text(encoding="utf-8"))
            candidates = [o for o in content.get("objects", []) if o.get("class", o.get("name")) == object_name]
            if len(candidates) > 1:
                raise ValueError("Multiple matching instances: disambiguate the object upstream")
            if not candidates:
                continue
            obj = candidates[0]
            p = np.asarray(obj["location"], dtype=float) * scale
            q = np.asarray(obj["quaternion_xyzw"], dtype=float)
            if p.shape != (3,) or q.shape != (4,) or not np.isfinite(np.r_[p, q]).all() or np.linalg.norm(q) < 1e-10:
                raise ValueError("Invalid DOPE pose")
            p = T[:3, :3] @ p + T[:3, 3]
            q = (Rotation.from_matrix(T[:3, :3]) * Rotation.from_quat(q)).as_quat()
            t = float(row["t"])
            arrival = float(row.get("available_at", t))
            if not np.isfinite([t, arrival]).all() or arrival < t:
                raise ValueError("DOPE arrival must be finite and no earlier than capture")
            observations.append(PoseObservation(t, p, q, "dope_offline", arrival))
    observations.sort(key=lambda obs: obs.available_at)
    write_poses(Path(output), observations)
    return len(observations)
