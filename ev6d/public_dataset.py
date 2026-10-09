"""Adapter for the public UE/YCB sequence used by Hybrid Object Tracking.

Ground-truth annotations and velocity.csv are NEVER input to the tracker.
All positions are meters, all times seconds, quaternions xyzw.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re

import cv2
import numpy as np
from scipy.spatial.transform import Rotation



@dataclass(frozen=True)
class Frame:
    stem: str
    t: float
    rgb_path: Path
    depth_path: Path
    mask_path: Path | None


@dataclass(frozen=True)
class PoseObservation:
    t: float
    available_at: float
    position: np.ndarray
    quaternion: np.ndarray
    source: str = "dope"


class PublicYCBSequence:
    """Load the published photorealistic1 directory, without custom conversion.

    Event log is imported with bimvee (IIT-YARP). The event clock and frame
    clock are relative to their respective recording starts; additional offset
    must be specified if the recording clocks differ. RGB, depth, and event
    pixels are treated as aligned ONLY when assume_coregistered is explicitly
    enabled. Instance/semantic mask is dataset annotation, not a detector.
    """

    def __init__(self, path, *, event_time_offset_s=0., timestamp_scale=1.,
                 assume_coregistered=False, mask_kind="instance"):
        path = Path(path).expanduser().resolve()
        self.root = path.parent if path.name == "photorealistic1" else path
        self.frames_dir = self.root / "photorealistic1"
        if not self.frames_dir.is_dir():
            raise FileNotFoundError(self.frames_dir)
        self.filter_params = self._json(self.root / "filter_param.json")
        cam = self._json(self.frames_dir / "_camera_settings.json")
        self.object_settings = self._json(self.frames_dir / "_object_settings.json")
        intrinsic = cam["camera_settings"][0]["intrinsic_settings"]
        self.K = np.array([[float(intrinsic["fx"]), 0., float(intrinsic["cx"])],
                           [0., float(intrinsic["fy"]), float(intrinsic["cy"])],
                           [0., 0., 1.]])
        self.mask_kind = mask_kind
        self.assume_coregistered = bool(assume_coregistered)
        if not self.assume_coregistered:
            raise ValueError(
                "Explicit --assume-coregistered is required for the UE/YCB "
                "rendering: verify RGB, depth and events share the same pixels."
            )
        if not np.isfinite([event_time_offset_s, timestamp_scale]).all() or timestamp_scale <= 0:
            raise ValueError("Invalid event-time offset or timestamp scale")
        images = sorted((p for p in self.frames_dir.glob("*.png")
                         if re.fullmatch(r"[^.]+\.png", p.name)), key=lambda p: p.name)
        if len(images) < 2:
            raise ValueError("Cannot find >=2 RGB frames named xx.png")
        frames = []
        for image in images:
            stem = image.stem
            annotation = self._json(self.frames_dir / (stem + ".json"))
            raw_t = float(annotation["timestamp"]) * timestamp_scale
            depth = self.frames_dir / (stem + ".depth.mm.16.png")
            if not depth.is_file():
                raise FileNotFoundError(depth)
            ext = "is" if mask_kind == "instance" else "cs"
            mask = self.frames_dir / (stem + "." + ext + ".png")
            frames.append(Frame(stem, raw_t, image, depth, mask if mask.is_file() else None))
        origin = frames[0].t
        self.frames = [Frame(f.stem, f.t-origin, f.rgb_path, f.depth_path, f.mask_path) for f in frames]
        if not np.all(np.diff([f.t for f in self.frames]) > 0):
            raise ValueError("RGB annotation timestamps must increase")
        first = cv2.imread(str(frames[0].depth_path), cv2.IMREAD_UNCHANGED)
        if first is None or first.ndim != 2 or first.dtype != np.uint16:
            raise ValueError("Expected single-channel 16-bit depth (millimetres)")
        self.shape = first.shape
        if cv2.imread(str(images[0]), cv2.IMREAD_COLOR).shape[:2] != self.shape:
            raise ValueError("RGB/depth image sizes mismatch; require explicit reprojection")
        self.start, self.end = 0., float(self.frames[-1].t)
        self.events = self._load_events(event_time_offset_s)
        if (not len(self.events) or not np.isfinite(self.events).all()
                or np.diff(self.events[:, 0]).min(initial=0) < 0):
            raise ValueError("Empty or unsorted events")
        if self.events[-1, 0] < 0 or self.events[0, 0] > self.end:
            raise ValueError("No overlap between event and RGB clocks; check offset")

    @staticmethod
    def _json(path):
        if not path.is_file():
            raise FileNotFoundError(path)
        return json.loads(path.read_text(encoding="utf-8"))

    def _load_events(self, offset):
        try:
            import bimvee
        except ImportError as error:
            raise ImportError("Install bimvee (pip install bimvee) to read binaryevents.log") from error
        log = self.frames_dir / "binaryevents.log"
        if not log.is_file():
            raise FileNotFoundError(log)
        # IIT-YARP import normally expects the recording parent directory.
        payload = bimvee.importIitYarp.importIitYarp(filePathOrName=str(self.root))
        if isinstance(payload, list):
            payload = payload[0]
        cameras = payload["data"]
        dvs = (cameras["left"] if "left" in cameras else next(iter(cameras.values())))["dvs"]
        ts = np.asarray(dvs["ts"], dtype=float)
        x = np.asarray(dvs["x"], dtype=float)
        y = np.asarray(dvs["y"], dtype=float)
        pol = np.asarray(dvs["pol"], dtype=np.uint8)
        if not (len(ts) == len(x) == len(y) == len(pol)):
            raise ValueError("Inconsistent event arrays")
        events = np.column_stack([ts + float(offset), x, y, pol])
        h, w = self.shape
        if np.any((x < 0) | (x >= w) | (y < 0) | (y >= h)):
            raise ValueError("Event pixels outside image; camera resolution mismatch")
        return events

    def windows(self, window_s):
        """Adjacent E-RAFT pairs [t-dt,t), [t,t+dt), output at t+dt."""
        if not 0 < window_s <= self.end / 2:
            raise ValueError("Sequence too short for E-RAFT's two event windows")
        count = int(np.floor((self.end-self.start)/window_s+1e-9))
        for index in range(1, count):
            start = self.start+index*window_s
            yield (start, start+window_s)

    def event_pair(self, start, end):
        dt = end-start
        idx = np.searchsorted(self.events[:, 0], [start-dt, start, end], side="left")
        return self.events[idx[0]:idx[1]], self.events[idx[1]:idx[2]]

    def source_geometry(self, timestamp, *, max_age_s=.03):
        """Use a depth/mask captured no later than the flow *source* time."""
        prior = [frame for frame in self.frames if frame.t <= timestamp+1e-10]
        h, w = self.shape
        if not prior:
            return np.full((h,w), np.nan), np.zeros((h,w), bool)
        frame = prior[-1]
        if timestamp-frame.t > max_age_s+1e-10:
            return np.full((h,w), np.nan), np.zeros((h,w), bool)
        raw = cv2.imread(str(frame.depth_path), cv2.IMREAD_UNCHANGED)
        if raw is None or raw.dtype != np.uint16 or raw.shape != self.shape:
            raise ValueError("Invalid depth frame " + str(frame.depth_path))
        depth = raw.astype(np.float64)/1000.
        if frame.mask_path is None:
            raise FileNotFoundError("Source-time target mask is missing: " + frame.stem)
        segmentation = cv2.imread(str(frame.mask_path), cv2.IMREAD_UNCHANGED)
        if segmentation is None or segmentation.shape[:2] != self.shape:
            raise ValueError("Invalid segmentation image")
        mask = (segmentation != 0) if segmentation.ndim == 2 else np.any(segmentation != 0, axis=2)
        return depth, (mask & np.isfinite(depth) & (depth > 0))

    def dope_observations(self, *, path=None, json_dir=None, translation_unit="m",
                          delay_s=0., T_event_rgb=None):
        """Read DOPE estimates, NOT the per-frame GT .json annotations.

        txt columns: x y z axis_x axis_y axis_z angle_rad (7); or
        t x y z axis_x axis_y axis_z angle_rad (8).
        Seven-column files must match the RGB frame count exactly.
        Alternatively use DOPE's <rgb_stem>.json output in json_dir.
        """
        scale = {"m": 1., "cm": .01, "mm": .001}.get(translation_unit)
        if scale is None or not np.isfinite(delay_s) or delay_s < 0:
            raise ValueError("Specify translation unit m/cm/mm and nonnegative DOPE delay")
        T = np.eye(4) if T_event_rgb is None else np.asarray(T_event_rgb, dtype=float)
        if T.shape != (4,4) or not np.allclose(T[3], [0,0,0,1]):
            raise ValueError("Invalid T_event_rgb")
        rotation = Rotation.from_matrix(T[:3,:3])
        measurements = []
        if json_dir is not None:
            directory = Path(json_dir)
            for frame in self.frames:
                path_json = directory / (frame.stem+".json")
                if not path_json.is_file():
                    continue
                objs = self._json(path_json).get("objects", [])
                if len(objs) != 1:
                    continue  # missing or ambiguous detection is NOT ground truth
                obj = objs[0]
                if obj.get("location") is None or obj.get("quaternion_xyzw") is None:
                    continue
                p = np.asarray(obj["location"], float)*scale
                q = Rotation.from_quat(np.asarray(obj["quaternion_xyzw"], float))
                measurements.append((frame.t,p,q))
        else:
            source = Path(path) if path is not None else self.root/"dope_poses_ycb.txt"
            if not source.is_file():
                raise FileNotFoundError(
                    f"Missing {source}; run NVIDIA DOPE on RGB images first. "
                    "Dataset xx.json is ground truth and cannot replace DOPE."
                )
            rows=[]
            for line in source.read_text(encoding="utf-8").splitlines():
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                try:
                    row=np.array([float(v) for v in line.replace(","," ").split()], dtype=float)
                except ValueError as error:
                    raise ValueError("DOPE txt must contain numeric pose rows") from error
                if len(row) not in (7,8):
                    raise ValueError("DOPE txt row must contain 7 or 8 numbers")
                rows.append(row)
            if not rows:
                raise ValueError("No DOPE observations found")
            if any(len(r)!=len(rows[0]) for r in rows):
                raise ValueError("Inconsistent DOPE txt columns")
            if len(rows[0])==7 and len(rows)!=len(self.frames):
                raise ValueError(
                    "Untimestamped DOPE txt must contain one row per RGB frame; "
                    "supply 8-column timestamped rows for sparse detections"
                )
            for i,row in enumerate(rows):
                t=float(row[0]) if len(row)==8 else self.frames[i].t
                values=row[1:] if len(row)==8 else row
                axis=values[3:6]
                angle=float(values[6])
                norm=np.linalg.norm(axis)
                if norm < 1e-12 and abs(angle)>1e-12:
                    raise ValueError("Nonzero DOPE angle requires a rotation axis")
                q=Rotation.from_rotvec(np.zeros(3) if norm<1e-12 else axis/norm*angle)
                measurements.append((t,values[:3]*scale,q))
        result=[]
        for t,p,q in measurements:
            if not np.isfinite(np.r_[t,p,q.as_quat()]).all() or p.shape!=(3,):
                raise ValueError("Nonfinite DOPE pose")
            result.append(PoseObservation(float(t),float(t+delay_s),
                          rotation.apply(p)+T[:3,3], (rotation*q).as_quat()))
        result.sort(key=lambda obs:(obs.available_at,obs.t))
        return result
