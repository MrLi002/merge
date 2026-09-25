"""Validate generated YCB suite and make a compact image preview."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ev6d.data import Dataset


def validate(root):
    root = Path(root)
    suite = json.loads((root / "suite.json").read_text(encoding="utf-8"))
    entries = suite["sequences"]
    thumbnails, rows = [], []
    for entry in entries:
        # Derive from stable object/speed identifiers. This also accepts older
        # suite.json files that recorded an absolute Windows path.
        path = root / entry["object"] / entry["speed"]
        ds = Dataset(path)
        assert len(ds.events) == entry["events"]
        assert len(ds.frames) == entry["frames"]
        assert len(ds.poses) == entry["poses"]
        assert len(ds.frames) == 1 + int(np.floor(ds.end * 60 + 1e-9))
        assert len(ds.poses) == 1 + int(np.floor(ds.end * 5 + 1e-9))
        np.testing.assert_allclose([p.t for p in ds.poses], np.arange(len(ds.poses)) / 5)
        assert {p.source for p in ds.poses} == {"simulated_noisy_pose_NOT_DOPE"}
        with np.load(path / "ground_truth.npz") as gt:
            assert gt["position"].shape == (len(gt["t"]), 3)
            assert gt["quaternion"].shape == (len(gt["t"]), 4)
            assert gt["velocity"].shape == (len(gt["t"]), 6)
            assert gt["t"][0] == ds.start and gt["t"][-1] == ds.end
            start_angular_speed = float(np.linalg.norm(gt["velocity"][0, 3:]))
        mask_pixels = []
        for frame in ds.frames:
            depth = ds.load_depth(frame)
            mask = np.load(path / frame["target_mask_event"], allow_pickle=False)
            assert mask.dtype == bool and mask.shape == depth.shape
            assert np.isfinite(depth).all() and np.all(depth > 0)
            mask_pixels.append(int(mask.sum()))
        assert min(mask_pixels) > 0, f"Object left camera view: {path}"
        mid = ds.frames[len(ds.frames) // 2]
        with Image.open(path / mid["rgb"]) as im:
            thumb = im.convert("RGB")
            thumb.thumbnail((320, 240))
            thumbnails.append((entry["object"], entry["speed"], thumb.copy()))
        rows.append({"object": entry["object"], "speed": entry["speed"],
                     "events": len(ds.events), "frames": len(ds.frames), "poses": len(ds.poses),
                     "mask_pixels_min": min(mask_pixels), "mask_pixels_max": max(mask_pixels),
                     "initial_angular_speed_rad_s": start_angular_speed,
                     "start_time": ds.start, "end_time": ds.end})
        print(f"Validated {entry['object']}/{entry['speed']}", flush=True)
    by_object = {r["object"]: {} for r in rows}
    for row in rows:
        by_object[row["object"]][row["speed"]] = row
    for name, pair in by_object.items():
        if {"regular", "fast"} <= pair.keys():
            assert np.isclose(pair["fast"]["initial_angular_speed_rad_s"],
                              3 * pair["regular"]["initial_angular_speed_rad_s"], rtol=1e-9), name
    report = {"status": "passed", "sequences": len(rows), "events": sum(r["events"] for r in rows),
              "rows": rows}
    (root / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    canvas = Image.new("RGB", (2 * 330, 4 * 270), "#e9edf2")
    draw = ImageDraw.Draw(canvas)
    for i, (name, speed, im) in enumerate(thumbnails):
        x, y = (i % 2) * 330 + 5, (i // 2) * 270 + 5
        draw.text((x + 5, y), f"{name} / {speed}", fill="#17202a")
        canvas.paste(im, (x + 5, y + 22))
    canvas.save(root / "preview.png")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    args = parser.parse_args()
    result = validate(args.dataset)
    print(json.dumps({k: result[k] for k in ("status", "sequences", "events")}, indent=2))
