"""Generated RGB-D/event sequence must satisfy the tracker input contract."""
import json

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("moderngl")

from ev6d.data import Dataset
from ev6d.ycb_synthetic import generate_ycb_sequence


def test_generated_sequence_is_loadable_and_clock_aligned(tmp_path):
    # A self-contained textured object avoids any external YCB download in CI.
    pixels = np.indices((16, 16)).sum(axis=0) % 2
    Image.fromarray(np.uint8(50 + pixels * 180)).convert("RGB").save(tmp_path / "pattern.png")
    (tmp_path / "mesh.mtl").write_text("newmtl patterned\nmap_Kd pattern.png\n", encoding="utf-8")
    obj = tmp_path / "mesh.obj"
    obj.write_text("\n".join([
        "mtllib mesh.mtl", "usemtl patterned",
        "v 0 -.09 -.09", "v 0 .09 -.09", "v 0 .09 .09", "v 0 -.09 .09",
        "vt 0 0", "vt 1 0", "vt 1 1", "vt 0 1",
        "f 1/1 2/2 3/3 4/4"]), encoding="utf-8")
    root = tmp_path / "sequence"
    result = generate_ycb_sequence(root, obj, "003_cracker_box",
                                   duration=.04, width=64, height=48,
                                   contrast_threshold=.02)
    ds = Dataset(root)
    assert result["frames"] == 3
    assert result["poses"] == 1  # A 40-ms sequence has only the t=0 5-Hz observation.
    assert result["events"] > 0
    np.testing.assert_allclose([f["depth_t"] for f in ds.frames], [0., 1/60, 2/60])
    assert ds.events[0, 0] >= 0 and ds.events[-1, 0] <= .04
    with np.load(root / "ground_truth.npz") as gt:
        assert len(gt["t"]) == 21 and gt["t"][-1] == pytest.approx(.04)
        assert gt["velocity"].shape == (21, 6)
    for frame in ds.frames:
        depth = ds.load_depth(frame)
        mask = np.load(root / frame["target_mask_event"])
        assert mask.shape == depth.shape == (48, 64)
        assert mask.any() and np.isfinite(depth).all()
    meta = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    assert meta["simulation"]["target_mask"].endswith("oracle segmentation")
    assert meta["simulation"]["pose_observations"].endswith("NOT DOPE")
