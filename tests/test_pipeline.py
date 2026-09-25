"""Causality/data contracts and a complete synthetic run, without accuracy tuning."""
import json
import subprocess
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ev6d.data import Dataset, convert_dope, read_poses
from ev6d.evaluation import evaluate_tracking
from ev6d.pipeline import load_config, run_tracking
from ev6d.synthetic import generate_dataset


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "dataset"
    generate_dataset(root, "static", duration=.053, width=32, height=24, render_hz=200)
    return root


def test_tracking_never_requires_ground_truth_and_preserves_endpoint(dataset, tmp_path):
    (dataset / "ground_truth.npz").unlink()
    result = tmp_path / "result"
    stats = run_tracking(dataset, result)
    with np.load(result / "trajectory.npz") as track:
        assert track["t"][-1] == .053
        assert np.all(np.diff(track["t"]) > 0)
        assert np.isfinite(track["position"]).all()
        np.testing.assert_allclose(np.linalg.norm(track["quaternion"], axis=1), 1., atol=1e-12)
        assert np.all(track["pose_variance"] >= 0)
    assert stats["input_events"] == 0
    assert stats["pose_updates"] == 1
    assert np.load(result / "flows.npy").shape == (0, 9)


def test_partial_configuration_merges_without_mutating_input():
    overrides = {"tick_s": .02, "velocity": {"normal_flow": False}}
    original = json.dumps(overrides)
    result = load_config(overrides)
    assert result["tick_s"] == .02 and result["velocity"]["normal_flow"] is False
    assert "flow" in result and "weighting" in result["velocity"]
    assert json.dumps(overrides) == original
    for bad in ({"unknown": 3}, {"flow": {"typo": 1}}, {"tick_s": float("nan")}, {"roi_margin_px": -.1}):
        with pytest.raises(ValueError):
            load_config(bad)


def test_late_pose_and_unavailable_depth_are_not_used(dataset, tmp_path):
    poses = read_poses(dataset / "poses.csv")
    from ev6d.data import write_poses
    poses[0].available_at = .025
    write_poses(dataset / "poses.csv", poses)
    meta = json.loads((dataset / "dataset.json").read_text())
    for frame in meta["frames"]:
        frame["depth_available_at"] = frame["depth_t"] + 1.
    (dataset / "dataset.json").write_text(json.dumps(meta))
    stats = run_tracking(dataset, tmp_path / "late")
    assert stats["late_pose_rejected"] == 1
    assert stats["pose_updates"] == 0
    assert stats["depth_valid_flows"] == 0
    assert stats["stale_depth_batches"] > 0


def test_dataset_rejects_nan_interval(dataset):
    meta = json.loads((dataset / "dataset.json").read_text())
    meta["end_time"] = float("nan")
    (dataset / "dataset.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="interval"):
        Dataset(dataset)


def test_dope_units_and_extrinsic_conversion(tmp_path):
    source = tmp_path / "json"
    source.mkdir()
    (source / "frame.json").write_text(json.dumps({"objects": [{"class": "box", "location": [100, 0, 50], "quaternion_xyzw": [0, 0, 0, 1]}]}))
    timestamps = tmp_path / "timestamps.csv"
    timestamps.write_text("file,t,available_at\nframe.json,1.0,1.1\n")
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    transform[:3, 3] = [.1, .2, .3]
    output = tmp_path / "nested" / "poses.csv"
    count = convert_dope(source, timestamps, output, "box", "cm", "rgb", {"T_event_rgb": transform})
    assert count == 1
    obs = read_poses(output)[0]
    np.testing.assert_allclose(obs.position, [.1, 1.2, .8], atol=1e-12)
    np.testing.assert_allclose(Rotation.from_quat(obs.quaternion).as_matrix(), transform[:3, :3], atol=1e-12)
    assert obs.t == 1. and obs.available_at == 1.1


def test_evaluation_slerp_sign_invariance_and_no_extrapolation(tmp_path):
    root, result = tmp_path / "gt", tmp_path / "result"
    root.mkdir()
    result.mkdir()
    times = np.array([0., 1.])
    p = np.array([[0., 0., 1.], [1., 0., 1.]])
    q = Rotation.from_euler("z", [[0.], [90.]], degrees=True).as_quat()
    np.savez(root / "ground_truth.npz", t=times, position=p, quaternion=q)
    np.savez(result / "trajectory.npz", t=[-.1, .5, 1.1], position=[[0., 0., 1.], [.5, 0., 1.], [1., 0., 1.]],
             quaternion=-Rotation.from_euler("z", [[0.], [45.], [90.]], degrees=True).as_quat())
    metrics = evaluate_tracking(root, result, make_plot=False)
    assert metrics["samples"] == 1 and metrics["excluded_outside_gt_support"] == 2
    assert metrics["position_rmse_m"] < 1e-12
    assert metrics["rotation_rmse_deg"] < 1e-12


def test_translation_end_to_end_produces_causal_flow(tmp_path):
    root, result = tmp_path / "motion", tmp_path / "tracked"
    generate_dataset(root, "translation", duration=.09, width=48, height=36, render_hz=500)
    stats = run_tracking(root, result)
    flow = np.load(result / "flows.npy")
    assert stats["flow_measurements"] > 0
    assert stats["depth_valid_flows"] > 0
    assert np.all(flow[:, 7] <= flow[:, 0])
    assert np.all(flow[:, 8] <= flow[:, 7])
    assert np.all(flow[:, 7] >= .05)
    assert stats["warmup_flow_measurements"] > 0
    metrics = evaluate_tracking(root, result, make_plot=False)
    assert np.isfinite(metrics["position_rmse_m"])


def test_module_entrypoint():
    completed = subprocess.run([sys.executable, "-m", "ev6d", "--help"], capture_output=True, text=True)
    assert completed.returncode == 0 and "convert-dope" in completed.stdout


def test_unavailable_depth_blocks_flow_despite_nonempty_event_stream(tmp_path):
    root = tmp_path / "motion"
    generate_dataset(root, "translation", duration=.09, width=48, height=36, render_hz=500)
    meta = json.loads((root / "dataset.json").read_text())
    for frame in meta["frames"]:
        frame["depth_available_at"] = frame["depth_t"]+1.
    (root / "dataset.json").write_text(json.dumps(meta))
    stats = run_tracking(root, tmp_path / "blocked_depth")
    assert stats["input_events"] > 0
    assert stats["target_events"] == 0
    assert stats["flow_measurements"] == 0
