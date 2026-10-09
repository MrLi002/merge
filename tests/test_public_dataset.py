"""Offline tests for the public UE/YCB adapter and no-ground-truth tracking policy."""
import json
from pathlib import Path
import cv2
import numpy as np
import pytest

from ev6d.public_dataset import PublicYCBSequence
from ev6d.dense_data import FlowResult


def example_sequence(tmp_path, monkeypatch, n=4):
    root=tmp_path/"scene"
    frames=root/"photorealistic1"
    frames.mkdir(parents=True)
    (root/"filter_param.json").write_text(json.dumps({"initial_state":{"mean":[0]*6}}))
    (frames/"_camera_settings.json").write_text(json.dumps({"camera_settings":[{
       "intrinsic_settings":{"fx":100.,"fy":100.,"cx":3.5,"cy":3.5}}]}))
    (frames/"_object_settings.json").write_text("{}")
    (frames/"binaryevents.log").write_bytes(b"mock-not-used")
    for i in range(n):
        stem=f"{i:04}"
        cv2.imwrite(str(frames/(stem+".png")),np.full((8,8,3),60,np.uint8))
        cv2.imwrite(str(frames/(stem+".depth.mm.16.png")),np.full((8,8),1000,np.uint16))
        cv2.imwrite(str(frames/(stem+".is.png")),np.ones((8,8),np.uint8))
        (frames/(stem+".json")).write_text(json.dumps({"timestamp":42+i*0.1,
                                                       "objects":[{"location":[999,999,999]}]}))
    (root/"dope_poses_ycb.txt").write_text("\n".join("0 0 1 0 0 1 0" for _ in range(n)))
    monkeypatch.setattr(PublicYCBSequence,"_load_events",
        lambda self,offset:np.asarray([[0.+offset,1,1,1],[.05+offset,2,1,0],
            [.11+offset,3,1,1],[.19+offset,4,1,0],[.21+offset,5,1,1]]))
    return root


def test_public_dataset_and_dope_not_gt(tmp_path,monkeypatch):
    root=example_sequence(tmp_path,monkeypatch)
    ds=PublicYCBSequence(root,assume_coregistered=True)
    assert len(ds.frames)==4
    assert ds.frames[0].t==pytest.approx(0.)
    assert ds.K[0,0]==100.
    assert [tuple(map(float,x)) for x in ds.event_pair(.1,.2)[0]][0][0]==pytest.approx(0.)
    depth,mask=ds.source_geometry(.1,max_age_s=.04)
    assert depth.shape==(8,8) and np.allclose(depth,1.)
    assert mask.sum()==64
    poses=ds.dope_observations()
    assert len(poses)==4
    assert np.allclose(poses[0].position,[0,0,1])
    assert not np.allclose(poses[0].position,[999,999,999])


def test_dope_missing_and_sparse_time(tmp_path,monkeypatch):
    root=example_sequence(tmp_path,monkeypatch)
    ds=PublicYCBSequence(root,assume_coregistered=True)
    source=root/"dope_poses_ycb.txt"
    source.write_text("0 0 0 1 0 0 1 0\n0.2 0 0 1 0 0 1 0")
    poses=ds.dope_observations()
    assert [p.t for p in poses]==pytest.approx([0,.2])
    source.write_text("0 0 1 0 0 1 0")
    with pytest.raises(ValueError, match="one row per RGB"):
        ds.dope_observations()


def test_flow_result_units_and_shape():
    sample=FlowResult(np.zeros((1,2,8,8),np.float32),0.1,0.2,
                      "event_rectified",np.ones((1,8,8),bool))
    assert sample.dt==pytest.approx(.1)
    assert sample.valid_mask.all()
