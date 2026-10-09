"""One production-facing path: E-RAFT -> six-dimensional KF -> DOPE -> pose UKF.

The sequence annotations/velocity.csv are not imported for tracking.
All execution here is offline causal replay, NOT a real-time camera driver.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
import time

import numpy as np

from .dense_filter import DenseVelocityConfig, DenseVelocityKF
from .eraft import ERAFTConfig, ERAFTFrontend
from .filters import PoseConfig, PoseUKF
from .pose_replay import PoseHistory
from .public_dataset import PublicYCBSequence


def _new_directory(path):
    path=Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Output is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_config(path):
    cfg={
      "eraft": {"num_bins":15,"iterations":12,"window_s":0.1,"normalize":True,"device":"cpu"},
      "velocity": {},
      "pose": {},
      "history": {"max_history_s":2.,"max_intervals":1000,"max_pose_measurements":1000},
      "max_depth_age_s":0.035,
      "flow_latency_s":0.,
      "dope_delay_s":0.,
      "dope_translation_unit":"m",
      "dope_max_hz":5.,
      "event_time_offset_s":0.,
      "timestamp_scale":1.,
      "mask_kind":"instance",
      "assume_coregistered":False,
      "T_event_rgb":None
    }
    if path:
        user=json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(user,dict) or set(user)-set(cfg):
            raise ValueError("Unknown top-level configuration keys")
        for k,v in user.items():
            if k in ("eraft","velocity","pose","history"):
                if not isinstance(v,dict):
                    raise ValueError(f"{k} must be a JSON object")
                if k=="eraft" and set(v)-set(cfg[k]):
                    raise ValueError("Unknown ERAFT configuration keys")
                if k=="history" and set(v)-set(cfg[k]):
                    raise ValueError("Unknown PoseHistory configuration keys")
                cfg[k].update(v)
            else:
                cfg[k]=v
    ERAFTConfig(**cfg["eraft"])
    DenseVelocityConfig(**cfg["velocity"])
    PoseConfig(**cfg["pose"])
    if cfg["max_depth_age_s"] < 0 or cfg["flow_latency_s"] < 0:
        raise ValueError("Depth age and flow latency must be nonnegative")
    return cfg


def track(sequence_dir, checkpoint, output, *, config=None, dope_txt=None,
          dope_json_dir=None, device=None, max_intervals=None,
          allow_annotation_mask=False):
    """Run the single path; writes result.csv (six velocities) and trajectory.csv.

    DOPE must be computed independently on the RGB images using NVIDIA's model.
    No GT pose or GT velocity is used for initialization or updates.
    """
    cfg=_read_config(config)
    if device:
        cfg["eraft"]["device"]=device
    sequence=PublicYCBSequence(
        sequence_dir, event_time_offset_s=cfg["event_time_offset_s"],
        timestamp_scale=cfg["timestamp_scale"],
        assume_coregistered=cfg["assume_coregistered"],
        mask_kind=cfg["mask_kind"])
    if not allow_annotation_mask:
        raise ValueError(
            "This public benchmark's xx.is.png/xx.cs.png are annotation masks. "
            "Explicit --allow-annotation-mask is required and the experiment "
            "will be labelled oracle_mask=True."
        )
    observations=sequence.dope_observations(
        path=dope_txt, json_dir=dope_json_dir,
        translation_unit=cfg["dope_translation_unit"],
        delay_s=cfg["dope_delay_s"], T_event_rgb=cfg["T_event_rgb"])
    if not observations:
        raise ValueError("No valid DOPE measurements. Never initialize from GT.")
    # Keep the first valid pose for initialization, then sample low-rate DOPE.
    observations=sorted(observations,key=lambda x:x.t)
    if cfg["dope_max_hz"] <= 0:
        raise ValueError("dope_max_hz must be positive")
    sampled=[observations[0]]
    for obs in observations[1:]:
        if obs.t-sampled[-1].t >= 1.0/cfg["dope_max_hz"]-1e-9:
            sampled.append(obs)
    observations=sampled
    first=observations[0]
    if first.available_at>first.t+1e-8:
        raise ValueError(
            "Delayed first DOPE measurement cannot initialize a causal tracker "
            "at its capture time. Provide an externally available initial pose "
            "or set zero first-pose delay for the offline benchmark."
        )
    intervals=list(sequence.windows(cfg["eraft"]["window_s"]))
    if max_intervals is not None:
        if max_intervals<1:
            raise ValueError("max_intervals must be positive")
        intervals=intervals[:max_intervals]
    intervals=[(s,e) for s,e in intervals if s>=first.t-1e-9]
    if not intervals:
        raise ValueError("No flow windows after initial DOPE timestamp")
    end=intervals[-1][1]
    flow_latency=float(cfg["flow_latency_s"])
    arrivals={i:e+flow_latency for i,(s,e) in enumerate(intervals)}
    output=_new_directory(output)
    engine=ERAFTFrontend(checkpoint, ERAFTConfig(**cfg["eraft"]))
    velocity=DenseVelocityKF(DenseVelocityConfig(**cfg["velocity"]))
    velocity.predict_to(first.t)
    history=PoseHistory(
        PoseUKF(first.position,first.quaternion,first.t,PoseConfig(**cfg["pose"])),
        **cfg["history"])
    source_schedule=defaultdict(list)
    arrival_schedule=defaultdict(list)
    pose_schedule=defaultdict(list)
    for index,(start,stop) in enumerate(intervals):
        source_schedule[start].append(index)
        arrival_schedule[arrivals[index]].append(index)
    for i,obs in enumerate(observations):
        if obs is first or obs.t> end+1e-9:
            continue
        pose_schedule[obs.available_at].append((i,obs))
    horizon=max([end, *arrivals.values(),*pose_schedule.keys()])
    timeline=sorted({first.t,horizon,*source_schedule,*arrival_schedule,*pose_schedule})
    cached_geometry={}
    velocities=[]
    trajectory=[]
    reports=[]
    timing={"network_s":0.,"velocity_kf_s":0.,"pose_ukf_s":0.}
    start_wall=time.perf_counter()
    for t in timeline:
        if t<first.t:
            continue
        stamp=time.perf_counter()
        history.predict_to(t,velocity.x.copy(),velocity.P.copy())
        velocity.predict_to(t)
        timing["pose_ukf_s"]+=time.perf_counter()-stamp
        for i in arrival_schedule.get(t,()):
            start,stop=intervals[i]
            previous,current=sequence.event_pair(start,stop)
            stamp=time.perf_counter()
            flow=engine.infer(previous,current,start,stop,
                sequence.shape[1],sequence.shape[0],available_at=t)
            timing["network_s"]+=time.perf_counter()-stamp
            depth,mask=cached_geometry.pop(i)
            valid=mask & flow.valid_mask[0] & np.isfinite(depth) & (depth>0)
            yy,xx=np.nonzero(valid)
            uv=np.column_stack((xx,yy)).astype(float)
            z=depth[yy,xx]
            displacement=flow.flow[0].transpose(1,2,0)[yy,xx]
            # Advanced NumPy indexing can yield [N,2] or [2,N] depending on version.
            displacement=np.asarray(displacement,float)
            stamp=time.perf_counter()
            detail=velocity.update(uv,z,displacement,sequence.K,flow.dt)
            timing["velocity_kf_s"]+=time.perf_counter()-stamp
            velocities.append(velocity.x.copy())
            reports.append({"t":t,"start":start,"end":stop,**detail,
                            "valid_event_support":bool(flow.valid_mask.any())})
        for i,obs in pose_schedule.get(t,()):
            stamp=time.perf_counter()
            history.update(obs.position,obs.quaternion,obs.t,obs.available_at,
                           measurement_id=f"dope_{i}")
            timing["pose_ukf_s"]+=time.perf_counter()-stamp
        for i in source_schedule.get(t,()):
            cached_geometry[i]=sequence.source_geometry(t,max_age_s=cfg["max_depth_age_s"])
        trajectory.append([t,*history.position,*history.quaternion,*velocity.x,
                          *(velocity.x[:3]+np.cross(velocity.x[3:],history.position))])
    with (output/"trajectory.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f)
        writer.writerow(["t","tx","ty","tz","qx","qy","qz","qw",
                         "vox","voy","voz","wx","wy","wz","vref_x","vref_y","vref_z"])
        writer.writerows(trajectory)
    # Compatible header convention with Hybrid-object-tracking src/kf/utils.py.
    velocity_path=sequence.root/"result.csv"
    if velocity_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing {velocity_path}; rename it first")
    with velocity_path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f)
        writer.writerow(["l_v_x","l_v_y","l_v_z","a_v_x","a_v_y","a_v_z"])
        writer.writerows(velocities)
    meta={"kind":"eraft_kf_dope_ukf","sequence":str(sequence.root),
          "checkpoint":str(checkpoint),"flow_intervals":len(intervals),
          "velocity_rows":len(velocities),"pose_input":"DOPE predictions only",
          "gt_used_for_tracking":False,"oracle_mask":True,
          "geometry_assumption":"co-registered RGB-D/events explicitly accepted",
          "flow_unit":"pixel/interval","velocity_unit":"m/s;rad/s",
          "poses_are_T_event_object":True,
          "first_dope_as_initial_pose":True,
          "result_csv_path":str(velocity_path),
          "elapsed_s":time.perf_counter()-start_wall,
          "timings_s":timing,"reports":reports,
          "warning":"Offline causal replay, not sensor real-time timing; result.csv omits timestamps"}
    (output/"diagnostics.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    return {"output":str(output),"result_csv":str(velocity_path),
            "flow_intervals":len(intervals),"pose_observations":len(observations),
            "tracker":"E-RAFT+KF+DOPE+UKF"}
