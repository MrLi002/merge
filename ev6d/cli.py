"""Public UE/YCB tracking CLI; no triplet, training, or data synthesis commands."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys

def _rgb_files(sequence):
    path=Path(sequence).expanduser().resolve()
    frames=path if path.name=="photorealistic1" else path/"photorealistic1"
    if not frames.is_dir():
        raise FileNotFoundError(frames)
    images=sorted(p for p in frames.glob("*.png") if re.fullmatch(r"[^.]+\.png",p.name))
    if not images:
        raise ValueError("No RGB-only xx.png images found")
    return images


def prepare_rgb(sequence, output):
    """Copy ONLY RGB images for NVIDIA DOPE; do not feed depth/mask/event pngs."""
    images=_rgb_files(sequence)
    destination=Path(output)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("DOPE RGB destination must be empty")
    destination.mkdir(parents=True,exist_ok=True)
    for image in images:
        shutil.copy2(image,destination/image.name)
    return len(images)


def main(argv=None):
    parser=argparse.ArgumentParser(description="E-RAFT + 6D velocity KF + RGB DOPE + pose UKF")
    cmds=parser.add_subparsers(dest="command",required=True)

    track=cmds.add_parser("track",help="Run the only tracking path")
    track.add_argument("--sequence",required=True,type=Path,help="Sequence root or photorealistic1")
    track.add_argument("--checkpoint",required=True,type=Path,help="Verified E-RAFT .tar/.pt")
    track.add_argument("--output",required=True,type=Path,help="New trajectory/diagnostic directory")
    track.add_argument("--config",type=Path,help="Tracking config JSON")
    track.add_argument("--dope-txt",type=Path,help="Defaults to sequence/dope_poses_ycb.txt")
    track.add_argument("--dope-json-dir",type=Path,help="Alternative: official DOPE per-frame JSON")
    track.add_argument("--device",choices=["cpu","cuda"])
    track.add_argument("--max-intervals",type=int)
    track.add_argument("--allow-annotation-mask",action="store_true",
                       help="Acknowledge that xx.is.png/xx.cs.png is oracle segmentation")

    rgb=cmds.add_parser("prepare-rgb",help="Copy xx.png only to a dedicated DOPE input folder")
    rgb.add_argument("--sequence",required=True,type=Path)
    rgb.add_argument("--output",required=True,type=Path)

    dope=cmds.add_parser("run-dope",help="Run official NVIDIA DOPE on prepared RGB images")
    dope.add_argument("--dope-repo",required=True,type=Path)
    dope.add_argument("--dope-python",type=Path,help="DOPE environment Python executable")
    dope.add_argument("--rgb-dir",required=True,type=Path)
    dope.add_argument("--output",required=True,type=Path)
    dope.add_argument("--weights",required=True,type=Path)
    dope.add_argument("--config",required=True,type=Path,help="NVIDIA DOPE config_pose.yaml")
    dope.add_argument("--camera",required=True,type=Path,help="RGB camera_info.yaml")
    dope.add_argument("--object",required=True,help="YCB object class in DOPE config")
    args=parser.parse_args(argv)

    if args.command=="prepare-rgb":
        n=prepare_rgb(args.sequence,args.output)
        print(f"Prepared {n} RGB images for DOPE in {args.output}")
    elif args.command=="run-dope":
        script=(args.dope_repo/"inference"/"inference.py").resolve()
        if not script.is_file():
            raise FileNotFoundError(script)
        command=[str(args.dope_python.resolve()) if args.dope_python else sys.executable,str(script),"--data",str(args.rgb_dir.resolve()),
                 "--outf",str(args.output.resolve()),"--weights",str(args.weights.resolve()),
                 "--config",str(args.config.resolve()),"--camera",str(args.camera.resolve()),
                 "--object",args.object]
        # DOPE depends on its own separate environment and trained object weights.
        subprocess.run(command,cwd=script.parent,check=True)
        print("Official DOPE outputs are in subfolders of",args.output)
    else:
        from .runner import track as run
        result=run(args.sequence,args.checkpoint,args.output,config=args.config,
                   dope_txt=args.dope_txt,dope_json_dir=args.dope_json_dir,
                   device=args.device,max_intervals=args.max_intervals,
                   allow_annotation_mask=args.allow_annotation_mask)
        import json
        print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
