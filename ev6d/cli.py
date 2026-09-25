"""Command-line entry points for generation, tracking, and offline evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Independent event/frame 6-DoF method reproduction")
    commands = parser.add_subparsers(dest="command", required=True)

    # 生成简化长方体合成数据，用于快速验证算法和缺测等场景
    generate = commands.add_parser("generate", help="Generate a small synthetic verification dataset")
    generate.add_argument("--output", required=True, type=Path)
    generate.add_argument("--scenario", default="mixed", choices=("static", "translation", "rotation", "mixed", "event_gap", "pose_gap", "outliers"))
    generate.add_argument("--duration", type=float, default=1.2)
    generate.add_argument("--seed", type=int, default=7)
    generate.add_argument("--render-hz", type=float, default=1000)
    generate.add_argument("--width", type=int, default=160)
    generate.add_argument("--height", type=int, default=120)

    # 下载四种官方 YCB 物体的网格和纹理；只下载模型，不生成事件数据
    fetch_ycb = commands.add_parser("fetch-ycb", help="Download four official YCB Google 16k models")
    fetch_ycb.add_argument("--assets", type=Path, default=Path("assets/ycb"))
    fetch_ycb.add_argument("--objects", nargs="+", choices=("003_cracker_box", "005_tomato_soup_can", "006_mustard_bottle", "010_potted_meat_can"))

    # 利用 YCB 模型渲染并生成事件、RGB-D、模拟位姿和真值；默认四物体 × 两种速度
    ycb = commands.add_parser("generate-ycb", help="Generate paper-inspired YCB RGB-D/event dataset")
    ycb.add_argument("--output", type=Path, required=True)
    ycb.add_argument("--assets", type=Path, default=Path("assets/ycb"))
    ycb.add_argument("--objects", nargs="+", choices=("003_cracker_box", "005_tomato_soup_can", "006_mustard_bottle", "010_potted_meat_can"))
    ycb.add_argument("--speeds", nargs="+", choices=("regular", "fast"), default=["regular", "fast"])
    ycb.add_argument("--fetch", action="store_true", help="Download official models if needed")
    ycb.add_argument("--duration", type=float, default=1.)
    ycb.add_argument("--seed", type=int, default=7)
    ycb.add_argument("--width", type=int, default=640)
    ycb.add_argument("--height", type=int, default=480)
    ycb.add_argument("--render-hz", type=float, default=500.)
    ycb.add_argument("--frame-hz", type=float, default=60.)
    ycb.add_argument("--pose-hz", type=float, default=5.)
    ycb.add_argument("--contrast-threshold", type=float, default=.2)
    ycb.add_argument("--threshold-sigma", type=float, default=0.)
    ycb.add_argument("--refractory-s", type=float, default=0.)
    ycb.add_argument("--supersample", type=int, default=1)

    # 读取一条序列，估计物体的六自由度轨迹；不读取真值
    track = commands.add_parser("track", help="Track without loading ground truth")
    track.add_argument("--dataset", required=True, type=Path)
    track.add_argument("--output", required=True, type=Path)
    track.add_argument("--config", type=Path)
    track.add_argument("--variant", default="full", choices=("full", "no_normal", "no_weight", "pose_only", "velocity_only"))

    # 将 track 的估计轨迹与数据集真值比较
    evaluate = commands.add_parser("evaluate", help="Offline metrics and plots against ground_truth.npz")
    evaluate.add_argument("--dataset", required=True, type=Path)
    evaluate.add_argument("--result", required=True, type=Path)
    evaluate.add_argument("--no-plot", action="store_true")

    # 将估计位姿和真值对齐，导出可离线打开的交互式三维轨迹
    visualize = commands.add_parser("visualize", help="Export interactive 3D estimate/ground-truth HTML")
    visualize.add_argument("--dataset", required=True, type=Path)
    visualize.add_argument("--result", required=True, type=Path)
    visualize.add_argument("--output", type=Path, help="HTML path (default: RESULT/trajectory_3d.html)")

    # 显示或保存跟踪算法的默认参数；本身不运行跟踪
    config = commands.add_parser("config", help="Print or save all default parameters")
    config.add_argument("--output", type=Path)

    inspect = commands.add_parser("check-data", help="Validate actual schema-1 tracking files and calibration")
    inspect.add_argument("--dataset", type=Path, required=True)
    inspect.add_argument("--metadata-only", action="store_true")

    dense_config = commands.add_parser("dense-config", help="Print/save E-RAFT and dense filtering defaults")
    dense_config.add_argument("--output", type=Path)
    for name, description in (("precompute-flow", "Save timestamped E-RAFT flow cache"),
                              ("track-dense", "Track from E-RAFT or a verified flow cache")):
        dense = commands.add_parser(name, help=description)
        dense.add_argument("--dataset", required=True, type=Path)
        dense.add_argument("--output", required=True, type=Path)
        dense.add_argument("--checkpoint", type=Path)
        dense.add_argument("--config", type=Path)
        dense.add_argument("--device", help="Override eraft.device, for example cpu or cuda")
        dense.add_argument("--max-intervals", type=int)
        if name == "track-dense":
            dense.add_argument("--flow-cache", type=Path)
            dense.add_argument("--allow-oracle-pose", action="store_true", help="Explicitly label known GT/synthetic initialization or pose inputs")
            dense.add_argument("--velocity-only", action="store_true")

    train = commands.add_parser("train-flow", help="Independent supervised DSEC E-RAFT training/fine-tuning")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--resume", type=Path)
    train.add_argument("--output", type=Path)
    train.add_argument("--max-steps", type=int)
    flow_eval = commands.add_parser("evaluate-flow", help="Supervised DSEC flow evaluation")
    flow_eval.add_argument("--config", type=Path, required=True)
    flow_eval.add_argument("--checkpoint", type=Path)
    flow_eval.add_argument("--split", choices=("train", "val"), default="val")
    flow_eval.add_argument("--max-batches", type=int)
    flow_eval.add_argument("--output", type=Path)
    cache_eval = commands.add_parser("evaluate-flow-cache", help="Full/target EPE against explicit source-plane interval GT")
    cache_eval.add_argument("--prediction", type=Path, required=True)
    cache_eval.add_argument("--ground-truth", type=Path, required=True)
    cache_eval.add_argument("--output", type=Path)
    fixture = commands.add_parser("flow-fixture", help="Create labelled synthetic DSEC-format training smoke data")
    fixture.add_argument("--output", type=Path, required=True)
    fixture.add_argument("--size", type=int, default=128)
    fixture.add_argument("--samples", type=int, default=2)
    reproject = commands.add_parser("reproject", help="Calibrated offline pose overlays on RGB images")
    reproject.add_argument("--dataset", type=Path, required=True)
    reproject.add_argument("--result", type=Path, required=True)
    reproject.add_argument("--output", type=Path)
    reproject.add_argument("--max-frames", type=int, default=12)

    # 把外部已经运行好的 DOPE 推理 JSON 转成代码需要的 poses.csv；它不运行或训练 DOPE
    convert = commands.add_parser("convert-dope", help="Convert existing DOPE inference JSON to poses.csv")
    convert.add_argument("--input", required=True, type=Path)
    convert.add_argument("--timestamps", required=True, type=Path)
    convert.add_argument("--output", required=True, type=Path)
    convert.add_argument("--object", required=True)
    convert.add_argument("--length-unit", required=True, choices=("m", "cm", "mm"))
    convert.add_argument("--pose-frame", required=True, choices=("rgb", "event"))
    convert.add_argument("--dataset", required=True, type=Path, help="Directory containing dataset.json with calibration")
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            from .synthetic import generate_dataset
            result = generate_dataset(args.output, args.scenario, args.duration, args.seed, args.render_hz, args.width, args.height)
        elif args.command == "fetch-ycb":
            from .assets import YCB_OBJECTS, fetch_ycb as fetch_models
            manifest = fetch_models(args.assets, args.objects or YCB_OBJECTS)
            result = {"objects": list(manifest["objects"]), "assets": str(args.assets.resolve())}
        elif args.command == "generate-ycb":
            from .assets import YCB_OBJECTS
            from .ycb_synthetic import generate_ycb_suite
            result = generate_ycb_suite(args.output, args.assets, objects=args.objects or YCB_OBJECTS,
                                        speeds=args.speeds, fetch=args.fetch, duration=args.duration,
                                        seed=args.seed, width=args.width, height=args.height,
                                        render_hz=args.render_hz, frame_hz=args.frame_hz,
                                        pose_hz=args.pose_hz, contrast_threshold=args.contrast_threshold,
                                        threshold_sigma=args.threshold_sigma,
                                        refractory_s=args.refractory_s, supersample=args.supersample)
        elif args.command == "track":
            from .pipeline import run_tracking
            result = run_tracking(args.dataset, args.output, args.config, args.variant)
        elif args.command == "evaluate":
            from .evaluation import evaluate_tracking
            result = evaluate_tracking(args.dataset, args.result, make_plot=not args.no_plot)
        elif args.command == "visualize":
            from .visualization import visualize_tracking
            result = visualize_tracking(args.dataset, args.result, args.output)
        elif args.command == "config":
            from .pipeline import load_config
            result = load_config()
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        elif args.command == "check-data":
            from .dense_data import DenseSequence
            result = DenseSequence(args.dataset).inspect(full=not args.metadata_only)
        elif args.command == "dense-config":
            from .dense_pipeline import load_dense_config
            result = load_dense_config()
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        elif args.command in ("track-dense", "precompute-flow"):
            from .dense_pipeline import load_dense_config, precompute_flow, run_dense_tracking
            cfg = load_dense_config(args.config)
            if args.device:
                cfg['eraft']['device'] = args.device
            if args.command == "precompute-flow":
                result = precompute_flow(args.dataset, args.output, checkpoint=args.checkpoint,
                                         config=cfg, max_intervals=args.max_intervals)
            else:
                if args.allow_oracle_pose:
                    cfg['allow_oracle_pose'] = True
                if args.velocity_only:
                    cfg['use_pose_observations'] = False
                result = run_dense_tracking(args.dataset, args.output, checkpoint=args.checkpoint,
                                            config=cfg, flow_cache=args.flow_cache,
                                            max_intervals=args.max_intervals)
        elif args.command == "train-flow":
            from .flow_training import train_flow
            result = train_flow(args.config, resume=args.resume, output_dir=args.output,
                                max_steps=args.max_steps)
        elif args.command == "evaluate-flow":
            from .flow_training import evaluate_flow
            result = evaluate_flow(args.config, checkpoint=args.checkpoint, split=args.split,
                                   max_batches=args.max_batches, output=args.output)
        elif args.command == "flow-fixture":
            from .dsec import create_synthetic_dsec_fixture
            path = create_synthetic_dsec_fixture(args.output, size=args.size, samples=args.samples)
            result = {'output': str(path.resolve()), 'synthetic': True, 'not_real_dsec': True}
        elif args.command == "evaluate-flow-cache":
            from .flow_evaluation import evaluate_flow_cache
            result = evaluate_flow_cache(args.prediction, args.ground_truth, args.output)
        elif args.command == "reproject":
            from .reporting import reproject_tracking
            result = reproject_tracking(args.dataset, args.result, args.output, args.max_frames)
        elif args.command == "convert-dope":
            from .data import convert_dope
            calibration = json.loads((args.dataset / "dataset.json").read_text(encoding="utf-8"))["calibration"]
            count = convert_dope(args.input, args.timestamps, args.output, args.object, args.length_unit, args.pose_frame, calibration)
            result = {"observations": count, "output": str(args.output)}
    except (ValueError, OSError, KeyError, TypeError, RuntimeError, ImportError) as exc:
        parser.exit(2, f"ev6d: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0
