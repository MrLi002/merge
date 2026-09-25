"""Reproducible small regression suite. Run from the repository root.

python scripts/validate_scenarios.py --output output/validation --duration 0.4
This runs the implementation, not the original paper's benchmark.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ev6d.evaluation import evaluate_tracking
from ev6d.pipeline import run_tracking
from ev6d.synthetic import SCENARIOS, generate_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/validation"))
    parser.add_argument("--duration", type=float, default=.4)
    parser.add_argument("--width", type=int, default=80)
    parser.add_argument("--height", type=int, default=60)
    parser.add_argument("--render-hz", type=float, default=1000)
    args = parser.parse_args()
    rows = []
    for scenario in SCENARIOS:
        data = args.output / "datasets" / scenario
        generate_dataset(data, scenario, args.duration, width=args.width, height=args.height, render_hz=args.render_hz)
        variants = ("full", "pose_only", "velocity_only", "no_normal", "no_weight") if scenario == "mixed" else ("full",)
        for variant in variants:
            result = args.output / "results" / scenario / variant
            runtime = run_tracking(data, result, variant=variant)
            metrics = evaluate_tracking(data, result, make_plot=True)
            row = {"scenario": scenario, "variant": variant, **metrics,
                   "processing_s": runtime["processing_s"], "flow_measurements": runtime["flow_measurements"]}
            rows.append(row)
            print(json.dumps(row), flush=True)
            (args.output / "summary.json").write_text(json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
