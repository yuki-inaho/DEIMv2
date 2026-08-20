"""Write a compact, machine-readable evidence summary for a DEIMv2 run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _metric(record: dict[str, Any]) -> float:
    values = record.get("test_coco_eval_bbox")
    if not isinstance(values, list) or not values:
        raise ValueError("log record does not contain test_coco_eval_bbox[0] (validation mAP)")
    return float(values[0])


def build_summary(run_directory: Path) -> dict[str, Any]:
    records = [json.loads(line) for line in (run_directory / "log.txt").read_text().splitlines() if line]
    if not records:
        raise ValueError("log.txt contains no JSON records")
    best = max(records, key=_metric)
    final = records[-1]
    return {
        "run_directory": str(run_directory),
        "epochs_completed": int(final["epoch"]) + 1,
        "best_validation": {
            "epoch": int(best["epoch"]),
            "bbox_map": _metric(best),
            "bbox_ap50": float(best["test_coco_eval_bbox"][1]),
        },
        "final_validation": {
            "epoch": int(final["epoch"]),
            "bbox_map": _metric(final),
            "bbox_ap50": float(final["test_coco_eval_bbox"][1]),
        },
        "checkpoints": sorted(path.name for path in run_directory.glob("*.pth")),
        "tensorboard_events": sorted(path.name for path in (run_directory / "summary").glob("events.out.tfevents.*")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Defaults to <run>/run_summary.json")
    args = parser.parse_args()
    summary = build_summary(args.run)
    output = args.output or args.run / "run_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
