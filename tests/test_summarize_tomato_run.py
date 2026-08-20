"""Tests for release-evidence extraction from DEIMv2 training logs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/deployment/summarize_tomato_run.py"


def test_summary_reports_best_and_final_metrics(tmp_path: Path) -> None:
    run = tmp_path / "run"
    (run / "summary").mkdir(parents=True)
    records = [
        {"epoch": 0, "test_coco_eval_bbox": [0.1, 0.3]},
        {"epoch": 1, "test_coco_eval_bbox": [0.25, 0.5]},
        {"epoch": 2, "test_coco_eval_bbox": [0.2, 0.45]},
    ]
    (run / "log.txt").write_text("".join(json.dumps(record) + "\n" for record in records))
    (run / "best_stg1.pth").touch()
    (run / "summary/events.out.tfevents.unit").touch()
    output = tmp_path / "summary.json"

    subprocess.run(
        [sys.executable, str(SCRIPT), "--run", str(run), "--output", str(output)],
        check=True,
        cwd=ROOT,
    )

    summary = json.loads(output.read_text())
    assert summary["epochs_completed"] == 3
    assert summary["best_validation"] == {"epoch": 1, "bbox_map": 0.25, "bbox_ap50": 0.5}
    assert summary["final_validation"] == {"epoch": 2, "bbox_map": 0.2, "bbox_ap50": 0.45}
    assert summary["checkpoints"] == ["best_stg1.pth"]
    assert summary["tensorboard_events"] == ["events.out.tfevents.unit"]
