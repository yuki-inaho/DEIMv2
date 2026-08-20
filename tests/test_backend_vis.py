"""Contract tests for runtime-independent DEIMv2 visualization helpers."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

backend_vis = import_module("tools.inference.backend_vis")
_draw = backend_vis._draw
_preprocess = backend_vis._preprocess


def test_preprocess_uses_eval_hw_and_original_hw(tmp_path: Path) -> None:
    """Runtimes receive NCHW float images and original sizes in [height, width]."""
    source = tmp_path / "input.png"
    Image.new("RGB", (17, 11), color=(10, 20, 30)).save(source)

    image, images, original_sizes = _preprocess(source, (8, 12))

    assert image.size == (17, 11)
    assert images.shape == (1, 3, 8, 12)
    assert images.dtype == np.float32
    assert original_sizes.dtype == np.int64
    np.testing.assert_array_equal(original_sizes, [[11, 17]])


def test_draw_counts_only_scores_at_threshold() -> None:
    """The summary count and rendering threshold have the same inclusive contract."""
    image = Image.new("RGB", (20, 20), color="black")
    labels = np.asarray([0, 0], dtype=np.int64)
    boxes = np.asarray([[1, 1, 9, 9], [10, 10, 18, 18]], dtype=np.float32)
    scores = np.asarray([0.5, 0.499], dtype=np.float32)

    rendered, detections = _draw(image, labels, boxes, scores, 0.5, "stem")

    assert detections == 1
    assert rendered.size == image.size
