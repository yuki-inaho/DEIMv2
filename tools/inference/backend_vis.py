"""Run a DEIMv2 detector with a selectable runtime and save a visual result.

The three runtimes intentionally share preprocessing and postprocessing
contracts: ``images`` is resized to the configured evaluation size and
``orig_target_sizes`` is ``[height, width]``.  Exported DEIMv2 models already
contain their postprocessor, so all backends return labels, xyxy boxes, and
scores in the original-image coordinate system.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _preprocess(image_path: Path, image_size: tuple[int, int]) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    eval_height, eval_width = image_size
    resized = image.resize((eval_width, eval_height), Image.BILINEAR)
    array = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    original_size = np.asarray([[height, width]], dtype=np.int64)
    return image, array, original_size


def _load_torch(config_path: Path, checkpoint_path: Path, device: str) -> tuple[Callable, tuple[int, int]]:
    import torch
    import torch.nn as nn

    from engine.core import YAMLConfig

    config = YAMLConfig(str(config_path), resume=str(checkpoint_path))
    if "HGNetv2" in config.yaml_cfg:
        config.yaml_cfg["HGNetv2"]["pretrained"] = False
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("ema", {}).get("module", checkpoint["model"])
    config.model.load_state_dict(state)

    class DeployModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = config.model.deploy()
            self.postprocessor = config.postprocessor.deploy()

        def forward(self, images, original_sizes):
            return self.postprocessor(self.model(images), original_sizes)

    model = DeployModel().eval().to(device)
    image_size = tuple(config.yaml_cfg["eval_spatial_size"])

    def infer(images: np.ndarray, original_sizes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with torch.inference_mode():
            labels, boxes, scores = model(
                torch.from_numpy(images).to(device), torch.from_numpy(original_sizes).to(device)
            )
        return tuple(value.detach().cpu().numpy() for value in (labels, boxes, scores))

    return infer, image_size


def _load_onnxruntime(model_path: Path, device: str) -> tuple[Callable, tuple[int, int]]:
    import onnxruntime as ort

    available = ort.get_available_providers()
    providers = ["CPUExecutionProvider"]
    if device.startswith("cuda") and "CUDAExecutionProvider" in available:
        providers.insert(0, "CUDAExecutionProvider")
    session = ort.InferenceSession(str(model_path), providers=providers)
    input_shape = session.get_inputs()[0].shape
    image_size = tuple(int(value) for value in input_shape[-2:])

    def infer(images: np.ndarray, original_sizes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = session.run(None, {"images": images, "orig_target_sizes": original_sizes})
        return tuple(np.asarray(value) for value in values)

    return infer, image_size


def _load_openvino(model_path: Path, device: str) -> tuple[Callable, tuple[int, int]]:
    import openvino as ov

    core = ov.Core()
    model = core.read_model(str(model_path))
    compiled = core.compile_model(model, device.upper())
    inputs = {item.get_any_name(): item for item in compiled.inputs}
    image_shape = inputs["images"].get_partial_shape()
    image_size = tuple(int(image_shape[index].get_length()) for index in (-2, -1))

    def infer(images: np.ndarray, original_sizes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = compiled({"images": images, "orig_target_sizes": original_sizes})
        return tuple(np.asarray(values[output]) for output in compiled.outputs)

    return infer, image_size


def _draw(image: Image.Image, labels: np.ndarray, boxes: np.ndarray, scores: np.ndarray, threshold: float, class_name: str) -> tuple[Image.Image, int]:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    keep = scores >= threshold
    count = int(np.count_nonzero(keep))
    for label, box, score in zip(labels[keep], boxes[keep], scores[keep]):
        x1, y1, x2, y2 = (int(round(value)) for value in box)
        text = f"{class_name if int(label) == 0 else int(label)} {float(score):.2f}"
        draw.rectangle((x1, y1, x2, y2), outline="lime", width=3)
        draw.text((x1 + 2, max(y1 - 12, 0)), text, fill="lime", font=font)
    return canvas, count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("torch", "onnxruntime", "openvino"), required=True)
    parser.add_argument("--model", type=Path, required=True, help=".pth for torch; .onnx/.xml for other backends")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="DEIMv2 YAML (required for --backend torch)")
    parser.add_argument("--device", default="cpu", help="cpu or cuda for torch/ONNX Runtime; CPU/GPU for OpenVINO")
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--class-name", default="stem")
    args = parser.parse_args()

    if args.backend == "torch":
        if args.config is None:
            parser.error("--config is required with --backend torch")
        runner, image_size = _load_torch(args.config, args.model, args.device)
    elif args.backend == "onnxruntime":
        runner, image_size = _load_onnxruntime(args.model, args.device)
    else:
        runner, image_size = _load_openvino(args.model, args.device)

    image, images, original_sizes = _preprocess(args.input, image_size)
    labels, boxes, scores = runner(images, original_sizes)
    result, count = _draw(image, labels[0], boxes[0], scores[0], args.threshold, args.class_name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)
    summary = {
        "backend": args.backend,
        "device": args.device,
        "input": str(args.input),
        "output": str(args.output),
        "eval_size_hw": list(image_size),
        "score_threshold": args.threshold,
        "detections": count,
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
