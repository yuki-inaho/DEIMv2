"""Convert a checked DEIMv2 ONNX detector to an OpenVINO IR model."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True, help="Input checked .onnx model")
    parser.add_argument("--output", type=Path, required=True, help="Output .xml IR path")
    parser.add_argument("--fp16", action="store_true", help="Compress IR weights to FP16")
    args = parser.parse_args()

    if args.onnx.suffix.lower() != ".onnx":
        parser.error("--onnx must name an .onnx model")
    if args.output.suffix.lower() != ".xml":
        parser.error("--output must name an .xml IR file")

    import openvino as ov

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = ov.convert_model(str(args.onnx))
    ov.save_model(model, str(args.output), compress_to_fp16=args.fp16)

    core = ov.Core()
    validated = core.read_model(str(args.output))
    core.compile_model(validated, "CPU")
    bin_path = args.output.with_suffix(".bin")
    print(f"OpenVINO IR validated: {args.output} ({args.output.stat().st_size} bytes)")
    print(f"OpenVINO weights: {bin_path} ({bin_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
