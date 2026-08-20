#!/usr/bin/env bash
# Finalize a completed tomato DEIMv2 run without silently accepting a failed run.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUN=${RUN:-"$ROOT/outputs/deimv2_hgnetv2_m_fruitbbox_muon_full_20260820"}
CONFIG=${CONFIG:-"$ROOT/../configs/deimv2/deimv2_hgnetv2_m_coco_tomato_muon.yml"}
SAMPLE=${SAMPLE:-"/workspace/data/tomato_fruits_bbox/fruits_detection_data_Jun23-2025/images_rgb/20240626_13-31-01_2024-06-26_13-40-32_rgb.jpg"}
LOG="$RUN/console.resume.log"

if ! rg -q '^Training time ' "$LOG"; then
    echo "Refusing finalization: the run has no normal 'Training time' marker: $LOG" >&2
    exit 1
fi

BEST="$RUN/best_stg2.pth"
if [[ ! -f "$BEST" ]]; then
    BEST="$RUN/best_stg1.pth"
fi
for required in "$BEST" "$CONFIG" "$SAMPLE"; do
    if [[ ! -f "$required" ]]; then
        echo "Required file is missing: $required" >&2
        exit 1
    fi
done

export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-/home/kasm-user/Desktop/DEIM_sandbox/.venv}
ONNX="${BEST%.pth}.onnx"
IR="${ONNX%.onnx}.xml"
IR_FP16="${ONNX%.onnx}_fp16.xml"
FINAL_LOG="$RUN/finalization.log"

cd "$ROOT"
{
    echo "Finalizing completed run: $RUN"
    echo "Selected checkpoint: $BEST"
    uv run --no-sync python tools/deployment/export_onnx.py --check -c "$CONFIG" -r "$BEST"
    uv run --no-sync python tools/deployment/convert_openvino.py --onnx "$ONNX" --output "$IR"
    uv run --no-sync python tools/deployment/convert_openvino.py --onnx "$ONNX" --output "$IR_FP16" --fp16
    uv run --no-sync python tools/inference/backend_vis.py --backend torch --device cuda \
        --config "$CONFIG" --model "$BEST" --input "$SAMPLE" --output "$RUN/torch.jpg"
    uv run --no-sync python tools/inference/backend_vis.py --backend onnxruntime --device cpu \
        --model "$ONNX" --input "$SAMPLE" --output "$RUN/onnxruntime.jpg"
    uv run --no-sync python tools/inference/backend_vis.py --backend openvino --device CPU \
        --model "$IR" --input "$SAMPLE" --output "$RUN/openvino.jpg"
    uv run --no-sync python tools/deployment/summarize_tomato_run.py --run "$RUN"
    sha256sum "$BEST" "$ONNX" "$IR" "${IR%.xml}.bin" "$IR_FP16" "${IR_FP16%.xml}.bin" \
        | tee "$RUN/artifacts.sha256"
} 2>&1 | tee "$FINAL_LOG"
