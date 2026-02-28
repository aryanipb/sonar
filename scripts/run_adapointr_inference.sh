#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$ROOT_DIR/env5/bin/python" ]]; then
    PYTHON_BIN="$ROOT_DIR/env5/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

CONFIG=""
CKPT=""
PC_ROOT=""
PC_FILE=""
OUT_DIR=""
DEVICE=""
SAVE_VIS=0
SKIP_EXISTING=0
MAKE_3D_VIEW=0
SHOW_3D=0
MAX_VIEW_POINTS=20000

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --ckpt)
      CKPT="$2"
      shift 2
      ;;
    --pc_root)
      PC_ROOT="$2"
      shift 2
      ;;
    --pc)
      PC_FILE="$2"
      shift 2
      ;;
    --out)
      OUT_DIR="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --save-vis)
      SAVE_VIS=1
      shift
      ;;
    --skip-existing)
      SKIP_EXISTING=1
      shift
      ;;
    --make-3d-view)
      MAKE_3D_VIEW=1
      shift
      ;;
    --show-3d)
      SHOW_3D=1
      shift
      ;;
    --max-view-points)
      MAX_VIEW_POINTS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 2
      ;;
  esac
done

if [[ -z "$CONFIG" || -z "$CKPT" ]]; then
  echo "Usage: $0 --config <yaml> --ckpt <pth> [--pc_root <dir> | --pc <file>] [--out <dir>] [--device <cpu|cuda:0>] [--save-vis] [--skip-existing] [--make-3d-view] [--show-3d] [--max-view-points <N>]"
  exit 2
fi

if [[ -z "$PC_ROOT" && -z "$PC_FILE" ]]; then
  echo "Error: either --pc_root or --pc is required"
  exit 2
fi

if [[ -z "$OUT_DIR" ]]; then
  ts="$(date +%Y%m%d_%H%M%S)"
  OUT_DIR="$ROOT_DIR/inference_result/$ts"
fi
mkdir -p "$OUT_DIR"

CMD=("$PYTHON_BIN" tools/inference.py "$CONFIG" "$CKPT" --out_pc_root "$OUT_DIR")
if [[ -n "$DEVICE" ]]; then
  CMD+=(--device "$DEVICE")
fi
if [[ "$SAVE_VIS" -eq 1 ]]; then
  CMD+=(--save_vis_img)
fi
if [[ "$SKIP_EXISTING" -eq 1 ]]; then
  CMD+=(--skip_existing)
fi
if [[ -n "$PC_ROOT" ]]; then
  CMD+=(--pc_root "$PC_ROOT")
else
  CMD+=(--pc "$PC_FILE")
fi

echo "[Run] Output directory: $OUT_DIR"
echo "[Run] Command: ${CMD[*]}"
"${CMD[@]}"

if [[ "$MAKE_3D_VIEW" -eq 1 ]]; then
  echo "[Run] Generating interactive 3D comparison view(s)"
  if [[ -n "$PC_ROOT" ]]; then
    shopt -s nullglob
    for in_file in "$PC_ROOT"/*; do
      if [[ ! -f "$in_file" ]]; then
        continue
      fi
      ext="${in_file##*.}"
      ext_lc="$(echo "$ext" | tr '[:upper:]' '[:lower:]')"
      case "$ext_lc" in
        pcd|ply|npy|txt|h5) ;;
        *) continue ;;
      esac
      name="$(basename "$in_file")"
      stem="${name%.*}"
      pred_file="$OUT_DIR/$stem/fine.npy"
      html_file="$OUT_DIR/$stem/actual_vs_pred_3d.html"
      if [[ -f "$pred_file" ]]; then
        VIEW_CMD=("$PYTHON_BIN" tools/visualize_actual_vs_pred.py --input_pc "$in_file" --pred_pc "$pred_file" --output_html "$html_file" --max_points "$MAX_VIEW_POINTS")
        if [[ "$SHOW_3D" -eq 1 ]]; then
          VIEW_CMD+=(--show)
        fi
        "${VIEW_CMD[@]}"
      else
        echo "[Run] Skipping 3D view for $name (missing $pred_file)"
      fi
    done
    shopt -u nullglob
  else
    base_name="$(basename "$PC_FILE")"
    stem="${base_name%.*}"
    pred_file="$OUT_DIR/$stem/fine.npy"
    html_file="$OUT_DIR/$stem/actual_vs_pred_3d.html"
    if [[ -f "$pred_file" ]]; then
      VIEW_CMD=("$PYTHON_BIN" tools/visualize_actual_vs_pred.py --input_pc "$PC_FILE" --pred_pc "$pred_file" --output_html "$html_file" --max_points "$MAX_VIEW_POINTS")
      if [[ "$SHOW_3D" -eq 1 ]]; then
        VIEW_CMD+=(--show)
      fi
      "${VIEW_CMD[@]}"
    else
      echo "[Run] Skipping 3D view (missing $pred_file)"
    fi
  fi
fi

echo "[Run] Inference finished"
