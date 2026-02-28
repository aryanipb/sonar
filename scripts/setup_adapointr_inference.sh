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

INSTALL_PIP=1
BUILD_EXTENSIONS="${BUILD_EXTENSIONS:-0}"
SMOKE_TEST=0
CONFIG_PATH="cfgs/PCN_models/AdaPoinTr.yaml"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --no-pip)
      INSTALL_PIP=0
      shift
      ;;
    --build-extensions)
      BUILD_EXTENSIONS=1
      shift
      ;;
    --smoke-test)
      SMOKE_TEST=1
      shift
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 2
      ;;
  esac
done

echo "[Setup] Using Python: $PYTHON_BIN"
"$PYTHON_BIN" --version

if [[ "$INSTALL_PIP" -eq 1 ]]; then
  echo "[Setup] Installing Python requirements"
  "$PYTHON_BIN" -m pip install --upgrade pip
  "$PYTHON_BIN" -m pip install -r requirements.txt
fi

if [[ "$BUILD_EXTENSIONS" -eq 1 ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    echo "[Setup] Building optional CUDA extensions via install.sh"
    bash install.sh || echo "[Setup] Optional extension build failed; inference can still work without Chamfer loss."
  else
    echo "[Setup] nvcc not found; skipping optional extension build"
  fi
fi

echo "[Setup] Validating environment"
if [[ "$SMOKE_TEST" -eq 1 ]]; then
  "$PYTHON_BIN" tools/check_inference_env.py --config "$CONFIG_PATH" --smoke_test
else
  "$PYTHON_BIN" tools/check_inference_env.py --config "$CONFIG_PATH"
fi

echo "[Setup] Done"
