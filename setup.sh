#!/usr/bin/env sh
# One-time setup for the silero-tts plugin: creates .venv with PyTorch (CPU)
# in the plugin folder and verifies the worker can import torch.
# Requires Python 3.9+ (or uv). Run from anywhere.
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/.venv"
VENV_PY="$VENV/bin/python"

if [ ! -x "$VENV_PY" ]; then
    echo "Creating .venv ..."
    if command -v python3 >/dev/null 2>&1; then
        python3 -m venv "$VENV"
    elif command -v uv >/dev/null 2>&1; then
        uv venv --seed "$VENV"
    else
        echo "neither python3 nor uv found" >&2
        exit 1
    fi
fi

if [ ! -x "$VENV_PY" ]; then
    echo "venv python not found: $VENV_PY" >&2
    exit 1
fi

echo "Installing PyTorch (CPU) + numpy ..."
if ! "$VENV_PY" -m pip install --upgrade pip numpy torch; then
    # A uv-created venv may have no pip bundled; retry through uv itself.
    echo "pip unavailable inside the venv, retrying with uv ..."
    uv pip install --python "$VENV_PY" numpy torch
fi

echo "Checking the worker runtime ..."
"$VENV_PY" -c "import sys, torch, numpy; print('python', sys.version.split()[0]); print('torch', torch.__version__); print('numpy', numpy.__version__)"

echo ""
echo "Done. First synthesis will download the model (~140 MB) once."