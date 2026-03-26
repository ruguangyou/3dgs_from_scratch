#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/yrg/miniconda3/envs/3dgs/bin/python}"
SEEDS="${1:-3}"

cd "${PROJECT_ROOT}"
PYTHONPATH="${PROJECT_ROOT}" "${PYTHON_BIN}" -m scripts.check_cuda_gradients --seeds "${SEEDS}" --device cuda