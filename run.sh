#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/yrg/miniconda3/envs/3dgs/bin/python}"
SEEDS="${2:-3}"

cd "${PROJECT_ROOT}"

if [ "$1" = "train" ]; then
    PYTHONPATH="${PROJECT_ROOT}" "${PYTHON_BIN}" -m scripts.train
elif [ "$1" = "evaluate" ]; then
    PYTHONPATH="${PROJECT_ROOT}" "${PYTHON_BIN}" -m scripts.evaluate
elif [ "$1" = "test" ]; then
    PYTHONPATH="${PROJECT_ROOT}" "${PYTHON_BIN}" -m scripts.test
elif [ "$1" = "gradcheck" ]; then
    PYTHONPATH="${PROJECT_ROOT}" "${PYTHON_BIN}" -m scripts.check_cuda_gradients --seeds "${SEEDS}" --device cuda
elif [ "$1" = "benchmark" ]; then
    PYTHONPATH="${PROJECT_ROOT}" "${PYTHON_BIN}" -m scripts.benchmark_rasterizers "${@:2}"
elif [ "$1" = "build" ]; then
    PYTHONPATH="${PROJECT_ROOT}" "${PYTHON_BIN}" setup.py build_ext --inplace
else
    echo "Usage: $0 {train|evaluate|test|gradcheck [seeds]|benchmark [args...]|build}"
    exit 1
fi