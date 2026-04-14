# 3DGS from Scratch

A from-scratch **3D Gaussian Splatting (3DGS)** learning/experiment repository with:

- A PyTorch-based training pipeline
- Two rasterization implementations: pure Torch and CUDA
- Custom CUDA ops (forward and backward)
- Training, evaluation, rendering test, and CUDA gradient consistency scripts

## Repository Structure

Core directories:

- `src/`: core implementation (data loading, Gaussian params, training strategy, Torch rasterizer, CUDA wrapper)
- `src/cuda/`: CUDA/C++ extension source
- `scripts/`: entry points for training, evaluation, test rendering, and gradient checks
- `scripts/benchmark_rasterizers.py`: training benchmark for Torch vs CUDA rasterizers
- `colmap_data/`: COLMAP data directory (input)
- `logs/`: checkpoints, eval images, TensorBoard logs, test inputs and outputs
- `run.sh`: unified command entry script

## Requirements

- Linux
- Python >= 3.10
- CUDA-enabled PyTorch
- Reconstructed COLMAP data

## Install & Build

Create environment:

```bash
conda env create -f environment.yml

conda activate 3dgs
```

Build the `cuda_rasterizer` extension module used by `src/cuda/wrapper.py`.

```bash
python setup.py build_ext --inplace
```

## Data Preparation

The project expects this default layout under `colmap_data/`:

```text
colmap_data/
  images/
  sparse/
    0/
```

On first training run, parsed inputs are cached to `colmap_data/input_data.pkl`.

## Run

- Training arguments are defined in `train()` in `scripts/train.py`.
- Evaluation entry is in `scripts/evaluate.py`.
- Test rendering entry is in `scripts/test.py`.
- CUDA gradient check entry is in `scripts/check_cuda_gradients.py`.
- Rasterizer benchmark entry is in `scripts/benchmark_rasterizers.py`.

Use the root script for all workflows:

```bash
./run.sh train
./run.sh evaluate
./run.sh test
./run.sh gradcheck          # default seeds=3
./run.sh gradcheck 5        # custom seed count
```

Benchmark Torch vs CUDA rasterizers during training:

```bash
./run.sh benchmark --max-steps 100 --repeats 3
```

Notes:

- If `colmap_data/input_data.pkl` does not exist yet, run one training pass first, or add `--no-load-cached-input`.
- The benchmark disables checkpoint saving to reduce I/O noise.
- For a more representative full-training comparison, use a larger `--max-steps` (for example, >500 so densification/pruning can start).

The benchmark reports:

- total training time
- average per-step time
- peak GPU memory allocated
- peak GPU memory reserved
- percentage improvement of CUDA relative to Torch

To override the interpreter:

```bash
PYTHON_BIN=/path/to/python ./run.sh train
```

## Outputs

- Training checkpoints: `logs/trained_gaussians_<max_steps>.pth`
- Evaluation visualizations: `logs/eval/frame_*.png`
- Test render sequence: `logs/test/novel_views/frame_*.png`
- TensorBoard logs: `logs/tensorboard/<timestamp>/`
