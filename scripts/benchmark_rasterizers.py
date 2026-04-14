from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from scripts.train import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark torch and CUDA rasterizers during training."
    )
    parser.add_argument("--worker", action="store_true", help="Run a single backend benchmark.")
    parser.add_argument("--backend", choices=["torch", "cuda"], default="cuda")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--resolution-warmup-steps", type=int, default=250)
    parser.add_argument("--sh-degree-warmup-steps", type=int, default=1000)
    parser.add_argument("--ssim-lambda", type=float, default=0.2)
    parser.add_argument("--ssim-warmup-steps", type=int, default=3000)
    parser.add_argument("--scale-reg", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260410)
    parser.add_argument(
        "--downsample-points",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Downsample the COLMAP point cloud before training.",
    )
    parser.add_argument("--initial-max-points", type=int, default=50000)
    parser.add_argument(
        "--load-cached-input",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use cached COLMAP input_data.pkl when available.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be greater than 0.")
    if args.repeats <= 0:
        raise ValueError("--repeats must be greater than 0.")
    if args.load_cached_input and not Path("colmap_data/input_data.pkl").exists():
        raise FileNotFoundError(
            "colmap_data/input_data.pkl not found. Run one training pass first, "
            "or use --no-load-cached-input."
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_megabytes(num_bytes: int) -> float:
    return num_bytes / (1024**2)


def compute_improvement(torch_value: float, cuda_value: float) -> float:
    if torch_value == 0:
        return 0.0
    return (torch_value - cuda_value) / torch_value * 100.0


def run_worker(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    set_seed(args.seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start_time = time.perf_counter()
    train(
        batch_size=args.batch_size,
        sh_degree=args.sh_degree,
        max_steps=args.max_steps,
        resolution_warmup_steps=args.resolution_warmup_steps,
        sh_degree_warmup_steps=args.sh_degree_warmup_steps,
        ssim_lambda=args.ssim_lambda,
        ssim_warmup_steps=args.ssim_warmup_steps,
        scale_reg=args.scale_reg,
        downsample_points=args.downsample_points,
        initial_max_points=args.initial_max_points,
        load_cached_input=args.load_cached_input,
        use_cuda_rasterizer=args.backend == "cuda",
        use_tensorboard=False,
        save_checkpoint=False,
    )
    torch.cuda.synchronize()
    total_time_sec = time.perf_counter() - start_time

    result = {
        "backend": args.backend,
        "batch_size": args.batch_size,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "total_time_sec": total_time_sec,
        "avg_step_time_ms": total_time_sec / args.max_steps * 1000.0,
        "peak_memory_allocated_mb": to_megabytes(torch.cuda.max_memory_allocated()),
        "peak_memory_reserved_mb": to_megabytes(torch.cuda.max_memory_reserved()),
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    print(json.dumps(result, ensure_ascii=False))


def build_worker_command(args: argparse.Namespace, backend: str, repeat_index: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "scripts.benchmark_rasterizers",
        "--worker",
        "--backend",
        backend,
        "--max-steps",
        str(args.max_steps),
        "--batch-size",
        str(args.batch_size),
        "--sh-degree",
        str(args.sh_degree),
        "--resolution-warmup-steps",
        str(args.resolution_warmup_steps),
        "--sh-degree-warmup-steps",
        str(args.sh_degree_warmup_steps),
        "--ssim-lambda",
        str(args.ssim_lambda),
        "--ssim-warmup-steps",
        str(args.ssim_warmup_steps),
        "--scale-reg",
        str(args.scale_reg),
        "--seed",
        str(args.seed + repeat_index),
        "--initial-max-points",
        str(args.initial_max_points),
        f"--{'downsample-points' if args.downsample_points else 'no-downsample-points'}",
        f"--{'load-cached-input' if args.load_cached_input else 'no-load-cached-input'}",
    ]


def parse_worker_result(stdout: str) -> dict[str, float | int | str]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise RuntimeError("Failed to parse worker benchmark output.")


def run_worker_process(command: list[str], env: dict[str, str]) -> tuple[int, str]:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        env=env,
        bufsize=1,
    )

    stdout_lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        stdout_lines.append(line)

    return_code = process.wait()
    return return_code, "".join(stdout_lines)


def summarize(results: list[dict[str, float | int | str]]) -> dict[str, float]:
    total_times = [float(item["total_time_sec"]) for item in results]
    avg_steps = [float(item["avg_step_time_ms"]) for item in results]
    allocs = [float(item["peak_memory_allocated_mb"]) for item in results]
    reserves = [float(item["peak_memory_reserved_mb"]) for item in results]

    def std(values: list[float]) -> float:
        return statistics.stdev(values) if len(values) > 1 else 0.0

    return {
        "mean_total_time_sec": statistics.mean(total_times),
        "std_total_time_sec": std(total_times),
        "mean_avg_step_time_ms": statistics.mean(avg_steps),
        "std_avg_step_time_ms": std(avg_steps),
        "mean_peak_memory_allocated_mb": statistics.mean(allocs),
        "std_peak_memory_allocated_mb": std(allocs),
        "mean_peak_memory_reserved_mb": statistics.mean(reserves),
        "std_peak_memory_reserved_mb": std(reserves),
    }


def run_parent(args: argparse.Namespace) -> None:
    logs_dir = Path("logs/benchmark")
    logs_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    project_root = str(Path(__file__).resolve().parents[1])
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        project_root if not existing_pythonpath else f"{project_root}:{existing_pythonpath}"
    )

    all_results: dict[str, list[dict[str, float | int | str]]] = {"torch": [], "cuda": []}
    for repeat_index in range(args.repeats):
        backend_order = ("torch", "cuda") if repeat_index % 2 == 0 else ("cuda", "torch")
        for backend in backend_order:
            command = build_worker_command(args, backend, repeat_index)
            print(
                f"\n>>> Running {backend} benchmark "
                f"({repeat_index + 1}/{args.repeats}) with max_steps={args.max_steps}",
                flush=True,
            )
            return_code, stdout = run_worker_process(command, env)
            if return_code != 0:
                raise RuntimeError(
                    f"Benchmark worker failed for backend={backend}, repeat={repeat_index}.\n"
                    f"STDOUT:\n{stdout}"
                )
            result = parse_worker_result(stdout)
            all_results[backend].append(result)
            print(
                f"[{backend}] repeat {repeat_index + 1}/{args.repeats}: "
                f"time={float(result['total_time_sec']):.3f}s, "
                f"peak_alloc={float(result['peak_memory_allocated_mb']):.2f}MB, "
                f"peak_reserved={float(result['peak_memory_reserved_mb']):.2f}MB"
            )

    torch_summary = summarize(all_results["torch"])
    cuda_summary = summarize(all_results["cuda"])
    improvement = {
        "time_improvement_pct": compute_improvement(
            torch_summary["mean_total_time_sec"], cuda_summary["mean_total_time_sec"]
        ),
        "allocated_memory_improvement_pct": compute_improvement(
            torch_summary["mean_peak_memory_allocated_mb"],
            cuda_summary["mean_peak_memory_allocated_mb"],
        ),
        "reserved_memory_improvement_pct": compute_improvement(
            torch_summary["mean_peak_memory_reserved_mb"],
            cuda_summary["mean_peak_memory_reserved_mb"],
        ),
    }

    report = {
        "config": {
            "repeats": args.repeats,
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "sh_degree": args.sh_degree,
            "resolution_warmup_steps": args.resolution_warmup_steps,
            "sh_degree_warmup_steps": args.sh_degree_warmup_steps,
            "ssim_lambda": args.ssim_lambda,
            "ssim_warmup_steps": args.ssim_warmup_steps,
            "scale_reg": args.scale_reg,
            "downsample_points": args.downsample_points,
            "initial_max_points": args.initial_max_points,
            "load_cached_input": args.load_cached_input,
        },
        "torch": {
            "summary": torch_summary,
            "runs": all_results["torch"],
        },
        "cuda": {
            "summary": cuda_summary,
            "runs": all_results["cuda"],
        },
        "improvement": improvement,
    }

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_path = logs_dir / f"rasterizer_benchmark_{timestamp}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print("\n=== Benchmark Summary ===")
    print(
        "Torch : "
        f"time={torch_summary['mean_total_time_sec']:.3f}±{torch_summary['std_total_time_sec']:.3f}s, "
        f"peak_alloc={torch_summary['mean_peak_memory_allocated_mb']:.2f}±"
        f"{torch_summary['std_peak_memory_allocated_mb']:.2f}MB"
    )
    print(
        "CUDA  : "
        f"time={cuda_summary['mean_total_time_sec']:.3f}±{cuda_summary['std_total_time_sec']:.3f}s, "
        f"peak_alloc={cuda_summary['mean_peak_memory_allocated_mb']:.2f}±"
        f"{cuda_summary['std_peak_memory_allocated_mb']:.2f}MB"
    )
    print(
        "CUDA improvement: "
        f"time={improvement['time_improvement_pct']:.2f}%, "
        f"peak_alloc={improvement['allocated_memory_improvement_pct']:.2f}%, "
        f"peak_reserved={improvement['reserved_memory_improvement_pct']:.2f}%"
    )
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    arguments = parse_args()
    validate_args(arguments)
    if arguments.worker:
        run_worker(arguments)
    else:
        run_parent(arguments)
