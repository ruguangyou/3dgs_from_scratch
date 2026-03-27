"""
Side-by-side training with CUDA and torch rasterizers.
Track loss, image center-of-mass, and parameter divergence per step.
"""

from __future__ import annotations

import math
import pickle

import torch

from src.dataset import Dataset
from src.gaussian import downsample_point_cloud, initialize
from src.cuda.wrapper import render as cuda_render
from src.torch_rasterizer import render as torch_render
from scripts.train import setup_optimizers


DEVICE = "cuda"
POINTS = 3000
SH_DEGREE = 3
IMAGE_SCALE = 0.5
STEPS = 300
SSIM_WARMUP_STEPS = 3000
SSIM_LAMBDA = 0.2
SCALE_REG = 0.01
GRAD_CLIP_NORM = 1.0
SH_DEGREE_INCREASE_STEP = 1000


def center_of_mass_norm(img: torch.Tensor) -> tuple[float, float]:
    """Normalized center offset from image center. Negative = upper-left."""
    lum = img.detach().float().mean(dim=-1)
    total = lum.sum()
    if total < 1e-12:
        return float("nan"), float("nan")
    H, W = lum.shape
    ys = torch.arange(H, device=img.device, dtype=torch.float32).unsqueeze(1)
    xs = torch.arange(W, device=img.device, dtype=torch.float32).unsqueeze(0)
    cy = float((ys * lum).sum() / total)
    cx = float((xs * lum).sum() / total)
    return (cx - (W - 1) / 2) / W, (cy - (H - 1) / 2) / H


def make_camera_sequence(length: int, total: int) -> list[int]:
    gen = torch.Generator().manual_seed(20260327)
    return torch.randint(0, total, (length,), generator=gen).tolist()


def clone_params(base: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone().requires_grad_(True) for k, v in base.items()}


def run_training(
    tag: str,
    render_fn,
    base_params: dict[str, torch.Tensor],
    dataset: Dataset,
    cam_seq: list[int],
) -> list[dict]:
    params = clone_params(base_params)
    optimizers = setup_optimizers(params, batch_size=1)
    means_sched = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=0.01 ** (1 / STEPS)
    )

    records: list[dict] = []
    for step, cam_idx in enumerate(cam_seq):
        scales = torch.exp(params["scales"])
        quats = params["quaternions"] / torch.norm(
            params["quaternions"], dim=1, keepdim=True
        ).clamp_min(1e-12)
        opac = torch.sigmoid(params["opacities"])

        cam = dataset[cam_idx]
        w2c = cam["world_to_camera"].to(DEVICE)
        K = cam["intrinsic"].to(DEVICE)
        target = cam["image"].to(DEVICE)

        img = render_fn(
            w2c,
            K,
            target.shape[1],
            target.shape[0],
            params["means"],
            scales,
            quats,
            opac,
            params["sh_coeffs_dc"],
            params["sh_coeffs_rest"],
        )
        rn = img / 255.0
        tn = target / 255.0
        l1_loss = torch.nn.functional.l1_loss(rn, tn)
        eff_ssim = SSIM_LAMBDA * min(step / SSIM_WARMUP_STEPS, 1.0)
        loss = (1 - eff_ssim) * l1_loss
        loss = loss + SCALE_REG * (params["scales"] ** 2).mean()
        loss.backward()

        if GRAD_CLIP_NORM > 0:
            torch.nn.utils.clip_grad_norm_(list(params.values()), GRAD_CLIP_NORM)

        sh_deg = min(SH_DEGREE, step // SH_DEGREE_INCREASE_STEP)
        if sh_deg < SH_DEGREE:
            params["sh_coeffs_rest"].grad[:, (sh_deg + 1) ** 2 - 1 :, :].zero_()

        for opt in optimizers.values():
            opt.step()
            opt.zero_grad()
        means_sched.step()
        with torch.no_grad():
            params["scales"].clamp_(min=-6.0, max=1.5)

        if step % 20 == 0 or step == STEPS - 1:
            cx, cy = center_of_mass_norm(img)
            rec = {
                "step": step,
                "loss": float(loss.item()),
                "l1": float(l1_loss.item()),
                "render_mean": float(rn.mean().item()),
                "cx": cx,
                "cy": cy,
                "means_norm": float(params["means"].detach().norm().item()),
                "scale_mean": float(params["scales"].detach().mean().item()),
                "scale_max": float(params["scales"].detach().max().item()),
                "opac_mean": float(opac.mean().item()),
            }
            records.append(rec)
            print(
                f"[{tag:5s}] step={step:3d} loss={rec['loss']:.6f} l1={rec['l1']:.6f} "
                f"rmean={rec['render_mean']:.4f} center=({cx:+.4f},{cy:+.4f}) "
                f"pos_norm={rec['means_norm']:.2f} scale_mean={rec['scale_mean']:.3f}"
            )

    return records


def main() -> None:
    torch.manual_seed(20260327)
    with open("colmap_data/input_data.pkl", "rb") as f:
        camera_data, points, rgbs = pickle.load(f)

    points, rgbs = downsample_point_cloud(points, rgbs, POINTS)
    pts = torch.from_numpy(points).float()
    rgb = torch.from_numpy(rgbs / 255.0).float()
    base_params = initialize(pts, rgb, sh_degree=SH_DEGREE)

    dataset = Dataset(camera_data, image_scale=IMAGE_SCALE, split="train")
    cam_seq = make_camera_sequence(STEPS, len(dataset))

    print(f"Training {POINTS} gaussians for {STEPS} steps\n")
    cuda_recs = run_training("cuda", cuda_render, base_params, dataset, cam_seq)
    torch.cuda.empty_cache()
    torch_recs = run_training("torch", torch_render, base_params, dataset, cam_seq)

    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(
        f"{'step':>4s}  {'cuda_loss':>10s} {'torch_loss':>10s} {'delta_loss':>10s}  "
        f"{'cuda_cx':>8s} {'cuda_cy':>8s} {'torch_cx':>8s} {'torch_cy':>8s}"
    )
    for c, t in zip(cuda_recs, torch_recs, strict=True):
        print(
            f"{c['step']:4d}  {c['loss']:10.6f} {t['loss']:10.6f} {c['loss'] - t['loss']:+10.6f}  "
            f"{c['cx']:+8.4f} {c['cy']:+8.4f} {t['cx']:+8.4f} {t['cy']:+8.4f}"
        )


if __name__ == "__main__":
    main()
