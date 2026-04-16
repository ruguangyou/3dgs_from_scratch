"""
Diagnose differences between CUDA and torch rasterizers.
Compare forward outputs and backward gradients on identical inputs.
"""

from __future__ import annotations

import pickle
from pathlib import Path
import sys

_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import torch

from src.dataset import Dataset
from src.gaussian import downsample_point_cloud, initialize
from src.cuda.wrapper import render as cuda_render
from src.cuda.wrapper import ProjectPointsFunction, RasterizeFunction
from src.torch_rasterizer import render as torch_render
import cuda_rasterizer


DEVICE = "cuda"
POINTS = 500
SH_DEGREE = 3
IMAGE_SCALE = 0.5


def compare_forward():
    """Render one frame with both rasterizers and compare pixel-by-pixel."""
    torch.manual_seed(42)
    with open("colmap_data/input_data.pkl", "rb") as f:
        camera_data, points, rgbs = pickle.load(f)

    points, rgbs = downsample_point_cloud(points, rgbs, POINTS)
    pts = torch.from_numpy(points).float()
    rgb = torch.from_numpy(rgbs / 255.0).float()
    params = initialize(pts, rgb, sh_degree=SH_DEGREE)

    dataset = Dataset(camera_data, image_scale=IMAGE_SCALE, split="train")
    camera = dataset[0]
    w2c = camera["world_to_camera"].to(DEVICE)
    K = camera["intrinsic"].to(DEVICE)
    target = camera["image"].to(DEVICE)
    H, W = target.shape[0], target.shape[1]

    scales = torch.exp(params["scales"])
    quats = params["quaternions"] / torch.norm(
        params["quaternions"], dim=1, keepdim=True
    ).clamp_min(1e-12)
    opac = torch.sigmoid(params["opacities"])

    with torch.no_grad():
        img_cuda, _, _ = cuda_render(
            w2c,
            K,
            W,
            H,
            params["means"],
            scales,
            quats,
            opac,
            params["sh_coeffs_dc"],
            params["sh_coeffs_rest"],
        )
        img_torch, _, _ = torch_render(
            w2c,
            K,
            W,
            H,
            params["means"],
            scales,
            quats,
            opac,
            params["sh_coeffs_dc"],
            params["sh_coeffs_rest"],
        )

    diff = (img_cuda - img_torch).abs()
    print(f"Forward comparison (H={H}, W={W}, N={POINTS}):")
    print(f"  CUDA  mean={img_cuda.mean():.4f} max={img_cuda.max():.4f}")
    print(f"  Torch mean={img_torch.mean():.4f} max={img_torch.max():.4f}")
    print(f"  AbsDiff mean={diff.mean():.6f} max={diff.max():.6f}")
    print(f"  Nonzero pixels CUDA:  {(img_cuda > 0).any(dim=-1).sum().item()}")
    print(f"  Nonzero pixels Torch: {(img_torch > 0).any(dim=-1).sum().item()}")

    if diff.max() > 1.0:
        # find where the difference is largest
        flat_idx = diff.sum(dim=-1).argmax()
        v_max = flat_idx // W
        u_max = flat_idx % W
        print(f"\n  Largest diff at pixel (u={u_max.item()}, v={v_max.item()}):")
        print(f"    CUDA:  {img_cuda[v_max, u_max]}")
        print(f"    Torch: {img_torch[v_max, u_max]}")


def compare_gradients():
    """Do one forward+backward with both and compare all gradients."""
    torch.manual_seed(42)
    with open("colmap_data/input_data.pkl", "rb") as f:
        camera_data, points, rgbs = pickle.load(f)

    points, rgbs = downsample_point_cloud(points, rgbs, POINTS)
    pts = torch.from_numpy(points).float()
    rgb = torch.from_numpy(rgbs / 255.0).float()
    base_params = initialize(pts, rgb, sh_degree=SH_DEGREE)

    dataset = Dataset(camera_data, image_scale=IMAGE_SCALE, split="train")
    camera = dataset[0]
    w2c = camera["world_to_camera"].to(DEVICE)
    K = camera["intrinsic"].to(DEVICE)
    target = camera["image"].to(DEVICE)
    H, W = target.shape[0], target.shape[1]

    results = {}
    for tag, render_fn in [("cuda", cuda_render), ("torch", torch_render)]:
        p = {name: t.detach().clone().requires_grad_(True) for name, t in base_params.items()}
        scales = torch.exp(p["scales"])
        quats = p["quaternions"] / torch.norm(p["quaternions"], dim=1, keepdim=True).clamp_min(
            1e-12
        )
        opac = torch.sigmoid(p["opacities"])

        img = render_fn(
            w2c, K, W, H, p["means"], scales, quats, opac, p["sh_coeffs_dc"], p["sh_coeffs_rest"]
        )
        loss = torch.nn.functional.l1_loss(img / 255.0, target / 255.0)
        loss.backward()
        results[tag] = {
            "loss": loss.item(),
            "image": img.detach(),
            "grads": {
                name: p[name].grad.detach().clone() for name in p if p[name].grad is not None
            },
        }

    print(f"\nGradient comparison:")
    print(f"  CUDA  loss={results['cuda']['loss']:.8f}")
    print(f"  Torch loss={results['torch']['loss']:.8f}")

    for name in results["cuda"]["grads"]:
        gc = results["cuda"]["grads"][name]
        gt = results["torch"]["grads"][name]
        cos = torch.nn.functional.cosine_similarity(
            gc.flatten().unsqueeze(0), gt.flatten().unsqueeze(0)
        ).item()
        rel = (gc - gt).norm().item() / max(gt.norm().item(), 1e-12)
        mean_cuda = gc.mean().item()
        mean_torch = gt.mean().item()
        print(
            f"  {name:20s} cos={cos:.6f} rel={rel:.6e} "
            f"mean_cuda={mean_cuda:+.6e} mean_torch={mean_torch:+.6e}"
        )


def compare_intermediate():
    """Compare intermediate rasterization outputs to find mismatch."""
    torch.manual_seed(42)
    with open("colmap_data/input_data.pkl", "rb") as f:
        camera_data, points, rgbs = pickle.load(f)

    points, rgbs = downsample_point_cloud(points, rgbs, POINTS)
    pts = torch.from_numpy(points).float()
    rgb = torch.from_numpy(rgbs / 255.0).float()
    params = initialize(pts, rgb, sh_degree=SH_DEGREE)

    dataset = Dataset(camera_data, image_scale=IMAGE_SCALE, split="train")
    camera = dataset[0]
    w2c = camera["world_to_camera"].to(DEVICE)
    K = camera["intrinsic"].to(DEVICE)
    target = camera["image"].to(DEVICE)
    H, W = target.shape[0], target.shape[1]

    scales = torch.exp(params["scales"]).detach()
    quats = (
        params["quaternions"]
        / torch.norm(params["quaternions"], dim=1, keepdim=True).clamp_min(1e-12)
    ).detach()
    opac = torch.sigmoid(params["opacities"]).detach()
    means = params["means"].detach()
    sh_dc = params["sh_coeffs_dc"].detach()
    sh_rest = params["sh_coeffs_rest"].detach()

    # --- CUDA intermediate ---
    pts_img_c, depths_c, cov_c, cov_inv_c, radii_c, mask_c = cuda_rasterizer.project_points(
        means.contiguous(),
        scales.contiguous(),
        quats.contiguous(),
        opac.contiguous(),
        w2c.contiguous(),
        K.contiguous(),
        0.01,
        100.0,
        1 / 255,
        0.5,
        128.0,
        W,
        H,
    )
    camera_pos = -w2c[:3, :3].t() @ w2c[:3, 3]
    colors_c = cuda_rasterizer.evaluate_spherical_harmonics(
        camera_pos.contiguous(),
        means.contiguous(),
        sh_dc.contiguous(),
        sh_rest.contiguous(),
        mask_c.contiguous(),
        False,
    )

    # --- Torch intermediate ---
    from src.gaussian import evaluate_spherical_harmonics, quaternion_to_rotation_matrix

    N = means.shape[0]
    points_cam = (w2c @ torch.cat([means, torch.ones(N, 1, device=DEVICE)], dim=1).t())[:3, :]
    depth_t = points_cam[2, :]
    x, y, z = points_cam.unbind(0)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u_t = fx * x / z + cx
    v_t = fy * y / z + cy

    # Compare projected points
    valid = mask_c
    print(f"\nIntermediate comparison (N={N}, visible={valid.sum().item()}):")
    diff_u = (pts_img_c[valid, 0] - u_t[valid]).abs()
    diff_v = (pts_img_c[valid, 1] - v_t[valid]).abs()
    print(f"  Projected u: max_diff={diff_u.max():.8f} mean_diff={diff_u.mean():.8f}")
    print(f"  Projected v: max_diff={diff_v.max():.8f} mean_diff={diff_v.mean():.8f}")

    # Compare colors
    view_dirs = means[valid] - camera_pos.unsqueeze(0)
    view_dirs = view_dirs / torch.norm(view_dirs, dim=1, keepdim=True)
    colors_t = evaluate_spherical_harmonics(sh_dc[valid], sh_rest[valid], view_dirs)
    diff_colors = (colors_c[valid] - colors_t).abs()
    print(f"  Colors: max_diff={diff_colors.max():.8f} mean_diff={diff_colors.mean():.8f}")

    # Compare covariance inverse
    S = torch.diag_embed(scales)
    R = quaternion_to_rotation_matrix(quats)
    cov_w = R @ S @ S @ R.transpose(1, 2)
    R_w2c = w2c[:3, :3]
    cov_cam = R_w2c.unsqueeze(0) @ cov_w @ R_w2c.t().unsqueeze(0)
    J = torch.zeros((N, 2, 3), device=DEVICE)
    invz = 1 / z
    invz2 = invz * invz
    J[:, 0, 0] = fx * invz
    J[:, 1, 1] = fy * invz
    J[:, 0, 2] = -fx * x * invz2
    J[:, 1, 2] = -fy * y * invz2
    cov_img_t = J @ cov_cam @ J.transpose(1, 2)
    cov_img_t = (cov_img_t + cov_img_t.transpose(1, 2)) / 2

    # Torch: compute inverse from full 2x2
    a_t = cov_img_t[:, 0, 0]
    b_t = cov_img_t[:, 0, 1]
    d_t = cov_img_t[:, 1, 1]
    det_t = (a_t * d_t - b_t * b_t).clamp_min(1e-12)
    inv_t = torch.stack([d_t / det_t, -b_t / det_t, a_t / det_t], dim=1)  # (N, 3)

    diff_cov_inv = (cov_inv_c[valid] - inv_t[valid]).abs()
    print(f"  CovInv: max_diff={diff_cov_inv.max():.8f} mean_diff={diff_cov_inv.mean():.8f}")


if __name__ == "__main__":
    compare_forward()
    compare_gradients()
    compare_intermediate()
