"""Diagnose gradient divergence between CUDA and torch renderers."""

import pickle
import torch
from fused_ssim import fused_ssim
from src.cuda.wrapper import render as cuda_render
from src.torch_rasterizer import render as torch_render
from src.gaussian import downsample_point_cloud, initialize
from src.dataset import Dataset


def main():
    torch.manual_seed(400)
    device = "cuda"

    with open("colmap_data/input_data.pkl", "rb") as f:
        camera_data, points, rgbs = pickle.load(f)

    points, rgbs = downsample_point_cloud(points, rgbs, 50000)
    points = torch.from_numpy(points).float()
    rgbs = torch.from_numpy(rgbs / 255.0).float()
    base_params = initialize(points, rgbs, sh_degree=3)

    train_dataset = Dataset(camera_data, image_scale=0.5, split="train")
    sample = train_dataset[0]
    w2c = sample["world_to_camera"].to(device)
    intrinsic = sample["intrinsic"].to(device)
    target = sample["image"].to(device)
    H, W = target.shape[0], target.shape[1]
    print(f"Image size: {W}x{H}, tiles: {(W + 15) // 16}x{(H + 15) // 16}")

    names = ["means", "scales", "quaternions", "opacities", "sh_coeffs_dc", "sh_coeffs_rest"]

    def clone_params():
        return {n: base_params[n].detach().clone().requires_grad_(True) for n in names}

    def prepare(params):
        scales = torch.exp(params["scales"])
        quats = params["quaternions"] / torch.norm(
            params["quaternions"], dim=1, keepdim=True
        ).clamp_min(1e-12)
        opac = torch.sigmoid(params["opacities"])
        return scales, quats, opac

    # forward comparison
    p1 = clone_params()
    s1, q1, o1 = prepare(p1)
    img_cuda, _, _ = cuda_render(
        w2c, intrinsic, W, H, p1["means"], s1, q1, o1, p1["sh_coeffs_dc"], p1["sh_coeffs_rest"]
    )

    p2 = clone_params()
    s2, q2, o2 = prepare(p2)
    img_torch = torch_render(
        w2c, intrinsic, W, H, p2["means"], s2, q2, o2, p2["sh_coeffs_dc"], p2["sh_coeffs_rest"]
    )

    diff = (img_cuda - img_torch).abs()
    print(f"\nForward image comparison:")
    print(f"  CUDA range:  [{img_cuda.min():.4f}, {img_cuda.max():.4f}]")
    print(f"  Torch range: [{img_torch.min():.4f}, {img_torch.max():.4f}]")
    print(f"  Max diff:    {diff.max():.6f}")
    print(f"  Mean diff:   {diff.mean():.6f}")
    print(f"  Pixels > 1:  {(diff > 1).sum().item()}")
    print(f"  Pixels > 10: {(diff > 10).sum().item()}")

    # backward comparison with L1 + SSIM (same as training)
    def compute_loss(img, tgt):
        l1 = torch.nn.functional.l1_loss(img, tgt)
        ssim = 1.0 - fused_ssim(
            img.permute(2, 0, 1).unsqueeze(0),
            tgt.permute(2, 0, 1).unsqueeze(0),
            padding="valid",
        )
        return 0.8 * l1 + 0.2 * ssim

    loss_cuda = compute_loss(img_cuda, target)
    loss_cuda.backward()

    loss_torch = compute_loss(img_torch, target)
    loss_torch.backward()

    print(f"\nL1+SSIM loss: cuda={loss_cuda.item():.6f} torch={loss_torch.item():.6f}")
    print(f"\nGradient comparison (L1+SSIM):")
    for n in names:
        gc = p1[n].grad.flatten()
        gt = p2[n].grad.flatten()
        cos = torch.dot(gc, gt).item() / max(gc.norm().item() * gt.norm().item(), 1e-12)
        rel = (gc - gt).norm().item() / max(gt.norm().item(), 1e-12)
        print(
            f"  {n:16s} cos={cos:.6f} rel={rel:.6e} "
            f"norm_cuda={gc.norm():.6e} norm_torch={gt.norm():.6e}"
        )

    # check which Gaussians have the biggest gradient difference
    diff_means = (p1["means"].grad - p2["means"].grad).norm(dim=1)
    topk = diff_means.topk(10)
    print(f"\nTop 10 Gaussians with biggest means gradient diff:")
    for i, (val, idx) in enumerate(zip(topk.values, topk.indices)):
        gc = p1["means"].grad[idx]
        gt = p2["means"].grad[idx]
        print(f"  [{i}] idx={idx.item()} diff={val:.6f} cuda={gc.tolist()} torch={gt.tolist()}")


if __name__ == "__main__":
    main()
