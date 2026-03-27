"""Check if clamp difference causes SSIM gradient divergence."""

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

    # Check pixel statistics
    p0 = clone_params()
    s0, q0, o0 = prepare(p0)
    with torch.no_grad():
        img_cuda = cuda_render(
            w2c, intrinsic, W, H, p0["means"], s0, q0, o0, p0["sh_coeffs_dc"], p0["sh_coeffs_rest"]
        )
        img_torch = torch_render(
            w2c, intrinsic, W, H, p0["means"], s0, q0, o0, p0["sh_coeffs_dc"], p0["sh_coeffs_rest"]
        )

    # check range before scaling (divide by 255)
    cuda_pre = img_cuda / 255.0
    torch_pre = img_torch / 255.0  # already clamped

    print(f"CUDA pre-scale range: [{cuda_pre.min():.8f}, {cuda_pre.max():.8f}]")
    print(f"Torch pre-scale range: [{torch_pre.min():.8f}, {torch_pre.max():.8f}]")
    print(f"CUDA pixels == 0: {(cuda_pre == 0).sum().item()} / {cuda_pre.numel()}")
    print(f"CUDA pixels < 0: {(cuda_pre < 0).sum().item()}")
    print(f"CUDA pixels > 1: {(cuda_pre > 1).sum().item()}")
    print(f"Torch pixels == 0: {(torch_pre == 0).sum().item()} / {torch_pre.numel()}")

    # Now test SSIM with CUDA render but WITH clamp
    def compute_ssim_loss(img, tgt):
        return 1.0 - fused_ssim(
            img.permute(2, 0, 1).unsqueeze(0),
            tgt.permute(2, 0, 1).unsqueeze(0),
            padding="valid",
        )

    # CUDA with clamp (like torch)
    p1 = clone_params()
    s1, q1, o1 = prepare(p1)
    img1_raw = cuda_render(
        w2c, intrinsic, W, H, p1["means"], s1, q1, o1, p1["sh_coeffs_dc"], p1["sh_coeffs_rest"]
    )
    img1_clamped = (img1_raw / 255.0).clamp(0, 1) * 255.0
    loss1 = compute_ssim_loss(img1_clamped, target)
    loss1.backward()

    # Torch (already has clamp)
    p2 = clone_params()
    s2, q2, o2 = prepare(p2)
    img2 = torch_render(
        w2c, intrinsic, W, H, p2["means"], s2, q2, o2, p2["sh_coeffs_dc"], p2["sh_coeffs_rest"]
    )
    loss2 = compute_ssim_loss(img2, target)
    loss2.backward()

    print(f"\nSSIM loss: cuda_clamped={loss1.item():.6f} torch={loss2.item():.6f}")
    print(f"\nGradient comparison (CUDA with clamp vs Torch):")
    for n in names:
        gc = p1[n].grad.flatten()
        gt = p2[n].grad.flatten()
        cos = torch.dot(gc, gt).item() / max(gc.norm().item() * gt.norm().item(), 1e-12)
        rel = (gc - gt).norm().item() / max(gt.norm().item(), 1e-12)
        print(f"  {n:16s} cos={cos:.6f} rel={rel:.6e}")


if __name__ == "__main__":
    main()
