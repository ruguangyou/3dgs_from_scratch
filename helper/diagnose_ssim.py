"""Check if SSIM gradient propagation differs between CUDA and torch paths."""

import pickle
import torch
from fused_ssim import fused_ssim
from src.cuda.wrapper import render as cuda_render
from src.torch_rasterizer import render as torch_render
from src.gaussian import downsample_point_cloud, initialize
from src.dataset import Dataset, resize_camera


def main():
    torch.manual_seed(400)
    device = "cuda"

    with open("colmap_data/input_data.pkl", "rb") as f:
        camera_data, points, rgbs = pickle.load(f)

    points, rgbs = downsample_point_cloud(points, rgbs, 50000)
    points = torch.from_numpy(points).float()
    rgbs = torch.from_numpy(rgbs / 255.0).float()
    base_params = initialize(points, rgbs, sh_degree=3)

    train_dataset = Dataset(camera_data, split="train")
    sample = train_dataset[0]
    sample = resize_camera(
        {
            "world_to_camera": sample["world_to_camera"].unsqueeze(0),
            "intrinsic": sample["intrinsic"].unsqueeze(0),
            "image": sample["image"].unsqueeze(0),
        },
        0.5,
    )
    w2c = sample["world_to_camera"].squeeze(0).to(device)
    intrinsic = sample["intrinsic"].squeeze(0).to(device)
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

    # capture gradient at rendered_image level
    grad_at_image = {}

    def hook_factory(name):
        def hook(grad):
            grad_at_image[name] = grad.clone()

        return hook

    # CUDA path - SSIM only
    p1 = clone_params()
    s1, q1, o1 = prepare(p1)
    img_cuda, _, _ = cuda_render(
        w2c, intrinsic, W, H, p1["means"], s1, q1, o1, p1["sh_coeffs_dc"], p1["sh_coeffs_rest"]
    )
    img_cuda.register_hook(hook_factory("cuda"))
    ssim_cuda = 1.0 - fused_ssim(
        img_cuda.permute(2, 0, 1).unsqueeze(0),
        target.permute(2, 0, 1).unsqueeze(0),
        padding="valid",
    )
    ssim_cuda.backward()

    # Torch path - SSIM only
    p2 = clone_params()
    s2, q2, o2 = prepare(p2)
    img_torch, _, _ = torch_render(
        w2c, intrinsic, W, H, p2["means"], s2, q2, o2, p2["sh_coeffs_dc"], p2["sh_coeffs_rest"]
    )
    img_torch.register_hook(hook_factory("torch"))
    ssim_torch = 1.0 - fused_ssim(
        img_torch.permute(2, 0, 1).unsqueeze(0),
        target.permute(2, 0, 1).unsqueeze(0),
        padding="valid",
    )
    ssim_torch.backward()

    # compare gradient AT the image level (should be ~same since images are ~same)
    gc_img = grad_at_image["cuda"].flatten()
    gt_img = grad_at_image["torch"].flatten()
    cos_img = torch.dot(gc_img, gt_img).item() / max(
        gc_img.norm().item() * gt_img.norm().item(), 1e-12
    )
    rel_img = (gc_img - gt_img).norm().item() / max(gt_img.norm().item(), 1e-12)
    print(f"SSIM gradient at image level: cos={cos_img:.6f} rel={rel_img:.6e}")
    print(f"  cuda norm={gc_img.norm():.6e} torch norm={gt_img.norm():.6e}")
    print(
        f"  max abs diff={grad_at_image['cuda'].abs().max():.6e} / {grad_at_image['torch'].abs().max():.6e}"
    )

    # compare parameter gradients
    print(f"\nSSIM-only parameter gradients:")
    for n in names:
        gc = p1[n].grad.flatten()
        gt = p2[n].grad.flatten()
        cos = torch.dot(gc, gt).item() / max(gc.norm().item() * gt.norm().item(), 1e-12)
        rel = (gc - gt).norm().item() / max(gt.norm().item(), 1e-12)
        print(
            f"  {n:16s} cos={cos:.6f} rel={rel:.6e} "
            f"norm_cuda={gc.norm():.6e} norm_torch={gt.norm():.6e}"
        )

    # Now test: use torch-computed SSIM gradient to backprop through CUDA rasterizer
    # by manually calling rasterize_backward with the same gradient
    print(f"\n--- Cross-check: SSIM grad from torch image → cuda backward ---")
    p3 = clone_params()
    s3, q3, o3 = prepare(p3)
    img3, _, _ = cuda_render(
        w2c, intrinsic, W, H, p3["means"], s3, q3, o3, p3["sh_coeffs_dc"], p3["sh_coeffs_rest"]
    )
    # use the torch gradient as upstream
    # to test: if we feed the same gradient, does cuda backward produce matching gradients?
    fake_loss = (img3 * grad_at_image["torch"].detach()).sum()
    fake_loss.backward()

    print(f"Parameter gradients (torch SSIM grad → cuda backward):")
    for n in names:
        gc = p3[n].grad.flatten()
        gt = p2[n].grad.flatten()
        cos = torch.dot(gc, gt).item() / max(gc.norm().item() * gt.norm().item(), 1e-12)
        rel = (gc - gt).norm().item() / max(gt.norm().item(), 1e-12)
        print(f"  {n:16s} cos={cos:.6f} rel={rel:.6e}")


if __name__ == "__main__":
    main()
