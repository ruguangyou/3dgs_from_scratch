"""Check if grad tensor contiguity causes the SSIM gradient divergence."""

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

    def compute_ssim_loss(img, tgt):
        return 1.0 - fused_ssim(
            img.permute(2, 0, 1).unsqueeze(0),
            tgt.permute(2, 0, 1).unsqueeze(0),
            padding="valid",
        )

    grad_info = {}

    def hook_factory(name):
        def hook(grad):
            grad_info[name] = {
                "contiguous": grad.is_contiguous(),
                "stride": grad.stride(),
                "shape": tuple(grad.shape),
            }

        return hook

    # Test 1: Direct SSIM on cuda_render output
    print("=== Test 1: Direct SSIM (failing case) ===")
    p1 = clone_params()
    s1, q1, o1 = prepare(p1)
    img1, _, _ = cuda_render(
        w2c, intrinsic, W, H, p1["means"], s1, q1, o1, p1["sh_coeffs_dc"], p1["sh_coeffs_rest"]
    )
    img1.register_hook(hook_factory("test1_img"))

    # also check grad at permuted level
    img1_perm = img1.permute(2, 0, 1).unsqueeze(0)
    img1_perm.register_hook(hook_factory("test1_perm"))

    loss1 = 1.0 - fused_ssim(img1_perm, target.permute(2, 0, 1).unsqueeze(0), padding="valid")
    loss1.backward()

    for k, v in grad_info.items():
        print(f"  {k}: contiguous={v['contiguous']} stride={v['stride']} shape={v['shape']}")
    grad_info.clear()

    # Test 2: SSIM with /255 * 255 wrapper (working case)
    print("\n=== Test 2: With /255*255 wrapper (working case) ===")
    p2 = clone_params()
    s2, q2, o2 = prepare(p2)
    img2_raw, _, _ = cuda_render(
        w2c, intrinsic, W, H, p2["means"], s2, q2, o2, p2["sh_coeffs_dc"], p2["sh_coeffs_rest"]
    )
    img2_raw.register_hook(hook_factory("test2_raw"))
    img2 = (img2_raw / 255.0).clamp(0, 1) * 255.0
    img2.register_hook(hook_factory("test2_clamped"))
    loss2 = compute_ssim_loss(img2, target)
    loss2.backward()

    for k, v in grad_info.items():
        print(f"  {k}: contiguous={v['contiguous']} stride={v['stride']} shape={v['shape']}")
    grad_info.clear()

    # Test 3: Direct SSIM but with .contiguous() before passing to SSIM
    print("\n=== Test 3: Direct SSIM with img.contiguous() ===")
    p3 = clone_params()
    s3, q3, o3 = prepare(p3)
    img3, _, _ = cuda_render(
        w2c, intrinsic, W, H, p3["means"], s3, q3, o3, p3["sh_coeffs_dc"], p3["sh_coeffs_rest"]
    )
    img3.register_hook(hook_factory("test3_img"))
    # Force contiguous before permute — shouldn't matter for forward but tests autograd
    loss3 = compute_ssim_loss(img3.contiguous(), target)
    loss3.backward()

    for k, v in grad_info.items():
        print(f"  {k}: contiguous={v['contiguous']} stride={v['stride']} shape={v['shape']}")

    # Compare all three against torch reference
    print("\n=== Comparing parameter gradients ===")
    p_ref = clone_params()
    s_ref, q_ref, o_ref = prepare(p_ref)
    img_ref, _, _ = torch_render(
        w2c,
        intrinsic,
        W,
        H,
        p_ref["means"],
        s_ref,
        q_ref,
        o_ref,
        p_ref["sh_coeffs_dc"],
        p_ref["sh_coeffs_rest"],
    )
    loss_ref = compute_ssim_loss(img_ref, target)
    loss_ref.backward()

    for test_name, params in [("test1_direct", p1), ("test2_div_mul", p2), ("test3_contig", p3)]:
        print(f"\n  {test_name} vs torch_ref:")
        for n in names:
            gc = params[n].grad.flatten()
            gt = p_ref[n].grad.flatten()
            cos = torch.dot(gc, gt).item() / max(gc.norm().item() * gt.norm().item(), 1e-12)
            rel = (gc - gt).norm().item() / max(gt.norm().item(), 1e-12)
            print(f"    {n:16s} cos={cos:.6f} rel={rel:.6e}")


if __name__ == "__main__":
    main()
