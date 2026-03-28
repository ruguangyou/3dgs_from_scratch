import logging
import pickle
import torch
from fused_ssim import fused_ssim
from PIL import Image
from src.dataset import Dataset


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def evaluate(ckpt_path, use_cuda_rasterizer=True):
    # load the trained model
    checkpoint = torch.load(ckpt_path)
    means = checkpoint["learnable_params"]["means"].cuda()
    scales = checkpoint["learnable_params"]["scales"].cuda()
    quaternions = checkpoint["learnable_params"]["quaternions"].cuda()
    opacities = checkpoint["learnable_params"]["opacities"].cuda()
    sh_coeffs_dc = checkpoint["learnable_params"]["sh_coeffs_dc"].cuda()
    sh_coeffs_rest = checkpoint["learnable_params"]["sh_coeffs_rest"].cuda()

    # load camera data for evaluation
    with open("colmap_data/input_data.pkl", "rb") as f:
        camera_data, _, _ = pickle.load(f)
    eval_dataset = Dataset(camera_data, split="eval")
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,  # no shuffling for evaluation
    )

    if use_cuda_rasterizer:
        from src.cuda.wrapper import render
    else:
        from src.torch_rasterizer import render

    eval_dataiter = iter(eval_dataloader)
    for idx, camera in enumerate(eval_dataiter):
        world_to_camera = camera["world_to_camera"].squeeze(0).cuda()  # (4, 4)
        intrinsic = camera["intrinsic"].squeeze(0).cuda()  # (3, 3)
        target_image = camera["image"].squeeze(0).cuda()  # (H, W, C)

        rendered_image, _, _ = render(
            world_to_camera,
            intrinsic,
            target_image.shape[1],
            target_image.shape[0],
            means,
            torch.exp(scales),
            quaternions / torch.norm(quaternions, dim=1, keepdim=True),
            torch.sigmoid(opacities),
            sh_coeffs_dc,
            sh_coeffs_rest,
        )

        # Normalize to [0, 1] so fused_ssim uses the correct c1/c2 constants (L=1).
        ssim_score = fused_ssim(
            rendered_image.permute(2, 0, 1).unsqueeze(0) / 255.0,
            target_image.permute(2, 0, 1).unsqueeze(0) / 255.0,
            padding="valid",
        ).item()
        logging.info(f"Evaluation image {idx}: SSIM = {ssim_score:.4f}")

        concat_image = torch.cat([rendered_image, target_image], dim=1)  # (H, 2*W, C)
        Image.fromarray(concat_image.cpu().byte().numpy()).save(f"logs/eval/frame_{idx}.png")


if __name__ == "__main__":
    ckpt_path = "logs/trained_gaussians_5000.pth"  # Update this path if needed
    evaluate(ckpt_path, use_cuda_rasterizer=True)
