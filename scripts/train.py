import logging
import math
import pickle
from datetime import datetime
import torch
from fused_ssim import fused_ssim
from tqdm import tqdm
from src.dataset import load_colmap, resize_camera, Dataset
from src.gaussian import initialize, downsample_point_cloud
from torch.utils.tensorboard import SummaryWriter


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def ensure_finite(name, tensor):
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"Non-finite values detected in {name}.")


def setup_optimizers(
    gaussians,
    batch_size,
    means_lr=1.6e-4,
    scales_lr=5e-3,
    quaternions_lr=1e-3,
    opacities_lr=5e-2,
    sh_dc_lr=2.5e-3,
    sh_rest_lr=2.5e-3 / 20,
):
    lr_dict = {
        "means": means_lr,
        "scales": scales_lr,
        "quaternions": quaternions_lr,
        "opacities": opacities_lr,
        "sh_coeffs_dc": sh_dc_lr,
        "sh_coeffs_rest": sh_rest_lr,
    }
    optimizers = {
        name: torch.optim.Adam(
            # scale learning rates by sqrt of batch size
            [{"params": gaussians[name], "lr": lr * math.sqrt(batch_size)}],
            eps=1e-15,
            betas=(1 - batch_size * (1 - 0.9), 1 - batch_size * (1 - 0.999)),
        )
        for name, lr in lr_dict.items()
    }
    return optimizers


def train(
    batch_size=1,
    sh_degree=3,
    max_steps=30000,
    resolution_warmup_steps=250,
    sh_degree_warmup_steps=1000,
    ssim_lambda=0.2,
    ssim_warmup_steps=3000,
    grad_clip_norm=1.0,
    scale_reg=0.01,
    opacity_reset_interval=3000,
    opacity_reset_value=0.01,
    downsample_points=False,
    initial_max_points=100000,  # ignored if downsample_points is False
    load_cached_input=True,
    use_cuda_rasterizer=True,
    use_tensorboard=True,
    tensorboard_log_dir="logs/tensorboard",
    tensorboard_image_interval=100,
):
    data_dir = "colmap_data"
    if load_cached_input:
        logging.info("Loading cached input...")
        with open(f"{data_dir}/input_data.pkl", "rb") as f:
            camera_data, points, rgbs = pickle.load(f)
    else:
        logging.info("Loading COLMAP data...")
        camera_data, points, rgbs = load_colmap(data_dir)
        with open(f"{data_dir}/input_data.pkl", "wb") as f:
            pickle.dump((camera_data, points, rgbs), f)

    if downsample_points:
        logging.info(f"Downsampling {points.shape[0]} points...")
        points, rgbs = downsample_point_cloud(
            points,
            rgbs,
            initial_max_points,
        )

    train_dataset = Dataset(camera_data, split="train")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    train_dataiter = iter(train_dataloader)

    logging.info(f"Initializing {points.shape[0]} gaussians...")
    points = torch.from_numpy(points).float()
    rgbs = torch.from_numpy(rgbs / 255.0).float()  # Normalize RGB values to [0, 1]
    learnable_params = initialize(points, rgbs, sh_degree=sh_degree)

    logging.info("Optimizer setup...")
    optimizers = setup_optimizers(learnable_params, batch_size)
    # exponential decay, lr_at_end = 0.01 * lr_at_start
    means_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=0.01 ** (1 / max_steps)
    )

    means = learnable_params["means"]
    sh_coeffs_dc = learnable_params["sh_coeffs_dc"]
    sh_coeffs_rest = learnable_params["sh_coeffs_rest"]

    if use_cuda_rasterizer:
        from src.cuda.wrapper import render
    else:
        from src.torch_rasterizer import render

    writer = None
    if use_tensorboard:
        run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(log_dir=f"{tensorboard_log_dir}/{run_name}")
        writer.add_text(
            "train/config",
            (
                f"batch_size={batch_size}, sh_degree={sh_degree}, max_steps={max_steps}, "
                f"ssim_lambda={ssim_lambda}, ssim_warmup_steps={ssim_warmup_steps}, "
                f"use_cuda_rasterizer={use_cuda_rasterizer}"
            ),
        )

    def to_tb_image(image_hwc: torch.Tensor) -> torch.Tensor:
        image = image_hwc.detach().float()
        image = image.permute(2, 0, 1)
        if image.max() > 1.0:
            image = image / 255.0
        return image.clamp(0.0, 1.0)

    logging.info("Training started...")
    try:
        for step in tqdm(range(max_steps), desc="Training"):
            # recalculate the intermediate variables that depend on the learnable parameters
            scales = torch.exp(learnable_params["scales"])
            quaternions = learnable_params["quaternions"]
            quaternions = quaternions / torch.norm(quaternions, dim=1, keepdim=True).clamp_min(
                1e-12
            )
            opacities = torch.sigmoid(learnable_params["opacities"])

            ensure_finite("scales", scales)
            ensure_finite("quaternions", quaternions)
            ensure_finite("opacities", opacities)

            try:
                camera = next(train_dataiter)
            except StopIteration:
                train_dataiter = iter(train_dataloader)
                camera = next(train_dataiter)

            if resolution_warmup_steps > 0:
                if step < resolution_warmup_steps:
                    warmup_scale = 0.25
                elif step < 2 * resolution_warmup_steps:
                    warmup_scale = 0.5
                else:
                    warmup_scale = 1.0
                camera = resize_camera(camera, warmup_scale)

            # batch size is 1, so squeeze the batch dimension
            world_to_camera = camera["world_to_camera"].squeeze(0).cuda()  # (4, 4)
            intrinsic = camera["intrinsic"].squeeze(0).cuda()  # (3, 3)
            target_image = camera["image"].squeeze(0).cuda()  # (H, W, C)

            rendered_image = render(
                world_to_camera,
                intrinsic,
                target_image.shape[1],
                target_image.shape[0],
                means,
                scales,
                quaternions,
                opacities,
                sh_coeffs_dc,
                sh_coeffs_rest,
            )
            ensure_finite("rendered_image", rendered_image)

            # Normalize to [0, 1] for loss computation.
            # render() outputs [0, 255] and target_image is also uint8 [0, 255].
            # fused_ssim expects [0, 1] (uses c1=(0.01)^2, c2=(0.03)^2 for L=1),
            # and scale_reg is calibrated for [0, 1] pixel loss.
            rendered_norm = rendered_image / 255.0
            target_norm = target_image / 255.0

            # L1 loss on pixel colors
            l1_loss = torch.nn.functional.l1_loss(rendered_norm, target_norm)
            # Dissimilarity SSIM on structural similarity
            ssim_loss = 1.0 - fused_ssim(
                rendered_norm.permute(2, 0, 1).unsqueeze(0),  # (H, W, C) -> (1, C, H, W)
                target_norm.permute(2, 0, 1).unsqueeze(0),  # (H, W, C) -> (1, C, H, W)
                padding="valid",  # no padding to avoid border artifacts
            )

            # Linear SSIM warmup: 0 for the first ssim_warmup_steps, then ramp to ssim_lambda
            if ssim_warmup_steps > 0 and step < ssim_warmup_steps:
                effective_ssim_lambda = ssim_lambda * step / ssim_warmup_steps
            else:
                effective_ssim_lambda = ssim_lambda

            loss = (1 - effective_ssim_lambda) * l1_loss + effective_ssim_lambda * ssim_loss
            # Scale regularization: penalize large Gaussians (mirrors gsplat's scale_reg).
            # Without a densification loop to prune oversized Gaussians, this is essential
            # to prevent scale explosion in the CUDA rasterizer.
            if scale_reg > 0.0:
                loss = loss + scale_reg * scales.mean()
            ensure_finite("loss", loss)
            loss.backward()

            for name, param in learnable_params.items():
                if param.grad is not None:
                    ensure_finite(f"{name}.grad", param.grad)

            # Clip gradients to prevent large-scale Gaussians from causing parameter explosion.
            # When a Gaussian covers many pixels, du^2 terms in grad_cov_inv accumulate across
            # all pixels via atomicAdd, potentially producing extremely large gradients.
            if grad_clip_norm is not None and grad_clip_norm > 0:
                all_params = list(learnable_params.values())
                torch.nn.utils.clip_grad_norm_(all_params, grad_clip_norm)

            # increase SH degree every 1000 steps, to progressively learn higher frequency details
            sh_degree_to_use = min(sh_degree, step // sh_degree_warmup_steps)
            # zero out gradients for unused SH coefficients
            if sh_degree_to_use < sh_degree:
                sh_coeffs_rest.grad[:, (sh_degree_to_use + 1) ** 2 - 1 :, :].zero_()

            for optimizer in optimizers.values():
                optimizer.step()
                optimizer.zero_grad()
            means_scheduler.step()

            # Periodic opacity reset: force all Gaussians back to near-zero opacity,
            # mimicking gsplat's DefaultStrategy reset_every=3000. Prevents bloated
            # Gaussians from accumulating indefinitely without a full densification loop.
            if opacity_reset_interval > 0 and step > 0 and step % opacity_reset_interval == 0:
                with torch.no_grad():
                    reset_logit = math.log(opacity_reset_value / (1.0 - opacity_reset_value))
                    learnable_params["opacities"].fill_(reset_logit)
                    # also reset the Adam second-moment to avoid stale momentum
                    for key in ("opacities",):
                        state = optimizers[key].state[learnable_params[key]]
                        if "exp_avg" in state:
                            state["exp_avg"].zero_()
                        if "exp_avg_sq" in state:
                            state["exp_avg_sq"].zero_()
                logging.info(f"Step {step}: opacity reset to {opacity_reset_value}")

            for name, param in learnable_params.items():
                ensure_finite(name, param)

            if writer is not None:
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/l1_loss", l1_loss.item(), step)
                writer.add_scalar("train/ssim_loss", ssim_loss.item(), step)
                writer.add_scalar("train/ssim_lambda", effective_ssim_lambda, step)
                writer.add_scalar("train/means_lr", optimizers["means"].param_groups[0]["lr"], step)
                # monitor scale magnitude to detect explosion early
                writer.add_scalar(
                    "train/scale_raw_max", learnable_params["scales"].max().item(), step
                )
                writer.add_scalar(
                    "train/scale_raw_min", learnable_params["scales"].min().item(), step
                )
                writer.add_scalar(
                    "train/scale_raw_mean", learnable_params["scales"].mean().item(), step
                )
                writer.add_scalar("train/opacity_mean", opacities.mean().item(), step)

                if step % tensorboard_image_interval == 0:
                    writer.add_image("train/rendered", to_tb_image(rendered_image), step)
                    writer.add_image("train/target", to_tb_image(target_image), step)
                    writer.add_image(
                        "train/concat",
                        to_tb_image(torch.cat([rendered_image, target_image], dim=1)),
                        step,
                    )
    finally:
        if writer is not None:
            writer.close()

    # save the learned parameters and training configuration for later use
    torch.save(
        {
            "learnable_params": learnable_params.state_dict(),
            "optimizers": {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
            "means_scheduler": means_scheduler.state_dict(),
            "training_config": {
                "batch_size": batch_size,
                "sh_degree": sh_degree,
                "max_steps": max_steps,
                "sh_degree_warmup_steps": sh_degree_warmup_steps,
                "ssim_lambda": ssim_lambda,
                "ssim_warmup_steps": ssim_warmup_steps,
                "grad_clip_norm": grad_clip_norm,
                "scale_reg": scale_reg,
                "opacity_reset_interval": opacity_reset_interval,
                "opacity_reset_value": opacity_reset_value,
                "initial_max_points": initial_max_points,
            },
        },
        f"logs/trained_gaussians_{max_steps}.pth",
    )
    logging.info("Training completed and model saved.")


if __name__ == "__main__":
    train(
        max_steps=5000,
        downsample_points=False,
        initial_max_points=50000,
        use_cuda_rasterizer=True,
        ssim_warmup_steps=3000,
        grad_clip_norm=1.0,
        scale_reg=0.01,
        opacity_reset_interval=0,
    )
