import logging
import math
import pickle
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fused_ssim import fused_ssim
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from dataset import load_colmap, Dataset


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def initialize(
    points: torch.Tensor,
    rgbs: torch.Tensor,
    init_scale: float = 1.0,
    init_opacity: float = 0.1,
    sh_degree: int = 3,
    device: str = "cuda",
):
    N = points.shape[0]

    # initalize gaussians size to be the average distance of the 3 nearest neighbors
    pts = points.cpu().numpy()
    knn = NearestNeighbors(n_neighbors=4).fit(pts)  # 3 closest neighbors + the point itself
    distances, _ = knn.kneighbors(pts)  # distances shape: (N, 4)
    dist = torch.from_numpy(distances[:, 1:]).float()  # 0th column is distance to itself
    avg_dist = torch.sqrt(torch.square(dist).mean(axis=1))  # avg_dist shape: (N,)
    # in log space, after exp the values will be positive
    scales = torch.log(avg_dist * init_scale).unsqueeze(-1).repeat(1, 3)

    quaternions = torch.rand(N, 4)
    # in logit space, after sigmoid the values will be constrained in (0, 1)
    opacities = torch.logit(torch.full((N,), init_opacity))

    sh_coeffs = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # (N, 16, 3) for degree 3
    # initialize the first SH coefficients (the constant term) with the RGB colors
    C0 = 0.28209479177387814  # DC component of SH basis
    sh_coeffs[:, 0, :] = (rgbs - 0.5) / C0  # Normalize to [-0.5, 0.5] and scale by C0

    learnable_params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(points),
            "scales": torch.nn.Parameter(scales),
            "quaternions": torch.nn.Parameter(quaternions),
            "opacities": torch.nn.Parameter(opacities),
            # dc and rest of the SH coefficients are separated to apply for different learning rates
            "sh_coeffs_dc": torch.nn.Parameter(sh_coeffs[:, 0, :]),
            "sh_coeffs_rest": torch.nn.Parameter(sh_coeffs[:, 1:, :]),
        }
    ).to(device)

    return learnable_params


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


def quaternion_to_rotation_matrix(quaternions):
    # quaternions shape: (N, 4) in (x, y, z, w) format
    x, y, z, w = quaternions.unbind(dim=1)

    R = torch.zeros((quaternions.shape[0], 3, 3), device=quaternions.device)
    R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x**2 + y**2)

    return R


def evaluate_spherical_harmonics(sh_coeffs, view_dirs):
    # sh_coeffs shape: (N, 16, 3) for degree 3
    # view_dirs shape: (N, 3)

    SH_C0 = 0.28209479177387814
    SH_C1_x = 0.4886025119029199
    SH_C1_y = 0.4886025119029199
    SH_C1_z = 0.4886025119029199
    SH_C2_xy = 1.0925484305920792
    SH_C2_xz = 1.0925484305920792
    SH_C2_yz = 1.0925484305920792
    SH_C2_zz = 0.31539156525252005
    SH_C2_xx_yy = 0.5462742152960396
    SH_C3_yxx_yyy = 0.5900435899266435
    SH_C3_xyz = 2.890611442640554
    SH_C3_yzz_yxx_yyy = 0.4570457994644658
    SH_C3_zzz_zxx_zyy = 0.3731763325901154
    SH_C3_xzz_xxx_xyy = 0.4570457994644658
    SH_C3_zxx_zyy = 1.445305721320277
    SH_C3_xxx_xyy = 0.5900435899266435

    x, y, z = view_dirs.unbind(dim=1)  # (N,)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z

    sh_basis = torch.stack(
        [
            torch.full_like(x, SH_C0),  # l=0, m=0
            -SH_C1_y * y,  # l=1, m=-1
            SH_C1_z * z,  # l=1, m=0
            -SH_C1_x * x,  # l=1, m=1
            SH_C2_xy * xy,  # l=2, m=-2
            SH_C2_yz * yz,  # l=2, m=-1
            SH_C2_zz * (3 * zz - 1),  # l=2, m=0
            SH_C2_xz * xz,  # l=2, m=1
            SH_C2_xx_yy * (xx - yy),  # l=2, m=2
            SH_C3_yxx_yyy * y * (xx - yy),  # l=3, m=-3
            SH_C3_xyz * (x * y * z),  # l=3, m=-2
            SH_C3_yzz_yxx_yyy * y * (4 * zz - xx - yy),  # l=3, m=-1
            SH_C3_zzz_zxx_zyy * z * (2 * zz - 3 * xx - 3 * yy),  # l=3, m=0
            SH_C3_xzz_xxx_xyy * x * (4 * zz - xx - yy),  # l=3, m=1
            SH_C3_zxx_zyy * z * (xx - yy),  # l=3, m=2
            SH_C3_xxx_xyy * x * (xx - 3 * yy),  # l=3, m=3
        ],
        dim=1,
    )  # (N, 16)

    return torch.sigmoid((sh_coeffs * sh_basis.unsqueeze(-1)).sum(dim=1))  # (N, 3)


def downsample_point_cloud(
    points: np.ndarray,
    rgbs: np.ndarray,
    max_points: int,
    seed: int = 42,
):
    if max_points <= 0 or points.shape[0] <= max_points:
        return points, rgbs

    rng = np.random.default_rng(seed)
    keep_idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[keep_idx], rgbs[keep_idx]


def resize_camera(camera, image_scale: float):
    if image_scale == 1.0:
        return camera

    image = camera["image"]
    world_to_camera = camera["world_to_camera"]
    intrinsic = camera["intrinsic"].clone()

    resized_image = F.interpolate(
        image.permute(0, 3, 1, 2),
        scale_factor=image_scale,
        mode="bilinear",
        align_corners=False,
        recompute_scale_factor=False,
    ).permute(0, 2, 3, 1)

    intrinsic[:, 0, 0] *= image_scale
    intrinsic[:, 1, 1] *= image_scale
    intrinsic[:, 0, 2] *= image_scale
    intrinsic[:, 1, 2] *= image_scale

    return {
        "world_to_camera": world_to_camera,
        "intrinsic": intrinsic,
        "image": resized_image,
        "camera_id": camera["camera_id"],
    }


def render(
    gaussians,
    camera,
    near_plane=0.01,
    far_plane=100.0,
    min_opacity=1 / 255,
    min_radius=0.5,
    max_radius=128.0,
    alpha_threshold=1e-4,
    transmitance_threshold=1e-4,
    device="cuda",
):
    means = gaussians["means"].to(device)  # (N, 3)
    scales = torch.exp(gaussians["scales"]).to(device)  # (N, 3), ensure positivity
    quaternions = gaussians["quaternions"].to(device)  # (N, 4), normalize to unit length
    quaternions = quaternions / torch.norm(quaternions, dim=1, keepdim=True)
    opacities = torch.sigmoid(gaussians["opacities"]).to(device)  # (N,), constrain to (0, 1)
    sh_coeffs = torch.cat(
        [
            gaussians["sh_coeffs_dc"].unsqueeze(1),  # (N, 3) -> (N, 1, 3)
            gaussians["sh_coeffs_rest"],  # (N, 15, 3) for degree 3
        ],
        dim=1,
    ).to(device)  # (N, 16, 3)

    # batch size is 1
    world_to_camera = camera["world_to_camera"].squeeze(0).to(device)  # (1, 4, 4) -> (4, 4)
    intrinsic = camera["intrinsic"].squeeze(0).to(device)  # (1, 3, 3) -> (3, 3)
    height, width = camera["image"].shape[1:3]  # (1, H, W, C)
    N = means.shape[0]

    # transform Gaussian centers to camera space
    points_cam = (
        world_to_camera @ torch.cat([means, torch.ones(N, 1).to(means.device)], dim=1).t()
    )[:3, :]  # camera space, (3, N)

    # filter out Gaussians that are behind the near plane or beyond the far plane
    depth = points_cam[2, :]  # (N,)
    valid_depth_mask = (depth > near_plane) & (depth < far_plane)
    points_cam = points_cam[:, valid_depth_mask]  # (3, M) where M <= N
    scales = scales[valid_depth_mask]  # (M, 3)
    quaternions = quaternions[valid_depth_mask]  # (M, 4)
    opacities = opacities[valid_depth_mask]  # (M,)
    sh_coeffs = sh_coeffs[valid_depth_mask]  # (M, 16, 3)
    M = points_cam.shape[1]

    # evaluate spherical harmonics at the view direction
    camera_pos = -world_to_camera[:3, :3].t() @ world_to_camera[:3, 3]  # (3,)
    view_dirs = means[valid_depth_mask] - camera_pos.unsqueeze(0)  # (M, 3)
    view_dirs = view_dirs / torch.norm(view_dirs, dim=1, keepdim=True)
    colors = evaluate_spherical_harmonics(sh_coeffs, view_dirs)  # (M, 3)

    # project Gaussian to image plane
    x, y, z = points_cam.unbind(0)  # (M,)
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    u_img = fx * x / z + cx  # (M,)
    v_img = fy * y / z + cy  # (M,)

    # build Gaussian covariance matrices from scales and orientations
    S = torch.diag_embed(scales)  # (M, 3, 3)
    R = quaternion_to_rotation_matrix(quaternions)  # (M, 3, 3)
    cov_world = R @ S @ S @ R.transpose(1, 2)  # (M, 3, 3)
    R_world_to_camera = world_to_camera[:3, :3]  # (3, 3)
    cov_cam = R_world_to_camera.unsqueeze(0) @ cov_world @ R_world_to_camera.t().unsqueeze(0)

    # project Gaussian covariances to image plane
    J = torch.zeros((M, 2, 3), device=means.device)  # Jacobian of projection
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    x, y, z = points_cam.unbind(0)  # (M,)
    invz = 1 / z
    invz2 = invz * invz
    J[:, 0, 0] = fx * invz
    J[:, 1, 1] = fy * invz
    J[:, 0, 2] = -fx * x * invz2
    J[:, 1, 2] = -fy * y * invz2
    cov_img = J @ cov_cam @ J.transpose(1, 2)  # (M, 2, 2)

    # valid mask for Gaussians that have positive definite covariances
    cov_img = (cov_img + cov_img.transpose(1, 2)) / 2  # ensure symmetry
    # eigvals = torch.linalg.eigvalsh(cov_img)  # (M, 2)
    # valid_mask = (eigvals > 1e-6).all(dim=1)  # (M,)
    valid_mask = torch.isfinite(cov_img).all(dim=(1, 2))  # (M,), filter out NaN or Inf
    u_img = u_img[valid_mask]  # (K,) where K <= M
    v_img = v_img[valid_mask]  # (K,)
    opacities = opacities[valid_mask]  # (K,)
    colors = colors[valid_mask]  # (K, 3)
    z = z[valid_mask]  # (K,)
    cov_img = cov_img[valid_mask]  # (K, 2, 2)

    # valid mask for Gaussians that project within the image boundaries
    extend = 3  # extend up to 3 standard deviations to account for the Gaussian tails
    radius_u = torch.sqrt(cov_img[:, 0, 0].clamp_min(1e-12)) * extend  # (K,)
    radius_v = torch.sqrt(cov_img[:, 1, 1].clamp_min(1e-12)) * extend  # (K,)
    radius = torch.maximum(radius_u, radius_v)
    cull_mask = (
        torch.isfinite(radius_u)
        & torch.isfinite(radius_v)
        & (opacities >= min_opacity)
        & (radius >= min_radius)
        & (radius <= max_radius)
    )
    within_image_mask = (
        (u_img > -radius_u)
        & (u_img < width + radius_u)
        & (v_img > -radius_v)
        & (v_img < height + radius_v)
    )
    within_image_mask = within_image_mask & cull_mask
    u_img = u_img[within_image_mask]  # (L,) where L <= K
    v_img = v_img[within_image_mask]  # (L,)
    opacities = opacities[within_image_mask]  # (L,)
    colors = colors[within_image_mask]  # (L, 3)
    z = z[within_image_mask]  # (L,)
    cov_img = cov_img[within_image_mask]  # (L, 2, 2)

    # global sorting by depth for correct compositing order (near to far)
    order = torch.argsort(z, descending=False)
    u_img = u_img[order]
    v_img = v_img[order]
    opacities = opacities[order]
    colors = colors[order]
    cov_img = cov_img[order]

    # tiling
    T = 256  # 16x16 pixel tiles
    radius_u = torch.sqrt(cov_img[:, 0, 0]) * extend  # (L,)
    radius_v = torch.sqrt(cov_img[:, 1, 1]) * extend  # (L,)

    # AABB of the Gaussian in image space
    u_min = torch.floor(u_img - radius_u).clamp(min=0, max=width - 1).long()
    u_max = torch.floor(u_img + radius_u).clamp(min=0, max=width - 1).long()
    v_min = torch.floor(v_img - radius_v).clamp(min=0, max=height - 1).long()
    v_max = torch.floor(v_img + radius_v).clamp(min=0, max=height - 1).long()

    # tile indices covered by the AABB
    tile_u_min = torch.floor(u_min / T).long()
    tile_u_max = torch.floor(u_max / T).long()
    tile_v_min = torch.floor(v_min / T).long()
    tile_v_max = torch.floor(v_max / T).long()

    # for each tile, find the Gaussians that overlap with it and composite them
    gaussian_chunk_size = 64
    rendered_image = torch.zeros((height, width, 3), device=means.device)  # (H, W, C)
    for tile_u in range((width - 1) // T + 1):
        for tile_v in range((height - 1) // T + 1):
            in_tile_mask = (
                (tile_u_min <= tile_u)
                & (tile_u_max >= tile_u)
                & (tile_v_min <= tile_v)
                & (tile_v_max >= tile_v)
            )  # (L,)
            if not in_tile_mask.any():
                continue

            gs_points = torch.stack(
                [u_img[in_tile_mask], v_img[in_tile_mask]], dim=1
            )  # (G, 2) where G <= L
            gs_colors = colors[in_tile_mask]  # (G, 3)
            gs_opacities = opacities[in_tile_mask]  # (G,)
            gs_covs = cov_img[in_tile_mask]  # (G, 2, 2)
            a = gs_covs[:, 0, 0]
            b = gs_covs[:, 0, 1]
            c = gs_covs[:, 1, 0]
            d = gs_covs[:, 1, 1]
            det = a * d - b * c
            inv_det = 1.0 / det.clamp_min(1e-12)
            gs_covs_inv = torch.stack(
                [
                    torch.stack([d * inv_det, -b * inv_det], dim=-1),
                    torch.stack([-c * inv_det, a * inv_det], dim=-1),
                ],
                dim=1,
            )  # (G, 2, 2)

            u0, u1 = tile_u * T, min((tile_u + 1) * T, width)
            v0, v1 = tile_v * T, min((tile_v + 1) * T, height)
            tile_h = v1 - v0
            tile_w = u1 - u0
            pixel_u = torch.arange(u0, u1, device=means.device)  # (tile_w,)
            pixel_v = torch.arange(v0, v1, device=means.device)  # (tile_h,)
            pixel_uu, pixel_vv = torch.meshgrid(pixel_u, pixel_v, indexing="xy")  # (tile_w, tile_h)
            pixel_coords = torch.stack([pixel_uu, pixel_vv], dim=-1).to(
                gs_points.dtype
            )  # (tile_w, tile_h, 2)

            transmitance = torch.ones((tile_h, tile_w), device=means.device, dtype=gs_points.dtype)
            contribution = torch.zeros(
                (tile_h, tile_w, 3), device=means.device, dtype=gs_points.dtype
            )

            G = gs_points.shape[0]
            for chunk_start in range(0, G, gaussian_chunk_size):
                chunk_end = min(chunk_start + gaussian_chunk_size, G)
                chunk_points = gs_points[chunk_start:chunk_end]  # (C, 2)
                chunk_colors = gs_colors[chunk_start:chunk_end]  # (C, 3)
                chunk_opacities = gs_opacities[chunk_start:chunk_end]  # (C,)
                chunk_covs_inv = gs_covs_inv[chunk_start:chunk_end]  # (C, 2, 2)

                for idx in range(chunk_points.shape[0]):
                    diff = pixel_coords - chunk_points[idx].view(1, 1, 2)  # (tile_h, tile_w, 2)
                    exponent = torch.einsum(
                        "...i,ij,...j->...", diff, chunk_covs_inv[idx], diff
                    )  # (tile_h, tile_w)
                    alpha = torch.exp(-0.5 * exponent) * chunk_opacities[idx]  # (tile_h, tile_w)
                    alpha = torch.where(alpha < alpha_threshold, torch.zeros_like(alpha), alpha)
                    contribution += (transmitance * alpha).unsqueeze(-1) * chunk_colors[idx].view(
                        1, 1, 3
                    )  # (tile_h, tile_w, 3)
                    transmitance *= 1 - alpha + 1e-10
                    transmitance = torch.where(
                        transmitance < transmitance_threshold,
                        torch.zeros_like(transmitance),
                        transmitance,
                    )

            rendered_image[v0:v1, u0:u1, :] = contribution

    return rendered_image * 255


def train():
    load_cached_input = True
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

    batch_size = 1
    sh_degree = 3
    max_steps = 10000
    sh_degree_increase_step = 1000
    ssim_lambda = 0.2
    initial_max_points = 10000
    initial_downsample_seed = 42
    image_scale = 0.5

    points, rgbs = downsample_point_cloud(
        points,
        rgbs,
        max_points=initial_max_points,
        seed=initial_downsample_seed,
    )

    train_dataset = Dataset(camera_data, split="train")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    train_dataiter = iter(train_dataloader)

    logging.info(f"Initializing {points.shape[0]} gaussians...")
    points = torch.from_numpy(points).float()
    rgbs = torch.from_numpy(rgbs / 255.0).float()  # Normalize RGB values to [0, 1]
    gaussians = initialize(points, rgbs, sh_degree=sh_degree)

    logging.info("Optimizer setup...")
    optimizers = setup_optimizers(gaussians, batch_size)
    # exponential decay, lr_at_end = 0.01 * lr_at_start
    means_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=0.01 ** (1 / max_steps)
    )

    logging.info("Training started...")
    for step in tqdm(range(max_steps), desc="Training"):
        camera = resize_camera(next(train_dataiter), image_scale=image_scale)

        rendered_image = render(gaussians, camera)
        cv2.imshow("Rendered Image", rendered_image.cpu().numpy())
        cv2.imshow("Target Image", camera["image"].cpu().numpy())
        cv2.waitKey(0)

        # L1 loss on pixel colors
        l1_loss = torch.nn.functional.l1_loss(rendered_image, camera["image"])
        # Dissimilarity SSIM on structural similarity
        ssim_loss = 1.0 - fused_ssim(
            rendered_image.permute(0, 3, 1, 2),  # (N, H, W, C) -> (N, C, H, W)
            camera["image"].permute(0, 3, 1, 2),  # NCHW format for pytorch
            padding="valid",  # no padding to avoid border artifacts
        )

        loss = (1 - ssim_lambda) * l1_loss + ssim_lambda * ssim_loss
        loss.backward()

        # increase SH degree every 1000 steps, to progressively learn higher frequency details
        sh_degree_to_use = min(sh_degree, step // sh_degree_increase_step)
        # zero out gradients for unused SH coefficients
        if sh_degree_to_use < sh_degree:
            gaussians["sh_coeffs_rest"].grad[:, (sh_degree_to_use + 1) ** 2 - 1 :, :].zero_()

        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad()
        means_scheduler.step()


if __name__ == "__main__":
    train()
