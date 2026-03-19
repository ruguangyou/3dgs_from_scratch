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
    # points shape: (N, 3)
    # rgbs shape: (N, 3) in [0, 1] range
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

    C0 = 0.28209479177387814  # DC component of SH basis
    # normalize RGB colors to [-0.5, 0.5] and scale by C0 and initialize sh_coeffs_dc
    sh_coeffs_dc = (rgbs - 0.5) / C0  # (N, 3) for the DC component of SH
    # dc and rest of the SH coefficients are separated to apply for different learning rates
    sh_coeffs_rest = torch.zeros(
        (N, (sh_degree + 1) ** 2 - 1, 3)
    )  # (N, 15, 3) for degree 3 excluding DC

    learnable_params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(points),
            "scales": torch.nn.Parameter(scales),
            "quaternions": torch.nn.Parameter(quaternions),
            "opacities": torch.nn.Parameter(opacities),
            "sh_coeffs_dc": torch.nn.Parameter(sh_coeffs_dc),
            "sh_coeffs_rest": torch.nn.Parameter(sh_coeffs_rest),
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


def evaluate_spherical_harmonics(sh_coeffs_dc, sh_coeffs_rest, view_dirs):
    # sh_coeffs_dc shape: (N, 3) for degree 0
    # sh_coeffs_rest shape: (N, 15, 3) for degree 1,2,3
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

    sh_basis_rest = torch.stack(
        [
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
    )  # (N, 15)

    sh_dc = sh_coeffs_dc * SH_C0  # (N, 3)
    sh_rest = (sh_coeffs_rest * sh_basis_rest.unsqueeze(-1)).sum(dim=1)  # (N, 3)

    return torch.sigmoid(sh_dc + sh_rest)  # (N, 3)


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
    camera,
    means,
    scales,
    quaternions,
    opacities,
    sh_coeffs_dc,
    sh_coeffs_rest,
    near_plane=0.01,
    far_plane=100.0,
    min_opacity=1 / 255,
    min_radius=0.5,
    max_radius=128.0,
    alpha_threshold=1e-4,
    transmittance_threshold=1e-4,
    chi_squared_threshold=9.21,  # 99% confidence interval for 2 DOF
    device="cuda",
):
    means = means.to(device)  # (N, 3)
    scales = scales.to(device)  # (N, 3)
    quaternions = quaternions.to(device)  # (N, 4)
    opacities = opacities.to(device)  # (N,)
    sh_coeffs_dc = sh_coeffs_dc.to(device)  # (N, 3)
    sh_coeffs_rest = sh_coeffs_rest.to(device)  # (N, 15, 3)

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
    sh_coeffs_dc = sh_coeffs_dc[valid_depth_mask]  # (M, 3)
    sh_coeffs_rest = sh_coeffs_rest[valid_depth_mask]  # (M, 15, 3)
    M = points_cam.shape[1]

    # evaluate spherical harmonics at the view direction
    camera_pos = -world_to_camera[:3, :3].t() @ world_to_camera[:3, 3]  # (3,)
    view_dirs = means[valid_depth_mask] - camera_pos.unsqueeze(0)  # (M, 3)
    view_dirs = view_dirs / torch.norm(view_dirs, dim=1, keepdim=True)
    colors = evaluate_spherical_harmonics(sh_coeffs_dc, sh_coeffs_rest, view_dirs)  # (M, 3)

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
    L = u_img.shape[0]

    # global sorting by depth for correct compositing order (near to far)
    order = torch.argsort(z, descending=False)
    u_img = u_img[order]
    v_img = v_img[order]
    opacities = opacities[order]
    colors = colors[order]
    cov_img = cov_img[order]

    # tiling
    T = 16  # 16x16 pixel tiles
    radius_u = torch.sqrt(cov_img[:, 0, 0]) * extend  # (L,)
    radius_v = torch.sqrt(cov_img[:, 1, 1]) * extend  # (L,)

    # AABB of the Gaussian in image space
    u_min = torch.floor(u_img - radius_u).clamp(min=0, max=width - 1).long()
    u_max = torch.floor(u_img + radius_u).clamp(min=0, max=width - 1).long()
    v_min = torch.floor(v_img - radius_v).clamp(min=0, max=height - 1).long()
    v_max = torch.floor(v_img + radius_v).clamp(min=0, max=height - 1).long()

    # tile indices covered by the AABB
    tile_u_min = torch.floor(u_min / T).long()  # (L,)
    tile_u_max = torch.floor(u_max / T).long()  # (L,)
    tile_v_min = torch.floor(v_min / T).long()  # (L,)
    tile_v_max = torch.floor(v_max / T).long()  # (L,)

    # number of tiles covered by each Gaussian
    num_tiles_u = tile_u_max - tile_u_min + 1  # (L,)
    num_tiles_v = tile_v_max - tile_v_min + 1  # (L,)

    # each Gaussian id will be repeated for the number of tiles it covers
    gaussian_ids = torch.repeat_interleave(
        torch.arange(L, device=means.device, dtype=torch.int64),
        num_tiles_u * num_tiles_v,
    )  # (sum(num_tiles_u * num_tiles_v),)

    # max number of tiles covered by any Gaussian, used for preallocating buffers
    max_num_tiles_u = num_tiles_u.max().item()
    max_num_tiles_v = num_tiles_v.max().item()

    # generate tile indices for each Gaussian
    span_indices_u = torch.arange(max_num_tiles_u, device=means.device)  # (max_num_tiles_u,)
    span_indices_v = torch.arange(max_num_tiles_v, device=means.device)  # (max_num_tiles_v,)
    tile_u = (
        tile_u_min.view(-1, 1, 1) + span_indices_u.view(1, -1, 1)  # (L, max_num_tiles_u, 1)
    ).expand(L, max_num_tiles_u, max_num_tiles_v)  # (L, max_num_tiles_u, max_num_tiles_v)
    tile_v = (
        tile_v_min.view(-1, 1, 1) + span_indices_v.view(1, 1, -1)  # (L, 1, max_num_tiles_v)
    ).expand(L, max_num_tiles_u, max_num_tiles_v)  # (L, max_num_tiles_u, max_num_tiles_v)

    # mask for valid tiles that are actually covered (some entries are just padding)
    mask = (
        (span_indices_u.view(1, -1, 1) < num_tiles_u.view(-1, 1, 1))  # (L, max_num_tiles_u, 1)
        & (span_indices_v.view(1, 1, -1) < num_tiles_v.view(-1, 1, 1))  # (L, 1, max_num_tiles_v)
    )  # (L, max_num_tiles_u, max_num_tiles_v)

    # compute tile ids for each Gaussian
    num_tiles_per_row = (width + T - 1) // T
    tile_ids = tile_v[mask] * num_tiles_per_row + tile_u[mask]  # (sum(num_tiles_u * num_tiles_v),)

    # scale up tile ids by number of Gaussians and add depth ordering
    # so that Gaussians in the same tile will be grouped together and sorted by depth
    z_ordering = torch.arange(L, device=means.device, dtype=torch.int64)
    scaled_tile_ids = tile_ids * (L + 1) + z_ordering[gaussian_ids]
    sorted_scaled_tile_ids, permutation = torch.sort(scaled_tile_ids)
    gaussian_ids = gaussian_ids[permutation]
    sorted_tile_ids = torch.div(sorted_scaled_tile_ids, L + 1, rounding_mode="floor")

    # precompute inverse of covariance matrices for Gaussian evaluation
    a = cov_img[:, 0, 0]
    b = cov_img[:, 0, 1]
    c = cov_img[:, 1, 0]
    d = cov_img[:, 1, 1]
    det = a * d - b * c
    inv_det = 1.0 / det.clamp_min(1e-12)
    covs_img_inv = torch.stack(
        [
            torch.stack([d * inv_det, -b * inv_det], dim=-1),
            torch.stack([-c * inv_det, a * inv_det], dim=-1),
        ],
        dim=1,
    )  # (L, 2, 2)

    rendered_image = torch.zeros((height, width, 3), device=means.device)  # (H, W, C)

    # iterate over tiles
    unique_tile_ids, counts = torch.unique_consecutive(sorted_tile_ids, return_counts=True)
    gs_count = 0
    for tile_id, count in zip(unique_tile_ids.tolist(), counts.tolist()):
        tu = tile_id % num_tiles_per_row
        tv = tile_id // num_tiles_per_row
        u0, v0 = tu * T, tv * T
        u1, v1 = min(u0 + T, width), min(v0 + T, height)
        if u1 <= u0 or v1 <= v0:
            continue  # skip invalid tiles

        gs_ids = gaussian_ids[gs_count : gs_count + count]  # (G,), number of Gaussians in this tile
        gs_u = u_img[gs_ids]  # (G,)
        gs_v = v_img[gs_ids]  # (G,)
        gs_colors = colors[gs_ids]  # (G, 3)
        gs_opacities = opacities[gs_ids]  # (G,)
        gs_covs_inv = covs_img_inv[gs_ids]  # (G, 2, 2)
        gs_count += count

        size_u, size_v = u1 - u0, v1 - v0
        pixel_u, pixel_v = torch.meshgrid(
            torch.arange(u0, u1, device=means.device),
            torch.arange(v0, v1, device=means.device),
            indexing="xy",
        )  # (size_v, size_u)
        du = pixel_u.view(-1, size_v, size_u) - gs_u.view(-1, 1, 1)  # (G, size_v, size_u)
        dv = pixel_v.view(-1, size_v, size_u) - gs_v.view(-1, 1, 1)  # (G, size_v, size_u)

        C11 = gs_covs_inv[:, 0, 0].view(-1, 1, 1)
        C12 = gs_covs_inv[:, 0, 1].view(-1, 1, 1)
        C21 = gs_covs_inv[:, 1, 0].view(-1, 1, 1)
        C22 = gs_covs_inv[:, 1, 1].view(-1, 1, 1)
        exponent = C11 * du * du + (C12 + C21) * du * dv + C22 * dv * dv  # (G, size_v, size_u)

        # zero out pixels outside the confidence interval
        chi_mask = (exponent < chi_squared_threshold).float()
        density = torch.exp(-0.5 * exponent) * chi_mask

        alpha = density * gs_opacities.view(-1, 1, 1)  # (G, size_v, size_u)
        alpha = torch.where(alpha < alpha_threshold, torch.zeros_like(alpha), alpha)

        one_minus_alpha = 1 - alpha + 1e-10
        transmittance = torch.ones_like(one_minus_alpha)  # (G, size_v, size_u)
        transmittance[1:, :, :] = torch.cumprod(one_minus_alpha[:-1, :, :], dim=0)
        transmittance = torch.where(
            transmittance < transmittance_threshold,
            torch.zeros_like(transmittance),
            transmittance,
        )

        contribution = ((transmittance * alpha).unsqueeze(-1) * gs_colors.view(-1, 1, 1, 3)).sum(
            dim=0
        )  # (size_v, size_u, 3)

        rendered_image[v0:v1, u0:u1, :] = contribution

    return rendered_image.clamp(0, 1) * 255


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
    debug = True

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
    learnable_params = initialize(points, rgbs, sh_degree=sh_degree)

    logging.info("Optimizer setup...")
    optimizers = setup_optimizers(learnable_params, batch_size)
    # exponential decay, lr_at_end = 0.01 * lr_at_start
    means_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=0.01 ** (1 / max_steps)
    )

    means = learnable_params["means"]
    scales = torch.exp(learnable_params["scales"])
    quaternions = learnable_params["quaternions"]
    quaternions = quaternions / torch.norm(quaternions, dim=1, keepdim=True)
    opacities = torch.sigmoid(learnable_params["opacities"])
    sh_coeffs_dc = learnable_params["sh_coeffs_dc"]
    sh_coeffs_rest = learnable_params["sh_coeffs_rest"]

    logging.info("Training started...")
    for step in tqdm(range(max_steps), desc="Training"):
        camera = resize_camera(next(train_dataiter), image_scale=image_scale)

        rendered_image = render(
            camera,
            means,
            scales,
            quaternions,
            opacities,
            sh_coeffs_dc,
            sh_coeffs_rest,
        )

        if debug and step % 100 == 0:
            cv2.imshow("Rendered Image", rendered_image.cpu().numpy())
            cv2.imshow("Target Image", camera["image"].cpu().numpy())
            cv2.waitKey(1)

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
            sh_coeffs_rest.grad[:, (sh_degree_to_use + 1) ** 2 - 1 :, :].zero_()

        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad()
        means_scheduler.step()


if __name__ == "__main__":
    train()
