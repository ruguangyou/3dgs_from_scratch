import torch
from src.gaussian import evaluate_spherical_harmonics, quaternion_to_rotation_matrix


LOW_PASS_FILTER = 0.3
FRUSTUM_CLAMP_FACTOR = 1.3


def render(
    world_to_camera,
    intrinsic,
    width,
    height,
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
    sh_sigmoid=False,
):
    world_to_camera = world_to_camera.to(device)  # (4, 4)
    intrinsic = intrinsic.to(device)  # (3, 3)
    means = means.to(device)  # (N, 3)
    scales = scales.to(device)  # (N, 3)
    quaternions = quaternions.to(device)  # (N, 4)
    opacities = opacities.to(device)  # (N,)
    sh_coeffs_dc = sh_coeffs_dc.to(device)  # (N, 3)
    sh_coeffs_rest = sh_coeffs_rest.to(device)  # (N, 15, 3)
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
    if M == 0:
        return torch.zeros((height, width, 3), device=means.device, dtype=means.dtype), None, None

    # evaluate spherical harmonics at the view direction
    camera_pos = -world_to_camera[:3, :3].t() @ world_to_camera[:3, 3]  # (3,)
    view_dirs = means[valid_depth_mask] - camera_pos.unsqueeze(0)  # (M, 3)
    view_dirs = view_dirs / torch.norm(view_dirs, dim=1, keepdim=True)
    colors = evaluate_spherical_harmonics(
        sh_coeffs_dc, sh_coeffs_rest, view_dirs, sh_sigmoid
    )  # (M, 3)

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
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    x, y, z = points_cam.unbind(0)  # (M,)
    invz = 1 / z
    lim_x = FRUSTUM_CLAMP_FACTOR * max(cx, width - cx) / fx
    lim_y = FRUSTUM_CLAMP_FACTOR * max(cy, height - cy) / fy
    x_ndc = x * invz
    y_ndc = y * invz
    x_ndc_clamped = x_ndc.clamp(min=-lim_x, max=lim_x)
    y_ndc_clamped = y_ndc.clamp(min=-lim_y, max=lim_y)
    x_ndc_for_cov = torch.where(torch.abs(x_ndc) <= lim_x, x_ndc, x_ndc_clamped.detach())
    y_ndc_for_cov = torch.where(torch.abs(y_ndc) <= lim_y, y_ndc, y_ndc_clamped.detach())
    J[:, 0, 0] = fx * invz
    J[:, 1, 1] = fy * invz
    J[:, 0, 2] = -fx * x_ndc_for_cov * invz
    J[:, 1, 2] = -fy * y_ndc_for_cov * invz
    cov_img = J @ cov_cam @ J.transpose(1, 2)  # (M, 2, 2)
    cov_img[:, 0, 0] = cov_img[:, 0, 0] + LOW_PASS_FILTER
    cov_img[:, 1, 1] = cov_img[:, 1, 1] + LOW_PASS_FILTER

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
    if u_img.shape[0] == 0:
        return torch.zeros((height, width, 3), device=means.device, dtype=means.dtype), None, None

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
    if L == 0:
        return torch.zeros((height, width, 3), device=means.device, dtype=means.dtype), None, None

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

    return (
        rendered_image.clamp(0, 1) * 255,
        None,
        None,
    )  # for compatibility with CUDA rasterizer output
