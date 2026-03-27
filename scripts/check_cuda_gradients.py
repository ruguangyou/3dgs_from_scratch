import argparse
import pickle
from dataclasses import dataclass

import torch
from fused_ssim import fused_ssim

import cuda_rasterizer
from src.cuda.wrapper import SphericalHarmonicsFunction
from src.cuda.wrapper import render as cuda_render
from src.dataset import Dataset
from src.gaussian import downsample_point_cloud
from src.gaussian import initialize
from src.gaussian import evaluate_spherical_harmonics as torch_eval_sh
from src.torch_rasterizer import render as torch_render


@dataclass(frozen=True)
class Thresholds:
    min_cos: float = 0.999
    max_rel: float = 1e-3


def metric(name: str, grad_cuda: torch.Tensor, grad_ref: torch.Tensor) -> tuple[str, float, float]:
    flat_cuda = grad_cuda.detach().flatten()
    flat_ref = grad_ref.detach().flatten()
    norm_cuda = torch.norm(flat_cuda).item()
    norm_ref = torch.norm(flat_ref).item()
    cosine = torch.dot(flat_cuda, flat_ref).item() / max(norm_cuda * norm_ref, 1e-12)
    rel = torch.norm(flat_cuda - flat_ref).item() / max(norm_ref, 1e-12)
    return name, cosine, rel


def check_result(tag: str, values: list[tuple[str, float, float]], thresholds: Thresholds) -> bool:
    print(f"\n[{tag}]")
    success = True
    for name, cosine, rel in values:
        print(f"  {name:16s} cos={cosine:.6f} rel={rel:.6e}")
        if cosine < thresholds.min_cos or rel > thresholds.max_rel:
            success = False
    return success


def build_reference_tile_layout(
    points_img: torch.Tensor,
    radii: torch.Tensor,
    depths: torch.Tensor,
    mask: torch.Tensor,
    width: int,
    height: int,
    tile_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tiles_per_row = (width + tile_size - 1) // tile_size
    num_tiles_per_col = (height + tile_size - 1) // tile_size
    unique_tiles = num_tiles_per_row * num_tiles_per_col

    entries: list[tuple[int, int, int]] = []
    for gaussian_id in range(points_img.shape[0]):
        if not bool(mask[gaussian_id].item()):
            continue

        u = float(points_img[gaussian_id, 0].item())
        v = float(points_img[gaussian_id, 1].item())
        radius_u = float(radii[gaussian_id, 0].item())
        radius_v = float(radii[gaussian_id, 1].item())

        u_min = min(max(0, int(torch.floor(torch.tensor(u - radius_u)).item())), width - 1)
        u_max = min(max(0, int(torch.floor(torch.tensor(u + radius_u)).item())), width - 1)
        v_min = min(max(0, int(torch.floor(torch.tensor(v - radius_v)).item())), height - 1)
        v_max = min(max(0, int(torch.floor(torch.tensor(v + radius_v)).item())), height - 1)
        if u_min > u_max or v_min > v_max:
            continue

        tile_u_min = u_min // tile_size
        tile_u_max = u_max // tile_size
        tile_v_min = v_min // tile_size
        tile_v_max = v_max // tile_size
        depth_bits = (
            depths[gaussian_id : gaussian_id + 1].detach().view(torch.int32).item() & 0xFFFFFFFF
        )

        for tile_v in range(tile_v_min, tile_v_max + 1):
            for tile_u in range(tile_u_min, tile_u_max + 1):
                tile_id = tile_v * num_tiles_per_row + tile_u
                entries.append((tile_id, depth_bits, gaussian_id))

    entries.sort(key=lambda item: (item[0], item[1]))
    total_tiles = len(entries)
    gaussian_ids_sorted = torch.tensor(
        [gaussian_id for _, _, gaussian_id in entries],
        dtype=torch.int32,
        device=points_img.device,
    )
    indexing_offset = torch.full(
        (unique_tiles,),
        total_tiles,
        dtype=torch.int32,
        device=points_img.device,
    )

    if total_tiles == 0:
        indexing_offset.zero_()
        return indexing_offset, gaussian_ids_sorted

    previous_tile_id: int | None = None
    for idx, (tile_id, _, _) in enumerate(entries):
        if previous_tile_id is None:
            indexing_offset[: tile_id + 1] = 0
            previous_tile_id = tile_id
            continue
        if tile_id != previous_tile_id:
            indexing_offset[previous_tile_id + 1 : tile_id + 1] = idx
            previous_tile_id = tile_id

    if previous_tile_id is not None:
        indexing_offset[previous_tile_id + 1 :] = total_tiles

    return indexing_offset, gaussian_ids_sorted


def rasterize_reference(
    indexing_offset: torch.Tensor,
    gaussian_ids_sorted: torch.Tensor,
    points_img: torch.Tensor,
    cov_inv_img: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    width: int,
    height: int,
    tile_size: int,
    alpha_threshold: float,
    transmittance_threshold: float,
    chi_squared_threshold: float,
) -> torch.Tensor:
    unique_tiles = indexing_offset.shape[0]
    total_tiles = gaussian_ids_sorted.shape[0]
    num_tiles_per_row = (width + tile_size - 1) // tile_size
    reference = torch.zeros((height, width, 3), device=points_img.device, dtype=points_img.dtype)

    for tile_id in range(unique_tiles):
        row = tile_id // num_tiles_per_row
        col = tile_id % num_tiles_per_row
        range_start = indexing_offset[tile_id].item()
        range_end = (
            indexing_offset[tile_id + 1].item() if tile_id + 1 < unique_tiles else total_tiles
        )

        for idx_in_tile in range(tile_size * tile_size):
            u = col * tile_size + (idx_in_tile % tile_size)
            v = row * tile_size + (idx_in_tile // tile_size)
            if u >= width or v >= height:
                continue

            transmittance = torch.tensor(1.0, device=points_img.device, dtype=points_img.dtype)
            pixel = torch.zeros(3, device=points_img.device, dtype=points_img.dtype)
            for gaussian_index in range(range_start, range_end):
                gaussian_id = gaussian_ids_sorted[gaussian_index]
                du = u - points_img[gaussian_id, 0]
                dv = v - points_img[gaussian_id, 1]
                inv_cov_00 = cov_inv_img[gaussian_id, 0]
                inv_cov_01 = cov_inv_img[gaussian_id, 1]
                inv_cov_11 = cov_inv_img[gaussian_id, 2]

                exponent = inv_cov_00 * du * du + 2.0 * inv_cov_01 * du * dv + inv_cov_11 * dv * dv
                if exponent.item() > chi_squared_threshold:
                    continue

                alpha = torch.exp(-0.5 * exponent) * opacities[gaussian_id]
                if alpha.item() < alpha_threshold:
                    continue

                weight = alpha * transmittance
                pixel = pixel + weight * colors[gaussian_id]
                transmittance = transmittance * (1.0 - alpha)
                if transmittance.item() < transmittance_threshold:
                    break

            reference[v, u] = pixel

    return reference


def check_project_points(seed: int, device: str, thresholds: Thresholds) -> bool:
    torch.manual_seed(seed)
    count = 256
    height, width = 64, 64
    world_to_camera = torch.eye(4, device=device, dtype=torch.float32)
    intrinsic = torch.tensor(
        [[60.0, 0.0, width / 2], [0.0, 60.0, height / 2], [0.0, 0.0, 1.0]],
        device=device,
        dtype=torch.float32,
    )

    points = (torch.rand(count, 3, device=device) - 0.5) * 1.0
    points[:, 2] = torch.rand(count, device=device) * 0.5 + 2.0
    points = points.detach().requires_grad_(True)
    scales = (torch.rand(count, 3, device=device) * 0.2 + 0.8).detach().requires_grad_(True)
    quaternions_raw = torch.randn(count, 4, device=device)
    quaternions = (
        (quaternions_raw / torch.norm(quaternions_raw, dim=1, keepdim=True))
        .detach()
        .requires_grad_(True)
    )
    opacities = torch.ones(count, device=device) * 0.8

    points_img, _, cov_img, cov_inv_img, _, mask = cuda_rasterizer.project_points(
        points,
        scales,
        quaternions,
        opacities,
        world_to_camera,
        intrinsic,
        0.01,
        100.0,
        1 / 255,
        0.1,
        256.0,
        width,
        height,
    )

    upstream_points = torch.randn_like(points_img)
    upstream_cov_inv = torch.randn_like(cov_inv_img) * 0.1
    grad_points_cuda, grad_scales_cuda, grad_quaternions_cuda = (
        cuda_rasterizer.project_points_backward(
            upstream_points,
            upstream_cov_inv,
            points,
            scales,
            quaternions,
            opacities,
            world_to_camera,
            intrinsic,
            cov_img,
            mask,
        )
    )

    points_camera = (world_to_camera[:3, :3] @ points.t()).t() + world_to_camera[:3, 3]
    x_camera = points_camera[:, 0]
    y_camera = points_camera[:, 1]
    z_camera = points_camera[:, 2]
    points_img_ref = torch.stack(
        [
            intrinsic[0, 0] * x_camera / z_camera + intrinsic[0, 2],
            intrinsic[1, 1] * y_camera / z_camera + intrinsic[1, 2],
        ],
        dim=-1,
    )

    qi = quaternions[:, 0]
    qj = quaternions[:, 1]
    qk = quaternions[:, 2]
    qr = quaternions[:, 3]
    rotation = torch.stack(
        [
            1 - 2 * qj * qj - 2 * qk * qk,
            2 * qi * qj - 2 * qk * qr,
            2 * qi * qk + 2 * qj * qr,
            2 * qi * qj + 2 * qk * qr,
            1 - 2 * qi * qi - 2 * qk * qk,
            2 * qj * qk - 2 * qi * qr,
            2 * qi * qk - 2 * qj * qr,
            2 * qj * qk + 2 * qi * qr,
            1 - 2 * qi * qi - 2 * qj * qj,
        ],
        dim=-1,
    ).reshape(count, 3, 3)
    scales_diag = torch.diag_embed(scales)
    cov_world = rotation @ scales_diag @ scales_diag @ rotation.transpose(1, 2)
    world_rotation = world_to_camera[:3, :3]
    cov_camera = world_rotation.unsqueeze(0) @ cov_world @ world_rotation.t().unsqueeze(0)

    inv_z = 1.0 / z_camera
    inv_z2 = inv_z * inv_z
    jacobian = torch.zeros((count, 2, 3), device=device)
    jacobian[:, 0, 0] = intrinsic[0, 0] * inv_z
    jacobian[:, 0, 2] = -intrinsic[0, 0] * x_camera * inv_z2
    jacobian[:, 1, 1] = intrinsic[1, 1] * inv_z
    jacobian[:, 1, 2] = -intrinsic[1, 1] * y_camera * inv_z2
    cov_image = jacobian @ cov_camera @ jacobian.transpose(1, 2)
    cov_image = 0.5 * (cov_image + cov_image.transpose(1, 2))
    a = cov_image[:, 0, 0]
    b = cov_image[:, 0, 1]
    c = cov_image[:, 1, 1]
    determinant = a * c - b * b
    cov_inv_ref = torch.stack([c / determinant, -b / determinant, a / determinant], dim=-1)

    loss = (points_img_ref * upstream_points).sum() + (cov_inv_ref * upstream_cov_inv).sum()
    grad_points_ref, grad_scales_ref, grad_quaternions_ref = torch.autograd.grad(
        loss,
        [points, scales, quaternions],
        retain_graph=False,
        create_graph=False,
    )

    valid_mask_3 = mask.float().unsqueeze(-1)
    values = [
        metric("points", grad_points_cuda * valid_mask_3, grad_points_ref * valid_mask_3),
        metric("scales", grad_scales_cuda * valid_mask_3, grad_scales_ref * valid_mask_3),
        metric(
            "quaternions", grad_quaternions_cuda * valid_mask_3, grad_quaternions_ref * valid_mask_3
        ),
    ]
    return check_result("project_points", values, thresholds)


def check_rasterize(seed: int, device: str, thresholds: Thresholds) -> bool:
    torch.manual_seed(seed)
    count = 16
    width, height = 16, 16
    tile_size = 16

    indexing_offset = torch.tensor([0], dtype=torch.int32, device=device)
    gaussian_ids_sorted = torch.arange(count, dtype=torch.int32, device=device)
    points_img = (torch.rand(count, 2, device=device) * 12.0 + 2.0).detach().requires_grad_(True)

    cov_inv_img = torch.zeros(count, 3, device=device)
    cov_inv_img[:, 0] = torch.rand(count, device=device) * 0.03 + 0.01
    cov_inv_img[:, 1] = (torch.rand(count, device=device) - 0.5) * 0.005
    cov_inv_img[:, 2] = torch.rand(count, device=device) * 0.03 + 0.01
    cov_inv_img = cov_inv_img.detach().requires_grad_(True)

    opacities = (torch.rand(count, device=device) * 0.25 + 0.55).detach().requires_grad_(True)
    colors = torch.rand(count, 3, device=device).detach().requires_grad_(True)

    alpha_threshold = 0.0
    transmittance_threshold = 0.0
    chi_squared_threshold = 20.0

    rendered = cuda_rasterizer.rasterize(
        indexing_offset,
        gaussian_ids_sorted,
        points_img,
        cov_inv_img,
        opacities,
        colors,
        width,
        height,
        tile_size,
        alpha_threshold,
        transmittance_threshold,
        chi_squared_threshold,
    )
    upstream = torch.randn_like(rendered)

    grad_points_cuda, grad_cov_cuda, grad_opacities_cuda, grad_colors_cuda = (
        cuda_rasterizer.rasterize_backward(
            upstream,
            indexing_offset,
            gaussian_ids_sorted,
            points_img,
            cov_inv_img,
            opacities,
            colors,
            width,
            height,
            tile_size,
            alpha_threshold,
            transmittance_threshold,
            chi_squared_threshold,
        )
    )

    reference = rasterize_reference(
        indexing_offset,
        gaussian_ids_sorted,
        points_img,
        cov_inv_img,
        opacities,
        colors,
        width,
        height,
        tile_size,
        alpha_threshold,
        transmittance_threshold,
        chi_squared_threshold,
    )

    loss = (reference * upstream).sum()
    grad_points_ref, grad_cov_ref, grad_opacities_ref, grad_colors_ref = torch.autograd.grad(
        loss,
        [points_img, cov_inv_img, opacities, colors],
        retain_graph=False,
        create_graph=False,
    )

    values = [
        metric("points_img", grad_points_cuda, grad_points_ref),
        metric("cov_inv", grad_cov_cuda, grad_cov_ref),
        metric("opacities", grad_opacities_cuda, grad_opacities_ref),
        metric("colors", grad_colors_cuda, grad_colors_ref),
    ]
    return check_result("rasterize", values, thresholds)


def check_rasterize_multitile(seed: int, device: str, thresholds: Thresholds) -> bool:
    torch.manual_seed(seed)
    width, height = 48, 48
    tile_size = 16

    points_base = torch.tensor(
        [
            [15.5, 15.5],
            [16.2, 15.8],
            [31.2, 7.5],
            [7.5, 31.2],
            [39.0, 39.0],
        ],
        device=device,
        dtype=torch.float32,
    )
    radii = torch.tensor(
        [
            [10.5, 10.5],
            [9.8, 9.2],
            [8.4, 7.6],
            [7.6, 8.8],
            [4.5, 4.5],
        ],
        device=device,
        dtype=torch.float32,
    )
    depths = torch.tensor([0.55, 0.62, 0.28, 0.91, 0.18], device=device, dtype=torch.float32)
    mask = torch.ones(points_base.shape[0], device=device, dtype=torch.bool)

    points_img = points_base.detach().clone().requires_grad_(True)
    cov_inv_img = (
        torch.tensor(
            [
                [0.085, 0.012, 0.072],
                [0.078, -0.009, 0.088],
                [0.094, 0.004, 0.109],
                [0.103, -0.006, 0.091],
                [0.162, 0.000, 0.149],
            ],
            device=device,
            dtype=torch.float32,
        )
        .detach()
        .requires_grad_(True)
    )
    opacities = (
        torch.tensor(
            [0.82, 0.67, 0.73, 0.58, 0.91],
            device=device,
            dtype=torch.float32,
        )
        .detach()
        .requires_grad_(True)
    )
    colors = (
        torch.tensor(
            [
                [0.90, 0.25, 0.10],
                [0.15, 0.80, 0.35],
                [0.30, 0.40, 0.95],
                [0.85, 0.55, 0.20],
                [0.60, 0.75, 0.90],
            ],
            device=device,
            dtype=torch.float32,
        )
        .detach()
        .requires_grad_(True)
    )

    alpha_threshold = 0.0
    transmittance_threshold = 0.0
    chi_squared_threshold = 9.21

    indexing_offset_cuda, gaussian_ids_sorted_cuda = cuda_rasterizer.compute_tile_intersection(
        points_img.detach(),
        radii,
        depths,
        mask,
        width,
        height,
        tile_size,
    )
    indexing_offset_ref, gaussian_ids_sorted_ref = build_reference_tile_layout(
        points_img.detach(),
        radii,
        depths,
        mask,
        width,
        height,
        tile_size,
    )

    layout_ok = torch.equal(indexing_offset_cuda, indexing_offset_ref) and torch.equal(
        gaussian_ids_sorted_cuda, gaussian_ids_sorted_ref
    )
    print(f"\n[rasterize_multitile_layout] match={layout_ok}")
    if not layout_ok:
        print(f"  indexing_offset_cuda={indexing_offset_cuda.tolist()}")
        print(f"  indexing_offset_ref ={indexing_offset_ref.tolist()}")
        print(f"  gaussian_ids_cuda  ={gaussian_ids_sorted_cuda.tolist()}")
        print(f"  gaussian_ids_ref   ={gaussian_ids_sorted_ref.tolist()}")
        return False

    rendered_cuda = cuda_rasterizer.rasterize(
        indexing_offset_cuda,
        gaussian_ids_sorted_cuda,
        points_img,
        cov_inv_img,
        opacities,
        colors,
        width,
        height,
        tile_size,
        alpha_threshold,
        transmittance_threshold,
        chi_squared_threshold,
    )
    rendered_ref = rasterize_reference(
        indexing_offset_ref,
        gaussian_ids_sorted_ref,
        points_img,
        cov_inv_img,
        opacities,
        colors,
        width,
        height,
        tile_size,
        alpha_threshold,
        transmittance_threshold,
        chi_squared_threshold,
    )

    xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=torch.float32)
    ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    upstream = torch.stack(
        [
            grid_x + 0.35 * grid_y,
            -0.25 * grid_x + 0.75 * grid_y,
            0.55 * grid_x - 0.45 * grid_y,
        ],
        dim=-1,
    )

    grad_points_cuda, grad_cov_cuda, grad_opacities_cuda, grad_colors_cuda = (
        cuda_rasterizer.rasterize_backward(
            upstream,
            indexing_offset_cuda,
            gaussian_ids_sorted_cuda,
            points_img,
            cov_inv_img,
            opacities,
            colors,
            width,
            height,
            tile_size,
            alpha_threshold,
            transmittance_threshold,
            chi_squared_threshold,
        )
    )

    loss_ref = (rendered_ref * upstream).sum()
    grad_points_ref, grad_cov_ref, grad_opacities_ref, grad_colors_ref = torch.autograd.grad(
        loss_ref,
        [points_img, cov_inv_img, opacities, colors],
        retain_graph=False,
        create_graph=False,
    )

    values = [
        metric("rendered", rendered_cuda, rendered_ref),
        metric("points_img", grad_points_cuda, grad_points_ref),
        metric("cov_inv", grad_cov_cuda, grad_cov_ref),
        metric("opacities", grad_opacities_cuda, grad_opacities_ref),
        metric("colors", grad_colors_cuda, grad_colors_ref),
    ]

    threshold_overrides = {
        "rendered": Thresholds(min_cos=0.999999, max_rel=1e-6),
        "points_img": Thresholds(min_cos=0.999, max_rel=2e-3),
        "cov_inv": Thresholds(min_cos=0.999, max_rel=2e-3),
        "opacities": Thresholds(min_cos=0.99999, max_rel=1e-5),
        "colors": Thresholds(min_cos=0.99999, max_rel=1e-5),
    }
    print("\n[rasterize_multitile]")
    success = True
    for name, cosine, rel in values:
        custom_threshold = threshold_overrides.get(name, thresholds)
        print(f"  {name:16s} cos={cosine:.6f} rel={rel:.6e}")
        if cosine < custom_threshold.min_cos or rel > custom_threshold.max_rel:
            success = False
    return success


def check_rasterize_multitile_truncated(seed: int, device: str, thresholds: Thresholds) -> bool:
    torch.manual_seed(seed)
    width, height = 48, 48
    tile_size = 16
    grid_u = torch.tensor([15.2, 15.9, 16.6, 17.3], device=device, dtype=torch.float32)
    grid_v = torch.tensor([15.1, 15.8, 16.5, 17.2], device=device, dtype=torch.float32)
    points_base = torch.cartesian_prod(grid_u, grid_v)
    radii = torch.full((points_base.shape[0], 2), 9.8, device=device, dtype=torch.float32)
    depths = torch.linspace(0.2, 1.0, points_base.shape[0], device=device, dtype=torch.float32)
    mask = torch.ones(points_base.shape[0], device=device, dtype=torch.bool)

    points_img = points_base.detach().clone().requires_grad_(True)
    cov_inv_img = torch.zeros(points_base.shape[0], 3, device=device, dtype=torch.float32)
    cov_inv_img[:, 0] = torch.linspace(0.072, 0.108, points_base.shape[0], device=device)
    cov_inv_img[:, 1] = torch.linspace(-0.007, 0.007, points_base.shape[0], device=device)
    cov_inv_img[:, 2] = torch.linspace(0.078, 0.115, points_base.shape[0], device=device)
    cov_inv_img = cov_inv_img.detach().requires_grad_(True)
    opacities = (
        torch.linspace(0.72, 0.94, points_base.shape[0], device=device)
        .detach()
        .requires_grad_(True)
    )
    colors = (
        torch.stack(
            [
                torch.linspace(0.15, 0.95, points_base.shape[0], device=device),
                torch.linspace(0.90, 0.20, points_base.shape[0], device=device),
                torch.linspace(0.25, 0.85, points_base.shape[0], device=device),
            ],
            dim=-1,
        )
        .detach()
        .requires_grad_(True)
    )

    alpha_threshold = 1e-4
    transmittance_threshold = 1e-4
    chi_squared_threshold = 9.21

    indexing_offset_cuda, gaussian_ids_sorted_cuda = cuda_rasterizer.compute_tile_intersection(
        points_img.detach(),
        radii,
        depths,
        mask,
        width,
        height,
        tile_size,
    )
    indexing_offset_ref, gaussian_ids_sorted_ref = build_reference_tile_layout(
        points_img.detach(),
        radii,
        depths,
        mask,
        width,
        height,
        tile_size,
    )

    layout_ok = torch.equal(indexing_offset_cuda, indexing_offset_ref) and torch.equal(
        gaussian_ids_sorted_cuda, gaussian_ids_sorted_ref
    )
    print(f"\n[rasterize_multitile_truncated_layout] match={layout_ok}")
    if not layout_ok:
        print(f"  indexing_offset_cuda={indexing_offset_cuda.tolist()}")
        print(f"  indexing_offset_ref ={indexing_offset_ref.tolist()}")
        print(f"  gaussian_ids_cuda  ={gaussian_ids_sorted_cuda.tolist()}")
        print(f"  gaussian_ids_ref   ={gaussian_ids_sorted_ref.tolist()}")
        return False

    rendered_cuda = cuda_rasterizer.rasterize(
        indexing_offset_cuda,
        gaussian_ids_sorted_cuda,
        points_img,
        cov_inv_img,
        opacities,
        colors,
        width,
        height,
        tile_size,
        alpha_threshold,
        transmittance_threshold,
        chi_squared_threshold,
    )
    rendered_ref = rasterize_reference(
        indexing_offset_ref,
        gaussian_ids_sorted_ref,
        points_img,
        cov_inv_img,
        opacities,
        colors,
        width,
        height,
        tile_size,
        alpha_threshold,
        transmittance_threshold,
        chi_squared_threshold,
    )

    xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=torch.float32)
    ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    upstream = torch.stack(
        [
            0.6 * grid_x + 0.4 * grid_y,
            -0.4 * grid_x + 0.9 * grid_y,
            0.5 * grid_x - 0.7 * grid_y,
        ],
        dim=-1,
    )

    grad_points_cuda, grad_cov_cuda, grad_opacities_cuda, grad_colors_cuda = (
        cuda_rasterizer.rasterize_backward(
            upstream,
            indexing_offset_cuda,
            gaussian_ids_sorted_cuda,
            points_img,
            cov_inv_img,
            opacities,
            colors,
            width,
            height,
            tile_size,
            alpha_threshold,
            transmittance_threshold,
            chi_squared_threshold,
        )
    )

    loss_ref = (rendered_ref * upstream).sum()
    grad_points_ref, grad_cov_ref, grad_opacities_ref, grad_colors_ref = torch.autograd.grad(
        loss_ref,
        [points_img, cov_inv_img, opacities, colors],
        retain_graph=False,
        create_graph=False,
    )

    values = [
        metric("rendered", rendered_cuda, rendered_ref),
        metric("points_img", grad_points_cuda, grad_points_ref),
        metric("cov_inv", grad_cov_cuda, grad_cov_ref),
        metric("opacities", grad_opacities_cuda, grad_opacities_ref),
        metric("colors", grad_colors_cuda, grad_colors_ref),
    ]

    threshold_overrides = {
        "rendered": Thresholds(min_cos=0.999999, max_rel=1e-6),
        "points_img": Thresholds(min_cos=0.999, max_rel=2e-3),
        "cov_inv": Thresholds(min_cos=0.999, max_rel=2e-3),
        "opacities": Thresholds(min_cos=0.99999, max_rel=1e-5),
        "colors": Thresholds(min_cos=0.99999, max_rel=1e-5),
    }
    print("\n[rasterize_multitile_truncated]")
    success = True
    for name, cosine, rel in values:
        custom_threshold = threshold_overrides.get(name, thresholds)
        print(f"  {name:16s} cos={cosine:.6f} rel={rel:.6e}")
        if cosine < custom_threshold.min_cos or rel > custom_threshold.max_rel:
            success = False
    return success


def check_sh(seed: int, device: str, thresholds: Thresholds) -> bool:
    torch.manual_seed(seed)
    count = 512
    camera_pos = torch.tensor([0.1, -0.2, 0.3], device=device)

    means_cuda = torch.randn(count, 3, device=device, requires_grad=True)
    sh_dc_cuda = torch.randn(count, 3, device=device, requires_grad=True)
    sh_rest_cuda = torch.randn(count, 15, 3, device=device, requires_grad=True)
    mask = torch.ones(count, device=device, dtype=torch.bool)

    means_ref = means_cuda.detach().clone().requires_grad_(True)
    sh_dc_ref = sh_dc_cuda.detach().clone().requires_grad_(True)
    sh_rest_ref = sh_rest_cuda.detach().clone().requires_grad_(True)

    colors_cuda = SphericalHarmonicsFunction.apply(
        camera_pos, means_cuda, sh_dc_cuda, sh_rest_cuda, mask
    )
    loss_cuda = (colors_cuda * 0.3).sum()
    loss_cuda.backward()

    view_dirs = means_ref - camera_pos.unsqueeze(0)
    view_dirs = view_dirs / torch.norm(view_dirs, dim=1, keepdim=True)
    colors_ref = torch_eval_sh(sh_dc_ref, sh_rest_ref, view_dirs)
    loss_ref = (colors_ref * 0.3).sum()
    loss_ref.backward()

    values = [
        metric("means", means_cuda.grad, means_ref.grad),
        metric("sh_dc", sh_dc_cuda.grad, sh_dc_ref.grad),
        metric("sh_rest", sh_rest_cuda.grad, sh_rest_ref.grad),
    ]
    return check_result("spherical_harmonics", values, thresholds)


def check_end_to_end(seed: int, device: str, thresholds: Thresholds) -> bool:
    torch.manual_seed(seed)
    count = 384
    height, width = 64, 64

    world_to_camera = torch.eye(4, device=device, dtype=torch.float32)
    intrinsic = torch.tensor(
        [[60.0, 0.0, width / 2.0], [0.0, 60.0, height / 2.0], [0.0, 0.0, 1.0]],
        device=device,
        dtype=torch.float32,
    )

    means = (torch.rand(count, 3, device=device) - 0.5) * 2.0
    means[:, 2] = torch.rand(count, device=device) * 2.0 + 1.5
    scales_raw = torch.randn(count, 3, device=device) * 0.1
    quaternions_raw = torch.randn(count, 4, device=device)
    opacities_raw = torch.randn(count, device=device) * 0.5
    sh_dc_raw = torch.randn(count, 3, device=device) * 0.2
    sh_rest_raw = torch.randn(count, 15, 3, device=device) * 0.05

    def make_parameters() -> dict[str, torch.Tensor]:
        return {
            "means": means.detach().clone().requires_grad_(True),
            "scales_raw": scales_raw.detach().clone().requires_grad_(True),
            "quaternions_raw": quaternions_raw.detach().clone().requires_grad_(True),
            "opacities_raw": opacities_raw.detach().clone().requires_grad_(True),
            "sh_dc_raw": sh_dc_raw.detach().clone().requires_grad_(True),
            "sh_rest_raw": sh_rest_raw.detach().clone().requires_grad_(True),
        }

    def forward(render_function, parameters: dict[str, torch.Tensor]) -> torch.Tensor:
        scales = torch.exp(parameters["scales_raw"])
        quaternions = parameters["quaternions_raw"] / torch.norm(
            parameters["quaternions_raw"], dim=1, keepdim=True
        )
        opacities = torch.sigmoid(parameters["opacities_raw"])
        image = render_function(
            world_to_camera,
            intrinsic,
            width,
            height,
            parameters["means"],
            scales,
            quaternions,
            opacities,
            parameters["sh_dc_raw"],
            parameters["sh_rest_raw"],
        )
        target = torch.zeros_like(image) + 64.0
        return torch.nn.functional.l1_loss(image, target)

    parameters_cuda = make_parameters()
    loss_cuda = forward(cuda_render, parameters_cuda)
    loss_cuda.backward()

    parameters_ref = make_parameters()
    loss_ref = forward(torch_render, parameters_ref)
    loss_ref.backward()

    print(f"\n[end_to_end] loss_cuda={loss_cuda.item():.6f} loss_torch={loss_ref.item():.6f}")
    values = [
        metric("means", parameters_cuda["means"].grad, parameters_ref["means"].grad),
        metric("scales_raw", parameters_cuda["scales_raw"].grad, parameters_ref["scales_raw"].grad),
        metric(
            "quaternions_raw",
            parameters_cuda["quaternions_raw"].grad,
            parameters_ref["quaternions_raw"].grad,
        ),
        metric(
            "opacities_raw",
            parameters_cuda["opacities_raw"].grad,
            parameters_ref["opacities_raw"].grad,
        ),
        metric("sh_dc_raw", parameters_cuda["sh_dc_raw"].grad, parameters_ref["sh_dc_raw"].grad),
        metric(
            "sh_rest_raw", parameters_cuda["sh_rest_raw"].grad, parameters_ref["sh_rest_raw"].grad
        ),
    ]

    threshold_overrides = {
        "means": Thresholds(min_cos=0.995, max_rel=5e-2),
    }
    success = True
    print("  gradients:")
    for name, cosine, rel in values:
        custom_threshold = threshold_overrides.get(name, thresholds)
        print(f"  {name:16s} cos={cosine:.6f} rel={rel:.6e}")
        if cosine < custom_threshold.min_cos or rel > custom_threshold.max_rel:
            success = False
    return success


def check_end_to_end_real_train_loss(seed: int, device: str, thresholds: Thresholds) -> bool:
    torch.manual_seed(seed)

    with open("colmap_data/input_data.pkl", "rb") as file:
        camera_data, points, rgbs = pickle.load(file)

    points, rgbs = downsample_point_cloud(points, rgbs, 50000)
    points = torch.from_numpy(points).float()
    rgbs = torch.from_numpy(rgbs / 255.0).float()
    base_params = initialize(points, rgbs, sh_degree=3)

    train_dataset = Dataset(camera_data, image_scale=0.5, split="train")
    sample = train_dataset[seed % len(train_dataset)]
    world_to_camera = sample["world_to_camera"].to(device)
    intrinsic = sample["intrinsic"].to(device)
    target_image = sample["image"].to(device)

    parameter_names = [
        "means",
        "scales",
        "quaternions",
        "opacities",
        "sh_coeffs_dc",
        "sh_coeffs_rest",
    ]

    def clone_parameters() -> dict[str, torch.Tensor]:
        return {
            name: base_params[name].detach().clone().requires_grad_(True)
            for name in parameter_names
        }

    def forward_loss(render_function, params: dict[str, torch.Tensor]) -> torch.Tensor:
        scales = torch.exp(params["scales"])
        quaternions = params["quaternions"] / torch.norm(
            params["quaternions"], dim=1, keepdim=True
        ).clamp_min(1e-12)
        opacities = torch.sigmoid(params["opacities"])
        rendered_image = render_function(
            world_to_camera,
            intrinsic,
            target_image.shape[1],
            target_image.shape[0],
            params["means"],
            scales,
            quaternions,
            opacities,
            params["sh_coeffs_dc"],
            params["sh_coeffs_rest"],
        )
        rendered_norm = rendered_image / 255.0
        target_norm = target_image / 255.0
        l1_loss = torch.nn.functional.l1_loss(rendered_norm, target_norm)
        ssim_loss = 1.0 - fused_ssim(
            rendered_norm.permute(2, 0, 1).unsqueeze(0),
            target_norm.permute(2, 0, 1).unsqueeze(0),
            padding="valid",
        )
        return 0.8 * l1_loss + 0.2 * ssim_loss

    params_cuda = clone_parameters()
    loss_cuda = forward_loss(cuda_render, params_cuda)
    loss_cuda.backward()

    params_ref = clone_parameters()
    loss_ref = forward_loss(torch_render, params_ref)
    loss_ref.backward()

    print(f"\n[end_to_end_real] loss_cuda={loss_cuda.item():.6f} loss_torch={loss_ref.item():.6f}")
    values = [
        metric("means", params_cuda["means"].grad, params_ref["means"].grad),
        metric("scales", params_cuda["scales"].grad, params_ref["scales"].grad),
        metric("quaternions", params_cuda["quaternions"].grad, params_ref["quaternions"].grad),
        metric("opacities", params_cuda["opacities"].grad, params_ref["opacities"].grad),
        metric("sh_coeffs_dc", params_cuda["sh_coeffs_dc"].grad, params_ref["sh_coeffs_dc"].grad),
        metric(
            "sh_coeffs_rest", params_cuda["sh_coeffs_rest"].grad, params_ref["sh_coeffs_rest"].grad
        ),
    ]

    threshold_overrides = {
        "means": Thresholds(min_cos=0.995, max_rel=1e-1),
    }
    success = True
    print("  gradients:")
    for name, cosine, rel in values:
        custom_threshold = threshold_overrides.get(name, thresholds)
        print(f"  {name:16s} cos={cosine:.6f} rel={rel:.6e}")
        if cosine < custom_threshold.min_cos or rel > custom_threshold.max_rel:
            success = False
    return success


def main() -> None:
    parser = argparse.ArgumentParser(description="CUDA gradient regression checks")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds to test")
    parser.add_argument("--device", type=str, default="cuda", help="Device for checks")
    parser.add_argument(
        "--skip-real-train-loss",
        action="store_true",
        help="Skip end-to-end real training loss gradient comparison",
    )
    args = parser.parse_args()

    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this check script.")

    thresholds = Thresholds()
    all_ok = True

    for seed in range(args.seeds):
        print(f"\n===== Seed {seed} =====")
        all_ok = check_project_points(seed, args.device, thresholds) and all_ok
        all_ok = check_rasterize(seed + 100, args.device, thresholds) and all_ok
        all_ok = check_rasterize_multitile(seed + 150, args.device, thresholds) and all_ok
        all_ok = check_rasterize_multitile_truncated(seed + 175, args.device, thresholds) and all_ok
        all_ok = check_sh(seed + 200, args.device, thresholds) and all_ok
        all_ok = check_end_to_end(seed + 300, args.device, thresholds) and all_ok
        if not args.skip_real_train_loss:
            all_ok = (
                check_end_to_end_real_train_loss(seed + 400, args.device, thresholds) and all_ok
            )

    if not all_ok:
        raise SystemExit(1)
    print("\nAll CUDA gradient regression checks passed.")


if __name__ == "__main__":
    main()
