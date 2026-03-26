import torch
import cuda_rasterizer


def stat(name, a, b):
    a = a.flatten()
    b = b.flatten()
    na = torch.norm(a).item()
    nb = torch.norm(b).item()
    cos = torch.dot(a, b).item() / max(na * nb, 1e-12)
    rel = torch.norm(a - b).item() / max(nb, 1e-12)
    print(f"{name:10s} cos={cos:.6f} rel={rel:.6f} na={na:.3e} nb={nb:.3e}")


def rasterize_ref(
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
):
    unique_tiles = indexing_offset.shape[0]
    total_tiles = gaussian_ids_sorted.shape[0]
    num_tiles_per_row = (width + tile_size - 1) // tile_size
    output = torch.zeros((height, width, 3), device=points_img.device, dtype=points_img.dtype)

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

            for i in range(range_start, range_end):
                gid = gaussian_ids_sorted[i]
                du = u - points_img[gid, 0]
                dv = v - points_img[gid, 1]
                inv_cov_00 = cov_inv_img[gid, 0]
                inv_cov_01 = cov_inv_img[gid, 1]
                inv_cov_11 = cov_inv_img[gid, 2]

                exponent = inv_cov_00 * du * du + 2.0 * inv_cov_01 * du * dv + inv_cov_11 * dv * dv
                if exponent.item() > chi_squared_threshold:
                    continue

                alpha = torch.exp(-0.5 * exponent) * opacities[gid]
                if alpha.item() < alpha_threshold:
                    continue

                weight = alpha * transmittance
                pixel = pixel + weight * colors[gid]
                transmittance = transmittance * (1.0 - alpha)
                if transmittance.item() < transmittance_threshold:
                    break

            output[v, u] = pixel

    return output


def main():
    torch.manual_seed(0)
    device = "cuda"
    N = 12
    width, height = 16, 16
    tile_size = 16

    indexing_offset = torch.tensor([0], dtype=torch.int32, device=device)
    gaussian_ids_sorted = torch.arange(N, dtype=torch.int32, device=device)

    points_img = (torch.rand(N, 2, device=device) * 12.0 + 2.0).detach().requires_grad_(True)
    cov_inv_img = torch.zeros(N, 3, device=device)
    cov_inv_img[:, 0] = torch.rand(N, device=device) * 0.03 + 0.01
    cov_inv_img[:, 1] = (torch.rand(N, device=device) - 0.5) * 0.005
    cov_inv_img[:, 2] = torch.rand(N, device=device) * 0.03 + 0.01
    cov_inv_img = cov_inv_img.detach().requires_grad_(True)
    opacities = (torch.rand(N, device=device) * 0.25 + 0.55).detach().requires_grad_(True)
    colors = torch.rand(N, 3, device=device).detach().requires_grad_(True)

    alpha_threshold = 0.0
    transmittance_threshold = 0.0
    chi_squared_threshold = 20.0

    out_cuda = cuda_rasterizer.rasterize(
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

    grad_out = torch.randn_like(out_cuda)
    gp_cuda, gcov_cuda, gop_cuda, gcol_cuda = cuda_rasterizer.rasterize_backward(
        grad_out,
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

    out_ref = rasterize_ref(
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
    loss = (out_ref * grad_out).sum()
    gp_ref, gcov_ref, gop_ref, gcol_ref = torch.autograd.grad(
        loss, [points_img, cov_inv_img, opacities, colors], retain_graph=False, create_graph=False
    )

    stat("points_img", gp_cuda, gp_ref)
    stat("cov_inv", gcov_cuda, gcov_ref)
    stat("opacities", gop_cuda, gop_ref)
    stat("colors", gcol_cuda, gcol_ref)


if __name__ == "__main__":
    main()
