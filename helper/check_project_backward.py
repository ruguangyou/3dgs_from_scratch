import torch
import cuda_rasterizer


def stat(name, a, b, m):
    a = (a * m).flatten()
    b = (b * m).flatten()
    na = torch.norm(a).item()
    nb = torch.norm(b).item()
    cos = torch.dot(a, b).item() / max(na * nb, 1e-12)
    rel = torch.norm(a - b).item() / max(nb, 1e-12)
    print(f"{name:8s} cos={cos:.6f} rel={rel:.6f} na={na:.3e} nb={nb:.3e}")


def main():
    torch.manual_seed(0)
    device = "cuda"
    N = 256
    h, w = 64, 64

    W = torch.eye(4, device=device, dtype=torch.float32)
    K = torch.tensor(
        [[60.0, 0.0, w / 2], [0.0, 60.0, h / 2], [0.0, 0.0, 1.0]],
        device=device,
        dtype=torch.float32,
    )

    points = (torch.rand(N, 3, device=device) - 0.5) * 1.0
    points[:, 2] = torch.rand(N, device=device) * 0.5 + 2.0
    points = points.detach().requires_grad_(True)
    scales = (torch.rand(N, 3, device=device) * 0.2 + 0.8).detach().requires_grad_(True)
    quat_raw = torch.randn(N, 4, device=device)
    quats = (quat_raw / torch.norm(quat_raw, dim=1, keepdim=True)).detach().requires_grad_(True)
    opac = torch.ones(N, device=device) * 0.8

    p_img, depth, cov, cov_inv, radii, mask = cuda_rasterizer.project_points(
        points, scales, quats, opac, W, K, 0.01, 100.0, 1 / 255, 0.1, 256.0, w, h
    )

    up_p = torch.randn_like(p_img)
    up_cov_inv = torch.randn_like(cov_inv) * 0.1

    gp_cuda, gs_cuda, gq_cuda = cuda_rasterizer.project_points_backward(
        up_p, up_cov_inv, points, scales, quats, opac, W, K, cov, mask
    )

    pts_cam = (W[:3, :3] @ points.t()).t() + W[:3, 3]
    x = pts_cam[:, 0]
    y = pts_cam[:, 1]
    z = pts_cam[:, 2]
    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]
    points_img_ref = torch.stack([u, v], dim=-1)

    qi, qj, qk, qr = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    R = torch.stack(
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
    ).reshape(N, 3, 3)
    S = torch.diag_embed(scales)
    Cw = R @ S @ S @ R.transpose(1, 2)
    W3 = W[:3, :3]
    Cc = W3.unsqueeze(0) @ Cw @ W3.t().unsqueeze(0)

    inv_z = 1.0 / z
    inv_z2 = inv_z * inv_z
    J = torch.zeros((N, 2, 3), device=device)
    J[:, 0, 0] = K[0, 0] * inv_z
    J[:, 0, 2] = -K[0, 0] * x * inv_z2
    J[:, 1, 1] = K[1, 1] * inv_z
    J[:, 1, 2] = -K[1, 1] * y * inv_z2
    Ci = J @ Cc @ J.transpose(1, 2)
    Ci = (Ci + Ci.transpose(1, 2)) * 0.5

    a = Ci[:, 0, 0]
    b = Ci[:, 0, 1]
    c = Ci[:, 1, 1]
    det = a * c - b * b
    cov_inv_ref = torch.stack([c / det, -b / det, a / det], dim=-1)

    loss = (points_img_ref * up_p).sum() + (cov_inv_ref * up_cov_inv).sum()
    gp_ref, gs_ref, gq_ref = torch.autograd.grad(
        loss, [points, scales, quats], retain_graph=False, create_graph=False
    )

    mask3 = mask.float().unsqueeze(-1)
    mask4 = mask.float().unsqueeze(-1)
    print("mask ratio", mask.float().mean().item())
    stat("points", gp_cuda, gp_ref, mask3)
    stat("scales", gs_cuda, gs_ref, mask3)
    stat("quats", gq_cuda, gq_ref, mask4)


if __name__ == "__main__":
    main()
