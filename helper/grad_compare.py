import torch
from src.cuda.wrapper import render as cuda_render
from src.torch_rasterizer import render as torch_render

torch.manual_seed(0)
device = "cuda"
N = 512
height, width = 64, 64

world_to_camera = torch.eye(4, device=device, dtype=torch.float32)
intrinsic = torch.tensor(
    [[60.0, 0.0, width / 2.0], [0.0, 60.0, height / 2.0], [0.0, 0.0, 1.0]],
    device=device,
    dtype=torch.float32,
)

means = (torch.rand(N, 3, device=device) - 0.5) * 2.0
means[:, 2] = torch.rand(N, device=device) * 2.0 + 1.5
scales_raw = torch.randn(N, 3, device=device) * 0.1
quaternions_raw = torch.randn(N, 4, device=device)
opacities_raw = torch.randn(N, device=device) * 0.5
sh_dc_raw = torch.randn(N, 3, device=device) * 0.2
sh_rest_raw = torch.randn(N, 15, 3, device=device) * 0.05


def make_params():
    params = {
        "means": means.detach().clone().requires_grad_(True),
        "scales_raw": scales_raw.detach().clone().requires_grad_(True),
        "quaternions_raw": quaternions_raw.detach().clone().requires_grad_(True),
        "opacities_raw": opacities_raw.detach().clone().requires_grad_(True),
        "sh_dc_raw": sh_dc_raw.detach().clone().requires_grad_(True),
        "sh_rest_raw": sh_rest_raw.detach().clone().requires_grad_(True),
    }
    return params


def forward(render_fn, params):
    scales = torch.exp(params["scales_raw"])
    quaternions = params["quaternions_raw"] / torch.norm(
        params["quaternions_raw"], dim=1, keepdim=True
    )
    opacities = torch.sigmoid(params["opacities_raw"])
    image = render_fn(
        world_to_camera,
        intrinsic,
        width,
        height,
        params["means"],
        scales,
        quaternions,
        opacities,
        params["sh_dc_raw"],
        params["sh_rest_raw"],
    )
    target = torch.zeros_like(image) + 64.0
    loss = torch.nn.functional.l1_loss(image, target)
    return loss, image


params_cuda = make_params()
loss_cuda, image_cuda = forward(cuda_render, params_cuda)
loss_cuda.backward()

params_torch = make_params()
loss_torch, image_torch = forward(torch_render, params_torch)
loss_torch.backward()


def grad_stats(name, g1, g2):
    g1f = g1.detach().flatten()
    g2f = g2.detach().flatten()
    n1 = torch.norm(g1f).item()
    n2 = torch.norm(g2f).item()
    cos = torch.dot(g1f, g2f).item() / (max(n1 * n2, 1e-12))
    diff = torch.norm(g1f - g2f).item() / (max(n2, 1e-12))
    print(f"{name:14s} | norm_cuda={n1:.4e} norm_torch={n2:.4e} cos={cos:.6f} rel_diff={diff:.6f}")


print(f"loss_cuda={loss_cuda.item():.6f} loss_torch={loss_torch.item():.6f}")
print(
    f"image_mean_cuda={image_cuda.mean().item():.4f} image_mean_torch={image_torch.mean().item():.4f}"
)

for key in ["means", "scales_raw", "quaternions_raw", "opacities_raw", "sh_dc_raw", "sh_rest_raw"]:
    grad_stats(key, params_cuda[key].grad, params_torch[key].grad)
