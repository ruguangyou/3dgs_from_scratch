import torch
from src.cuda.wrapper import SphericalHarmonicsFunction
from src.gaussian import evaluate_spherical_harmonics as torch_eval_sh

torch.manual_seed(2)
device = "cuda"
N = 512

camera_pos = torch.tensor([0.1, -0.2, 0.3], device=device)
means_a = torch.randn(N, 3, device=device, requires_grad=True)
shdc_a = torch.randn(N, 3, device=device, requires_grad=True)
shrest_a = torch.randn(N, 15, 3, device=device, requires_grad=True)
mask = torch.ones(N, device=device, dtype=torch.bool)

means_b = means_a.detach().clone().requires_grad_(True)
shdc_b = shdc_a.detach().clone().requires_grad_(True)
shrest_b = shrest_a.detach().clone().requires_grad_(True)

colors_cuda = SphericalHarmonicsFunction.apply(camera_pos, means_a, shdc_a, shrest_a, mask)
loss_cuda = (colors_cuda * 0.3).sum()
loss_cuda.backward()

view_dirs = means_b - camera_pos.unsqueeze(0)
view_dirs = view_dirs / torch.norm(view_dirs, dim=1, keepdim=True)
colors_torch = torch_eval_sh(shdc_b, shrest_b, view_dirs)
loss_torch = (colors_torch * 0.3).sum()
loss_torch.backward()


def stat(name, g1, g2):
    g1f = g1.flatten()
    g2f = g2.flatten()
    n1 = torch.norm(g1f).item()
    n2 = torch.norm(g2f).item()
    cos = torch.dot(g1f, g2f).item() / max(n1 * n2, 1e-12)
    rel = torch.norm(g1f - g2f).item() / max(n2, 1e-12)
    print(f"{name:10s} cos={cos:.6f} rel={rel:.6f} n1={n1:.3e} n2={n2:.3e}")


stat("means", means_a.grad, means_b.grad)
stat("sh_dc", shdc_a.grad, shdc_b.grad)
stat("sh_rest", shrest_a.grad, shrest_b.grad)
