
import torch
from PIL import Image
import numpy as np
from main import render, resize_camera


def evaluate():
    pos = torch.load("test_data/pos_7000.pt").cuda()
    opacity_raw = torch.load("test_data/opacity_raw_7000.pt").cuda()
    f_dc = torch.load("test_data/f_dc_7000.pt").cuda()
    f_rest = torch.load("test_data/f_rest_7000.pt").cuda()
    scale_raw = torch.load("test_data/scale_raw_7000.pt").cuda()
    q_raw = torch.load("test_data/q_rot_7000.pt").cuda()
    sh = torch.empty((pos.shape[0], 16, 3), device=pos.device, dtype=pos.dtype)
    sh[:, 0] = f_dc
    sh[:, 1:, 0] = f_rest[:, :15]  # R
    sh[:, 1:, 1] = f_rest[:, 15:30]  # G
    sh[:, 1:, 2] = f_rest[:, 30:45]  # B
    gaussians = {
        "means": pos,
        "opacities": opacity_raw,
        "sh_coeffs_dc": sh[:, 0, :],
        "sh_coeffs_rest": sh[:, 1:, :],
        "scales": scale_raw,
        "quaternions": q_raw,
    }

    cam_parameters = np.load("test_data/cam_meta.npy", allow_pickle=True).item()
    orbit_c2ws = torch.load("test_data/kitchen_orbit.pt").cuda()
    height = cam_parameters["height"]
    width = cam_parameters["width"]
    fx, fy = cam_parameters["fx"], cam_parameters["fy"]
    cx, cy = width / 2, height / 2
    intrinsic = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32).cuda()

    with torch.no_grad():
        for i, c2w in enumerate(orbit_c2ws):
            w2c = torch.zeros((4, 4), dtype=torch.float32).cuda()
            w2c[:3, :3] = c2w[:3, :3].t()
            w2c[:3, 3] = -(w2c[:3, :3] @ c2w[:3, 3])
            w2c[3, 3] = 1.0
            camera = {
                "world_to_camera": w2c.unsqueeze(0),
                "intrinsic": intrinsic.unsqueeze(0),
                "image": torch.zeros(
                    (1, height, width, 3), dtype=torch.float32
                ).cuda(),  # dummy image
                "camera_id": 0,
            }
            camera = resize_camera(camera, image_scale=0.5)

            print("rendering frame", i)
            img = render(gaussians, camera)
            print("saving frame", i)
            
            Image.fromarray((img.cpu().detach().numpy()).astype(np.uint8)).save(
                f"test_data/novel_views/frame_{i:04d}.png"
            )


if __name__ == "__main__":
    evaluate()
