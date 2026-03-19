import torch
from PIL import Image
import numpy as np
from main import render


def evaluate():
    pos = torch.load("test_data/pos_7000.pt").cuda()  # (N, 3)
    scale_raw = torch.load("test_data/scale_raw_7000.pt").cuda()  # (N,)
    scale = torch.exp(scale_raw)
    opacity_raw = torch.load("test_data/opacity_raw_7000.pt").cuda()  # (N,)
    opacity = torch.sigmoid(opacity_raw)
    q_raw = torch.load("test_data/q_rot_7000.pt").cuda()  # (N, 4)
    q = q_raw / torch.norm(q_raw, dim=1, keepdim=True)
    f_dc = torch.load("test_data/f_dc_7000.pt").cuda()  # (N, 3)
    f_rest = torch.load("test_data/f_rest_7000.pt").cuda()
    f_rest = f_rest.reshape(-1, 3, 15).permute(0, 2, 1)  # (N, 15, 3)

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
            image_scale = 0.5
            camera = {
                "world_to_camera": w2c.unsqueeze(0),
                "intrinsic": (intrinsic * image_scale).unsqueeze(0),
                "image": torch.zeros(
                    (1, int(height * image_scale), int(width * image_scale), 3), dtype=torch.float32
                ).cuda(),  # dummy image
            }

            print("rendering frame", i)
            img = render(
                camera,
                pos,
                scale,
                q,
                opacity,
                f_dc,
                f_rest,
            )
            print("saving frame", i)

            Image.fromarray((img.cpu().detach().numpy()).astype(np.uint8)).save(
                f"test_data/novel_views/frame_{i:04d}.png"
            )


if __name__ == "__main__":
    evaluate()
