"""Verify: does resize_camera corrupt intrinsics via shared numpy memory?"""
import numpy as np, torch
from src.dataset import resize_camera

# Simulate a camera with float32 intrinsic (same as Camera.__init__)
intrinsic_np = np.array([[500.0, 0, 400], [0, 500, 300], [0, 0, 1]], dtype=np.float32)

print(f"Original intrinsic fx={intrinsic_np[0,0]:.1f} fy={intrinsic_np[1,1]:.1f}")

for i in range(5):
    camera = {
        "world_to_camera": torch.eye(4),
        "intrinsic": torch.from_numpy(intrinsic_np).float(),
        "image": torch.randn(100, 100, 3),
    }
    result = resize_camera(camera, 0.5)
    print(f"  After access {i+1}: original fx={intrinsic_np[0,0]:.4f}  "
          f"resized fx={result['intrinsic'][0,0]:.4f}")
