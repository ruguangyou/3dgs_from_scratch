import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from pycolmap import SceneManager
from torch.nn import functional as F
from tqdm import tqdm


def load_colmap(data_dir):
    scene_manager = SceneManager(f"{data_dir}/sparse/0")
    scene_manager.load_cameras()
    scene_manager.load_images()
    scene_manager.load_points3D()

    camera_data = {}
    for image in tqdm(scene_manager.images.values(), desc="Processing cameras"):
        camera_id = image.camera_id

        # camera extrinsics in world-to-camera format
        rotation = image.R()
        translation = image.tvec.reshape(3, 1)
        world_to_camera = np.vstack((np.hstack((rotation, translation)), np.array([[0, 0, 0, 1]])))

        # camera intrinsics
        camera = scene_manager.cameras[camera_id]
        intrinsic = np.array([[camera.fx, 0, camera.cx], [0, camera.fy, camera.cy], [0, 0, 1]])

        # distortion parameters
        model = camera.camera_type
        if model == 0 or model == "SIMPLE_PINHOLE":
            params = np.empty(0, dtype=np.float32)
            camtype = "perspective"
        elif model == 1 or model == "PINHOLE":
            params = np.empty(0, dtype=np.float32)
            camtype = "perspective"
        elif model == 2 or model == "SIMPLE_RADIAL":
            params = np.array([camera.k1, 0.0, 0.0, 0.0], dtype=np.float32)
            camtype = "perspective"
        elif model == 3 or model == "RADIAL":
            params = np.array([camera.k1, camera.k2, 0.0, 0.0], dtype=np.float32)
            camtype = "perspective"
        elif model == 4 or model == "OPENCV":
            params = np.array([camera.k1, camera.k2, camera.p1, camera.p2], dtype=np.float32)
            camtype = "perspective"
        elif model == 5 or model == "OPENCV_FISHEYE":
            params = np.array([camera.k1, camera.k2, camera.k3, camera.k4], dtype=np.float32)
            camtype = "fisheye"
        else:
            raise ValueError(f"Unsupported camera type: {model}")

        # image data
        image_size = (camera.width, camera.height)
        image_file = imageio.imread(f"{data_dir}/images/{image.name}")[..., :3]  # ensure RGB format

        camera_data[camera_id] = Camera(
            world_to_camera, intrinsic, params, image_size, image_file, camtype
        )

        # 3D points and RGB colors
        points = scene_manager.points3D.astype(np.float32)
        rgbs = scene_manager.point3D_colors.astype(np.uint8)

    return camera_data, points, rgbs


def resize_camera(camera, image_scale: float):
    if image_scale == 1.0:
        return camera

    resized_image = (
        F.interpolate(
            camera["image"].unsqueeze(0).permute(0, 3, 1, 2),  # (H, W, C) -> (1, C, H, W)
            scale_factor=image_scale,
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=False,
        )
        .permute(0, 2, 3, 1)
        .squeeze(0)
    )  # (1, C, H, W) -> (H, W, C)

    return {
        "world_to_camera": camera["world_to_camera"],
        "intrinsic": camera["intrinsic"] * image_scale,
        "image": resized_image,
    }


class Camera:
    def __init__(
        self,
        world_to_camera,
        intrinsic,
        distortion_params,
        image_size,
        image_file,
        camtype,
    ):
        # undistortion
        width, height = image_size
        if len(distortion_params) == 0:
            pass
        elif camtype == "perspective":
            intrinsic_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                intrinsic, distortion_params, image_size, 0
            )
            mapx, mapy = cv2.initUndistortRectifyMap(
                intrinsic,
                distortion_params,
                None,
                intrinsic_undist,
                image_size,
                cv2.CV_32FC1,
            )
            mask = None
        elif camtype == "fisheye":
            fx = intrinsic[0, 0]
            fy = intrinsic[1, 1]
            cx = intrinsic[0, 2]
            cy = intrinsic[1, 2]
            grid_x, grid_y = np.meshgrid(
                np.arange(width, dtype=np.float32),
                np.arange(height, dtype=np.float32),
                indexing="xy",
            )  # shape (height, width)
            x_normalized = (grid_x - cx) / fx
            y_normalized = (grid_y - cy) / fy
            theta = np.sqrt(x_normalized**2 + y_normalized**2)  # angle from the optical axis
            r = (
                1.0
                + distortion_params[0] * theta**2
                + distortion_params[1] * theta**4
                + distortion_params[2] * theta**6
                + distortion_params[3] * theta**8
            )  #  radial distortion factor

            # undistorted coordinates in the original image
            mapx = (fx * r * x_normalized + cx).astype(np.float32)
            mapy = (fy * r * y_normalized + cy).astype(np.float32)

            # valid pixel mask after undistortion, shape (height, width)
            mask = np.logical_and(
                np.logical_and(mapx > 0, mapx < width - 1),
                np.logical_and(mapy > 0, mapy < height - 1),
            )

            y_indices, x_indices = np.nonzero(mask)
            y_min, y_max = y_indices.min(), y_indices.max()
            x_min, x_max = x_indices.min(), x_indices.max()
            intrinsic_undist = np.array(
                [[fx, 0, cx - x_min], [0, fy, cy - y_min], [0, 0, 1]], dtype=np.float32
            )
            roi_undist = (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)
        else:
            raise ValueError(f"Unsupported camera type: {camtype}")

        image = cv2.remap(image_file, mapx, mapy, interpolation=cv2.INTER_LINEAR)
        x, y, w, h = roi_undist

        self.image = image[y : y + h, x : x + w]
        self.world_to_camera = world_to_camera
        self.intrinsic = intrinsic_undist


class Dataset:
    def __init__(self, camera_data, image_scale=1.0, split="train", test_every=8):
        self.camera_data = camera_data
        self.image_scale = image_scale
        camera_ids = sorted(camera_data.keys())
        if split == "train":
            self.camera_ids = [cid for i, cid in enumerate(camera_ids) if i % test_every != 0]
        else:
            self.camera_ids = [cid for i, cid in enumerate(camera_ids) if i % test_every == 0]

    def __len__(self):
        return len(self.camera_ids)

    def __getitem__(self, idx):
        idx = idx % len(self.camera_ids)  # wrap around for safety
        camera_id = self.camera_ids[idx]
        data = self.camera_data[camera_id]
        camera = {
            "world_to_camera": torch.from_numpy(data.world_to_camera).float(),
            "intrinsic": torch.from_numpy(data.intrinsic).float(),
            "image": torch.from_numpy(data.image).float(),
        }
        return resize_camera(camera, self.image_scale)
