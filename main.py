import logging
import math
import pickle
import numpy as np
import torch
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from dataset import load_colmap, Dataset


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def initialize(
    points: torch.Tensor,
    rgbs: torch.Tensor,
    init_scale: float = 1.0,
    init_opacity: float = 0.1,
    sh_degree: int = 3,
    device: str = "cuda",
):
    N = points.shape[0]

    # initalize gaussians size to be the average distance of the 3 nearest neighbors
    pts = points.cpu().numpy()
    knn = NearestNeighbors(n_neighbors=4).fit(pts)  # 3 closest neighbors + the point itself
    distances, _ = knn.kneighbors(pts)  # distances shape: (N, 4)
    dist = torch.from_numpy(distances[:, 1:]).float()  # 0th column is distance to itself
    avg_dist = torch.sqrt(torch.square(dist).mean(axis=1))  # avg_dist shape: (N,)
    # in log space, after exp the values will be positive
    scales = torch.log(avg_dist * init_scale).unsqueeze(-1).repeat(1, 3)

    quaternions = torch.rand(N, 4)
    # in logit space, after sigmoid the values will be constrained in (0, 1)
    opacities = torch.logit(torch.full((N,), init_opacity))

    sh_coeffs = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # (N, 16, 3) for degree 3
    # initialize the first SH coefficients (the constant term) with the RGB colors
    C0 = 0.28209479177387814  # DC component of SH basis
    sh_coeffs[:, 0, :] = (rgbs - 0.5) / C0  # Normalize to [-0.5, 0.5] and scale by C0

    learnable_params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(points),
        "scales": torch.nn.Parameter(scales),
        "quaternions": torch.nn.Parameter(quaternions),
        "opacities": torch.nn.Parameter(opacities),
        # dc and rest of the SH coefficients are separated to apply for different learning rates
        "sh_coeffs_dc": torch.nn.Parameter(sh_coeffs[:, 0, :]),
        "sh_coeffs_rest": torch.nn.Parameter(sh_coeffs[:, 1:, :]),
    }).to(device)

    return learnable_params


def setup_optimizers(
    gaussians,
    batch_size,
    means_lr=1.6e-4,
    scales_lr=5e-3,
    quaternions_lr=1e-3,
    opacities_lr=5e-2,
    sh_dc_lr=2.5e-3,
    sh_rest_lr=2.5e-3 / 20,
):
    lr_dict = {
        "means": means_lr,
        "scales": scales_lr,
        "quaternions": quaternions_lr,
        "opacities": opacities_lr,
        "sh_coeffs_dc": sh_dc_lr,
        "sh_coeffs_rest": sh_rest_lr,
    }
    optimizers = {
        name: torch.optim.Adam(
            # scale learning rates by sqrt of batch size
            [{"params": gaussians[name], "lr": lr * math.sqrt(batch_size)}],
            eps=1e-15,
            betas=(1 - batch_size*(1-0.9), 1 - batch_size*(1-0.999)),
        )
        for name, lr in lr_dict.items()
    }
    return optimizers


def train():
    load_cached_input = True
    data_dir="colmap_data"
    if load_cached_input:
        logging.info("Loading cached input...")
        with open(f"{data_dir}/input_data.pkl", "rb") as f:
            camera_data, points, rgbs = pickle.load(f)
    else:
        logging.info("Loading COLMAP data...")
        camera_data, points, rgbs = load_colmap(data_dir)
        with open(f"{data_dir}/input_data.pkl", "wb") as f:
            pickle.dump((camera_data, points, rgbs), f)

    batch_size = 1
    train_dataset = Dataset(camera_data, split="train")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_dataiter = iter(train_dataloader)

    logging.info(f"Initializing {points.shape[0]} gaussians...")
    points = torch.from_numpy(points).float()
    rgbs = torch.from_numpy(rgbs / 255.0).float()  # Normalize RGB values to [0, 1]
    gaussians = initialize(points, rgbs)

    logging.info("Optimizer setup...")
    optimizers = setup_optimizers(gaussians, batch_size)

    logging.info("Training started...")
    max_steps = 100
    for step in tqdm(range(max_steps), desc="Training"):
        data = next(train_dataiter)


if __name__ == "__main__":
    train()
