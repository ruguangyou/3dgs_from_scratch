import logging
import numpy as np
import torch
from tqdm import tqdm

from dataset import load_colmap, Dataset


logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)


def train():
    logging.info("Loading COLMAP data...")
    camera_data = load_colmap(data_dir="colmap_data")

    train_dataset = Dataset(camera_data, split="train")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True)
    train_dataiter = iter(train_dataloader)

    logging.info("Training started...")
    max_steps = 1000
    for step in tqdm(range(max_steps), desc="Training"):
        data = next(train_dataiter)


if __name__ == "__main__":
    train()