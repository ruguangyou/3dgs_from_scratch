from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tensorboard.backend.event_processing import event_accumulator


SELECTED_STEPS = [0, 100, 200, 300, 500, 800, 1000]


def load_image(image_event: object) -> np.ndarray:
    encoded = image_event.encoded_image_string
    image = Image.open(io.BytesIO(encoded)).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def center_of_mass(image: np.ndarray) -> tuple[float, float]:
    luminance = image.mean(axis=2)
    total = float(luminance.sum())
    if total <= 1e-12:
        return float("nan"), float("nan")
    ys, xs = np.indices(luminance.shape)
    cy = float((ys * luminance).sum() / total)
    cx = float((xs * luminance).sum() / total)
    return cx, cy


def normalized_offset(image: np.ndarray) -> tuple[float, float]:
    height, width = image.shape[:2]
    cx, cy = center_of_mass(image)
    return (cx - (width - 1) / 2.0) / width, (cy - (height - 1) / 2.0) / height


def summarize_scalars(event_acc: event_accumulator.EventAccumulator) -> None:
    scalar_tags = [
        "train/loss",
        "train/l1_loss",
        "train/ssim_loss",
        "train/ssim_lambda",
        "train/scale_raw_max",
        "train/scale_raw_mean",
        "train/opacity_mean",
    ]
    print("SCALAR SNAPSHOTS")
    for tag in scalar_tags:
        events = {event.step: event.value for event in event_acc.Scalars(tag)}
        values = [events.get(step) for step in SELECTED_STEPS if step in events]
        formatted = ", ".join(
            f"{step}:{value:.6f}" for step, value in zip(SELECTED_STEPS, values, strict=False)
        )
        print(f"{tag}\t{formatted}")


def summarize_images(event_acc: event_accumulator.EventAccumulator) -> None:
    rendered_events = {event.step: event for event in event_acc.Images("train/rendered")}
    target_events = {event.step: event for event in event_acc.Images("train/target")}

    print("IMAGE SNAPSHOTS")
    print("step\trender_dx\trender_dy\ttarget_dx\ttarget_dy\trender_mean\ttarget_mean\timage_l1")
    for step in SELECTED_STEPS:
        if step not in rendered_events or step not in target_events:
            continue
        rendered = load_image(rendered_events[step])
        target = load_image(target_events[step])
        render_dx, render_dy = normalized_offset(rendered)
        target_dx, target_dy = normalized_offset(target)
        image_l1 = float(np.abs(rendered - target).mean())
        print(
            f"{step}\t{render_dx:+.4f}\t{render_dy:+.4f}\t{target_dx:+.4f}\t{target_dy:+.4f}\t"
            f"{rendered.mean():.4f}\t{target.mean():.4f}\t{image_l1:.4f}"
        )


def main() -> None:
    path = Path(sys.argv[1])
    event_acc = event_accumulator.EventAccumulator(
        str(path),
        size_guidance={
            event_accumulator.SCALARS: 0,
            event_accumulator.IMAGES: 0,
        },
    )
    event_acc.Reload()
    summarize_scalars(event_acc)
    summarize_images(event_acc)


if __name__ == "__main__":
    main()
