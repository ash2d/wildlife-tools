"""Simple example for fine-tuning DINOv2 with the ``BasicTrainer``.

This script loads image metadata from a CSV file and trains a DINOv2 backbone
using the ``InfoNCELoss`` objective.  It mirrors the workflow of the
``exploring/dino_integrate_train.ipynb`` notebook in a runnable Python file.

Possible improvements:
    * Integrate `wandb` for experiment tracking by logging the loss inside the
      training loop or via an ``epoch_callback`` in ``BasicTrainer``.
    * Configure ``logging`` to write training progress to a file for easier
      debugging when running long jobs.
    * Create a simple SLURM ``sbatch`` script that invokes this module so it can
      be scheduled on a GPU cluster.
"""

import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import wandb
from torch.optim import SGD
from transformers import AutoImageProcessor, AutoModel

from wildlife_tools.data import ImageDataset
from wildlife_tools.train import BasicTrainer, InfoNCELoss, set_seed


class SafeImageDataset(ImageDataset):
    """ImageDataset that skips unreadable images and logs their paths."""

    def __init__(self, *args, log_file: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_file = log_file

    def __getitem__(self, idx):
        attempts = 0
        start_idx = idx
        while attempts < len(self.metadata):
            data = self.metadata.iloc[idx]
            img_path = os.path.join(self.root, data[self.col_path]) if self.root else data[self.col_path]
            try:
                img = self.get_image(img_path)
            except (FileNotFoundError, ValueError):
                if self.log_file:
                    with open(self.log_file, "a") as f:
                        f.write(f"{img_path}\n")
                idx = (idx + 1) % len(self.metadata)
                attempts += 1
                continue

            if self.transform:
                img = self.transform(img)
            if self.load_label:
                return img, self.labels[idx]
            return img

        raise RuntimeError("No valid images found starting from index " f"{start_idx}")


class DINOv2Wrapper(nn.Module):
    """Extract CLS embeddings from a DINOv2 backbone."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(x)
        return outputs.last_hidden_state[:, 0]


def _attention_overlays(model, images, processor):
    """Create attention map overlays for a batch of images."""
    with torch.no_grad():
        outputs = model.backbone(images, output_attentions=True)
    attn = outputs.attentions[-1][:, :, 0, 1:]
    attn = attn.mean(dim=1)
    side = int(attn.shape[-1] ** 0.5)
    attn = attn.reshape(images.shape[0], side, side)
    attn = F.interpolate(attn.unsqueeze(1), size=images.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)

    overlays = []
    for img, m in zip(images, attn):
        for t, mean, std in zip(img, processor.image_mean, processor.image_std):
            t.mul_(std).add_(mean)
        img_np = img.permute(1, 2, 0).cpu().numpy()
        m_np = m.cpu().numpy()
        m_np = (m_np - m_np.min()) / (m_np.max() - m_np.min() + 1e-6)
        heatmap = plt.get_cmap("viridis")(m_np)[..., :3]
        overlay = np.clip(0.6 * img_np + 0.4 * heatmap, 0, 1)
        overlays.append(overlay)
    return overlays


def _parse_model_size(model_name: str) -> str:
    """Return the DINOv2 model size from its name."""
    return model_name.split("-")[-1]


def _run_name(model_name: str, num_images: int, training_type: str) -> str:
    size = _parse_model_size(model_name)
    return f"dino-{size}_{training_type}_{num_images}"


def main(
    csv_path: str,
    root_dir: str,
    epochs: int = 3,
    batch_size: int = 16,
    project: str = "dinov2-infonce",
    log_bad_images: str | None = None,
    model_name: str = "facebook/dinov2-small",
    output_dir: str = "/gws/nopw/j04/iecdt/dash/embeddings/models",
):
    df = pd.read_csv(csv_path)
    run_name = _run_name(model_name, len(df), "infonce")

    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    backbone = AutoModel.from_pretrained(model_name)

    # freeze all parameters in the backbone
    for param in backbone.parameters():
        param.requires_grad = False

    # The DINOv2 model from Hugging Face has an encoder with a list of layers.
    # unfreeze the last two layers. DINOv2-small has 12 layers (0-11).
    layers_to_train = [10, 11]
    for name, param in backbone.named_parameters():
        for layer_num in layers_to_train:
            if f"encoder.layer.{layer_num}." in name:
                param.requires_grad = True

    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=processor.image_mean, std=processor.image_std),
        ]
    )

    dataset = SafeImageDataset(df, root=root_dir, transform=transform, log_file=log_bad_images)
    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    wandb.init(project=project, name=run_name)

    objective = InfoNCELoss(temperature=0.1)
    params_to_train = []
    for name, param in backbone.named_parameters():
        if param.requires_grad:
            params_to_train.append(param)
            # print(f"Training parameter: {name}")

    # If objective has learnable parameters (like ArcFaceLoss), add them
    if hasattr(objective, "parameters"):
        params_to_train.extend(objective.parameters())

    # params = itertools.chain(backbone.parameters())
    optimizer = SGD(params=params_to_train, lr=0.001, momentum=0.9)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(0)

    def epoch_callback(trainer, epoch_data):
        wandb.log({"train_loss": epoch_data["train_loss_epoch_avg"], "epoch": trainer.epoch})
        with torch.no_grad():
            imgs = torch.stack([dataset[i][0] for i in range(min(4, len(dataset)))])
            imgs = imgs.to(device)
            overlays = _attention_overlays(trainer.model, imgs, processor)
            wandb.log(
                {
                    "examples": [wandb.Image((o * 255).astype(np.uint8)) for o in overlays],
                    "epoch": trainer.epoch,
                }
            )
        trainer.save(os.path.join(run_dir, f"model_epoch_{trainer.epoch}.pt"))

    trainer = BasicTrainer(
        dataset=dataset,
        model=DINOv2Wrapper(backbone),
        objective=objective,
        optimizer=optimizer,
        epochs=epochs,
        device=device,
        batch_size=batch_size,
        epoch_callback=epoch_callback,
    )
    trainer.train()
    wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train DINOv2 with InfoNCE loss")
    parser.add_argument("--csv", help="CSV file with columns 'path' and 'identity'")
    parser.add_argument("--root", help="Root directory for images")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--project", type=str, default="dinov2-infonce", help="wandb project name")
    parser.add_argument(
        "--log-bad-images",
        type=str,
        default=None,
        help="File to log paths of unreadable images",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="facebook/dinov2-small",
        help="DINOv2 model name",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/gws/nopw/j04/iecdt/dash/embeddings/models",
        help="Directory to save model checkpoints",
    )
    args = parser.parse_args()

    main(
        args.csv,
        args.root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        project=args.project,
        log_bad_images=args.log_bad_images,
        model_name=args.model_name,
        output_dir=args.output_dir,
    )
