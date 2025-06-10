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

import pandas as pd
import itertools
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.optim import SGD
from transformers import AutoImageProcessor, AutoModel

from wildlife_tools.train import BasicTrainer, InfoNCELoss, set_seed
from wildlife_tools.data import ImageDataset


class DINOv2Wrapper(nn.Module):
    """Extract CLS embeddings from a DINOv2 backbone."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(x)
        return outputs.last_hidden_state[:, 0]


def main(csv_path: str, root_dir: str, epochs: int = 3, batch_size: int = 16):
    df = pd.read_csv(csv_path)

    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small", use_fast=True)
    backbone = AutoModel.from_pretrained("facebook/dinov2-small")

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])

    dataset = ImageDataset(df, root=root_dir, transform=transform)

    objective = InfoNCELoss(temperature=0.1)

    params = itertools.chain(backbone.parameters())
    optimizer = SGD(params=params, lr=0.001, momentum=0.9)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(0)
    trainer = BasicTrainer(
        dataset=dataset,
        model=DINOv2Wrapper(backbone),
        objective=objective,
        optimizer=optimizer,
        epochs=epochs,
        device=device,
        batch_size=batch_size,
    )
    trainer.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train DINOv2 with InfoNCE loss")
    parser.add_argument("csv", help="CSV file with columns 'path' and 'identity'")
    parser.add_argument("root", help="Root directory for images")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    main(args.csv, args.root, epochs=args.epochs, batch_size=args.batch_size)
