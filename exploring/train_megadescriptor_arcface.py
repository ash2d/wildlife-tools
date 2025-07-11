"""Simple example for fine-tuning MegaDescriptor with the ``BasicTrainer``.

This script loads image metadata from a CSV file and trains a MegaDescriptor
backbone using the ``ArcFaceLoss`` objective.  It mirrors the workflow of the
``baselines/training`` notebooks in a runnable Python file.
"""

import itertools
import os
import pandas as pd
import torch
import timm
import torchvision.transforms as T
from torch.optim import SGD
import wandb

from wildlife_tools.train import ArcFaceLoss, BasicTrainer, set_seed
from wildlife_tools.data import ImageDataset, SafeImageDataset


def _parse_model_size(model_name: str) -> str:
    part = model_name.split("MegaDescriptor-")[-1].split("-")[0]
    mapping = {"T": "tiny", "S": "small", "B": "base", "L": "large"}
    return mapping.get(part.upper(), part.lower())


def _run_name(model_name: str, num_images: int, training_type: str) -> str:
    size = _parse_model_size(model_name)
    return f"mgd-{size}_{training_type}_{num_images}"


def _get_embedding_size(model: torch.nn.Module) -> int:
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224)
        out = model(dummy)
        if isinstance(out, (list, tuple)):
            out = out[0]
        if hasattr(out, "last_hidden_state"):
            out = out.last_hidden_state
        return out.shape[1]


def main(
    csv_path: str,
    root_dir: str,
    model_name: str = "hf-hub:BVRA/MegaDescriptor-T-224",
    epochs: int = 3,
    batch_size: int = 16,
    output_dir: str = "/gws/nopw/j04/iecdt/dash/embeddings/models",
):
    df = pd.read_csv(csv_path)
    run_name = _run_name(model_name, len(df), "arcface")

    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    log_bad_images = "/home/users/dash/guppies/embeddings/wildlife-tools/exploring/unreadable_images.txt"
    dataset = SafeImageDataset(df, root=root_dir, transform=transform, log_file=log_bad_images)
    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    wandb.init(project="MGD-ArcFace", name=run_name)

    backbone = timm.create_model(model_name, num_classes=0, pretrained=True)
    embedding_size = _get_embedding_size(backbone)

    objective = ArcFaceLoss(
        num_classes=dataset.num_classes,
        embedding_size=embedding_size,
        margin=0.5,
        scale=64,
    )

    params = itertools.chain(backbone.parameters(), objective.parameters())
    optimizer = SGD(params=params, lr=0.001, momentum=0.9)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(0)

    def epoch_callback(trainer, epoch_data):
        wandb.log({"train_loss": epoch_data["train_loss_epoch_avg"], "epoch": trainer.epoch})
        trainer.save(os.path.join(run_dir, f"model_epoch_{trainer.epoch}.pt"))

    trainer = BasicTrainer(
        dataset=dataset,
        model=backbone,
        objective=objective,
        optimizer=optimizer,
        epochs=epochs,
        device=device,
        batch_size=batch_size,
        epoch_callback=epoch_callback,
    )
    trainer.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train MegaDescriptor with ArcFace loss")
    parser.add_argument("--csv", help="CSV file with columns 'path' and 'identity'")
    parser.add_argument("--root", help="Root directory for images")
    parser.add_argument("--model-name", type=str, default="hf-hub:BVRA/MegaDescriptor-T-224", help="timm model name")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/gws/nopw/j04/iecdt/dash/embeddings/models",
        help="Where to save model checkpoints",
    )
    args = parser.parse_args()

    main(
        args.csv,
        args.root,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )
