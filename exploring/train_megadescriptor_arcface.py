"""Simple example for fine-tuning MegaDescriptor with the ``BasicTrainer``.

This script loads image metadata from a CSV file and trains a MegaDescriptor
backbone using the ``ArcFaceLoss`` objective.  It mirrors the workflow of the
``baselines/training`` notebooks in a runnable Python file.
"""

import itertools
import pandas as pd
import torch
import timm
import torchvision.transforms as T
from torch.optim import SGD

from wildlife_tools.train import ArcFaceLoss, BasicTrainer, set_seed
from wildlife_tools.data import ImageDataset


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
):
    df = pd.read_csv(csv_path)

    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    dataset = ImageDataset(df, root=root_dir, transform=transform)

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
    trainer = BasicTrainer(
        dataset=dataset,
        model=backbone,
        objective=objective,
        optimizer=optimizer,
        epochs=epochs,
        device=device,
        batch_size=batch_size,
    )
    trainer.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train MegaDescriptor with ArcFace loss")
    parser.add_argument("csv", help="CSV file with columns 'path' and 'identity'")
    parser.add_argument("root", help="Root directory for images")
    parser.add_argument("--model-name", type=str, default="hf-hub:BVRA/MegaDescriptor-T-224", help="timm model name")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    main(
        args.csv,
        args.root,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
