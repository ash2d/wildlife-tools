import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from transformers import AutoImageProcessor, AutoModel
import timm

from wildlife_tools.data import SafeImageDataset
from wildlife_tools.features import DeepFeatures
from wildlife_tools.similarity import CosineSimilarity


def load_model(model_name: str):
    """Load DINOv2 or MegaDescriptor model with appropriate transform."""
    if "dinov2" in model_name.lower():
        processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        backbone = AutoModel.from_pretrained(model_name)

        class DINOv2Wrapper(nn.Module):
            def __init__(self, backbone):
                super().__init__()
                self.backbone = backbone

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out = self.backbone(x)
                return out.last_hidden_state[:, 0]

        transform = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=processor.image_mean, std=processor.image_std),
            ]
        )
        return DINOv2Wrapper(backbone), transform
    else:
        backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        transform = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        return backbone, transform


def average_precision(ranked_labels: np.ndarray, true_label: str) -> float:
    relevant = ranked_labels == true_label
    if not np.any(relevant):
        return 0.0
    precisions = []
    hits = 0
    for i, is_rel in enumerate(relevant, 1):
        if is_rel:
            hits += 1
            precisions.append(hits / i)
    return float(np.mean(precisions))


def cmc_curve(ranked_labels: np.ndarray, true_label: str, max_rank: int = 5) -> np.ndarray:
    out = np.zeros(max_rank, dtype=float)
    matches = np.where(ranked_labels == true_label)[0]
    if len(matches) == 0:
        return out
    first = matches[0]
    if first < max_rank:
        out[first:] = 1.0
    return out


def evaluate(sim_matrix: np.ndarray, query_labels: np.ndarray, db_labels: np.ndarray, *, max_rank: int = 5, distance: bool = False):
    idx = np.argsort(sim_matrix, axis=1)
    if not distance:
        idx = idx[:, ::-1]
    ranked = db_labels[idx]

    rank1 = np.mean(ranked[:, 0] == query_labels)
    rank5 = np.mean([ql in ranked[i, :5] for i, ql in enumerate(query_labels)])
    m_ap = np.mean([average_precision(ranked[i], query_labels[i]) for i in range(len(query_labels))])
    cmc = np.mean([cmc_curve(ranked[i], query_labels[i], max_rank=max_rank) for i in range(len(query_labels))], axis=0)
    return {
        "rank1": float(rank1),
        "rank5": float(rank5),
        "mAP": float(m_ap),
        "CMC": cmc.tolist(),
    }


def main(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, transform = load_model(args.model)
    extractor = DeepFeatures(model, batch_size=32, device=device)

    db_df = pd.read_csv(args.database)
    query_df = pd.read_csv(args.query)

    db_ds = SafeImageDataset(db_df, root=args.root, transform=transform, log_file=args.log_file)
    query_ds = SafeImageDataset(query_df, root=args.root, transform=transform, log_file=args.log_file)

    with torch.no_grad():
        db_features = extractor(db_ds)
        query_features = extractor(query_ds)

    cos = CosineSimilarity()
    sim_cos = cos(query_features, db_features)
    dist_euc = np.linalg.norm(
        query_features.features[:, None, :] - db_features.features[None, :, :], axis=2
    )

    metrics_cos = evaluate(sim_cos, query_ds.labels_string, db_ds.labels_string, max_rank=5, distance=False)
    metrics_euc = evaluate(dist_euc, query_ds.labels_string, db_ds.labels_string, max_rank=5, distance=True)

    with open(args.output, "w") as f:
        f.write(f"model: {args.model}\n")
        f.write(f"database_csv: {args.database}\n")
        f.write(f"query_csv: {args.query}\n\n")

        f.write("Cosine Similarity:\n")
        f.write(f"  Rank-1: {metrics_cos['rank1']:.4f}\n")
        f.write(f"  Rank-5: {metrics_cos['rank5']:.4f}\n")
        f.write(f"  mAP: {metrics_cos['mAP']:.4f}\n")
        f.write(f"  CMC: {metrics_cos['CMC']}\n\n")

        f.write("Euclidean Distance:\n")
        f.write(f"  Rank-1: {metrics_euc['rank1']:.4f}\n")
        f.write(f"  Rank-5: {metrics_euc['rank5']:.4f}\n")
        f.write(f"  mAP: {metrics_euc['mAP']:.4f}\n")
        f.write(f"  CMC: {metrics_euc['CMC']}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate re-id embeddings")
    parser.add_argument("model", help="Model name or path")
    parser.add_argument("database", help="CSV with database images")
    parser.add_argument("query", help="CSV with query images")
    parser.add_argument("--root", required=True, help="Root directory for image paths")
    parser.add_argument("--output", default="results.txt", help="Where to store the metrics")
    parser.add_argument("--log-file", default="unreadable_images.txt", help="File to log unreadable image paths")
    args = parser.parse_args()
    main(args)
