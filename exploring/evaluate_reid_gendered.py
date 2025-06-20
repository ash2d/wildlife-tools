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
import os


def load_model(model_name: str):
    """Load DINOv2 or MegaDescriptor model with appropriate transform."""
    if "dinov2" in model_name.lower():
        processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        backbone = AutoModel.from_pretrained(model_name)
        checkpoint = torch.load(args.save_path, map_location=device)
        new_checkpoint = {k.replace("backbone.", ""): v for k, v in checkpoint.items()}
        backbone.load_state_dict(new_checkpoint, strict=False)  # strict=False ignores unexpected keys

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
        checkpoint = torch.load(args.save_path, weights_only=False)
        backbone.load_state_dict(checkpoint['model'])
        transform = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
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

def evaluate_stream(
    query_features: np.ndarray,
    query_labels: np.ndarray,
    db_features: np.ndarray,
    db_labels: np.ndarray,
    *,
    max_rank: int = 5,
):
    """Compute metrics without allocating a full similarity matrix."""

    db_norm = db_features / np.linalg.norm(db_features, axis=1, keepdims=True)
    query_norm = query_features / np.linalg.norm(query_features, axis=1, keepdims=True)

    rank1_c, rank5_c, aps_c = [], [], []
    rank1_e, rank5_e, aps_e = [], [], []
    cmc_cos = np.zeros(max_rank, dtype=float)
    cmc_euc = np.zeros(max_rank, dtype=float)

    for q_feat, q_norm_feat, q_label in zip(query_features, query_norm, query_labels):
        sims_cos = np.dot(db_norm, q_norm_feat)
        idx_cos = np.argsort(sims_cos)[::-1]
        ranked_cos = db_labels[idx_cos]

        rank1_c.append(ranked_cos[0] == q_label)
        rank5_c.append(q_label in ranked_cos[:5])
        aps_c.append(average_precision(ranked_cos, q_label))
        cmc_cos += cmc_curve(ranked_cos, q_label, max_rank=max_rank)

        dist_euc = np.linalg.norm(db_features - q_feat, axis=1)
        idx_euc = np.argsort(dist_euc)
        ranked_euc = db_labels[idx_euc]

        rank1_e.append(ranked_euc[0] == q_label)
        rank5_e.append(q_label in ranked_euc[:5])
        aps_e.append(average_precision(ranked_euc, q_label))
        cmc_euc += cmc_curve(ranked_euc, q_label, max_rank=max_rank)

    n = len(query_features)
    metrics_cos = {
        "rank1": float(np.mean(rank1_c)),
        "rank5": float(np.mean(rank5_c)),
        "mAP": float(np.mean(aps_c)),
        "CMC": (cmc_cos / n).tolist(),
    }
    metrics_euc = {
        "rank1": float(np.mean(rank1_e)),
        "rank5": float(np.mean(rank5_e)),
        "mAP": float(np.mean(aps_e)),
        "CMC": (cmc_euc / n).tolist(),
    }

    return metrics_cos, metrics_euc

def gender(path,df_db, df_query):
    df_ids = pd.read_csv(path)
    df_ids = df_ids[['FishID', 'individual_id']]
    # Create gender columns based on first letter (F = female, M = male)
    df_ids['gender'] = df_ids['FishID'].str[0].map({
        'F': 'female', 
        'M': 'male',
        'f': 'female', 
        'm': 'male'
    })
    #dictionary to map individual_id to gender
    id_to_gender = dict(zip(df_ids['individual_id'], df_ids['gender']))
    df_db['gender'] = df_db['identity'].map(id_to_gender)
    df_query['gender'] = df_query['identity'].map(id_to_gender)
    df_db_m = df_db[df_db['gender'] == 'male']
    df_db_f = df_db[df_db['gender'] == 'female']
    df_query_m = df_query[df_query['gender'] == 'male']
    df_query_f = df_query[df_query['gender'] == 'female']

    return df_db_m, df_db_f, df_query_m, df_query_f


def main(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, transform = load_model(args.model)
    extractor = DeepFeatures(model, batch_size=32, device=device)
    path = '/home/users/dash/guppies/pre_embeddings/merged_2m_id_split_4m.csv'

    db_df = pd.read_csv(args.database)
    query_df = pd.read_csv(args.query)
    df_db_m, df_db_f, df_query_m, df_query_f = gender(path, db_df, query_df)

    db_ds_m = SafeImageDataset(df_db_m, root=args.root, transform=transform, log_file=args.log_file)
    db_ds_f = SafeImageDataset(df_db_f, root=args.root, transform=transform, log_file=args.log_file)
    query_ds_m = SafeImageDataset(df_query_m, root=args.root, transform=transform, log_file=args.log_file)
    query_ds_f = SafeImageDataset(df_query_f, root=args.root, transform=transform, log_file=args.log_file)
    
    with torch.no_grad():
        db_features_m = extractor(db_ds_m)
        db_features_f = extractor(db_ds_f)
        query_features_m = extractor(query_ds_m)
        query_features_f = extractor(query_ds_f)

    # cos = CosineSimilarity()
    # sim_cos = cos(query_features, db_features)
    # dist_euc = np.linalg.norm(
    #     query_features.features[:, None, :] - db_features.features[None, :, :], axis=2
    # )

    # metrics_cos = evaluate(sim_cos, query_ds.labels_string, db_ds.labels_string, max_rank=5, distance=False)
    # metrics_euc = evaluate(dist_euc, query_ds.labels_string, db_ds.labels_string, max_rank=5, distance=True)
    
    metrics_cos_m, metrics_euc_m = evaluate_stream(
        query_features_m.features,
        query_ds_m.labels_string,
        db_features_m.features,
        db_ds_m.labels_string,
        max_rank=5,
    )
    metrics_cos_f, metrics_euc_f = evaluate_stream(
        query_features_f.features,
        query_ds_f.labels_string,
        db_features_f.features,
        db_ds_f.labels_string,
        max_rank=5,
    )
    # Use the directory of save_path for output if save_path is provided
    output_path = args.output
    if args.save_path:
        output_dir = os.path.dirname(args.save_path)
        output_path = os.path.join(output_dir, args.output)

    with open(output_path, "w") as f:
        f.write(f"model: {args.model}\n")
        f.write(f"database_csv: {args.database}\n")
        f.write(f"query_csv: {args.query}\n\n")
        f.write('male results:\n')
        f.write("Cosine Similarity:\n")
        f.write(f"  Rank-1: {metrics_cos_m['rank1']:.4f}\n")
        f.write(f"  Rank-5: {metrics_cos_m['rank5']:.4f}\n")
        f.write(f"  mAP: {metrics_cos_m['mAP']:.4f}\n")
        f.write(f"  CMC: {metrics_cos_m['CMC']}\n\n")

        f.write("Euclidean Distance:\n")
        f.write(f"  Rank-1: {metrics_euc_m['rank1']:.4f}\n")
        f.write(f"  Rank-5: {metrics_euc_m['rank5']:.4f}\n")
        f.write(f"  mAP: {metrics_euc_m['mAP']:.4f}\n")
        f.write(f"  CMC: {metrics_euc_m['CMC']}\n")
        f.write('female results:\n')
        f.write("Cosine Similarity:\n")
        f.write(f"  Rank-1: {metrics_cos_f['rank1']:.4f}\n")
        f.write(f"  Rank-5: {metrics_cos_f['rank5']:.4f}\n")
        f.write(f"  mAP: {metrics_cos_f['mAP']:.4f}\n")
        f.write(f"  CMC: {metrics_cos_f['CMC']}\n\n")

        f.write("Euclidean Distance:\n")
        f.write(f"  Rank-1: {metrics_euc_f['rank1']:.4f}\n")
        f.write(f"  Rank-5: {metrics_euc_f['rank5']:.4f}\n")
        f.write(f"  mAP: {metrics_euc_f['mAP']:.4f}\n")
        f.write(f"  CMC: {metrics_euc_f['CMC']}\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate re-id embeddings")
    parser.add_argument("--model", default="hf-hub:BVRA/MegaDescriptor-T-224", help="Model name, e.g. hf-hub:BVRA/MegaDescriptor-T-224 or facebook/dinov2-small")
    parser.add_argument("--save-path", help="Path to the model checkpoint if needed")
    parser.add_argument("--database", help="CSV with database images")
    parser.add_argument("--query", help="CSV with query images")
    parser.add_argument("--root", required=True, help="Root directory for image paths")
    parser.add_argument("--output", default="results.txt", help="Where to store the metrics")
    parser.add_argument("--log-file", default="unreadable_images.txt", help="File to log unreadable image paths")
    args = parser.parse_args()
    main(args)
