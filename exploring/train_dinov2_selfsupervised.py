"""Self-supervised training script for DINOv2 using a simple student–teacher setup.

The script mirrors ``exploring/train_dinov2_infonce_wandb.py`` but trains in a
self-supervised manner.  Two random crops of each image are generated and the
student is trained to match the teacher outputs using the DINO loss.  Metrics
are logged to ``wandb`` and a local loss file, and the teacher model is saved at
the end of every epoch.
"""

import itertools
import os
from typing import Iterable

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import wandb
from torch.optim import SGD
from transformers import AutoImageProcessor, AutoModel

from wildlife_tools.data import ImageDataset
from wildlife_tools.train import set_seed


class SelfSupervisedDataset(ImageDataset):
    """Dataset returning two augmented views of each image."""

    def __init__(self, *args, transform_student=None, transform_teacher=None, **kwargs):
        super().__init__(*args, transform=None, load_label=False, **kwargs)
        if transform_student is None:
            raise ValueError("transform_student must be provided")
        self.transform_student = transform_student
        self.transform_teacher = transform_teacher or transform_student

    def __getitem__(self, idx):
        data = self.metadata.iloc[idx]
        img_path = os.path.join(self.root, data[self.col_path]) if self.root else data[self.col_path]
        img = self.get_image(img_path)
        img_s = self.transform_student(img)
        img_t = self.transform_teacher(img)
        return img_s, img_t


class DINOHead(nn.Module):
    """Projection head used in DINO."""

    def __init__(self, in_dim: int, out_dim: int = 65536, hidden_dim: int = 2048, bottleneck_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last = nn.Linear(bottleneck_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        return self.last(x)


class DINOLoss(nn.Module):
    """Cross entropy between student and teacher outputs with centering."""

    def __init__(
        self,
        out_dim: int,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ) -> None:
        super().__init__()
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum

    def forward(self, student_out: torch.Tensor, teacher_out: torch.Tensor) -> torch.Tensor:
        t_out = F.softmax((teacher_out - self.center) / self.teacher_temp, dim=-1)
        s_out = F.log_softmax(student_out / self.student_temp, dim=-1)
        loss = -torch.sum(t_out * s_out, dim=-1).mean()
        self.center = self.center * self.center_momentum + (1 - self.center_momentum) * t_out.mean(dim=0, keepdim=True)
        return loss


class DINOv2Wrapper(nn.Module):
    """Backbone with projection head."""

    def __init__(self, backbone: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        emb = out.last_hidden_state[:, 0]
        return self.head(emb)


def update_ema(student: nn.Module, teacher: nn.Module, momentum: float) -> None:
    with torch.no_grad():
        for ps, pt in zip(student.parameters(), teacher.parameters()):
            pt.data.mul_(momentum).add_(ps.data, alpha=1 - momentum)


def main(
    csv_path: str,
    root_dir: str,
    epochs: int = 10,
    batch_size: int = 32,
    ema: float = 0.996,
    project: str = "dinov2-selfsup",
    output_dir: str | None = None,
    log_file: str | None = None,
) -> None:
    df = pd.read_csv(csv_path)

    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small", use_fast=True)
    student_backbone = AutoModel.from_pretrained("facebook/dinov2-small")
    teacher_backbone = AutoModel.from_pretrained("facebook/dinov2-small")

    for p in teacher_backbone.parameters():
        p.requires_grad = False

    out_dim = 65536
    student_head = DINOHead(student_backbone.config.hidden_size, out_dim)
    teacher_head = DINOHead(teacher_backbone.config.hidden_size, out_dim)

    for p_s, p_t in zip(student_head.parameters(), teacher_head.parameters()):
        p_t.data.copy_(p_s.data)
        p_t.requires_grad = False

    student = DINOv2Wrapper(student_backbone, student_head)
    teacher = DINOv2Wrapper(teacher_backbone, teacher_head)

    criterion = DINOLoss(out_dim)

    params: Iterable[torch.nn.Parameter] = itertools.chain(student.parameters())
    optimizer = SGD(params=params, lr=0.001, momentum=0.9)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    student.to(device)
    teacher.to(device)
    criterion.to(device)

    transform_student = T.Compose(
        [
            T.RandomResizedCrop(224, scale=(0.4, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.4, 0.4, 0.2, 0.1),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            T.Normalize(mean=processor.image_mean, std=processor.image_std),
        ]
    )

    transform_teacher = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=processor.image_mean, std=processor.image_std),
        ]
    )

    dataset = SelfSupervisedDataset(
        df,
        root=root_dir,
        transform_student=transform_student,
        transform_teacher=transform_teacher,
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    wandb.init(project=project)
    run_dir = output_dir or "runs"
    run_dir = os.path.join(run_dir, f"run-{wandb.run.id}")
    os.makedirs(run_dir, exist_ok=True)
    log_file = log_file or os.path.join(run_dir, "loss.log")

    set_seed(0)
    for epoch in range(epochs):
        student.train()
        losses = []
        for img_s, img_t in loader:
            img_s = img_s.to(device)
            img_t = img_t.to(device)

            s_out = student(img_s)
            with torch.no_grad():
                t_out = teacher(img_t)
            loss = criterion(s_out, t_out)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema(student, teacher, ema)

            losses.append(loss.item())
        epoch_loss = sum(losses) / len(losses)
        print(f"Epoch {epoch}: loss={epoch_loss:.4f}")
        wandb.log({"loss": epoch_loss, "epoch": epoch})
        with open(log_file, "a") as f:
            f.write(f"{epoch},{epoch_loss:.6f}\n")
        torch.save(
            teacher.state_dict(),
            os.path.join(run_dir, f"teacher_epoch_{epoch}.pt"),
        )

    wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Self-supervised DINOv2 training")
    parser.add_argument("--csv", required=True, help="CSV file with a column 'path'")
    parser.add_argument("--root", required=True, help="Root directory for images")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--ema", type=float, default=0.996, help="Teacher EMA momentum")
    parser.add_argument("--project", type=str, default="dinov2-selfsup", help="wandb project name")
    parser.add_argument("--output-dir", type=str, default="runs", help="Directory to save models and logs")
    parser.add_argument("--log-file", type=str, default=None, help="File to log epoch losses")
    args = parser.parse_args()

    main(
        args.csv,
        args.root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        ema=args.ema,
        project=args.project,
        output_dir=args.output_dir,
        log_file=args.log_file,
    )
