"""Self-supervised training script for DINOv2 using a simple student–teacher setup.

Two random crops of each image are generated and the student is trained to match the
teacher outputs using the DINO loss.  Metrics are logged to ``wandb`` and a local loss
file, and the teacher model is saved at the end of every epoch.
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
from transformers import AutoImageProcessor, AutoModel, AutoConfig

from wildlife_tools.data import ImageDataset
from wildlife_tools.train import set_seed


def _parse_model_size(model_name: str) -> str:
    return model_name.split("-")[-1]


def _run_name(model_name: str, num_images: int, training_type: str) -> str:
    size = _parse_model_size(model_name)
    return f"dino-{size}_{training_type}_{num_images}"


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


def get_cosine_lr_schedule(initial_lr, final_lr, current_step, total_steps):
    """Calculate learning rate based on cosine decay schedule."""
    progress = current_step / total_steps
    cosine_decay = 0.5 * (1 + torch.cos(torch.tensor(progress * torch.pi)))
    return final_lr + (initial_lr - final_lr) * cosine_decay


def main(
    csv_path: str,
    root_dir: str,
    epochs: int = 10,
    batch_size: int = 32,
    ema: float = 0.996,
    project: str = "dinov2-selfsup",
    output_dir: str | None = None,
    log_file: str | None = None,
    initial_lr: float = 0.00001,
    final_lr: float = 0.000001,
    model_name: str = "facebook/dinov2-small",
) -> None:
    df = pd.read_csv(csv_path)
    run_name = _run_name(model_name, len(df), "selfsup")
    # ------------------------------------------------------------------
    #   STUDENT & TEACHER BACKBONES – initialise from pretrained
    # ------------------------------------------------------------------

    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    student_backbone = AutoModel.from_pretrained("facebook/dinov2-small")
    teacher_backbone = AutoModel.from_pretrained("facebook/dinov2-small")

    # # Initialize student and teacher backbones with random weights
    # config = AutoConfig.from_pretrained(model_name)
    # student_backbone = AutoModel.from_config(config)
    # teacher_backbone = AutoModel.from_config(config)
    teacher_backbone.eval()                                   # freeze BN/Dropout
    for p in teacher_backbone.parameters():
        p.requires_grad = False

    out_dim = 2048
    student_head = DINOHead(student_backbone.config.hidden_size, out_dim)
    teacher_head = DINOHead(teacher_backbone.config.hidden_size, out_dim)
    for p_s, p_t in zip(student_head.parameters(),
                        teacher_head.parameters()):
        p_t.data.copy_(p_s.data)
        p_t.requires_grad = False

    student = DINOv2Wrapper(student_backbone, student_head)
    teacher = DINOv2Wrapper(teacher_backbone, teacher_head)


    criterion = DINOLoss(out_dim)

    params: Iterable[torch.nn.Parameter] = itertools.chain(student.parameters())
    #  ---- AdamW: backbone lr = 2e-4 , head lr = 2e-3  ----

    base_lr = 2e-4
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for n, p in student.named_parameters()
                        if "head" not in n], "lr": base_lr},
            {"params": student_head.parameters(), "lr": base_lr * 10},
        ],
        betas=(0.9, 0.999),
        weight_decay=0.04,
    )
    # optimizer = SGD(params=params, lr=initial_lr, momentum=0.9)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    student.to(device)
    teacher.to(device)
    criterion.to(device)

    mean, std = processor.image_mean, processor.image_std

    global_crop = T.Compose([
        T.RandomResizedCrop(224, scale=(0.25, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.4, 0.4, 0.2, 0.1),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(), T.Normalize(mean, std),
    ])

    local_crop = T.Compose([
        T.RandomResizedCrop(96, scale=(0.05, 0.25)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.4, 0.4, 0.2, 0.1),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(), T.Normalize(mean, std),
    ])

    class MultiCropDataset(ImageDataset):
        """Return 2 global + 6 local crops per image."""
        def __init__(self, *args, **kw):
            super().__init__(*args, transform=None, load_label=False, **kw)

        def __getitem__(self, idx):
            data = self.metadata.iloc[idx]
            img_path = os.path.join(self.root, data[self.col_path]) if self.root else data[self.col_path]
            img = self.get_image(img_path)
            crops = [global_crop(img) for _ in range(2)]
            crops += [local_crop(img) for _ in range(6)]
            return crops
    # transform_student = T.Compose(
    #     [
    #         T.RandomResizedCrop(224, scale=(0.4, 1.0)),
    #         T.RandomHorizontalFlip(),
    #         T.ColorJitter(0.4, 0.4, 0.2, 0.1),
    #         T.RandomGrayscale(p=0.2),
    #         T.ToTensor(),
    #         T.Normalize(mean=processor.image_mean, std=processor.image_std),
    #     ]
    # )

    # transform_teacher = T.Compose(
    #     [
    #         T.Resize(256),
    #         T.CenterCrop(224),
    #         T.ToTensor(),
    #         T.Normalize(mean=processor.image_mean, std=processor.image_std),
    #     ]
    # )

    # dataset = SelfSupervisedDataset(
    #     df,
    #     root=root_dir,
    #     transform_student=transform_student,
    #     transform_teacher=transform_teacher,
    # )
    dataset = MultiCropDataset(df, root=root_dir)
    def multi_crop_collate(batch):
        # batch: list length B, each item is a list of 8 tensors
        transposed = list(zip(*batch))           # len 8, each is tuple len B
        return [torch.stack(v, 0) for v in transposed]   # list of 8 tensors [B,C,H,W]
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                         drop_last=True,collate_fn=multi_crop_collate)

    run_dir = os.path.join(output_dir or "runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    wandb.init(project=project, name=run_name)
    log_file = log_file or os.path.join(run_dir, "loss.log")

    # Calculate total steps for the cosine schedule
    total_steps = epochs * len(loader)
    current_step = 0
    def teacher_temperature(step, warmup=3000, t0=0.04, t1=0.07):
        if step > warmup: return t1
        return t0 + (t1 - t0) * step / warmup

    def ema_momentum(step, base_m=0.996, total=total_steps):
        # cosine ramp from 0.9 → 0.996
        return 0.9 + (base_m - 0.9) * (1 + torch.cos(
            torch.tensor(step / total * torch.pi))) / 2

    set_seed(0)
    for epoch in range(epochs):
        student.train()
        for crops in loader:          # crops is length-8 list
            current_step += 1

            # -------- learning-rate cosine schedule ----------
            lr = get_cosine_lr_schedule(base_lr, base_lr * 0.01,
                                        current_step, total_steps)
            for g in optimizer.param_groups:
                g["lr"] = lr * (10 if g is optimizer.param_groups[1] else 1)

            # -------- push all views to GPU ------------------
            crops = [c.to(device, non_blocking=True) for c in crops]
            student_out = [student(v) for v in crops]                # 8 tensors
            with torch.no_grad():
                teacher_out = [teacher(v) for v in crops[:2]]        # 2 globals

            # -------- DINOLoss over views --------------------
            temp = teacher_temperature(current_step)
            criterion.teacher_temp = temp                            # update T
            loss = 0.
            for so in student_out:
                for to in teacher_out:
                    loss += criterion(so, to)
            loss /= (len(student_out) * len(teacher_out))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            # -------- EMA update -----------------------------
            m = ema_momentum(current_step)
            update_ema(student, teacher, m)

        print(f"Epoch {epoch}: loss={loss.item():.4f}, lr={lr:.6f}")
        epoch_loss = loss.item()
        # print(f"Epoch {epoch}: loss={epoch_loss:.4f}, lr={current_lr:.6f}")
        wandb.log({"loss": epoch_loss, "epoch": epoch})
        # Create log file if it doesn't exist
        if not os.path.exists(os.path.dirname(log_file)):
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
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
    parser.add_argument("--project", type=str, default="dinov2-selfsup-100000", help="wandb project name")
    parser.add_argument("--output-dir", type=str, default="runs", help="Directory to save models and logs")
    parser.add_argument("--log-file", type=str, default=None, help="File to log epoch losses")
    parser.add_argument("--initial-lr", type=float, default=0.00001, help="Initial learning rate")
    parser.add_argument("--final-lr", type=float, default=0.000001, help="Final learning rate")
    parser.add_argument("--model-name", type=str, default="facebook/dinov2-small", help="DINOv2 model name")
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
        initial_lr=args.initial_lr,
        final_lr=args.final_lr,
        model_name=args.model_name,
    )
