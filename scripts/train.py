#!/usr/bin/env python3
"""Train the proposed self-supervised denoiser and paper ablations."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from complex_blindspot_dmri.data import DWIShellDataset
from complex_blindspot_dmri.masking import (
    generate_paired_masks,
    mixed_fill,
    neighbor_fill,
)
from complex_blindspot_dmri.models import ComplexCNN, ComplexNAFNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_complex_l1(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Masked complex reconstruction objective used by the proposed method."""
    denominator = mask.sum().clamp_min(1)
    real_loss = (torch.abs(prediction.real - target.real) * mask).sum() / denominator
    imag_loss = (torch.abs(prediction.imag - target.imag) * mask).sum() / denominator
    return 0.5 * (real_loss + imag_loss)


def sample_directions(dwi: torch.Tensor, count: int) -> torch.Tensor:
    """Sample the same number of target directions independently per subject."""
    batch, directions = dwi.shape[:2]
    count = min(count, directions)
    return torch.stack(
        [dwi[i, torch.randperm(directions, device=dwi.device)[:count]] for i in range(batch)]
    )


def build_model(name: str, num_slices: int, num_targets: int) -> torch.nn.Module:
    model_class = ComplexNAFNet if name == "nafnet" else ComplexCNN
    return model_class(
        dwi_channels=num_slices,
        out_channels=1,
        num_slices=num_slices,
        support_k=0,
        max_targets=num_targets,
    )


def remove_parallel_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key.replace("module.module.", "").replace("module.", ""): value
        for key, value in state_dict.items()
    }


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    device: torch.device,
) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(remove_parallel_prefix(state), strict=True)
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return int(checkpoint.get("epoch", -1)) + 1, float(
        checkpoint.get("best_val_loss", float("inf"))
    )


def save_checkpoint(
    path: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    best_val_loss: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "config": vars(args),
        },
        path,
    )


def run_epoch(
    loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    context = torch.enable_grad() if training else torch.no_grad()
    progress = tqdm(loader, desc="train" if training else "validation", leave=False)

    with context:
        for batch in progress:
            dwi = batch["dwi"].to(device, non_blocking=True)
            target = sample_directions(dwi, args.num_targets)
            batch_size, target_count, slices, height, width = target.shape
            target_flat = target.reshape(batch_size * target_count, slices, height, width)

            target_mask, hidden_mask = generate_paired_masks(
                target_flat.shape,
                device,
                target_probability=args.mask_probability,
                pair_probability=args.pair_probability,
                pair_radius_min=args.pair_radius_min,
                pair_radius_max=args.pair_radius_max,
                include_diagonal=not args.no_diagonal,
            )
            if args.corruption == "mixed":
                corrupted = mixed_fill(
                    target_flat,
                    target_mask,
                    hidden_mask,
                    local_radius_min=args.fill_radius_min,
                    local_radius_max=args.fill_radius_max,
                    include_diagonal=not args.no_diagonal,
                    hidden_global_probability=args.hidden_global_probability,
                    max_global_shift=args.max_global_shift,
                )
            else:
                corrupted = neighbor_fill(
                    target_flat,
                    target_mask,
                    hidden_mask,
                    include_diagonal=not args.no_diagonal,
                )

            prediction = model(
                corrupted.reshape(batch_size, target_count, slices, height, width)
            ).reshape(batch_size * target_count, 1, height, width)
            center = slices // 2
            target_center = target_flat[:, center : center + 1]
            loss = masked_complex_l1(prediction, target_center, target_mask)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total += float(loss.item())
            progress.set_postfix(loss=f"{loss.item():.5f}")
    return total / max(len(loader), 1)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"device: {device}")

    train_dataset = DWIShellDataset(args.train_manifest, args.num_slices)
    validation_dataset = (
        DWIShellDataset(args.validation_manifest, args.num_slices)
        if args.validation_manifest
        else None
    )
    common_loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **common_loader_options)
    validation_loader = (
        DataLoader(validation_dataset, shuffle=False, **common_loader_options)
        if validation_dataset
        else None
    )

    model = build_model(args.model, args.num_slices, args.num_targets).to(device)
    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-2, betas=(0.9, 0.9)
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.minimum_learning_rate
    )
    start_epoch, best_loss = 0, float("inf")
    if args.resume:
        start_epoch, best_loss = load_checkpoint(
            args.resume, model, optimizer, scheduler, device
        )

    output = Path(args.output_dir)
    for epoch in range(start_epoch, args.epochs):
        train_loss = run_epoch(train_loader, model, device, args, optimizer)
        validation_loss = (
            run_epoch(validation_loader, model, device, args, None)
            if validation_loader
            else train_loss
        )
        scheduler.step()
        print(
            f"epoch {epoch + 1:03d}/{args.epochs:03d} "
            f"train={train_loss:.6f} validation={validation_loss:.6f}"
        )

        improved = validation_loss < best_loss
        if improved:
            best_loss = validation_loss
            save_checkpoint(
                output / "model_best.pth",
                epoch,
                model,
                optimizer,
                scheduler,
                best_loss,
                args,
            )
        if (epoch + 1) % args.save_interval == 0 or epoch + 1 == args.epochs:
            save_checkpoint(
                output / "checkpoint_latest.pth",
                epoch,
                model,
                optimizer,
                scheduler,
                best_loss,
                args,
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Self-supervised complex-valued blind-spot DWI denoising"
    )
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/proposed"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--model", choices=["nafnet", "complex-cnn"], default="nafnet")
    parser.add_argument("--corruption", choices=["mixed", "neighbor"], default="mixed")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--num-slices", type=int, default=1, help="Use 1 for the paper's 2D setting"
    )
    parser.add_argument("--num-targets", type=int, default=1)
    parser.add_argument("--mask-probability", type=float, default=0.10)
    parser.add_argument("--pair-probability", type=float, default=0.50)
    parser.add_argument("--pair-radius-min", type=int, default=1)
    parser.add_argument("--pair-radius-max", type=int, default=2)
    parser.add_argument("--fill-radius-min", type=int, default=1)
    parser.add_argument("--fill-radius-max", type=int, default=2)
    parser.add_argument("--hidden-global-probability", type=float, default=0.15)
    parser.add_argument("--max-global-shift", type=int, default=8)
    parser.add_argument("--no-diagonal", action="store_true")
    parser.add_argument("--save-interval", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str)
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
