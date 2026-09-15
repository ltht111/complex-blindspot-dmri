"""Leakage-aware masking and mixed filling used for self-supervised training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _shift_with_replicate_padding(
    image: torch.Tensor, shift_y: int, shift_x: int
) -> torch.Tensor:
    if shift_y == 0 and shift_x == 0:
        return image
    height, width = image.shape[-2:]
    pad_top = max(shift_y, 0)
    pad_bottom = max(-shift_y, 0)
    pad_left = max(shift_x, 0)
    pad_right = max(-shift_x, 0)
    padded = F.pad(
        image,
        (pad_left, pad_right, pad_top, pad_bottom),
        mode="replicate",
    )
    return padded[..., pad_bottom : pad_bottom + height, pad_right : pad_right + width]


def _random_global_shift(image: torch.Tensor, max_shift: int = 8) -> torch.Tensor:
    shift_y = int(
        torch.randint(-max_shift, max_shift + 1, (1,), device=image.device).item()
    )
    shift_x = int(
        torch.randint(-max_shift, max_shift + 1, (1,), device=image.device).item()
    )
    return _shift_with_replicate_padding(image, shift_y, shift_x)


def build_direction_offsets(
    radius_min: int = 1,
    radius_max: int = 2,
    include_diagonal: bool = True,
) -> list[tuple[int, int]]:
    if radius_min < 1 or radius_max < radius_min:
        raise ValueError("Require 1 <= radius_min <= radius_max.")
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if include_diagonal:
        directions.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])
    return [
        (dy * radius, dx * radius)
        for radius in range(radius_min, radius_max + 1)
        for dy, dx in directions
    ]


def generate_paired_masks(
    shape: tuple[int, ...],
    device: torch.device,
    target_probability: float = 0.10,
    pair_probability: float = 0.50,
    pair_radius_min: int = 1,
    pair_radius_max: int = 2,
    include_diagonal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate target pixels and nearby hidden partners.

    Args:
        shape: Image tensor shape ``[B, C, H, W]``.
        target_probability: Bernoulli probability for target loss locations.
        pair_probability: Probability that a target receives one hidden partner.
    """
    batch, _, height, width = shape
    target = torch.rand((batch, 1, height, width), device=device) < target_probability
    hidden = torch.zeros_like(target)
    batch_idx, ys, xs = torch.where(target[:, 0])
    if ys.numel() == 0 or pair_probability <= 0:
        return target, hidden

    keep = torch.rand((ys.numel(),), device=device) < pair_probability
    if not keep.any():
        return target, hidden
    batch_idx, ys, xs = batch_idx[keep], ys[keep], xs[keep]

    offsets = build_direction_offsets(
        pair_radius_min, pair_radius_max, include_diagonal
    )
    offset_index = torch.randint(0, len(offsets), (ys.numel(),), device=device)
    dy = torch.tensor([p[0] for p in offsets], device=device, dtype=ys.dtype)
    dx = torch.tensor([p[1] for p in offsets], device=device, dtype=xs.dtype)
    ny = torch.clamp(ys + dy[offset_index], 0, height - 1)
    nx = torch.clamp(xs + dx[offset_index], 0, width - 1)
    hidden[batch_idx, 0, ny, nx] = True
    hidden &= ~target
    return target, hidden


def mixed_fill(
    image: torch.Tensor,
    target_mask: torch.Tensor,
    hidden_mask: torch.Tensor,
    local_radius_min: int = 1,
    local_radius_max: int = 2,
    include_diagonal: bool = True,
    hidden_global_probability: float = 0.15,
    max_global_shift: int = 8,
) -> torch.Tensor:
    """Corrupt target pixels globally and hidden partners locally.

    ``image`` is ``[B, C, H, W]`` and may be complex valued. Target pixels are
    filled from a globally shifted image. Hidden partners are filled from an
    unmasked local neighbor, with global filling as a fallback.
    """
    batch, _, height, width = image.shape
    output = image.clone()
    offsets = build_direction_offsets(
        local_radius_min, local_radius_max, include_diagonal
    )

    for batch_index in range(batch):
        global_fill = _random_global_shift(
            image[batch_index : batch_index + 1], max_global_shift
        )[0]
        target = target_mask[batch_index, 0]
        hidden = hidden_mask[batch_index, 0]
        occupied = target | hidden
        output[batch_index, :, target] = global_fill[:, target]

        hidden_y, hidden_x = torch.where(hidden)
        for y, x in zip(hidden_y.tolist(), hidden_x.tolist()):
            if (
                hidden_global_probability > 0
                and torch.rand((), device=image.device).item()
                < hidden_global_probability
            ):
                output[batch_index, :, y, x] = global_fill[:, y, x]
                continue

            candidates = [
                (y + dy, x + dx)
                for dy, dx in offsets
                if 0 <= y + dy < height
                and 0 <= x + dx < width
                and not occupied[y + dy, x + dx]
            ]
            if not candidates:
                output[batch_index, :, y, x] = global_fill[:, y, x]
                continue
            chosen = candidates[
                int(torch.randint(len(candidates), (), device=image.device).item())
            ]
            output[batch_index, :, y, x] = image[batch_index, :, chosen[0], chosen[1]]
    return output


def neighbor_fill(
    image: torch.Tensor,
    target_mask: torch.Tensor,
    hidden_mask: torch.Tensor,
    include_diagonal: bool = True,
) -> torch.Tensor:
    """Traditional Noise2Void-style random-neighbor corruption ablation."""
    _, _, height, width = image.shape
    output = image.clone()
    occupied = target_mask | hidden_mask
    offsets = build_direction_offsets(1, 1, include_diagonal)
    for batch_index in range(image.shape[0]):
        fill_y, fill_x = torch.where(occupied[batch_index, 0])
        for y, x in zip(fill_y.tolist(), fill_x.tolist()):
            valid = [
                (y + dy, x + dx)
                for dy, dx in offsets
                if 0 <= y + dy < height and 0 <= x + dx < width
            ]
            clean = [(ny, nx) for ny, nx in valid if not occupied[batch_index, 0, ny, nx]]
            candidates = clean or valid
            if not candidates:
                continue
            chosen = candidates[
                int(torch.randint(len(candidates), (), device=image.device).item())
            ]
            output[batch_index, :, y, x] = image[batch_index, :, chosen[0], chosen[1]]
    return output
