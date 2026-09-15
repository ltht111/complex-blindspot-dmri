"""Dataset utilities for complex-valued diffusion MRI slices."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class DWIShellDataset(Dataset):
    """Load a JSON manifest of complex-valued DWI slice files.

    Each NPZ must contain arrays named ``real`` and ``imag``. Relative paths in
    the manifest are resolved from the directory containing the manifest.
    """

    def __init__(self, manifest: str | Path, num_slices: int = 1):
        self.manifest = Path(manifest).expanduser().resolve()
        with self.manifest.open("r", encoding="utf-8") as stream:
            self.samples = json.load(stream)
        if not self.samples:
            raise ValueError(f"Dataset is empty: {self.manifest}")
        if num_slices < 1 or num_slices % 2 == 0:
            raise ValueError("num_slices must be a positive odd integer.")
        self.num_slices = int(num_slices)
        self.offset = self.num_slices // 2

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve(self, value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else self.manifest.parent / path

    @staticmethod
    def _load_complex_npz(path: Path) -> np.ndarray:
        with np.load(path) as data:
            if "real" not in data or "imag" not in data:
                raise KeyError(f"NPZ must contain real and imag arrays: {path}")
            return data["real"] + 1j * data["imag"]

    def _adjacent_paths(self, center_value: str) -> list[Path]:
        center = self._resolve(center_value)
        if self.num_slices == 1:
            return [center]
        match = re.search(r"(\d+)(?=\D*$)", center.name)
        if not match:
            return [center] * self.num_slices
        index = int(match.group(1))
        width = len(match.group(1))
        paths = []
        for adjacent in range(index - self.offset, index + self.offset + 1):
            name = (
                center.name[: match.start()]
                + str(adjacent).zfill(width)
                + center.name[match.end() :]
            )
            candidate = center.parent / name
            paths.append(candidate if candidate.exists() else center)
        return paths

    def _load_stack(self, paths: list[str]) -> list[np.ndarray]:
        return [
            np.stack([self._load_complex_npz(p) for p in self._adjacent_paths(path)])
            for path in paths
        ]

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.samples[index]
        dwi = np.stack(self._load_stack(item["dwi_paths"]), axis=0)
        direction_count = dwi.shape[0]
        scale = float(np.percentile(np.abs(dwi), 99))
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        dwi = dwi / scale

        bvals = np.asarray(
            item.get("dwi_bvals", [1000.0] * direction_count), dtype=np.float32
        ).reshape(-1)
        bvecs = np.asarray(
            item.get("dwi_bvecs", [[0.0, 0.0, 0.0]] * direction_count),
            dtype=np.float32,
        )
        if bvals.shape[0] != direction_count or bvecs.shape != (direction_count, 3):
            raise ValueError(f"Gradient metadata mismatch in sample {index}.")
        norms = np.linalg.norm(bvecs, axis=1, keepdims=True)
        bvecs = bvecs / np.where(norms < 1e-8, 1.0, norms)

        return {
            "dwi": torch.from_numpy(dwi.astype(np.complex64)),
            "bval": torch.from_numpy(bvals),
            "bvec": torch.from_numpy(bvecs),
            "scale": scale,
            "meta": {
                "group": item.get("group", "unknown"),
                "slice_name": item.get("slice_name", f"sample_{index}"),
            },
        }

