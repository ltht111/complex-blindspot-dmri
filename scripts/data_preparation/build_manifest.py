import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def parse_b_value(folder_name: str) -> Optional[int]:
    match = re.search(r"_b(\d+)", folder_name)
    return int(match.group(1)) if match else None


def parse_dir_index(folder_name: str) -> int:
    match = re.search(r"dir(\d+)", folder_name)
    return int(match.group(1)) if match else 10**9


def read_bval_file(path: Path) -> Optional[np.ndarray]:
    if path is None or not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore").strip().split()
    if not text:
        return None
    return np.asarray([float(x) for x in text], dtype=np.float32)


def read_bvec_file(path: Path) -> Optional[np.ndarray]:
    if path is None or not path.exists():
        return None
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").strip().splitlines():
        vals = [float(x) for x in line.split()]
        if vals:
            rows.append(vals)
    if not rows:
        return None
    arr = np.asarray(rows, dtype=np.float32)
    if arr.shape[0] == 3:
        arr = arr.T
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Unsupported bvec shape in {path}: {arr.shape}")
    return arr


def find_first_file(folder: Path, patterns: List[str]) -> Optional[Path]:
    for pattern in patterns:
        matches = sorted(folder.glob(pattern))
        if matches:
            return matches[0]
    return None


def load_subject_gradients(subject_dir: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    bval_path = find_first_file(subject_dir, ["*.bval", "*.bvals"]) 
    bvec_path = find_first_file(subject_dir, ["*.bvec", "*.bvecs"]) 
    bvals = read_bval_file(bval_path) if bval_path else None
    bvecs = read_bvec_file(bvec_path) if bvec_path else None
    return bvals, bvecs


def get_dir_metadata(folder: Path, subj_bvals: Optional[np.ndarray], subj_bvecs: Optional[np.ndarray]) -> Dict:
    dir_idx = parse_dir_index(folder.name)
    folder_b = parse_b_value(folder.name)

    bval = float(folder_b) if folder_b is not None else None
    bvec = None

    if subj_bvals is not None and dir_idx < len(subj_bvals):
        bval = float(subj_bvals[dir_idx])
    if subj_bvecs is not None and dir_idx < len(subj_bvecs):
        bvec = subj_bvecs[dir_idx].astype(np.float32).tolist()

    if bvec is None:
        bvec = [0.0, 0.0, 0.0]

    return {
        "dir_idx": int(dir_idx),
        "bval": float(bval) if bval is not None else 0.0,
        "bvec": bvec,
    }


def prepare_dataset_index(root_dir, output_json="dataset.json", b0_threshold=50, npz_dirname="npz"):
    root = Path(root_dir)
    dataset_index = []

    subjects = sorted([p for p in root.iterdir() if p.is_dir()])
    print(f"Found {len(subjects)} subjects in {root}")

    for subj in subjects:
        npz_root = subj / npz_dirname
        if not npz_root.exists():
            print(f"  [Skip] {subj.name}: no '{npz_dirname}' folder")
            continue

        subj_bvals, subj_bvecs = load_subject_gradients(subj)

        subdirs = sorted([d for d in npz_root.iterdir() if d.is_dir()], key=lambda p: parse_dir_index(p.name))

        b0_dirs: List[Path] = []
        dwi_dirs: List[Path] = []
        dwi_meta: Dict[str, Dict] = {}

        for d in subdirs:
            meta = get_dir_metadata(d, subj_bvals, subj_bvecs)
            b_val = meta["bval"]
            if b_val <= b0_threshold:
                b0_dirs.append(d)
            else:
                dwi_dirs.append(d)
                dwi_meta[d.name] = meta

        if not b0_dirs:
            print(f"  [Warning] {subj.name}: no b0 (<= {b0_threshold}) found, skip")
            continue
        if not dwi_dirs:
            print(f"  [Warning] {subj.name}: no DWI found, skip")
            continue

        b0_dirs.sort(key=lambda p: parse_dir_index(p.name))
        dwi_dirs.sort(key=lambda p: parse_dir_index(p.name))

        ref_b0_dir = b0_dirs[0]
        slice_files = sorted(list(ref_b0_dir.glob("*.npz")), key=lambda p: p.name)
        if len(slice_files) == 0:
            print(f"  [Warning] {subj.name}: empty b0 folder {ref_b0_dir.name}, skip")
            continue

        added = 0
        for s_file in slice_files:
            slice_name = s_file.name
            b0_paths = [str(b_dir / slice_name) for b_dir in b0_dirs if (b_dir / slice_name).exists()]

            sample_dwi_paths = []
            sample_bvals = []
            sample_bvecs = []
            sample_dir_indices = []

            for d_dir in dwi_dirs:
                sp = d_dir / slice_name
                if not sp.exists():
                    continue
                meta = dwi_meta[d_dir.name]
                sample_dwi_paths.append(str(sp))
                sample_bvals.append(float(meta["bval"]))
                sample_bvecs.append(meta["bvec"])
                sample_dir_indices.append(int(meta["dir_idx"]))

            if b0_paths and sample_dwi_paths:
                dataset_index.append({
                    "group": subj.name,
                    "slice_name": slice_name,
                    "b0_paths": b0_paths,
                    "dwi_paths": sample_dwi_paths,
                    "dwi_bvals": sample_bvals,
                    "dwi_bvecs": sample_bvecs,
                    "dwi_dir_indices": sample_dir_indices,
                })
                added += 1

        print(f"  [OK] {subj.name}: added {added} slices")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(dataset_index, f, indent=4, ensure_ascii=False)

    print(f"Done! Index saved to {output_json} with {len(dataset_index)} samples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="总训练目录（下面是若干被试文件夹）")
    parser.add_argument("--out", default="train.json", help="输出 json 文件名")
    parser.add_argument("--b0_threshold", type=int, default=50, help="b0 阈值(<=该值认为是 b0)")
    parser.add_argument("--npz_dirname", default="npz", help="npz 子目录名（默认 npz）")
    args = parser.parse_args()

    prepare_dataset_index(args.root, args.out, b0_threshold=args.b0_threshold, npz_dirname=args.npz_dirname)
