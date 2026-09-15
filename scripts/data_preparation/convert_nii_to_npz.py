import argparse
import numpy as np
from pathlib import Path
import nibabel as nib


def load_bvals(bval_path: Path) -> np.ndarray:
    txt = bval_path.read_text().strip().split()
    return np.asarray([float(x) for x in txt], dtype=np.float32)


def normalize_phase_to_radians(phase: np.ndarray) -> np.ndarray:
    """
    尝试把相位统一到弧度范围（尽量稳健，但不可能100%覆盖所有存储方式）。
    - 常见：[-pi, pi] 或 [0, 2pi]
    - 也可能是“度”
    """
    pmax = float(np.nanmax(phase))
    pmin = float(np.nanmin(phase))

    # 1) 看起来像“度”（比如最大到 180/360）
    if pmax > 2 * np.pi * 1.1 and pmax <= 360 * 1.1:
        phase = np.deg2rad(phase)

    # 2) 看起来像 [0, 2pi]：把它移到 [-pi, pi]
    pmax = float(np.nanmax(phase))
    pmin = float(np.nanmin(phase))
    if pmin >= -0.1 and pmax <= 2 * np.pi * 1.1:
        # 将 [0, 2pi] 平移到 [-pi, pi]（中心更一致）
        phase = phase - np.pi

    return phase.astype(np.float32)


def convert_one_subject(
    subj_dir: Path,
    mag_name: str,
    phase_name: str,
    bval_name: str,
    b0_threshold: float,
    out_dirname: str,
    slice_name_fmt: str = "slice_{:03d}.npz",
):
    mag_path = subj_dir / mag_name
    phase_path = subj_dir / phase_name
    bval_path = subj_dir / bval_name

    if not (mag_path.exists() and phase_path.exists() and bval_path.exists()):
        print(f"[Skip] Missing files in {subj_dir.name}")
        return 0

    mag_img = nib.load(str(mag_path))
    phase_img = nib.load(str(phase_path))

    mag = mag_img.get_fdata(dtype=np.float32)
    phase = phase_img.get_fdata(dtype=np.float32)

    if mag.shape != phase.shape:
        raise ValueError(f"{subj_dir.name}: mag shape {mag.shape} != phase shape {phase.shape}")

    if mag.ndim != 4:
        raise ValueError(f"{subj_dir.name}: Expect 4D NIfTI (X,Y,Z,V), got shape={mag.shape}")

    bvals = load_bvals(bval_path)
    V = mag.shape[3]
    if len(bvals) != V:
        raise ValueError(f"{subj_dir.name}: bvals length {len(bvals)} != volumes {V}")

    phase = normalize_phase_to_radians(phase)

    out_root = subj_dir / out_dirname
    out_root.mkdir(parents=True, exist_ok=True)

    saved = 0
    X, Y, Z, V = mag.shape

    for v in range(V):
        b = float(bvals[v])
        b_int = int(np.round(b))

        # 目录名保持与你旧 prepare_data.py 的解析兼容：dirXXX_bYYY
        vol_dir = out_root / f"dir{v:03d}_b{b_int}"
        vol_dir.mkdir(parents=True, exist_ok=True)

        # (X,Y,Z) 单卷
        mag_v = mag[..., v]
        phase_v = phase[..., v]

        # 转复数：real = mag*cos(phase), imag = mag*sin(phase)
        real_v = mag_v * np.cos(phase_v)
        imag_v = mag_v * np.sin(phase_v)

        for z in range(Z):
            # 注意：dataset_multi.py 通过文件名中数字找相邻切片，所以 slice_{:03d} 很关键
            out_path = vol_dir / slice_name_fmt.format(z)
            np.savez_compressed(
                out_path,
                real=real_v[:, :, z].astype(np.float32),
                imag=imag_v[:, :, z].astype(np.float32),
            )
            saved += 1

    print(f"[OK] {subj_dir.name}: saved {saved} npz files into {out_root}")
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="总训练数据文件夹（下面是若干被试小文件夹）")
    ap.add_argument("--mag_name", default="mag.nii.gz", help="幅值 NIfTI 文件名")
    ap.add_argument("--phase_name", default="phase.nii.gz", help="相位 NIfTI 文件名")
    ap.add_argument("--bval_name", default="bvals", help="bval 文件名（每个被试一个）")
    ap.add_argument("--b0_threshold", type=float, default=50.0, help="<=该值认为是 b0")
    ap.add_argument("--out_dirname", default="npz", help="npz 输出子目录名（放在每个被试文件夹下）")
    args = ap.parse_args()
    

    root = Path(args.root)
    assert root.exists(), f"root not found: {root}"

    total = 0
    for subj in sorted([p for p in root.iterdir() if p.is_dir()]):
        total += convert_one_subject(
            subj_dir=subj,
            mag_name=args.mag_name,
            phase_name=args.phase_name,
            bval_name=args.bval_name,
            b0_threshold=args.b0_threshold,
            out_dirname=args.out_dirname,
        )

    print(f"Done. Total npz files saved: {total}")


if __name__ == "__main__":
    main()