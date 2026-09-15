import argparse
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm

from complex_blindspot_dmri.models import ComplexCNN, ComplexNAFNet


def fix_complex_state_dict(state_dict):
    fixed = {}
    for k, v in state_dict.items():
        new_k = k.replace("module.module.", "").replace("module.", "")
        fixed[new_k] = v
    return fixed


def load_bvals(path: str) -> np.ndarray:
    return np.loadtxt(path, dtype=np.float32).reshape(-1)


def load_bvecs(path: str) -> np.ndarray:
    bvec = np.loadtxt(path, dtype=np.float32)
    bvec = np.asarray(bvec, dtype=np.float32)
    if bvec.ndim == 1:
        if bvec.size % 3 != 0:
            raise ValueError(f"Invalid bvec shape from {path}: {bvec.shape}")
        bvec = bvec.reshape(3, -1)
    if bvec.shape[0] == 3 and bvec.shape[1] != 3:
        bvec = bvec.T
    if bvec.ndim != 2 or bvec.shape[1] != 3:
        raise ValueError(f"bvec must be [N,3] or [3,N], got {bvec.shape}")
    norms = np.linalg.norm(bvec, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    return (bvec / norms).astype(np.float32)


def percentile99_scale_complex(x: np.ndarray) -> float:
    mag = np.abs(x)
    scale = np.percentile(mag, 99)
    if scale == 0 or not np.isfinite(scale):
        scale = 1.0
    return float(scale)

def wrap_phase_to_pi(phase_np: np.ndarray) -> np.ndarray:
    return (phase_np + np.pi) % (2 * np.pi) - np.pi


def print_phase_stats(phase_np: np.ndarray, prefix: str = "phase") -> None:
    finite = np.isfinite(phase_np)
    valid = phase_np[finite]
    if valid.size == 0:
        print(f"{prefix}: no finite voxels")
        return

    frac_pi = np.mean((valid >= -np.pi) & (valid <= np.pi))
    frac_2pi = np.mean((valid >= 0.0) & (valid <= 2.0 * np.pi))

    print(f"{prefix} min/max/mean/std: "
          f"{valid.min():.6f}, {valid.max():.6f}, {valid.mean():.6f}, {valid.std():.6f}")
    print(f"{prefix} p1/p50/p99: "
          f"{np.percentile(valid,1):.6f}, {np.percentile(valid,50):.6f}, {np.percentile(valid,99):.6f}")
    print(f"{prefix} fraction in [-pi, pi]: {frac_pi:.4f}")
    print(f"{prefix} fraction in [0, 2pi]:  {frac_2pi:.4f}")


def convert_phase_if_needed(phase_np: np.ndarray) -> np.ndarray:
    """
    Auto-detect and convert phase to wrapped radians in [-pi, pi].

    Cases:
    1) already in [-pi, pi] -> keep
    2) mostly in [0, 2pi]   -> wrap to [-pi, pi]
    3) looks like integer-coded phase (e.g. around [-4096, 4095]) -> scale to radians, then wrap
    4) fallback -> wrap only
    """
    finite = np.isfinite(phase_np)
    valid = phase_np[finite]
    if valid.size == 0:
        raise ValueError("phase contains no finite voxels")

    frac_pi = np.mean((valid >= -np.pi) & (valid <= np.pi))
    frac_2pi = np.mean((valid >= 0.0) & (valid <= 2.0 * np.pi))

    vmin = float(valid.min())
    vmax = float(valid.max())

    # Case 1: already wrapped radians
    if frac_pi > 0.99:
        print("[phase] looks already in [-pi, pi], keep unchanged")
        return phase_np.astype(np.float32)

    # Case 2: radians in [0, 2pi]
    if frac_2pi > 0.99:
        print("[phase] looks like radians in [0, 2pi], wrapping to [-pi, pi]")
        return wrap_phase_to_pi(phase_np.astype(np.float32)).astype(np.float32)

    # Case 3: integer-coded phase, typical vendor-like range near [-4096, 4095]
    # Map approximately [-4096, 4095] -> [-pi, pi]
    if vmin <= -4000 and vmax >= 4000:
        print("[phase] detected integer-coded phase near [-4096, 4095], converting to radians")
        phase_rad = (phase_np.astype(np.float32) / 4096.0) * np.pi
        phase_rad = wrap_phase_to_pi(phase_rad)
        return phase_rad.astype(np.float32)

    # Fallback
    print("[phase] abnormal range but not matched to a known encoding, applying wrap only as fallback")
    return wrap_phase_to_pi(phase_np.astype(np.float32)).astype(np.float32)


def build_slice_stack(volume_z_hw: torch.Tensor, z: int, num_slices: int) -> torch.Tensor:
    # volume_z_hw: [Z,H,W]
    zdim = volume_z_hw.shape[0]
    half = num_slices // 2
    idxs = [min(max(i, 0), zdim - 1) for i in range(z - half, z + half + 1)]
    return volume_z_hw[idxs]  # [S,H,W]


@torch.no_grad()
def run_model_in_chunks(model, target_batch, chunk_size=16, device="cuda"):
    n = target_batch.shape[0]
    out = []
    for i in range(0, n, chunk_size):
        t = target_batch[i:i + chunk_size].to(device, non_blocking=True)
        pred = model(t)
        out.append(pred.cpu())
    return torch.cat(out, dim=0)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading model...")
    model_class = ComplexNAFNet if args.model == "nafnet" else ComplexCNN
    model = model_class(
        dwi_channels=args.num_slices,
        #out_channels=args.num_slices,
        out_channels=1,
        num_slices=args.num_slices,
        support_k=0,
        max_targets=args.chunk_size,
    ).to(device)
    model.eval()

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = fix_complex_state_dict(checkpoint.get("model_state_dict", checkpoint))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)}")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)}")

    print("Loading NIfTI...")
    mag_img = nib.load(args.mag)
    phase_img = nib.load(args.phase)
    affine = mag_img.affine
    header = mag_img.header.copy()

    # mag_np = mag_img.get_fdata(dtype=np.float32)
    # phase_np = phase_img.get_fdata(dtype=np.float32)
    # if mag_np.ndim == 3:
    #     mag_np = mag_np[..., None]
    #     phase_np = phase_np[..., None]
    # if mag_np.shape != phase_np.shape:
    #     raise ValueError(f"mag shape {mag_np.shape} != phase shape {phase_np.shape}")

    # complex_np = (mag_np * np.exp(1j * phase_np)).astype(np.complex64)  # [X,Y,Z,N]
    mag_np = mag_img.get_fdata(dtype=np.float32)
    phase_np = phase_img.get_fdata(dtype=np.float32)
    if mag_np.ndim == 3:
      mag_np = mag_np[..., None]
      phase_np = phase_np[..., None]
    if mag_np.shape != phase_np.shape:
      raise ValueError(f"mag shape {mag_np.shape} != phase shape {phase_np.shape}")

    print_phase_stats(phase_np, prefix="phase before conversion")
    phase_np = convert_phase_if_needed(phase_np)
    print_phase_stats(phase_np, prefix="phase after conversion")

    complex_np = (mag_np * np.exp(1j * phase_np)).astype(np.complex64)  # [X,Y,Z,N]

    print("Loading bvals/bvecs...")
    bvals = load_bvals(args.bval)
    bvecs = load_bvecs(args.bvec)
    if complex_np.shape[-1] != len(bvals) or complex_np.shape[-1] != len(bvecs):
        raise ValueError(
            f"NIfTI volumes ({complex_np.shape[-1]}) / bvals ({len(bvals)}) / bvecs ({len(bvecs)}) do not match"
        )

    b0_indices = np.where(bvals <= args.b0_threshold)[0]
    dwi_indices = np.where(bvals > args.b0_threshold)[0]
    print(f"Found {len(b0_indices)} b0 volumes, {len(dwi_indices)} DWI volumes")
    if len(dwi_indices) == 0:
        raise ValueError("No DWI volumes found")

    dwi_data = complex_np[..., dwi_indices]  # [X,Y,Z,D]
    scale_dwi = percentile99_scale_complex(dwi_data)
    print(f"scale_dwi = {scale_dwi:.6f}")

    dwi_norm = (dwi_data / scale_dwi).astype(np.complex64)
    dwi_tensor = torch.from_numpy(dwi_norm).permute(3, 2, 0, 1).contiguous()  # [D,Z,H,W]

    d, zdim, h, w = dwi_tensor.shape
    print(f"DWI tensor shape: {tuple(dwi_tensor.shape)}")
    denoised = torch.zeros_like(dwi_tensor)

    print("Running single-target inference...")
    for z in tqdm(range(zdim), desc="Processing slices"):
        target_list = []
        for t_idx in range(d):
            target_stack = build_slice_stack(dwi_tensor[t_idx], z, args.num_slices)  # [S,H,W]
            target_list.append(target_stack)

        target_batch = torch.stack(target_list, dim=0).unsqueeze(1)  # [D,1,S,H,W]
        pred_batch = run_model_in_chunks(
            model=model,
            target_batch=target_batch,
            chunk_size=args.chunk_size,
            device=device,
        )  # [D,1,S,H,W]

        center = args.num_slices // 2
        #denoised[:, z] = pred_batch[:, 0, center]
        denoised[:, z] = pred_batch[:, 0, 0]

    denoised = denoised * scale_dwi
    denoised_dwi_np = denoised.permute(2, 3, 1, 0).numpy().astype(np.complex64)  # [X,Y,Z,D]

    out_complex_full = np.zeros_like(complex_np, dtype=np.complex64)
    out_complex_full[..., b0_indices] = complex_np[..., b0_indices]
    out_complex_full[..., dwi_indices] = denoised_dwi_np

    out_mag = np.abs(out_complex_full).astype(np.float32)
    out_phase = np.angle(out_complex_full).astype(np.float32)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    mag_out_path = f"{out_prefix}_mag.nii.gz"
    phase_out_path = f"{out_prefix}_phase.nii.gz"
    bval_out_path = f"{out_prefix}.bval"
    bvec_out_path = f"{out_prefix}.bvec"

    nib.save(nib.Nifti1Image(out_mag, affine, header), mag_out_path)
    nib.save(nib.Nifti1Image(out_phase, affine, header), phase_out_path)
    shutil.copyfile(args.bval, bval_out_path)
    shutil.copyfile(args.bvec, bvec_out_path)

    print("\nDone.")
    print(f"Saved magnitude: {mag_out_path}")
    print(f"Saved phase    : {phase_out_path}")
    print(f"Copied bval    : {bval_out_path}")
    print(f"Copied bvec    : {bvec_out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for single-target self-supervised complex dMRI denoiser")
    parser.add_argument("--mag", type=str, required=True, help="Input magnitude NIfTI")
    parser.add_argument("--phase", type=str, required=True, help="Input phase NIfTI")
    parser.add_argument("--bval", type=str, required=True, help="bval file")
    parser.add_argument("--bvec", type=str, required=True, help="bvec file")
    parser.add_argument("--checkpoint", type=str, required=True, help="model checkpoint")
    parser.add_argument("--out-prefix", type=str, required=True, help="output prefix")
    parser.add_argument("--model", choices=["nafnet", "complex-cnn"], default="nafnet")
    parser.add_argument("--b0-threshold", type=float, default=50.0, help="threshold for b0 volumes")
    parser.add_argument("--num-slices", type=int, default=1, help="must match training")
    parser.add_argument("--chunk-size", type=int, default=16, help="number of target directions processed per forward")
    args = parser.parse_args()
    main(args)
