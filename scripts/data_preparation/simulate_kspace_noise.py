#!/usr/bin/env python3
# 03_simulate_kspace_noise_and_phase.py
import argparse
import os
import math
import numpy as np
import nibabel as nib

try:
    from scipy.ndimage import gaussian_filter
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

def wrap_phase(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2*np.pi) - np.pi

def _fade(t):
    return t*t*t*(t*(t*6 - 15) + 10)

def _lerp(a, b, t):
    return a + t*(b-a)

def perlin2d(shape, res, rng):
    H, W = shape
    rh, rw = res
    if rh <= 0 or rw <= 0:
        raise ValueError("res must be positive")
    angles = rng.uniform(0, 2*np.pi, size=(rh+1, rw+1))
    grads = np.stack([np.cos(angles), np.sin(angles)], axis=-1)  
    ys = np.linspace(0, rh, H, endpoint=False)
    xs = np.linspace(0, rw, W, endpoint=False)
    y0 = ys.astype(int)
    x0 = xs.astype(int)
    y1 = y0 + 1
    x1 = x0 + 1
    fy = ys - y0
    fx = xs - x0
    fy_f = _fade(fy)[:, None]      
    fx_f = _fade(fx)[None, :]      
    y0b = y0[:, None]
    y1b = y1[:, None]
    x0b = x0[None, :]
    x1b = x1[None, :]

    def dot(gy, gx, dy, dx):
        g = grads[gy, gx]  
        return g[...,0]*dx + g[...,1]*dy

    n00 = dot(y0b, x0b, fy[:, None], fx[None, :])
    n10 = dot(y1b, x0b, (fy-1)[:, None], fx[None, :])
    n01 = dot(y0b, x1b, fy[:, None], (fx-1)[None, :])
    n11 = dot(y1b, x1b, (fy-1)[:, None], (fx-1)[None, :])

    nx0 = _lerp(n00, n01, fx_f)
    nx1 = _lerp(n10, n11, fx_f)
    nxy = _lerp(nx0, nx1, fy_f)
    return np.clip(nxy, -1.0, 1.0)

def make_smooth_phase_4d(shape_3d, n_vol, method, rng,
                        perlin_res=(6,6), blur_sigma=24.0,
                        phase_scale=1.0, slice_wise=True, z_smooth=1.5):
    X, Y, Z = shape_3d
    phase = np.zeros((X, Y, Z, n_vol), dtype=np.float32)

    for v in range(n_vol):
        vol = np.zeros((X, Y, Z), dtype=np.float32)
        if slice_wise:
            for z in range(Z):
                if method == "perlin":
                    n = perlin2d((X, Y), perlin_res, rng).astype(np.float32)
                elif method == "blur":
                    base = rng.normal(0, 1, size=(X, Y)).astype(np.float32)
                    if _HAVE_SCIPY:
                        n = gaussian_filter(base, sigma=blur_sigma).astype(np.float32)
                    else:
                        k = np.fft.fft2(base, norm="ortho")
                        fx = np.fft.fftfreq(Y)
                        fy = np.fft.fftfreq(X)
                        FY, FX = np.meshgrid(fx, fy)
                        r = np.sqrt(FX*FX + FY*FY)
                        cutoff = 0.08
                        H = (r <= cutoff).astype(np.float32)
                        n = np.fft.ifft2(k * H, norm="ortho").real.astype(np.float32)
                else:
                    raise ValueError("phase method must be perlin or blur")

                n = n / (np.max(np.abs(n)) + 1e-8)
                vol[..., z] = n

            if z_smooth > 0 and _HAVE_SCIPY:
                vol = gaussian_filter(vol, sigma=(0.0, 0.0, z_smooth)).astype(np.float32)
        else:
            base = rng.normal(0, 1, size=(X, Y, Z)).astype(np.float32)
            if _HAVE_SCIPY:
                vol = gaussian_filter(base, sigma=blur_sigma).astype(np.float32)
            else:
                vol = base
            vol = vol / (np.max(np.abs(vol)) + 1e-8)

        phi = phase_scale * np.pi * vol
        phase[..., v] = wrap_phase(phi).astype(np.float32)

    return phase

def make_random_lowpass(shape_2d, rng, cutoff_range=(0.18, 0.45), roll_range=(0.03, 0.12), anisotropy=True):
    X, Y = shape_2d
    fx = np.fft.fftfreq(Y)
    fy = np.fft.fftfreq(X)
    FY, FX = np.meshgrid(fx, fy)
    cutoff = rng.uniform(*cutoff_range)
    roll = rng.uniform(*roll_range)
    if anisotropy:
        ax = rng.uniform(0.8, 1.4)
        ay = rng.uniform(0.8, 1.4)
    else:
        ax = ay = 1.0
    r = np.sqrt((FX/ax)**2 + (FY/ay)**2)
    H = np.zeros_like(r, dtype=np.float32)
    inside = r <= cutoff
    trans = (r > cutoff) & (r < (cutoff + roll))
    H[inside] = 1.0
    H[trans] = 0.5 * (1 + np.cos(np.pi * (r[trans] - cutoff) / (roll + 1e-12)))
    return H

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noiseless_mag", required=True, help="Path to the CLEANED noiseless_mag.nii.gz")
    ap.add_argument("--bval", required=True)
    ap.add_argument("--bvec", required=True)
    ap.add_argument("--out_root", required=True)
    # 【核心修改】参数改为 noise_level_pct (百分比)
    ap.add_argument("--noise_level_pct", required=True, help="Comma separated noise percentages (e.g., 2,4,6)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--phase_method", choices=["perlin", "blur"], default="perlin")
    ap.add_argument("--perlin_res", type=str, default="6,6")
    ap.add_argument("--blur_sigma", type=float, default=24.0)
    ap.add_argument("--phase_scale", type=float, default=1.0)
    ap.add_argument("--slice_wise", type=int, default=1)
    ap.add_argument("--z_smooth", type=float, default=1.5)
    ap.add_argument("--apply_lowpass", type=int, default=1)
    ap.add_argument("--cutoff_range", type=str, default="0.25,0.52")
    ap.add_argument("--roll_range", type=str, default="0.03,0.12")
    ap.add_argument("--anisotropy", type=int, default=1)
    ap.add_argument("--mask", default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"[Info] Loading cleaned magnitude: {args.noiseless_mag}")
    img = nib.load(args.noiseless_mag)
    mag = img.get_fdata(dtype=np.float32)
    X, Y, Z, N = mag.shape

    mask = None
    if args.mask is not None:
        mask_data = nib.load(args.mask).get_fdata(dtype=np.float32)
        mask_data = np.squeeze(mask_data)
        if mask_data.ndim > 3:
            mask_data = mask_data[..., 0]
        mask = mask_data > 0.5

    # 提取 b0 图像作为物理底噪的基准
    b0_mag = mag[..., 0]
    if mask is not None:
        valid_b0 = b0_mag[mask]
    else:
        valid_b0 = b0_mag[b0_mag > 0]
    
    # 获取 b0 的 95% 分位数作为基准信号强度
    b0_ref = float(np.percentile(valid_b0, 95)) if valid_b0.size > 0 else float(np.max(b0_mag))
    print(f"[Info] Reference b0 signal for noise calculation: {b0_ref:.2f}")

    noise_pcts =[float(x) for x in args.noise_level_pct.split(",") if x.strip() != ""]
    pr = tuple(int(x) for x in args.perlin_res.split(","))
    cutoff_range = tuple(float(x) for x in args.cutoff_range.split(","))
    roll_range = tuple(float(x) for x in args.roll_range.split(","))

    # 1. 生成空间相关的合成相位
    phase = make_smooth_phase_4d(
        (X, Y, Z), N, method=args.phase_method, rng=rng, perlin_res=pr,
        blur_sigma=args.blur_sigma, phase_scale=args.phase_scale,
        slice_wise=bool(args.slice_wise), z_smooth=args.z_smooth
    )

    # 2. 组装纯净 Ground Truth 复数图像
    clean_complex = mag * (np.cos(phase) + 1j*np.sin(phase))

    bval_txt = open(args.bval, "r").read()
    bvec_txt = open(args.bvec, "r").read()

    # 3. 循环注入对应百分比的 K 空间相关性噪声
    for pct in noise_pcts:
        outdir = os.path.join(args.out_root, f"Noise_{pct:g}pct")
        os.makedirs(outdir, exist_ok=True)

        # 【核心逻辑】根据 b0 参考信号的百分比计算全局恒定底噪
        sigma_img = b0_ref * (pct / 100.0) 
        sigma_ri = sigma_img / math.sqrt(2.0)

        noisy_complex = np.zeros_like(clean_complex, dtype=np.complex64)

        for v in range(N):
            for z in range(Z):
                im2 = clean_complex[:, :, z, v]
                k = np.fft.fft2(im2, norm="ortho")

                noise_k = (rng.normal(0, sigma_ri, size=(X, Y)).astype(np.float32)
                           + 1j * rng.normal(0, sigma_ri, size=(X, Y)).astype(np.float32))

                if args.apply_lowpass:
                    H = make_random_lowpass(
                        (X, Y), rng,
                        cutoff_range=cutoff_range,
                        roll_range=roll_range,
                        anisotropy=bool(args.anisotropy)
                    )
                    noise_k = noise_k * H
                    # 确保滤波后整体能量不变，维持设定的噪声百分比
                    power_after = np.mean(np.abs(H)**2) + 1e-12
                    noise_k = noise_k / math.sqrt(power_after)

                k_noisy = k + noise_k
                im_noisy = np.fft.ifft2(k_noisy, norm="ortho")
                noisy_complex[:, :, z, v] = im_noisy.astype(np.complex64)

        mag_out = np.abs(noisy_complex).astype(np.float32)
        phase_out = wrap_phase(np.angle(noisy_complex).astype(np.float32))
        clean_mag_out = np.abs(clean_complex).astype(np.float32)
        clean_phase_out = wrap_phase(np.angle(clean_complex).astype(np.float32))

        # 将 Mask 外部的背景严格置零
        if mask is not None:
            mag_out[~mask] = 0.0
            phase_out[~mask] = 0.0
            clean_mag_out[~mask] = 0.0
            clean_phase_out[~mask] = 0.0

        nib.save(nib.Nifti1Image(mag_out, img.affine, img.header), os.path.join(outdir, "mag.nii.gz"))
        nib.save(nib.Nifti1Image(phase_out, img.affine, img.header), os.path.join(outdir, "phase.nii.gz"))
        nib.save(nib.Nifti1Image(clean_mag_out, img.affine, img.header), os.path.join(outdir, "clean_mag.nii.gz"))
        nib.save(nib.Nifti1Image(clean_phase_out, img.affine, img.header), os.path.join(outdir, "clean_phase.nii.gz"))

        with open(os.path.join(outdir, "bvals"), "w") as f:
            f.write(bval_txt)
        with open(os.path.join(outdir, "bvecs"), "w") as f:
            f.write(bvec_txt)

        meta = {
            "noise_level_pct": pct,
            "b0_reference_signal": float(b0_ref),
            "sigma_img": float(sigma_img),
            "phase_method": args.phase_method,
            "apply_lowpass": bool(args.apply_lowpass),
        }
        with open(os.path.join(outdir, "sim_meta.json"), "w") as f:
            import json
            json.dump(meta, f, indent=2)

        print(f"[OK] {pct}% Noise generated in {outdir} (sigma={sigma_img:.4f})")

if __name__ == "__main__":
    main()