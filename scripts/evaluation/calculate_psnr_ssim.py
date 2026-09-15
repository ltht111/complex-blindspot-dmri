import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from skimage.metrics import structural_similarity as ssim


def load_bvals(bval_path):
    """读取 bval 文件并返回 numpy 数组"""
    with open(bval_path, "r") as f:
        bvals = np.array([float(x) for x in f.read().split()], dtype=np.float32)
    return bvals


def compute_psnr(gt_mask_voxels, pred_mask_voxels, data_range):
    """
    标准 PSNR:
        PSNR = 10 * log10((data_range^2) / MSE)
    其中 data_range 必须是固定的、全局一致的动态范围。
    """
    diff = gt_mask_voxels.astype(np.float64) - pred_mask_voxels.astype(np.float64)
    mse = np.mean(diff ** 2)

    if mse == 0:
        return float("inf"), 0.0

    psnr = 10.0 * np.log10((float(data_range) ** 2) / mse)
    return float(psnr), float(mse)


def determine_data_range(gt_img, mask, dwi_indices, mode="global", manual_value=None):
    """
    统一确定 PSNR / SSIM 的 data_range

    mode:
      - global: 使用所有 DWI volume 在掩膜内的 GT 全局极差
      - normalized: 固定为 1.0（适用于已归一化到 [0,1]）
      - manual: 使用用户手动指定值
    """
    if mode == "normalized":
        return 1.0

    if mode == "manual":
        if manual_value is None or manual_value <= 0:
            raise ValueError("manual 模式下必须提供正数 manual_value")
        return float(manual_value)

    if mode == "global":
        # 所有 DWI volume 的 GT 掩膜内体素汇总后统一算动态范围
        gt_dwi = gt_img[..., dwi_indices]  # [X,Y,Z,D]
        mask4d = np.repeat(mask[..., None], len(dwi_indices), axis=3)
        voxels = gt_dwi[mask4d]

        if voxels.size == 0:
            raise ValueError("掩膜内没有体素，无法计算 data_range")

        vmin = float(np.min(voxels))
        vmax = float(np.max(voxels))
        data_range = vmax - vmin

        if data_range <= 0 or not np.isfinite(data_range):
            raise ValueError("global data_range 非法，请检查输入图像或掩膜")

        return data_range

    raise ValueError(f"未知 data_range mode: {mode}")


def calculate_dti_metrics(
    gt_path,
    target_path,
    mask_path,
    bval_path,
    b0_threshold=50,
    data_range_mode="global",
    manual_data_range=None,
):
    """
    计算基于脑掩膜和 bval 过滤的平均 PSNR 与 SSIM

    更标准的做法：
    - PSNR 使用统一固定的 data_range，而不是每个 volume 自己的 max
    - SSIM 也使用同一个 data_range
    """

    # 1. 加载数据
    gt_img = nib.load(gt_path).get_fdata(dtype=np.float32)
    target_img = nib.load(target_path).get_fdata(dtype=np.float32)
    mask = nib.load(mask_path).get_fdata().astype(bool)
    bvals = load_bvals(bval_path)

    # 2. 基本校验
    if gt_img.shape != target_img.shape:
        raise ValueError("Ground truth 和 Target 图像维度不一致！")
    if gt_img.ndim != 4:
        raise ValueError(f"输入图像必须是 4D NIfTI，当前 gt_img.ndim = {gt_img.ndim}")
    if gt_img.shape[:3] != mask.shape:
        raise ValueError("掩膜的 3D 维度与图像空间维度不一致！")
    if gt_img.shape[3] != len(bvals):
        raise ValueError("bval 的数量与图像第四维(方向数)不匹配！")

    # 3. 筛选 DWI 方向
    dwi_indices = np.where(bvals > b0_threshold)[0]
    if len(dwi_indices) == 0:
        raise ValueError("没有检测到 DWI 方向，请检查 b0_threshold 或 bval 文件。")

    print(
        f"检测到 {len(bvals)} 个总方向。"
        f"排除了 {len(bvals) - len(dwi_indices)} 个 b0 图像，"
        f"将对 {len(dwi_indices)} 个 DWI 方向计算指标。"
    )

    # 4. 确定统一 data_range
    data_range = determine_data_range(
        gt_img=gt_img,
        mask=mask,
        dwi_indices=dwi_indices,
        mode=data_range_mode,
        manual_value=manual_data_range,
    )
    print(f"统一 data_range = {data_range:.6f} (mode={data_range_mode})")

    psnr_list = []
    ssim_list = []
    mse_list = []

    # 5. 逐个 DWI 方向计算
    for idx in dwi_indices:
        gt_vol = gt_img[..., idx]
        target_vol = target_img[..., idx]

        # 只在掩膜内算 MSE / PSNR
        gt_mask_voxels = gt_vol[mask]
        target_mask_voxels = target_vol[mask]

        psnr, mse = compute_psnr(gt_mask_voxels, target_mask_voxels, data_range=data_range)
        psnr_list.append(psnr)
        mse_list.append(mse)

        # SSIM 通常按整幅 3D volume 算，但脑外区域置零，减少背景影响
        gt_vol_masked = np.where(mask, gt_vol, 0.0)
        target_vol_masked = np.where(mask, target_vol, 0.0)

        vol_ssim = ssim(
            gt_vol_masked,
            target_vol_masked,
            data_range=data_range,
        )
        ssim_list.append(float(vol_ssim))

    # 6. 汇总
    avg_psnr = float(np.mean(psnr_list))
    avg_ssim = float(np.mean(ssim_list))
    avg_mse = float(np.mean(mse_list))

    print("-" * 40)
    print(f"最终平均 MSE  (仅限 DWI, mask 内): {avg_mse:.6f}")
    print(f"最终平均 PSNR (仅限 DWI, mask 内): {avg_psnr:.4f} dB")
    print(f"最终平均 SSIM (仅限 DWI)        : {avg_ssim:.4f}")

    return {
        "avg_mse": avg_mse,
        "avg_psnr": avg_psnr,
        "avg_ssim": avg_ssim,
        "per_volume_psnr": psnr_list,
        "per_volume_ssim": ssim_list,
        "per_volume_mse": mse_list,
        "dwi_indices": dwi_indices.tolist(),
        "data_range": float(data_range),
        "data_range_mode": data_range_mode,
    }


def main():
    parser = argparse.ArgumentParser(description="Calculate masked DWI PSNR and SSIM with standardized data_range.")
    parser.add_argument("--gt", type=str, required=True, help="Ground truth magnitude NIfTI")
    parser.add_argument("--target", type=str, required=True, help="Predicted / denoised magnitude NIfTI")
    parser.add_argument("--mask", type=str, required=True, help="Brain mask NIfTI")
    parser.add_argument("--bval", type=str, required=True, help="bval file")
    parser.add_argument("--b0-threshold", type=float, default=50.0, help="b0 threshold")
    parser.add_argument(
        "--data-range-mode",
        type=str,
        default="global",
        choices=["global", "normalized", "manual"],
        help="How to determine unified data_range for PSNR/SSIM",
    )
    parser.add_argument(
        "--manual-data-range",
        type=float,
        default=None,
        help="Used only when --data-range-mode manual",
    )
    args = parser.parse_args()

    results = calculate_dti_metrics(
        gt_path=args.gt,
        target_path=args.target,
        mask_path=args.mask,
        bval_path=args.bval,
        b0_threshold=args.b0_threshold,
        data_range_mode=args.data_range_mode,
        manual_data_range=args.manual_data_range,
    )

    print("-" * 40)
    print("评估完成。")
    print(f"data_range_mode: {results['data_range_mode']}")
    print(f"data_range     : {results['data_range']:.6f}")


if __name__ == "__main__":
    main()