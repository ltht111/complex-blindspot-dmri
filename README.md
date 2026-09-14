# ComplexBlindSpot-dMRI

Official research implementation of a self-supervised, complex-valued diffusion MRI (dMRI) denoiser designed for spatially correlated reconstruction noise.

The method combines:

- paired hidden pixels that enlarge the local blind region;
- mixed filling, with globally shifted values at target pixels and local unmasked values at paired hidden pixels;
- a complex-valued encoder-decoder built from Complex NAFBlocks; and
- masked reconstruction loss on the real and imaginary components.

The code is organized for the 2D configuration described in the paper. Each diffusion direction is denoised independently. No private imaging data, trained weights, or machine-specific dataset manifests are included.

## Repository layout

```text
src/complex_blindspot_dmri/
  data.py                 Dataset and manifest loading
  masking.py              Paired masking and corruption strategies
  models/
    complex_nafnet.py     Proposed Complex NAFNet backbone
    complex_cnn.py        Complex CNN ablation backbone
scripts/
  train.py                Proposed method and learning-based ablations
  infer.py                NIfTI magnitude/phase inference
  data_preparation/       NIfTI conversion, manifest building, simulation
  evaluation/             PSNR and SSIM evaluation
examples/
  manifest.example.json   Dataset-manifest schema
tests/                    Lightweight unit tests
```

## Installation

Python 3.9 or newer is recommended. Install a PyTorch build appropriate for your CUDA version first, then install this repository:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e .
```

The network uses [`complexPyTorch`](https://github.com/wavefrontshaping/complexPyTorch) for complex convolution layers.

## Data format

Training samples are stored as compressed NPZ files containing two 2D arrays:

```python
np.savez_compressed(path, real=image.real.astype("float32"), imag=image.imag.astype("float32"))
```

The JSON manifest contains one record per slice. See [`examples/manifest.example.json`](examples/manifest.example.json). Relative paths are resolved from the manifest's directory.

For magnitude and phase NIfTI inputs, create NPZ slices and a manifest with:

```bash
python scripts/data_preparation/convert_nii_to_npz.py \
  --root /path/to/subjects \
  --mag_name mag.nii.gz \
  --phase_name phase.nii.gz \
  --bval_name bvals

python scripts/data_preparation/build_manifest.py \
  --root /path/to/subjects \
  --out train.local.json
```

Each subject directory should contain the magnitude image, phase image, b-values, b-vectors, and the generated `npz/` directory. Split subjects before constructing the training and validation manifests to avoid data leakage.

## Training

Train the proposed 2D model:

```bash
python scripts/train.py \
  --train-manifest train.local.json \
  --validation-manifest validation.local.json \
  --output-dir checkpoints/proposed \
  --num-slices 1 \
  --model nafnet \
  --corruption mixed
```

Important defaults matching the released configuration are a target-mask probability of 0.10, paired-neighbor probability of 0.50, pair/fill radii of 1-2 pixels, and a masked complex L1 objective. All settings are saved in each checkpoint under `config`.

The learning-based ablations use the same entry point:

```bash
# Traditional Noise2Void-style local-neighbor filling
python scripts/train.py --train-manifest train.local.json --corruption neighbor

# Replace Complex NAFBlocks with the complex CNN backbone
python scripts/train.py --train-manifest train.local.json --model complex-cnn
```

## Inference

Inference expects matched 4D magnitude and phase NIfTI images plus FSL-style b-values and b-vectors:

```bash
python scripts/infer.py \
  --mag input_mag.nii.gz \
  --phase input_phase.nii.gz \
  --bval input.bval \
  --bvec input.bvec \
  --checkpoint checkpoints/proposed/model_best.pth \
  --out-prefix outputs/subject_001 \
  --num-slices 1
```

The command writes denoised magnitude and phase images and copies the gradient files alongside them. Phase is represented in radians and normalized to `[-pi, pi]` when necessary.

## Synthetic correlated-noise data

The paper's synthetic-data helper creates smooth phase, transforms each slice to k-space, injects complex Gaussian noise with an optional random low-pass profile, and reconstructs noisy complex data:

```bash
python scripts/data_preparation/simulate_kspace_noise.py \
  --noiseless_mag clean_mag.nii.gz \
  --bval bvals \
  --bvec bvecs \
  --out_root simulated \
  --noise_level_pct 2,3,4,5 \
  --mask brain_mask.nii.gz
```

## Evaluation

Calculate mask-restricted PSNR and whole-volume masked SSIM over DWI volumes:

```bash
python scripts/evaluation/calculate_psnr_ssim.py \
  --gt reference_mag.nii.gz \
  --target outputs/subject_001_mag.nii.gz \
  --mask brain_mask.nii.gz \
  --bval input.bval
```

## Reproducibility notes

- The repository does not redistribute HCP, locally acquired, or OpenNeuro images.
- Subject-level train/validation/test splits must be constructed independently by users with the relevant data access and ethics approvals.
- Trained weights can be attached to a GitHub Release after publication; checkpoint files are intentionally ignored by Git.
- The current repository covers the proposed method, two learning-based ablations, data conversion, correlated-noise simulation, inference, and image-level evaluation. External MPPCA and NORDIC implementations should be obtained from their original sources and cited separately.

## Citation

The manuscript citation and DOI will be added after publication. In the meantime, GitHub can generate citation metadata from [`CITATION.cff`](CITATION.cff).

## License

This project is released under the [MIT License](LICENSE).

