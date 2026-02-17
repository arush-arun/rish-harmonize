# rish-harmonize

RISH harmonization for multi-site diffusion MRI data.

Implements Rotationally Invariant Spherical Harmonic (RISH) harmonization with two modes:

- **Signal-level SH** (recommended): Fits spherical harmonics directly to the DWI signal at each b-shell, following [De Luca et al. 2025](https://doi.org/10.1002/mrm.30467) and [Mirzaalian et al. 2016](https://doi.org/10.1016/j.neuroimage.2016.04.041). Captures shell-specific scanner effects.
- **FOD-level**: Operates on fiber orientation distribution (FOD) images from constrained spherical deconvolution (CSD).

Supports both template-based harmonization (reference site averaging) and RISH-GLM joint modeling (no matched cohorts required).

## Requirements

- Python >= 3.9
- [MRtrix3](https://www.mrtrix.org/) (runtime dependency for `amp2sh`, `sh2amp`, `mrconvert`, `mrmath`, `mrfilter`, `dwiextract`, `mrcalc`)
- NumPy, NiBabel, SciPy

## Installation

```bash
pip install -e .
```

## Quick start

### Signal-level SH harmonization

**Step 1: Create a reference template** from the reference site's DWI data (all registered to a common template space):

```bash
rish-harmonize create-template \
    --mode signal \
    --image-list ref_dwis.txt \
    --mask-list ref_masks.txt \
    -o template/ \
    --lmax 6
```

Where `ref_dwis.txt` is a text file with one DWI path per line.

**Step 2: Harmonize target subjects:**

```bash
rish-harmonize harmonize \
    --mode signal \
    --target target_dwi.mif \
    --template template/ \
    --mask mask.mif \
    -o harmonized_dwi.mif
```

### RISH-GLM (joint site + covariate model)

For multi-site studies without matched cohorts:

```bash
rish-harmonize rish-glm \
    --manifest manifest.csv \
    --reference-site SiteA \
    --mask group_mask.mif \
    -o output/ \
    --harmonize
```

**Manifest CSV format:**

```csv
subject,site,dwi_path,age,sex
sub-01,SiteA,/data/siteA/sub-01/dwi_reg.mif,32.5,0
sub-02,SiteB,/data/siteB/sub-02/dwi_reg.mif,45.1,1
```

Use `fod_path` instead of `dwi_path` for FOD-level mode.

## Commands

| Command | Description |
|---------|-------------|
| `detect-shells` | Show b-value shell structure of a DWI image |
| `extract-rish` | Extract RISH features from an SH image |
| `create-template` | Create RISH reference template from a site |
| `harmonize` | Harmonize a target image against a template |
| `rish-glm` | Fit RISH-GLM joint model and harmonize |
| `site-effect` | Test for site effects via permutation testing |

Run `rish-harmonize <command> --help` for detailed options.

## Signal-level SH pipeline

```
DWI (multi-shell, registered to template space)
    |
    v
detect shells (b=1000, b=3000, ...)
    |
    v
separate per shell (DW volumes only, no b=0)
    |
    v
amp2sh per shell --> SH coefficients
    |
    v
extract RISH per shell per order (l=0, l=2, l=4, ...)
    |
    v
compute scale maps:  scale_l = RISH_ref / RISH_target
    |
    v
apply scale maps to SH coefficients
    |
    v
sh2amp per shell --> harmonized DW signal
    |
    v
rejoin shells + original b=0
    |
    v
harmonized DWI
```

b=0 volumes are carried through unmodified (no angular dependence).

## Key parameters

- `--lmax`: Maximum SH order. Auto-determined from number of directions if not specified. Rule: (lmax+1)(lmax+2)/2 <= n_directions.
- `--smoothing`: Gaussian FWHM (mm) for scale map smoothing. Default: 3.0.
- `--clip-min / --clip-max`: Scale factor clipping range. Default: 0.5-2.0.

## References

- De Luca A, et al. "RISH-GLM: Rotationally invariant spherical harmonic general linear model for quantitative dMRI harmonization." *Magn Reson Med.* 2025.
- Mirzaalian H, et al. "Inter-site and inter-scanner diffusion MRI data harmonization." *NeuroImage.* 2016.
- Cetin Karayumak S, et al. "Retrospective harmonization of multi-site diffusion MRI data acquired with different acquisition parameters." *NeuroImage.* 2019.

## License

MIT
