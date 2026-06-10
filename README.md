# rish-harmonize

RISH harmonization for multi-site diffusion MRI data.

Implements Rotationally Invariant Spherical Harmonic (RISH) harmonization with two modes:

- **Signal-level SH** (recommended): Fits spherical harmonics directly to the DWI signal at each b-shell, following [De Luca et al. 2025](https://pubmed.ncbi.nlm.nih.gov/40407799/) 
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

SH fitting and harmonization happen in native space; RISH comparison and scale map computation happen in template space. See `workflow.md` for full details.

**Step 1: Extract RISH features** in native space (for each subject, both sites):

```bash
rish-harmonize extract-native-rish dwi.mif \
    -o sub01/native_rish/ \
    --mask mask.mif \
    --consistent-with all_dwis.txt
```

**Step 2: Warp RISH to template space** (user-provided registration):

```bash
for f in sub01/native_rish/b3000/rish/rish_l*.mif; do
    mrtransform "$f" -linear affine.txt -template template_grid.mif \
        -interp linear "sub01/template_rish/b3000/rish/$(basename $f)"
done
```

**Step 3: Create reference template** from reference-site RISH (in template space):

```bash
rish-harmonize create-template --mode signal \
    --rish-list ref_rish_dirs.txt \
    -o template/
```

**Step 4: Compute scale maps** in template space (for each target subject):

```bash
rish-harmonize compute-scale-maps \
    --ref-rish template/ \
    --target-rish sub01/template_rish/ \
    -o sub01/scale_maps_template/ \
    --mask template_mask.mif
```

**Step 5: Warp scale maps back to native space** and re-mask (interpolation at boundaries needs cleanup):

```bash
for f in sub01/scale_maps_template/b3000/scale_l*.mif; do
    out="sub01/scale_maps_native/b3000/$(basename $f)"
    mrtransform "$f" -linear affine.txt -inverse -template native_mask.mif \
        -interp linear "$out"
    # Re-mask: brain = warped value, background = 1.0
    mrcalc "$out" native_mask.mif -mult \
        native_mask.mif 1 -sub -neg -add "$out" -force
done
```

**Step 6: Apply harmonization** in native space:

```bash
rish-harmonize apply-harmonization dwi.mif \
    --scale-maps sub01/scale_maps_native/ \
    -o dwi_harmonized.mif \
    --lmax-json sub01/native_rish/shell_meta.json
```

### RISH-GLM (joint site + covariate model)

For multi-site studies without matched cohorts. Supports pre-extracted
template-space RISH (`rish_dir`) for the native↔template signal workflow:

```csv
subject,site,rish_dir,age,sex
sub-01,SiteA,/data/template_rish/sub-01/,25.0,0
sub-02,SiteB,/data/template_rish/sub-02/,30.0,1
```

```bash
rish-harmonize rish-glm \
    --manifest manifest_rish.csv \
    --reference-site SiteA \
    --mask group_mask.mif \
    -o output/
```

This outputs template-space scale maps. Warp them to native space, re-mask,
and apply with `apply-harmonization`. See `workflow.md` for full details.

For FOD-level mode, use `fod_path` in the manifest and add `--harmonize`
for direct harmonization in template space.

## Commands

| Command | Description |
|---------|-------------|
| `detect-shells` | Show b-value shell structure of a DWI image |
| `extract-rish` | Extract RISH features from an SH image |
| `extract-native-rish` | Extract per-shell RISH from native-space DWI |
| `create-template` | Create RISH reference template (signal or FOD mode) |
| `compute-scale-maps` | Compute per-shell scale maps from ref/target RISH |
| `apply-harmonization` | Apply native-space scale maps to DWI |
| `harmonize` | Harmonize a target FOD against a template |
| `rish-glm` | Fit RISH-GLM joint model and harmonize |
| `site-effect` | Test for site effects via permutation testing |
| `qc-report` | Generate QC visualization figures |

Run `rish-harmonize <command> --help` for detailed options.

## Modes

The `rish-glm` command supports three modes, auto-detected from manifest columns:

| Mode | Manifest column | Description |
|------|----------------|-------------|
| `signal` | `dwi_path` | **Recommended.** Fits SH to raw DWI per shell, extracts RISH, fits GLM jointly. Handles shell detection, lmax selection, and SH fitting internally. |
| `signal_rish` | `rish_dir` | Pre-extracted RISH directories (e.g., when RISH has already been warped to template space). Skips SH fitting; goes straight to GLM. |
| `fod` | `fod_path` | Operates on FOD images from CSD. Use `--harmonize` to apply scale maps directly in template space. |

Other commands operate on specific inputs and do not require a mode flag.

## CLI Reference

### `detect-shells`

Show b-value shell structure of a DWI image.

```
rish-harmonize detect-shells [-h] [--b0-threshold B0_THRESHOLD] dwi
```

| Argument | Description |
|----------|-------------|
| `dwi` | Input DWI image |
| `--b0-threshold` | B-value threshold for b=0 (default: 50) |

### `extract-rish`

Extract RISH features from an SH coefficient image (single image, single shell).

```
rish-harmonize extract-rish [-h] -o OUTPUT [--lmax LMAX] [--mask MASK]
                            [--threads THREADS] input
```

| Argument | Description |
|----------|-------------|
| `input` | Input SH coefficient image |
| `-o, --output` | Output directory |
| `--lmax` | Maximum SH order |
| `--mask` | Brain mask |
| `--threads` | Number of threads |

### `extract-native-rish`

Extract per-shell RISH from a native-space DWI. Splits shells, fits SH per shell, and computes RISH features. Use `--consistent-with` to ensure the same lmax across all subjects.

```
rish-harmonize extract-native-rish [-h] -o OUTPUT [--mask MASK] [--lmax LMAX]
                                   [--consistent-with FILE] [--threads THREADS]
                                   dwi
```

| Argument | Description |
|----------|-------------|
| `dwi` | Input DWI image (native space) |
| `-o, --output` | Output directory |
| `--mask` | Brain mask |
| `--lmax` | Maximum SH order (applied to all shells, capped by data) |
| `--consistent-with` | Text file listing all DWI images; computes minimum lmax across all subjects per shell |
| `--threads` | Number of threads |

### `create-template`

Create a RISH reference template by averaging RISH features across reference-site subjects.

```
rish-harmonize create-template [-h] --mode {signal,fod} [--rish-list RISH_LIST]
                               [--image-list IMAGE_LIST] [--mask-list MASK_LIST]
                               -o OUTPUT [--lmax LMAX] [--lmax-json LMAX_JSON]
                               [--threads THREADS]
```

| Argument | Description |
|----------|-------------|
| `--mode` | Harmonization mode (`signal` or `fod`) |
| `--rish-list` | (signal) Text file with one RISH directory per line |
| `--image-list` | (fod) Text file with one FOD image path per line |
| `--mask-list` | Text file with one mask path per line |
| `-o, --output` | Output directory |
| `--lmax` | (fod) Maximum SH order |
| `--lmax-json` | (signal) JSON file with shell_lmax (from `extract-native-rish`) |
| `--threads` | Number of threads |

### `compute-scale-maps`

Compute per-shell scale maps from reference and target RISH in template space.

```
rish-harmonize compute-scale-maps [-h] --ref-rish REF_RISH --target-rish TARGET_RISH
                                  -o OUTPUT [--mask MASK] [--smoothing SMOOTHING]
                                  [--clip-min CLIP_MIN] [--clip-max CLIP_MAX]
                                  [--threads THREADS]
```

| Argument | Description |
|----------|-------------|
| `--ref-rish` | Reference RISH directory (template space) |
| `--target-rish` | Target RISH directory (template space) |
| `-o, --output` | Output directory for scale maps |
| `--mask` | Brain mask (template space) |
| `--smoothing` | Scale map smoothing FWHM in mm (default: 3.0) |
| `--clip-min` | Minimum scale factor (default: 0.5) |
| `--clip-max` | Maximum scale factor (default: 2.0) |
| `--threads` | Number of threads |

### `apply-harmonization`

Apply native-space scale maps to a DWI. Fits SH per shell, multiplies by the corresponding scale map, then reconstructs the DWI signal.

```
rish-harmonize apply-harmonization [-h] --scale-maps SCALE_MAPS -o OUTPUT
                                   [--lmax-json LMAX_JSON] [--threads THREADS]
                                   dwi
```

| Argument | Description |
|----------|-------------|
| `dwi` | Input DWI image (native space) |
| `--scale-maps` | Scale maps directory (native space) |
| `-o, --output` | Output harmonized DWI |
| `--lmax-json` | JSON file with shell_lmax for consistency |
| `--threads` | Number of threads |

### `harmonize`

Harmonize a target FOD image against a reference template. FOD-mode shortcut that computes scale maps and applies them in one step.

```
rish-harmonize harmonize [-h] --target TARGET --template TEMPLATE [--mask MASK]
                         -o OUTPUT [--lmax LMAX] [--smoothing SMOOTHING]
                         [--clip-min CLIP_MIN] [--clip-max CLIP_MAX]
                         [--threads THREADS]
```

| Argument | Description |
|----------|-------------|
| `--target` | Target FOD image |
| `--template` | Template directory |
| `--mask` | Brain mask |
| `-o, --output` | Output path |
| `--lmax` | Maximum SH order |
| `--smoothing` | Scale map smoothing FWHM in mm (default: 3.0) |
| `--clip-min` | Minimum scale factor (default: 0.5) |
| `--clip-max` | Maximum scale factor (default: 2.0) |
| `--threads` | Number of threads |

### `rish-glm`

Fit a joint RISH-GLM model across all sites and compute scale maps. Supports covariates (age, sex, etc.) via the manifest CSV. Use `--harmonize` to also apply scale maps to target-site DWIs.

```
rish-harmonize rish-glm [-h] --manifest MANIFEST --reference-site REFERENCE_SITE
                        [--mode {signal,signal_rish,fod}] [--mask MASK] -o OUTPUT
                        [--lmax LMAX] [--harmonize] [--smoothing SMOOTHING]
                        [--clip-min CLIP_MIN] [--clip-max CLIP_MAX]
                        [--threads THREADS]
```

| Argument | Description |
|----------|-------------|
| `--manifest` | CSV with columns: subject, site, dwi_path/fod_path/rish_dir, [covariates] |
| `--reference-site` | Name of the reference site |
| `--mode` | Mode: `signal`, `signal_rish`, or `fod` (auto-detected from manifest) |
| `--mask` | Group brain mask |
| `-o, --output` | Output directory |
| `--lmax` | Maximum SH order |
| `--harmonize` | Also harmonize all target-site DWIs |
| `--smoothing` | Scale map smoothing FWHM in mm (default: 3.0) |
| `--clip-min` | Minimum scale factor (default: 0.5) |
| `--clip-max` | Maximum scale factor (default: 2.0) |
| `--threads` | Number of threads |

### `site-effect`

Test for residual site effects via permutation testing on RISH or image data.

```
rish-harmonize site-effect [-h] --site-list SITE_LIST --mask MASK -o OUTPUT
                           [--n-permutations N_PERMUTATIONS] [--seed SEED]
```

| Argument | Description |
|----------|-------------|
| `--site-list` | CSV with columns: site, rish_path/image_path |
| `--mask` | Group brain mask |
| `-o, --output` | Output directory |
| `--n-permutations` | Number of permutations (default: 5000) |
| `--seed` | Random seed (default: 42) |

### `qc-report`

Generate QC visualization figures from pipeline outputs (GLM scale maps and/or site effect comparisons).

```
rish-harmonize qc-report [-h] [--glm-output GLM_OUTPUT]
                         [--site-effect-dir SITE_EFFECT_DIR] -o OUTPUT [--dpi DPI]
```

| Argument | Description |
|----------|-------------|
| `--glm-output` | GLM output directory (with scale_maps/) |
| `--site-effect-dir` | Site effect comparison directory (with b*/pre/ and b*/post/) |
| `-o, --output` | Output directory for figures |
| `--dpi` | Figure DPI (default: 150) |

## Signal-level SH pipeline

```
Native space                    Template space

1. Extract RISH per subject
   DWI -> shells -> amp2sh -> RISH
   (scalar, rotationally invariant)
        |
        |  2. Warp RISH to template
        |     (mrtransform, standard scalar warp)
        |------------------------------>|
        |                               |
        |                    3. Average reference RISH
        |                       across ref-site subjects
        |                       -> reference template
        |
        |                    4. Compute scale maps
        |                       ref_RISH / target_RISH
        |                       per shell, per order
        |                               |
        |  5. Warp scale maps  <--------|
        |     back to native
        |
6. Apply scale maps to native SH
   SH x scale -> sh2amp -> harmonized DWI
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
