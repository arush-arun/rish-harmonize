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

Run `rish-harmonize <command> --help` for detailed options.

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
