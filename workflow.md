# Signal-level SH RISH Harmonization Workflow

## Overview

RISH features are scalar and rotationally invariant, so they can be warped
between native and template space using standard (non-reorienting) registration.
SH fitting and harmonization happen in native space; scale map computation
happens in template space.

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

## Step-by-step

### Step 1: Extract RISH in native space

For each subject (both reference and target sites):

```bash
rish-harmonize extract-native-rish dwi.mif \
    -o sub01/native_rish/ \
    --mask mask.mif \
    --consistent-with all_dwis.txt
```

`--consistent-with` takes a text file listing all DWI images across all
subjects (both reference and target sites). It computes the minimum number
of directions per shell across all subjects and derives a consistent lmax,
so that every subject uses the same SH basis.

**This is critical**: all subjects must use the same lmax per shell so that
their RISH features have the same number of orders and are directly
comparable. Downstream steps (`create-template`, `compute-scale-maps`) will
reject inputs with mismatched orders.

Output structure:

```
sub01/native_rish/
    shell_meta.json
    b1500/
        dwi_dw_only.mif
        sh.mif
        directions.txt
        rish/
            rish_l0.mif
            rish_l2.mif
            ...
    b3000/
        ...
```

### Step 2: Warp RISH to template space

Use your existing registration transforms (from preprocessing) to warp the
scalar RISH images to template space. For example with MRtrix3:

```bash
for f in sub01/native_rish/b3000/rish/rish_l*.mif; do
    mrtransform "$f" \
        -warp native2template_warp.mif \
        -interp linear \
        "sub01/template_rish/b3000/rish/$(basename $f)"
done
```

Repeat for each b-shell. Maintain the same directory structure
(`b{value}/rish/rish_l{order}.mif`) so downstream tools can discover the files.

### Step 3: Create reference template

Average template-space RISH across reference-site subjects:

```bash
# ref_rish_dirs.txt contains one directory per reference subject:
#   sub01/template_rish/
#   sub02/template_rish/
#   ...

rish-harmonize create-template --mode signal \
    --rish-list ref_rish_dirs.txt \
    -o template/ \
    --lmax-json sub01/native_rish/shell_meta.json
```

Output: `template/template_meta.json` with per-shell averaged RISH and lmax.

### Step 4: Compute scale maps in template space

For each target-site subject:

```bash
rish-harmonize compute-scale-maps \
    --ref-rish template/ \
    --target-rish sub01/template_rish/ \
    -o sub01/scale_maps_template/ \
    --mask template_mask.mif \
    --smoothing 3.0 \
    --clip-min 0.5 \
    --clip-max 2.0
```

Scale maps are `ref_RISH / target_RISH` per shell per order, smoothed and
clipped to prevent extreme corrections.

### Step 5: Warp scale maps back to native space

Warp scale maps to native space, then re-mask. Re-masking is needed because
interpolation at brain boundaries creates artifacts; we reset background
voxels to 1.0 (no scaling) using the native mask.

```bash
for f in sub01/scale_maps_template/b3000/scale_l*.mif; do
    out="sub01/scale_maps_native/b3000/$(basename $f)"
    # Inverse warp
    mrtransform "$f" \
        -warp template2native_warp.mif \
        -interp linear "$out"
    # Re-mask: brain = warped value, background = 1.0
    mrcalc "$out" native_mask.mif -mult \
        native_mask.mif 1 -sub -neg -add "$out" -force
done
```

### Step 6: Apply harmonization in native space

```bash
rish-harmonize apply-harmonization dwi.mif \
    --scale-maps sub01/scale_maps_native/ \
    -o dwi_harmonized.mif \
    --lmax-json sub01/native_rish/shell_meta.json
```

This performs: detect shells -> separate DW-only -> amp2sh -> multiply SH
coefficients by scale maps -> sh2amp -> rejoin shells with original b=0
volumes.

## Key details

- **b=0 volumes** bypass the SH pipeline entirely (no angular dependence)
  and are carried through unchanged.
- **lmax consistency** is enforced via `--consistent-with`: uses the minimum
  number of directions per shell across all subjects to determine lmax.
- **RISH features** are defined as `theta_l = sqrt(sum_m |c_lm|^2)` and are
  scalar images safe to warp with standard registration.
- **Scale maps** are `ref_theta_l / target_theta_l`, smoothed (default
  FWHM=3mm) and clipped (default 0.5-2.0) to prevent extreme corrections.
- **amp2sh** does not support masking in MRtrix3 3.0.8; SH fitting runs on
  the full image. Masking is applied during RISH extraction and scale map
  computation.

## FOD-level workflow (alternative)

For FOD-level harmonization, everything happens in template space (FODs are
already registered). This is simpler but collapses multi-shell information:

```bash
# Create template from reference FODs
rish-harmonize create-template --mode fod \
    --image-list ref_fods.txt \
    --mask-list ref_masks.txt \
    -o template/

# Harmonize a target FOD
rish-harmonize harmonize \
    --target target_fod.mif \
    --template template/ \
    --mask mask.mif \
    -o fod_harmonized.mif
```

## RISH-GLM (joint model with covariates)

For multi-site studies with confounding variables (age, sex, etc.), use the
RISH-GLM approach which fits a joint model.

### Signal-level (native↔template workflow)

Use a manifest with `rish_dir` pointing to pre-extracted template-space RISH
directories (from `extract-native-rish` + warping to template):

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

This fits the GLM in template space and outputs template-space scale maps
per target site. Then warp scale maps to native space, re-mask, and apply:

```bash
# Warp scale maps to native space
for f in output/scale_maps/SiteB/b3000/scale_l*.mif; do
    out="native_scale_maps/b3000/$(basename $f)"
    mrtransform "$f" -warp template2native_warp.mif -interp linear "$out"
    mrcalc "$out" native_mask.mif -mult \
        native_mask.mif 1 -sub -neg -add "$out" -force
done

# Apply in native space
rish-harmonize apply-harmonization dwi.mif \
    --scale-maps native_scale_maps/ \
    -o dwi_harmonized.mif \
    --lmax-json output/glm/shell_lmax.json
```

Note: `--harmonize` is not supported in `signal_rish` mode since scale maps
are in template space and must be warped back to native before application.

### FOD-level (single-space workflow)

For FOD-level mode, everything is in template space and `--harmonize`
applies harmonization directly:

```bash
rish-harmonize rish-glm \
    --manifest manifest.csv \
    --reference-site SiteA \
    --mask group_mask.mif \
    -o output/ \
    --harmonize
```

The manifest CSV should have columns: `subject`, `site`, `fod_path`,
and any covariates (e.g. `age`, `sex`).
