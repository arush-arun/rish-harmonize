# UK Biobank RISH-GLM Harmonization Example

Multi-site diffusion MRI harmonization of 15 UK Biobank subjects across 4 imaging sites using the RISH-GLM approach (De Luca et al. 2025).

## Dataset

- **Subjects:** 15 UKB participants (QSIPrep-preprocessed, ACPC space)
- **Shells:** b=0, b=1000, b=2000
- **Sites:** 4 UKB imaging centres

| Site  | Subjects | Role      |
|-------|----------|-----------|
| 11025 | 7        | Reference |
| 11026 | 3        | Target    |
| 11027 | 3        | Target    |
| 11028 | 2        | Target    |

Site 11025 is chosen as reference because it has the most subjects.

## Prerequisites

- **MRtrix3** (mrconvert, dwi2response, dwifod, mtnormalise, population_template)
- **ANTs** (only for FA-template approach)
- **rish-harmonize** (`pip install -e .`)
- QSIPrep-preprocessed DWI data

## Pipeline Scripts

| Script | Description |
|--------|-------------|
| `setup_bunya.sh` | One-time setup on UQ Bunya HPC (venv, modules, data checks) |
| `config.sh` | Local paths, subject list, site assignments, shared helper functions |
| `config_bunya.sh` | Bunya HPC path overrides (Neurodesk modules, scratch paths) |
| `run_fod_template.sh` | Full FOD-template pipeline (local, 11 steps) |
| `run_fa_template.sh` | Full FA-template pipeline (local, alternative approach) |
| `submit_fod_pipeline.slurm` | SLURM submission wrapper for Bunya (FOD) |
| `submit_fa_pipeline.slurm` | SLURM submission wrapper for Bunya (FA) |
| `test_new_features_bunya.sh` | Validation script for diagnostics, provenance, and deterministic seeds |

## How to Run

### Local

```bash
# Edit config.local.sh with your paths (QSIPREP, SITE_CSV, etc.)
cp config.sh config.local.sh
# Edit paths in config.local.sh...

# Run all steps
./run_fod_template.sh

# Or run a single step (e.g., step 8 = RISH-GLM)
./run_fod_template.sh 8
```

### Bunya HPC

```bash
# First-time setup
bash setup_bunya.sh

# Submit full pipeline
sbatch submit_fod_pipeline.slurm

# Or run a single step
sbatch submit_fod_pipeline.slurm 8
```

## Pipeline Steps (FOD Template)

### Step 1: Convert DWI to MIF
Converts QSIPrep NIfTI outputs to MRtrix `.mif` format with embedded gradient tables.

### Step 2: Estimate Response Functions & Compute FODs
Runs `dwi2response dhollander` per subject, averages response functions across subjects, then computes multi-shell multi-tissue CSD FODs (`dwi2fod msmt_csd`).

### Step 3: Multi-tissue Intensity Normalisation
Runs `mtnormalise` to correct for global intensity differences before template building.

### Step 4: Build FOD Population Template
Runs MRtrix `population_template` to build an unbiased study-specific template from WM FODs. This is the most time-consuming step (several hours). Produces per-subject warp fields.

### Step 5: Extract Native RISH
Runs `rish-harmonize extract-native-rish` per subject. For each b-shell, fits spherical harmonics to the DWI signal and computes rotationally invariant features: `rish_l0, rish_l2, ..., rish_l8`.

### Step 6: Warp RISH to Template
Applies the per-subject warps from step 4 to bring all RISH feature maps into template space. RISH features are scalars (rotationally invariant), so standard spatial warping is used without SH reorientation.

### Step 7: Create Group Mask
Intersects all per-subject brain masks warped to template space. Only voxels inside every subject's mask are included, ensuring no edge artifacts.

### Step 8: Run RISH-GLM
Runs `rish-harmonize rish-glm` with a manifest CSV listing each subject's site and template-space RISH directory. Fits a joint GLM across sites per shell and SH order, then computes per-site scale maps: `scale_l{order}_{site}.mif`.

### Step 9: Verify
Prints scale map statistics (mean, median, std within the group mask) for visual sanity checking.

### Step 10: Site Effect Comparison
Runs `rish-harmonize site-effect` on RISH l0 maps before and after harmonization (per shell). Computes voxel-wise F-statistics via permutation testing to quantify how much site effect remains.

### Step 11: Apply Harmonization to Native DWI
For each target-site subject:
1. **Compute inverse warp** from the population template warpfull (`warpconvert -from 2`) to get template→native deformation fields
2. **Warp scale maps to native space** using `mrtransform` with the inverse warp
3. **Re-mask** scale maps in native space — sets voxels outside the brain mask to 1.0 (neutral scaling), preventing interpolation artifacts at brain boundaries
4. **Apply harmonization** using `rish-harmonize apply-harmonization`, which decomposes the DWI per shell into SH coefficients, multiplies each order's coefficients by the corresponding scale map, and reconstructs the harmonized DWI

Reference-site subjects are left unchanged (their scale factor is identity by definition).

## Output Structure

```
pipeline_output_fod/
  mif/                          # Step 1: converted DWI + masks
    sub-*/dwi.mif, mask.mif
  responses/                    # Step 2: tissue response functions
  fods/                         # Step 2: per-subject FOD images
  fods_normalised/              # Step 3: intensity-normalised FODs
  population_template/          # Step 4: template + warp fields
    template_fod.mif
    warps/sub-*_warp.mif
  native_rish/                  # Step 5: per-subject native RISH
    sub-*/b1000/rish/rish_l0.mif, rish_l2.mif, ...
    sub-*/b2000/rish/rish_l0.mif, rish_l2.mif, ...
  template_rish/                # Step 6: RISH warped to template
    sub-*/b1000/rish/rish_l0.mif, ...
  template_masks/               # Step 7: group mask
    group_mask.mif
  glm_output/                   # Step 8: RISH-GLM results
    glm/
      shell_lmax.json           # b-value -> max SH order mapping
      b1000/rish_glm_model/     # fitted model per shell
        rish_glm_model.json
        beta_site_*_l*.mif      # per-site GLM coefficients
      b2000/rish_glm_model/
        ...
    scale_maps/
      11026/                    # per target site
        scale_maps_meta.json    # paths + provenance
        b1000/
          scale_l0_11026.mif    # scale maps per SH order
          scale_l2_11026.mif
          ...
          scale_map_diagnostics.json  # clipping stats
        b2000/
          ...
      11027/
        ...
      11028/
        ...
  site_effect_comparison/       # Step 10: pre/post comparison
    b1000/
      pre/summary.json          # pre-harmonization statistics
      post/summary.json         # post-harmonization statistics
    b2000/
      ...
  harmonized/                   # Step 11: harmonized native DWI
    sub-*/                      # target-site subjects only
      inverse_warp.mif          # template -> native deformation field
      native_scale_maps/        # scale maps in native space
        b1000/scale_l0.mif, scale_l2.mif, ...
        b2000/scale_l0.mif, ...
      dwi_harmonized.mif        # final harmonized DWI
```

## Understanding the Outputs

### shell_lmax.json

Maps each b-value shell to its maximum spherical harmonics order, determined by taking the minimum across all subjects (ensures consistency):

```json
{"shell_lmax": {"1000": 8, "2000": 8}}
```

This means both shells support SH fitting up to lmax=8 (orders l0, l2, l4, l6, l8).

### Scale Maps (`scale_l{order}_{site}.mif`)

Per-voxel multiplicative correction factors for each target site and SH order. A scale map value of 1.2 at a voxel means the target site's RISH is 20% lower than the reference, so it needs to be multiplied by 1.2 to match.

- **Values near 1.0:** little site difference at that voxel
- **Values > 1.0:** target site has lower signal (needs boosting)
- **Values < 1.0:** target site has higher signal (needs dampening)
- **Clipped to [0.5, 2.0]:** prevents extreme corrections from noise or registration errors

### scale_map_diagnostics.json

Per-SH-order statistics within the brain mask:

| Field | Meaning |
|-------|---------|
| `mean` | Average scale factor (ideal: ~1.0) |
| `median` | Median scale factor |
| `std` | Spatial variability of correction |
| `pct_clipped_min` | % voxels hitting the 0.5 floor |
| `pct_clipped_max` | % voxels hitting the 2.0 ceiling |
| `pct_clipped_total` | Total % clipped (red flag if >20%) |

High clipping percentages indicate the site difference is large relative to the allowed correction range, possibly due to registration issues, acquisition protocol differences, or too few subjects in that site.

### Site Effect summary.json

Results of voxel-wise permutation testing for site effects:

| Field | Meaning |
|-------|---------|
| `mean_effect_size` | Mean partial eta-squared across voxels. Measures the proportion of variance explained by site. Lower is better. |
| `median_effect_size` | Median partial eta-squared. More robust to outlier voxels. |
| `percent_significant_permutation` | % of voxels with significant site effect (permutation p<0.05). |
| `percent_significant_fdr` | Same but FDR-corrected (Benjamini-Hochberg). More conservative. |
| `n_permutations` | Number of permutations used. |
| `seed` | Random seed for reproducibility. |

### Interpreting Pre vs Post Comparison

After harmonization, you expect:

- **Mean eta-squared should decrease** (site explains less variance)
- **% significant voxels may increase or decrease** depending on sample size and permutation count; with small samples (n=2 at site 11028), permutation p-values have limited resolution

Our results with 15 subjects:

| Shell  | Pre eta-sq | Post eta-sq | Reduction |
|--------|-----------|-------------|-----------|
| b=1000 | 0.1420    | 0.0514      | 64%       |
| b=2000 | 0.1344    | 0.0548      | 59%       |

FDR-corrected significance was 0% in all conditions, consistent with the small sample sizes per site.

### QC Report Figures

Generate visualizations from the pipeline outputs:

```bash
rish-harmonize qc-report \
    --glm-output glm_output/ \
    --site-effect-dir site_effect_comparison/ \
    -o qc_figures/
```

This produces:
- **`site_effect_comparison.png`** — Grouped bar chart of mean eta-squared and % significant voxels, pre vs post, per b-shell
- **`scale_map_heatmap_b{shell}.png`** — Heatmap of mean scale factors across target sites and SH orders, annotated with clipping percentages

## Applying Harmonization to DWI

Step 11 of the pipeline automates this. To run it standalone:

```bash
./run_fod_template.sh 11
```

This handles the full warp-back workflow per subject: computing inverse warps, transforming scale maps to native space, re-masking, and applying harmonization. The harmonized DWI is saved to `harmonized/sub-*/dwi_harmonized.mif`.

For custom workflows or other registration tools (e.g., ANTs), the key steps are:

```bash
# 1. Warp scale maps from template to native space
#    (use your registration tool's inverse transform)
mrtransform scale_l0_11026.mif scale_l0_native.mif \
    -warp inverse_deformation.mif -interp linear

# 2. Re-mask: set voxels outside brain to 1.0 (neutral scaling)
mrcalc scale_l0_native.mif mask.mif -mult \
       mask.mif 1 -sub -neg -add \
       scale_l0_remasked.mif -force

# 3. Apply harmonization to DWI
rish-harmonize apply-harmonization dwi.mif \
    --scale-maps native_scale_maps/ \
    -o dwi_harmonized.mif \
    --lmax-json shell_lmax.json
```

## References

- De Luca, A. et al. (2025). Cross-site harmonization of diffusion MRI data without matched training subjects. *Magnetic Resonance in Medicine*, 94(4), 1750-1762.
