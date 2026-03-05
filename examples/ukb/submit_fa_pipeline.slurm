#!/bin/bash
#SBATCH --job-name=rish_fa
#SBATCH --account=a_aibn
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/user/uqahonne/ukb/bed_analysis/rish_harmonize/logs/fa_pipeline_%j.out
#SBATCH --error=/scratch/user/uqahonne/ukb/bed_analysis/rish_harmonize/logs/fa_pipeline_%j.err
# ==========================================================================
# UKB RISH-GLM pipeline — FA-based population template (ANTs)
#
# Builds an unbiased template from FA maps using ANTs SyN registration.
# Simpler but loses crossing fiber information during registration.
#
# Usage:
#   sbatch submit_fa_pipeline.slurm          # Run all steps
#   sbatch submit_fa_pipeline.slurm 3        # Run only step 3
#
# Steps:
#   1. Convert DWI to .mif
#   2. Compute FA maps
#   3. Build FA population template (ANTs)  [most time-consuming]
#   4. Extract native RISH
#   5. Warp RISH to template (ANTs transforms)
#   6. Create group mask
#   7. Run RISH-GLM
#   8. Verify
#   9. Site effect comparison (pre vs post harmonization)
#  10. Apply harmonization to native-space DWI
#  11. Generate QC report figures
# ==========================================================================

set -euo pipefail

# Source Bunya config (loads modules, sets paths, defines helpers)
SCRIPT_DIR="/scratch/user/uqahonne/ukb/bed_analysis/rish_harmonize"
if [[ -f "$SCRIPT_DIR/config_bunya.local.sh" ]]; then
    source "$SCRIPT_DIR/config_bunya.local.sh"
else
    source "$SCRIPT_DIR/config_bunya.sh"
fi

OUT="/scratch/user/uqahonne/ukb/bed_analysis/rish_harmonize/pipeline_output_fa"
mkdir -p "$OUT"

STEP="${1:-all}"

echo "================================================================"
echo "  RISH-GLM FA Pipeline — Bunya"
echo "  Job ID: ${SLURM_JOB_ID:-local}"
echo "  Node:   $(hostname)"
echo "  CPUs:   ${SLURM_CPUS_PER_TASK:-$NTHREADS}"
echo "  Step:   $STEP"
echo "  Output: $OUT"
echo "  Start:  $(date)"
echo "================================================================"

# ==========================================================================
# Step 1: Convert DWI to .mif
# ==========================================================================
if [[ "$STEP" == "all" || "$STEP" == "1" ]]; then
    step_convert_dwi "$OUT"
fi

# ==========================================================================
# Step 2: Compute FA maps
# ==========================================================================
if [[ "$STEP" == "all" || "$STEP" == "2" ]]; then
echo ""
echo "================================================================"
echo "  Step 2: Computing FA maps"
echo "================================================================"

FA_DIR="$OUT/fa_maps"
mkdir -p "$FA_DIR"

for SUBJ in "${SUBJECTS[@]}"; do
    FA_OUT="$FA_DIR/${SUBJ}_FA.nii.gz"
    if [[ -f "$FA_OUT" ]]; then
        echo "  [$SUBJ] Already exists, skipping"
        continue
    fi

    MASK_NII=$(find_mask "$SUBJ")
    TMP_DT="$FA_DIR/${SUBJ}_dt.mif"
    dwi2tensor "$OUT/mif/$SUBJ/dwi.mif" "$TMP_DT" -mask "$MASK_NII" -force -quiet
    tensor2metric "$TMP_DT" -fa "$FA_OUT" -mask "$MASK_NII" -force -quiet
    rm -f "$TMP_DT"

    echo "  [$SUBJ] FA computed"
done
echo "  Done."
fi

# ==========================================================================
# Step 3: Build unbiased FA population template (ANTs)
# ==========================================================================
if [[ "$STEP" == "all" || "$STEP" == "3" ]]; then
echo ""
echo "================================================================"
echo "  Step 3: Building FA population template (ANTs)"
echo "  This may take several hours..."
echo "================================================================"

TEMPLATE_DIR="$OUT/population_template"
mkdir -p "$TEMPLATE_DIR"

if [[ -f "$TEMPLATE_DIR/template0.nii.gz" ]]; then
    echo "  Template already exists, skipping"
else
    FA_LIST=""
    for SUBJ in "${SUBJECTS[@]}"; do
        FA_LIST="$FA_LIST $OUT/fa_maps/${SUBJ}_FA.nii.gz"
    done

    cd "$TEMPLATE_DIR"
    antsMultivariateTemplateConstruction2.sh \
        -d 3 \
        -o "$TEMPLATE_DIR/" \
        -n 0 \
        -r 1 \
        -i 4 \
        -m CC \
        -t SyN \
        -j "$NTHREADS" \
        $FA_LIST

    echo "  Template built: $TEMPLATE_DIR/template0.nii.gz"
fi

mrinfo "$TEMPLATE_DIR/template0.nii.gz" -size -spacing 2>/dev/null || true
fi

# ==========================================================================
# Step 4: Extract native RISH
# ==========================================================================
if [[ "$STEP" == "all" || "$STEP" == "4" ]]; then
    step_extract_rish "$OUT"
fi

# ==========================================================================
# Step 5: Warp RISH to template (ANTs transforms)
# ==========================================================================
if [[ "$STEP" == "all" || "$STEP" == "5" ]]; then
echo ""
echo "================================================================"
echo "  Step 5: Warping RISH to template (ANTs)"
echo "================================================================"

TEMPLATE="$OUT/population_template/template0.nii.gz"
TEMPLATE_DIR="$OUT/population_template"

for SUBJ in "${SUBJECTS[@]}"; do
    FA_BASE="${SUBJ}_FA"
    WARP=$(ls "$TEMPLATE_DIR"/input*-${FA_BASE}-1Warp.nii.gz 2>/dev/null | head -1)
    AFFINE=$(ls "$TEMPLATE_DIR"/input*-${FA_BASE}-0GenericAffine.mat 2>/dev/null | head -1)

    if [[ -z "$WARP" || -z "$AFFINE" ]]; then
        echo "  [$SUBJ] WARNING: Transforms not found, skipping"
        continue
    fi

    for BDIR in "$OUT/native_rish/$SUBJ"/b*/; do
        BVAL=$(basename "$BDIR")
        TEMPLATE_RISH_DIR="$OUT/template_rish/$SUBJ/$BVAL/rish"
        mkdir -p "$TEMPLATE_RISH_DIR"

        # Use a temp dir under $OUT (not /tmp) so files are visible
        # inside Singularity/Apptainer containers on HPC
        ANTS_TMP="$OUT/tmp"
        mkdir -p "$ANTS_TMP"

        for RISH_FILE in "$BDIR/rish"/rish_l*.mif; do
            FNAME=$(basename "$RISH_FILE" .mif)
            OUTFILE="$TEMPLATE_RISH_DIR/${FNAME}.mif"
            [[ -f "$OUTFILE" ]] && continue

            TMP_IN="$ANTS_TMP/rish_${SUBJ}_${BVAL}_${FNAME}.nii.gz"
            TMP_OUT="$ANTS_TMP/rish_${SUBJ}_${BVAL}_${FNAME}_template.nii.gz"
            mrconvert "$RISH_FILE" "$TMP_IN" -force -quiet

            antsApplyTransforms \
                -d 3 -i "$TMP_IN" -r "$TEMPLATE" \
                -t "$WARP" -t "$AFFINE" \
                -o "$TMP_OUT" --interpolation Linear

            mrconvert "$TMP_OUT" "$OUTFILE" -force -quiet
            rm -f "$TMP_IN" "$TMP_OUT"
        done
    done
    echo "  [$SUBJ] Warped"
done
echo "  Done."
fi

# ==========================================================================
# Step 6: Create group mask
# ==========================================================================
if [[ "$STEP" == "all" || "$STEP" == "6" ]]; then
echo ""
echo "================================================================"
echo "  Step 6: Creating group mask"
echo "================================================================"

TEMPLATE="$OUT/population_template/template0.nii.gz"
TEMPLATE_DIR="$OUT/population_template"
MASK_DIR="$OUT/template_masks"
mkdir -p "$MASK_DIR"

if [[ -f "$MASK_DIR/group_mask.mif" ]]; then
    echo "  Already exists, skipping"
else
    MASK_ARGS=""
    for SUBJ in "${SUBJECTS[@]}"; do
        FA_BASE="${SUBJ}_FA"
        WARP=$(ls "$TEMPLATE_DIR"/input*-${FA_BASE}-1Warp.nii.gz 2>/dev/null | head -1)
        AFFINE=$(ls "$TEMPLATE_DIR"/input*-${FA_BASE}-0GenericAffine.mat 2>/dev/null | head -1)
        NATIVE_MASK=$(find_mask "$SUBJ")
        TMPL_MASK="$MASK_DIR/${SUBJ}_mask_template.nii.gz"

        if [[ ! -f "$TMPL_MASK" ]]; then
            antsApplyTransforms \
                -d 3 -i "$NATIVE_MASK" -r "$TEMPLATE" \
                -t "$WARP" -t "$AFFINE" \
                -o "$TMPL_MASK" --interpolation NearestNeighbor
        fi
        MASK_ARGS="$MASK_ARGS $TMPL_MASK"
    done

    mrmath $MASK_ARGS min "$MASK_DIR/group_mask.mif" -force -quiet
    NVOX=$(mrstats "$MASK_DIR/group_mask.mif" -output count -ignorezero 2>/dev/null)
    echo "  Group mask: $NVOX voxels"
fi
fi

# ==========================================================================
# Step 7: Run RISH-GLM
# ==========================================================================
if [[ "$STEP" == "all" || "$STEP" == "7" ]]; then
    step_run_glm "$OUT" "$OUT/template_masks/group_mask.mif"
fi

# ==========================================================================
# Step 8: Verify
# ==========================================================================
if [[ "$STEP" == "all" || "$STEP" == "8" ]]; then
    step_verify "$OUT" "$OUT/template_masks/group_mask.mif"
fi

# ==========================================================================
# Step 9: Site effect comparison (pre vs post harmonization)
# ==========================================================================
if [[ "$STEP" == "all" || "$STEP" == "9" ]]; then
    step_site_effect_comparison "$OUT" "$OUT/template_masks/group_mask.mif"
fi

# ==========================================================================
# Step 10: Apply harmonization to native-space DWI
# ==========================================================================
if [[ "$STEP" == "all" || "$STEP" == "10" ]]; then
echo ""
echo "================================================================"
echo "  Step 10: Applying harmonization to native DWI"
echo "================================================================"

GLM_DIR="$OUT/glm_output"
TEMPLATE_DIR="$OUT/population_template"
TEMPLATE="$TEMPLATE_DIR/template0.nii.gz"
HARMONIZED_DIR="$OUT/harmonized"
LMAX_JSON="$GLM_DIR/glm/shell_lmax.json"
ANTS_TMP="$OUT/tmp"
mkdir -p "$HARMONIZED_DIR" "$ANTS_TMP"

if [[ ! -f "$LMAX_JSON" ]]; then
    echo "  ERROR: shell_lmax.json not found. Run step 7 first."
    exit 1
fi

for SUBJ in "${SUBJECTS[@]}"; do
    SITE="${SITE_MAP[$SUBJ]}"

    if [[ "$SITE" == "$REF_SITE" ]]; then
        echo "  [$SUBJ] Reference site ($REF_SITE), skipping"
        continue
    fi

    HARM_OUT="$HARMONIZED_DIR/$SUBJ/dwi_harmonized.mif"
    if [[ -f "$HARM_OUT" ]]; then
        echo "  [$SUBJ] Already harmonized, skipping"
        continue
    fi

    SCALE_DIR="$GLM_DIR/scale_maps/$SITE"
    if [[ ! -d "$SCALE_DIR" ]]; then
        echo "  [$SUBJ] WARNING: No scale maps for site $SITE, skipping"
        continue
    fi

    # Find ANTs transforms
    FA_BASE="${SUBJ}_FA"
    AFFINE=$(ls "$TEMPLATE_DIR"/input*-${FA_BASE}-0GenericAffine.mat 2>/dev/null | head -1)
    INV_WARP=$(ls "$TEMPLATE_DIR"/input*-${FA_BASE}-1InverseWarp.nii.gz 2>/dev/null | head -1)

    if [[ -z "$INV_WARP" || -z "$AFFINE" ]]; then
        echo "  [$SUBJ] WARNING: Transforms not found, skipping"
        continue
    fi

    NATIVE_SCALES="$HARMONIZED_DIR/$SUBJ/native_scale_maps"
    NATIVE_MASK="$OUT/mif/$SUBJ/mask.mif"
    mkdir -p "$HARMONIZED_DIR/$SUBJ" "$NATIVE_SCALES"

    # Native-space reference for ANTs
    NATIVE_REF="$ANTS_TMP/${SUBJ}_ref.nii.gz"
    if [[ ! -f "$NATIVE_REF" ]]; then
        mrconvert "$OUT/mif/$SUBJ/dwi.mif" "$NATIVE_REF" \
            -coord 3 0 -force -quiet
    fi

    for BDIR in "$SCALE_DIR"/b*/; do
        [[ ! -d "$BDIR" ]] && continue
        BVAL=$(basename "$BDIR")
        NATIVE_B_DIR="$NATIVE_SCALES/$BVAL"
        mkdir -p "$NATIVE_B_DIR"

        for SCALE_MIF in "$BDIR"/scale_l*_${SITE}.mif; do
            [[ ! -f "$SCALE_MIF" ]] && continue
            FNAME=$(basename "$SCALE_MIF" "_${SITE}.mif")
            NATIVE_SCALE="$NATIVE_B_DIR/${FNAME}.mif"
            [[ -f "$NATIVE_SCALE" ]] && continue

            TMP_SCALE_NII="$ANTS_TMP/${SUBJ}_${BVAL}_${FNAME}.nii.gz"
            TMP_NATIVE_NII="$ANTS_TMP/${SUBJ}_${BVAL}_${FNAME}_native.nii.gz"
            mrconvert "$SCALE_MIF" "$TMP_SCALE_NII" -force -quiet

            antsApplyTransforms \
                -d 3 -i "$TMP_SCALE_NII" -r "$NATIVE_REF" \
                -t "[${AFFINE},1]" -t "$INV_WARP" \
                -o "$TMP_NATIVE_NII" --interpolation Linear

            TMP_WARPED="$NATIVE_B_DIR/${FNAME}_warped.mif"
            mrconvert "$TMP_NATIVE_NII" "$TMP_WARPED" -force -quiet
            rm -f "$TMP_SCALE_NII" "$TMP_NATIVE_NII"

            mrcalc "$TMP_WARPED" "$NATIVE_MASK" -mult \
                   "$NATIVE_MASK" 1 -sub -neg -add \
                   "$NATIVE_SCALE" -force -quiet
            rm -f "$TMP_WARPED"
        done
    done

    rm -f "$NATIVE_REF"

    rish-harmonize apply-harmonization \
        "$OUT/mif/$SUBJ/dwi.mif" \
        --scale-maps "$NATIVE_SCALES" \
        -o "$HARM_OUT" \
        --lmax-json "$LMAX_JSON" \
        --threads "$NTHREADS"

    echo "  [$SUBJ] Harmonized -> $HARM_OUT"
done
echo "  Done."
fi

# ==========================================================================
# Step 11: Generate QC report figures
# ==========================================================================
if [[ "$STEP" == "all" || "$STEP" == "11" ]]; then
echo ""
echo "================================================================"
echo "  Step 11: Generating QC report figures"
echo "================================================================"

QC_DIR="$OUT/qc_figures"
mkdir -p "$QC_DIR"

QC_ARGS=(-o "$QC_DIR")

if [[ -d "$OUT/glm_output" ]]; then
    QC_ARGS+=(--glm-output "$OUT/glm_output")
fi

if [[ -d "$OUT/site_effect_comparison" ]]; then
    QC_ARGS+=(--site-effect-dir "$OUT/site_effect_comparison")
fi

rish-harmonize qc-report "${QC_ARGS[@]}"
echo "  Done."
fi

echo ""
echo "================================================================"
echo "  FA-template pipeline complete!"
echo "  Output: $OUT"
echo "  End:    $(date)"
echo "================================================================"
