#!/bin/bash
# ==========================================================================
# UKB RISH-GLM pipeline — FA-based population template (ANTs)
#
# Builds an unbiased template from FA maps using ANTs SyN registration.
# Simpler but loses crossing fiber information during registration.
#
# Steps:
#   1. Convert DWI to .mif
#   2. Compute FA maps
#   3. Build FA population template (ANTs)
#   4. Extract native RISH
#   5. Warp RISH to template (ANTs transforms)
#   6. Create group mask
#   7. Run RISH-GLM
#   8. Verify
#
# Usage: ./run_fa_template.sh [step]
# ==========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "$SCRIPT_DIR/config.local.sh" ]]; then
    source "$SCRIPT_DIR/config.local.sh"
else
    source "$SCRIPT_DIR/config.sh"
fi

OUT="/home/uqahonne/uq/rish-harmonize/examples/ukb/pipeline_output_fa"
mkdir -p "$OUT"

STEP="${1:-all}"

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
        -o "$TEMPLATE_DIR/template" \
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
    # ANTs names transforms after input filename
    FA_BASE="${SUBJ}_FA"
    WARP="$TEMPLATE_DIR/template${FA_BASE}1Warp.nii.gz"
    AFFINE="$TEMPLATE_DIR/template${FA_BASE}0GenericAffine.mat"

    if [[ ! -f "$WARP" || ! -f "$AFFINE" ]]; then
        echo "  [$SUBJ] WARNING: Transforms not found, skipping"
        continue
    fi

    for BDIR in "$OUT/native_rish/$SUBJ"/b*/; do
        BVAL=$(basename "$BDIR")
        TEMPLATE_RISH_DIR="$OUT/template_rish/$SUBJ/$BVAL/rish"
        mkdir -p "$TEMPLATE_RISH_DIR"

        for RISH_FILE in "$BDIR/rish"/rish_l*.mif; do
            FNAME=$(basename "$RISH_FILE" .mif)
            OUTFILE="$TEMPLATE_RISH_DIR/${FNAME}.mif"
            [[ -f "$OUTFILE" ]] && continue

            TMP_IN="/tmp/rish_${SUBJ}_${BVAL}_${FNAME}.nii.gz"
            TMP_OUT="/tmp/rish_${SUBJ}_${BVAL}_${FNAME}_template.nii.gz"
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
        WARP="$TEMPLATE_DIR/template${FA_BASE}1Warp.nii.gz"
        AFFINE="$TEMPLATE_DIR/template${FA_BASE}0GenericAffine.mat"
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

echo ""
echo "================================================================"
echo "  FA-template pipeline complete! Output: $OUT"
echo "================================================================"
