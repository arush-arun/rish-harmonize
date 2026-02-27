#!/bin/bash
# ==========================================================================
# UKB RISH-GLM pipeline — FOD-based population template (MRtrix)
#
# Builds an unbiased template from WM FODs using MRtrix population_template.
# Preserves crossing fiber information for anatomically accurate registration.
#
# Steps:
#   1. Convert DWI to .mif
#   2. Estimate response functions & compute MSMT-CSD FODs
#   3. Multi-tissue intensity normalisation (mtnormalise)
#   4. Build FOD population template (MRtrix)
#   5. Extract native RISH
#   6. Warp RISH to template (MRtrix warps)
#   7. Create group mask
#   8. Run RISH-GLM
#   9. Verify
#  10. Site effect comparison (pre vs post harmonization)
#
# Usage: ./run_fod_template.sh [step]
# ==========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/config.sh"

OUT="/home/uqahonne/uq/rish-harmonize/examples/ukb/pipeline_output_fod"
mkdir -p "$OUT"

STEP="${1:-all}"

# ==========================================================================
# Step 1: Convert DWI to .mif
# ==========================================================================
if [[ "$STEP" == "all" || "$STEP" == "1" ]]; then
    step_convert_dwi "$OUT"
fi

# ==========================================================================
# Step 2: Estimate response functions & compute MSMT-CSD FODs
# ==========================================================================
if [[ "$STEP" == "all" || "$STEP" == "2" ]]; then
echo ""
echo "================================================================"
echo "  Step 2: Estimating response functions & computing FODs"
echo "================================================================"

RESPONSE_DIR="$OUT/responses"
FOD_DIR="$OUT/fods"
mkdir -p "$RESPONSE_DIR" "$FOD_DIR"

# 2a. Estimate per-subject tissue response functions
echo "  --- Estimating per-subject response functions ---"
for SUBJ in "${SUBJECTS[@]}"; do
    WM_RESP="$RESPONSE_DIR/${SUBJ}_wm.txt"
    GM_RESP="$RESPONSE_DIR/${SUBJ}_gm.txt"
    CSF_RESP="$RESPONSE_DIR/${SUBJ}_csf.txt"

    if [[ -f "$WM_RESP" && -f "$GM_RESP" && -f "$CSF_RESP" ]]; then
        echo "  [$SUBJ] Response functions already exist, skipping"
        continue
    fi

    dwi2response dhollander \
        "$OUT/mif/$SUBJ/dwi.mif" \
        "$WM_RESP" "$GM_RESP" "$CSF_RESP" \
        -mask "$OUT/mif/$SUBJ/mask.mif" \
        -nthreads "$NTHREADS" -force -quiet

    echo "  [$SUBJ] Response functions estimated"
done

# 2b. Average response functions across subjects
echo "  --- Averaging response functions ---"
AVG_WM="$RESPONSE_DIR/avg_wm.txt"
AVG_GM="$RESPONSE_DIR/avg_gm.txt"
AVG_CSF="$RESPONSE_DIR/avg_csf.txt"

if [[ -f "$AVG_WM" && -f "$AVG_GM" && -f "$AVG_CSF" ]]; then
    echo "  Average responses already exist, skipping"
else
    WM_LIST=() GM_LIST=() CSF_LIST=()
    for SUBJ in "${SUBJECTS[@]}"; do
        WM_LIST+=("$RESPONSE_DIR/${SUBJ}_wm.txt")
        GM_LIST+=("$RESPONSE_DIR/${SUBJ}_gm.txt")
        CSF_LIST+=("$RESPONSE_DIR/${SUBJ}_csf.txt")
    done

    responsemean "${WM_LIST[@]}" "$AVG_WM" -force -quiet
    responsemean "${GM_LIST[@]}" "$AVG_GM" -force -quiet
    responsemean "${CSF_LIST[@]}" "$AVG_CSF" -force -quiet
    echo "  Average response functions computed"
fi

# 2c. Compute MSMT-CSD FODs using averaged response functions
echo "  --- Computing MSMT-CSD FODs ---"
for SUBJ in "${SUBJECTS[@]}"; do
    SUBJ_FOD_DIR="$FOD_DIR/$SUBJ"
    mkdir -p "$SUBJ_FOD_DIR"

    WM_FOD="$SUBJ_FOD_DIR/wm_fod.mif"
    GM_FOD="$SUBJ_FOD_DIR/gm.mif"
    CSF_FOD="$SUBJ_FOD_DIR/csf.mif"

    if [[ -f "$WM_FOD" ]]; then
        echo "  [$SUBJ] FODs already exist, skipping"
        continue
    fi

    dwi2fod msmt_csd \
        "$OUT/mif/$SUBJ/dwi.mif" \
        "$AVG_WM" "$WM_FOD" \
        "$AVG_GM" "$GM_FOD" \
        "$AVG_CSF" "$CSF_FOD" \
        -mask "$OUT/mif/$SUBJ/mask.mif" \
        -nthreads "$NTHREADS" -force -quiet

    echo "  [$SUBJ] FODs computed"
done
echo "  Done."
fi

# ==========================================================================
# Step 3: Multi-tissue intensity normalisation
# ==========================================================================
if [[ "$STEP" == "all" || "$STEP" == "3" ]]; then
echo ""
echo "================================================================"
echo "  Step 3: Multi-tissue intensity normalisation (mtnormalise)"
echo "================================================================"

FOD_DIR="$OUT/fods"
NORM_DIR="$OUT/fods_normalised"
mkdir -p "$NORM_DIR"

for SUBJ in "${SUBJECTS[@]}"; do
    SUBJ_NORM_DIR="$NORM_DIR/$SUBJ"
    mkdir -p "$SUBJ_NORM_DIR"

    WM_NORM="$SUBJ_NORM_DIR/wm_fod.mif"
    GM_NORM="$SUBJ_NORM_DIR/gm.mif"
    CSF_NORM="$SUBJ_NORM_DIR/csf.mif"

    if [[ -f "$WM_NORM" ]]; then
        echo "  [$SUBJ] Already normalised, skipping"
        continue
    fi

    mtnormalise \
        "$FOD_DIR/$SUBJ/wm_fod.mif" "$WM_NORM" \
        "$FOD_DIR/$SUBJ/gm.mif" "$GM_NORM" \
        "$FOD_DIR/$SUBJ/csf.mif" "$CSF_NORM" \
        -mask "$OUT/mif/$SUBJ/mask.mif" \
        -nthreads "$NTHREADS" -force -quiet

    echo "  [$SUBJ] Normalised"
done
echo "  Done."
fi

# ==========================================================================
# Step 4: Build FOD population template (MRtrix)
# ==========================================================================
if [[ "$STEP" == "all" || "$STEP" == "4" ]]; then
echo ""
echo "================================================================"
echo "  Step 4: Building FOD population template (MRtrix)"
echo "  This may take several hours..."
echo "================================================================"

TEMPLATE_DIR="$OUT/population_template"
WARP_DIR="$OUT/warps"
MASK_DIR_TMPL="$OUT/template_masks"
mkdir -p "$TEMPLATE_DIR" "$WARP_DIR" "$MASK_DIR_TMPL"

if [[ -f "$TEMPLATE_DIR/fod_template.mif" ]]; then
    echo "  Template already exists, skipping"
else
    # Create input directory structure for population_template
    # It expects a flat directory of FOD images
    TEMPLATE_INPUT="$TEMPLATE_DIR/input_fods"
    TEMPLATE_MASKS="$TEMPLATE_DIR/input_masks"
    mkdir -p "$TEMPLATE_INPUT" "$TEMPLATE_MASKS"

    for SUBJ in "${SUBJECTS[@]}"; do
        ln -sf "$OUT/fods_normalised/$SUBJ/wm_fod.mif" "$TEMPLATE_INPUT/${SUBJ}.mif"
        ln -sf "$OUT/mif/$SUBJ/mask.mif" "$TEMPLATE_MASKS/${SUBJ}.mif"
    done

    population_template \
        "$TEMPLATE_INPUT" \
        "$TEMPLATE_DIR/fod_template.mif" \
        -mask_dir "$TEMPLATE_MASKS" \
        -warp_dir "$WARP_DIR" \
        -template_mask "$MASK_DIR_TMPL/template_mask.mif" \
        -nthreads "$NTHREADS" \
        -force

    echo "  Template built: $TEMPLATE_DIR/fod_template.mif"
fi

mrinfo "$TEMPLATE_DIR/fod_template.mif" -size -spacing 2>/dev/null || true
fi

# ==========================================================================
# Step 5: Extract native RISH
# ==========================================================================
if [[ "$STEP" == "all" || "$STEP" == "5" ]]; then
    step_extract_rish "$OUT"
fi

# ==========================================================================
# Step 6: Warp RISH to template (MRtrix warps)
# ==========================================================================
if [[ "$STEP" == "all" || "$STEP" == "6" ]]; then
echo ""
echo "================================================================"
echo "  Step 6: Warping RISH to template (MRtrix warps)"
echo "================================================================"

WARP_DIR="$OUT/warps"
DEFORM_DIR="$OUT/warps_deformation"
TEMPLATE="$OUT/population_template/fod_template.mif"
mkdir -p "$DEFORM_DIR"

for SUBJ in "${SUBJECTS[@]}"; do
    SUBJ_WARP="$WARP_DIR/${SUBJ}.mif"

    if [[ ! -f "$SUBJ_WARP" ]]; then
        echo "  [$SUBJ] WARNING: Warp not found ($SUBJ_WARP), skipping"
        continue
    fi

    # Convert 5D warpfull to 4D deformation field (subject -> template)
    SUBJ_DEFORM="$DEFORM_DIR/${SUBJ}.mif"
    if [[ ! -f "$SUBJ_DEFORM" ]]; then
        warpconvert "$SUBJ_WARP" warpfull2deformation \
            -template "$TEMPLATE" \
            "$SUBJ_DEFORM" -force -quiet
    fi

    for BDIR in "$OUT/native_rish/$SUBJ"/b*/; do
        BVAL=$(basename "$BDIR")
        TEMPLATE_RISH_DIR="$OUT/template_rish/$SUBJ/$BVAL/rish"
        mkdir -p "$TEMPLATE_RISH_DIR"

        for RISH_FILE in "$BDIR/rish"/rish_l*.mif; do
            FNAME=$(basename "$RISH_FILE" .mif)
            OUTFILE="$TEMPLATE_RISH_DIR/${FNAME}.mif"
            [[ -f "$OUTFILE" ]] && continue

            mrtransform "$RISH_FILE" "$OUTFILE" \
                -warp "$SUBJ_DEFORM" \
                -interp linear \
                -nthreads "$NTHREADS" \
                -force -quiet
        done
    done
    echo "  [$SUBJ] Warped"
done
echo "  Done."
fi

# ==========================================================================
# Step 7: Create group mask
# ==========================================================================
if [[ "$STEP" == "all" || "$STEP" == "7" ]]; then
echo ""
echo "================================================================"
echo "  Step 7: Creating group mask"
echo "================================================================"

MASK_DIR="$OUT/template_masks"
mkdir -p "$MASK_DIR"

if [[ -f "$MASK_DIR/group_mask.mif" ]]; then
    echo "  Already exists, skipping"
else
    # population_template already produced a template_mask.mif
    # Use it directly as the group mask
    if [[ -f "$MASK_DIR/template_mask.mif" ]]; then
        cp "$MASK_DIR/template_mask.mif" "$MASK_DIR/group_mask.mif"
        NVOX=$(mrstats "$MASK_DIR/group_mask.mif" -output count -ignorezero 2>/dev/null)
        echo "  Group mask from population_template: $NVOX voxels"
    else
        # Fallback: warp individual masks and intersect
        echo "  template_mask.mif not found, warping individual masks..."
        DEFORM_DIR="$OUT/warps_deformation"
        MASK_ARGS=""
        for SUBJ in "${SUBJECTS[@]}"; do
            SUBJ_DEFORM="$DEFORM_DIR/${SUBJ}.mif"
            TMPL_MASK="$MASK_DIR/${SUBJ}_mask_template.mif"
            if [[ ! -f "$TMPL_MASK" ]]; then
                mrtransform "$OUT/mif/$SUBJ/mask.mif" "$TMPL_MASK" \
                    -warp "$SUBJ_DEFORM" \
                    -interp nearest \
                    -nthreads "$NTHREADS" \
                    -force -quiet
            fi
            MASK_ARGS="$MASK_ARGS $TMPL_MASK"
        done
        mrmath $MASK_ARGS min "$MASK_DIR/group_mask.mif" -force -quiet
        NVOX=$(mrstats "$MASK_DIR/group_mask.mif" -output count -ignorezero 2>/dev/null)
        echo "  Group mask (intersection): $NVOX voxels"
    fi
fi
fi

# ==========================================================================
# Step 8: Run RISH-GLM
# ==========================================================================
if [[ "$STEP" == "all" || "$STEP" == "8" ]]; then
    step_run_glm "$OUT" "$OUT/template_masks/group_mask.mif"
fi

# ==========================================================================
# Step 9: Verify
# ==========================================================================
if [[ "$STEP" == "all" || "$STEP" == "9" ]]; then
    step_verify "$OUT" "$OUT/template_masks/group_mask.mif"
fi

# ==========================================================================
# Step 10: Site effect comparison (pre vs post harmonization)
# ==========================================================================
if [[ "$STEP" == "all" || "$STEP" == "10" ]]; then
    step_site_effect_comparison "$OUT" "$OUT/template_masks/group_mask.mif"
fi

echo ""
echo "================================================================"
echo "  FOD-template pipeline complete! Output: $OUT"
echo "================================================================"
