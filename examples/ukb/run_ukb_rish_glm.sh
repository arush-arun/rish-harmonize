#!/bin/bash
# ==========================================================================
# UK Biobank RISH-GLM harmonization pipeline
#
# 14 subjects across 4 UKB imaging sites (11025/11026/11027/11028)
# QSIPrep-preprocessed DWI in ACPC space
# Multi-shell: b=0/1000/2000
#
# Builds an unbiased study-specific population template from FA maps
# (following De Luca et al. 2025 RISH-GLM approach), then:
#   1. Compute FA maps for template building
#   2. Build unbiased population template (ANTs)
#   3. Extract native RISH (per-shell SH -> RISH)
#   4. Warp RISH to population template space
#   5. Create group brain mask in template space
#   6. Run RISH-GLM (signal_rish mode)
#   7. Verify outputs
# ==========================================================================

set -euo pipefail
export PATH="/home/uqahonne/anaconda3/envs/mrtrix3_env/bin:$PATH"

QSIPREP="/home/uqahonne/uq/ukb/qsiprep_output_bed_controls"
SITE_CSV="/home/uqahonne/uq/ukb/bids_data_bed_controls_recovered/ukb_site_info.csv"
OUT="/home/uqahonne/uq/rish-harmonize/examples/ukb/pipeline_output"
NTHREADS=4

mkdir -p "$OUT"

# 14 usable subjects with site assignments
declare -A SITE_MAP=(
    [sub-1028566]=11025 [sub-1526049]=11025 [sub-1538152]=11025
    [sub-1556312]=11025 [sub-1563465]=11025 [sub-1573362]=11025
    [sub-1598122]=11025
    [sub-1022877]=11027 [sub-1591722]=11027 [sub-4124624]=11027
    [sub-1595060]=11026 [sub-3579967]=11026 [sub-3291205]=11026
    [sub-1598698]=11028 [sub-1818155]=11028
)
SUBJECTS=(${!SITE_MAP[@]})
SUBJECTS=($(printf '%s\n' "${SUBJECTS[@]}" | sort))

REF_SITE="11025"  # Largest site as reference

# Allow running individual steps: ./run_ukb_rish_glm.sh [step]
STEP="${1:-all}"

# ==========================================================================
# Helper: find the main preprocessed DWI for a subject
# QSIPrep has two naming patterns:
#   - Concatenated: sub-{id}_ses-01_space-ACPC_desc-preproc_dwi.nii.gz
#   - Separate:     sub-{id}_ses-01_acq-AP_space-ACPC_desc-preproc_dwi.nii.gz
# ==========================================================================
find_dwi() {
    local SUBJ=$1
    local DWI_DIR="$QSIPREP/$SUBJ/ses-01/dwi"
    local CONCAT="$DWI_DIR/${SUBJ}_ses-01_space-ACPC_desc-preproc_dwi.nii.gz"
    local AP="$DWI_DIR/${SUBJ}_ses-01_acq-AP_space-ACPC_desc-preproc_dwi.nii.gz"
    if [[ -f "$CONCAT" ]]; then echo "$CONCAT"
    elif [[ -f "$AP" ]]; then echo "$AP"
    else echo "ERROR" >&2; return 1; fi
}

find_mask() {
    local SUBJ=$1
    local DWI_DIR="$QSIPREP/$SUBJ/ses-01/dwi"
    local CONCAT="$DWI_DIR/${SUBJ}_ses-01_space-ACPC_desc-brain_mask.nii.gz"
    local AP="$DWI_DIR/${SUBJ}_ses-01_acq-AP_space-ACPC_desc-brain_mask.nii.gz"
    if [[ -f "$CONCAT" ]]; then echo "$CONCAT"
    elif [[ -f "$AP" ]]; then echo "$AP"
    else echo "ERROR" >&2; return 1; fi
}

# ==========================================================================
# Step 1: Compute FA maps for template building
# ==========================================================================
if [[ "$STEP" == "all" || "$STEP" == "1" ]]; then
echo ""
echo "================================================================"
echo "  Step 1: Computing FA maps from QSIPrep DWI"
echo "================================================================"

FA_DIR="$OUT/fa_maps"
mkdir -p "$FA_DIR"

for SUBJ in "${SUBJECTS[@]}"; do
    FA_OUT="$FA_DIR/${SUBJ}_FA.nii.gz"
    if [[ -f "$FA_OUT" ]]; then
        echo "  [$SUBJ] FA already exists, skipping"
        continue
    fi

    DWI_NII=$(find_dwi "$SUBJ")
    BVAL="${DWI_NII%.nii.gz}.bval"
    BVEC="${DWI_NII%.nii.gz}.bvec"
    MASK_NII=$(find_mask "$SUBJ")

    # DWI -> tensor -> FA
    TMP_MIF="$FA_DIR/${SUBJ}_dwi.mif"
    TMP_DT="$FA_DIR/${SUBJ}_dt.mif"
    mrconvert "$DWI_NII" "$TMP_MIF" -fslgrad "$BVEC" "$BVAL" -force -quiet
    dwi2tensor "$TMP_MIF" "$TMP_DT" -mask "$MASK_NII" -force -quiet
    tensor2metric "$TMP_DT" -fa "$FA_OUT" -mask "$MASK_NII" -force -quiet
    rm -f "$TMP_MIF" "$TMP_DT"

    echo "  [$SUBJ] FA computed"
done
echo "  Done."
fi

# ==========================================================================
# Step 2: Build unbiased population template from FA maps (ANTs)
# ==========================================================================
if [[ "$STEP" == "all" || "$STEP" == "2" ]]; then
echo ""
echo "================================================================"
echo "  Step 2: Building unbiased population template (ANTs)"
echo "  This may take several hours..."
echo "================================================================"

TEMPLATE_DIR="$OUT/population_template"
mkdir -p "$TEMPLATE_DIR"

if [[ -f "$TEMPLATE_DIR/template0.nii.gz" ]]; then
    echo "  Template already exists, skipping"
else
    # Collect FA paths
    FA_LIST=""
    for SUBJ in "${SUBJECTS[@]}"; do
        FA_LIST="$FA_LIST $OUT/fa_maps/${SUBJ}_FA.nii.gz"
    done

    # ANTs multivariate template construction
    # -d 3: 3D images
    # -o: output prefix (inside template dir)
    # -n 0: no bias field correction (FA maps don't need it)
    # -r 1: use rigid-body for initial alignment
    # -i 4: 4 iterations
    # -m CC: cross-correlation similarity metric
    # -t SyN: SyN nonlinear registration
    # -j $NTHREADS: parallel threads
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

# Show template info
mrinfo "$TEMPLATE_DIR/template0.nii.gz" -size -spacing 2>/dev/null || true
fi

# ==========================================================================
# Step 3: Extract RISH features in native (ACPC) space
# ==========================================================================
if [[ "$STEP" == "all" || "$STEP" == "3" ]]; then
echo ""
echo "================================================================"
echo "  Step 3: Extracting native RISH features"
echo "================================================================"

# Create DWI list for consistent lmax
DWI_LIST="$OUT/all_dwis.txt"
> "$DWI_LIST"
for SUBJ in "${SUBJECTS[@]}"; do
    # Need .mif for rish-harmonize; convert if not done
    MIF_DIR="$OUT/mif/$SUBJ"
    mkdir -p "$MIF_DIR"
    if [[ ! -f "$MIF_DIR/dwi.mif" ]]; then
        DWI_NII=$(find_dwi "$SUBJ")
        BVAL="${DWI_NII%.nii.gz}.bval"
        BVEC="${DWI_NII%.nii.gz}.bvec"
        MASK_NII=$(find_mask "$SUBJ")
        mrconvert "$DWI_NII" "$MIF_DIR/dwi.mif" -fslgrad "$BVEC" "$BVAL" -force -quiet
        mrconvert "$MASK_NII" "$MIF_DIR/mask.mif" -force -quiet
        echo "  [$SUBJ] Converted to .mif"
    fi
    echo "$MIF_DIR/dwi.mif" >> "$DWI_LIST"
done
echo "  DWI list: $(wc -l < "$DWI_LIST") entries"

for SUBJ in "${SUBJECTS[@]}"; do
    RISH_DIR="$OUT/native_rish/$SUBJ"
    if [[ -f "$RISH_DIR/shell_meta.json" ]]; then
        echo "  [$SUBJ] Already extracted, skipping"
        continue
    fi

    echo "  [$SUBJ] Extracting..."
    rish-harmonize extract-native-rish \
        "$OUT/mif/$SUBJ/dwi.mif" \
        -o "$RISH_DIR" \
        --mask "$OUT/mif/$SUBJ/mask.mif" \
        --consistent-with "$DWI_LIST" \
        --threads "$NTHREADS"
done
echo "  Done."
fi

# ==========================================================================
# Step 4: Warp RISH to population template space
#
# ANTs template construction produces per-subject transforms:
#   template{SUBJ_FA}1Warp.nii.gz      (nonlinear warp)
#   template{SUBJ_FA}0GenericAffine.mat (affine)
# These warp subject -> template space.
# ==========================================================================
if [[ "$STEP" == "all" || "$STEP" == "4" ]]; then
echo ""
echo "================================================================"
echo "  Step 4: Warping RISH to population template space"
echo "================================================================"

TEMPLATE="$OUT/population_template/template0.nii.gz"
TEMPLATE_DIR="$OUT/population_template"

for SUBJ in "${SUBJECTS[@]}"; do
    # Find the ANTs transforms for this subject
    # ANTs names them after the input filename: template{basename}1Warp.nii.gz
    FA_BASE="${SUBJ}_FA"
    WARP="$TEMPLATE_DIR/template${FA_BASE}1Warp.nii.gz"
    AFFINE="$TEMPLATE_DIR/template${FA_BASE}0GenericAffine.mat"

    if [[ ! -f "$WARP" || ! -f "$AFFINE" ]]; then
        echo "  [$SUBJ] WARNING: Transforms not found, skipping"
        echo "    Expected: $WARP"
        echo "    Expected: $AFFINE"
        continue
    fi

    for BDIR in "$OUT/native_rish/$SUBJ"/b*/; do
        BVAL=$(basename "$BDIR")
        TEMPLATE_RISH_DIR="$OUT/template_rish/$SUBJ/$BVAL/rish"
        mkdir -p "$TEMPLATE_RISH_DIR"

        for RISH_FILE in "$BDIR/rish"/rish_l*.mif; do
            FNAME=$(basename "$RISH_FILE" .mif)
            OUTFILE="$TEMPLATE_RISH_DIR/${FNAME}.mif"

            if [[ -f "$OUTFILE" ]]; then
                continue
            fi

            # .mif -> .nii.gz -> ANTs warp -> .mif
            TMP_IN="/tmp/rish_${SUBJ}_${BVAL}_${FNAME}.nii.gz"
            TMP_OUT="/tmp/rish_${SUBJ}_${BVAL}_${FNAME}_template.nii.gz"
            mrconvert "$RISH_FILE" "$TMP_IN" -force -quiet

            antsApplyTransforms \
                -d 3 \
                -i "$TMP_IN" \
                -r "$TEMPLATE" \
                -t "$WARP" \
                -t "$AFFINE" \
                -o "$TMP_OUT" \
                --interpolation Linear

            mrconvert "$TMP_OUT" "$OUTFILE" -force -quiet
            rm -f "$TMP_IN" "$TMP_OUT"
        done
    done
    echo "  [$SUBJ] Warped to template"
done
echo "  Done."
fi

# ==========================================================================
# Step 5: Create group brain mask in template space
# ==========================================================================
if [[ "$STEP" == "all" || "$STEP" == "5" ]]; then
echo ""
echo "================================================================"
echo "  Step 5: Creating group brain mask in template space"
echo "================================================================"

TEMPLATE="$OUT/population_template/template0.nii.gz"
TEMPLATE_BUILD_DIR="$OUT/population_template"
MASK_DIR="$OUT/template_masks"
mkdir -p "$MASK_DIR"

if [[ -f "$MASK_DIR/group_mask.mif" ]]; then
    echo "  Group mask already exists, skipping"
else
    MASK_ARGS=""
    for SUBJ in "${SUBJECTS[@]}"; do
        FA_BASE="${SUBJ}_FA"
        WARP="$TEMPLATE_BUILD_DIR/template${FA_BASE}1Warp.nii.gz"
        AFFINE="$TEMPLATE_BUILD_DIR/template${FA_BASE}0GenericAffine.mat"
        NATIVE_MASK=$(find_mask "$SUBJ")
        MNI_MASK="$MASK_DIR/${SUBJ}_mask_template.nii.gz"

        if [[ ! -f "$MNI_MASK" ]]; then
            antsApplyTransforms \
                -d 3 \
                -i "$NATIVE_MASK" \
                -r "$TEMPLATE" \
                -t "$WARP" \
                -t "$AFFINE" \
                -o "$MNI_MASK" \
                --interpolation NearestNeighbor
        fi
        MASK_ARGS="$MASK_ARGS $MNI_MASK"
    done

    # Intersection of all masks (conservative group mask)
    mrmath $MASK_ARGS min "$MASK_DIR/group_mask.mif" -force -quiet
    NVOX=$(mrstats "$MASK_DIR/group_mask.mif" -output count -ignorezero 2>/dev/null)
    echo "  Group mask created (intersection of ${#SUBJECTS[@]} subjects, $NVOX voxels)"
fi
fi

# ==========================================================================
# Step 6: Run RISH-GLM (signal_rish mode)
# ==========================================================================
if [[ "$STEP" == "all" || "$STEP" == "6" ]]; then
echo ""
echo "================================================================"
echo "  Step 6: Running RISH-GLM"
echo "================================================================"

# Build manifest CSV
MANIFEST="$OUT/manifest_rish.csv"
echo "subject,site,rish_dir" > "$MANIFEST"
for SUBJ in "${SUBJECTS[@]}"; do
    SITE="${SITE_MAP[$SUBJ]}"
    echo "${SUBJ},${SITE},$OUT/template_rish/$SUBJ/" >> "$MANIFEST"
done

echo "  Manifest:"
cat "$MANIFEST"
echo ""
echo "  Reference site: $REF_SITE"
echo ""

rish-harmonize rish-glm \
    --manifest "$MANIFEST" \
    --reference-site "$REF_SITE" \
    --mask "$OUT/template_masks/group_mask.mif" \
    -o "$OUT/glm_output/"
fi

# ==========================================================================
# Step 7: Verify outputs and show statistics
# ==========================================================================
if [[ "$STEP" == "all" || "$STEP" == "7" ]]; then
echo ""
echo "================================================================"
echo "  Step 7: Verifying outputs"
echo "================================================================"

GROUP_MASK="$OUT/template_masks/group_mask.mif"
GLM_DIR="$OUT/glm_output"

if [[ -f "$GLM_DIR/glm/shell_lmax.json" ]]; then
    echo "  [OK] shell_lmax.json:"
    cat "$GLM_DIR/glm/shell_lmax.json"
else
    echo "  [FAIL] shell_lmax.json missing"
fi

# Scale map stats for each target site
for SITE_DIR in "$GLM_DIR"/scale_maps/*/; do
    SITE=$(basename "$SITE_DIR")
    echo ""
    echo "  Scale maps for site $SITE:"
    if [[ -f "$SITE_DIR/scale_maps_meta.json" ]]; then
        echo "  [OK] scale_maps_meta.json exists"
        for SCALE_L0 in "$SITE_DIR"/b*/scale_l0_*.mif; do
            if [[ -f "$SCALE_L0" ]]; then
                BSHELL=$(basename "$(dirname "$SCALE_L0")")
                read -r MEAN MEDIAN STD <<< $(mrstats "$SCALE_L0" -mask "$GROUP_MASK" -output mean,median,std 2>/dev/null)
                echo "    $BSHELL l0: mean=$MEAN median=$MEDIAN std=$STD"
            fi
        done
    else
        echo "  [FAIL] scale_maps_meta.json missing"
    fi
done

echo ""
echo "  Site distribution:"
for SITE in 11025 11026 11027 11028; do
    COUNT=0
    for SUBJ in "${SUBJECTS[@]}"; do
        [[ "${SITE_MAP[$SUBJ]}" == "$SITE" ]] && ((COUNT++)) || true
    done
    echo "    Site $SITE: $COUNT subjects"
done
fi

echo ""
echo "================================================================"
echo "  Pipeline complete!"
echo "  Output: $OUT"
echo "================================================================"
