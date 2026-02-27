#!/bin/bash
# ==========================================================================
# Shared configuration for UKB RISH-GLM pipelines
#
# 14 subjects across 4 UKB imaging sites (11025/11026/11027/11028)
# QSIPrep-preprocessed DWI in ACPC space, multi-shell (b=0/1000/2000)
# ==========================================================================

export PATH="/home/uqahonne/anaconda3/envs/mrtrix3_env/bin:$PATH"

QSIPREP="/home/uqahonne/uq/ukb/qsiprep_output_bed_controls"
SITE_CSV="/home/uqahonne/uq/ukb/bids_data_bed_controls_recovered/ukb_site_info.csv"
NTHREADS=4

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

# ==========================================================================
# Helpers: find QSIPrep files (handles concat vs acq-AP naming)
# ==========================================================================
find_dwi() {
    local SUBJ=$1
    local DWI_DIR="$QSIPREP/$SUBJ/ses-01/dwi"
    local CONCAT="$DWI_DIR/${SUBJ}_ses-01_space-ACPC_desc-preproc_dwi.nii.gz"
    local AP="$DWI_DIR/${SUBJ}_ses-01_acq-AP_space-ACPC_desc-preproc_dwi.nii.gz"
    if [[ -f "$CONCAT" ]]; then echo "$CONCAT"
    elif [[ -f "$AP" ]]; then echo "$AP"
    else echo "ERROR: No DWI for $SUBJ" >&2; return 1; fi
}

find_mask() {
    local SUBJ=$1
    local DWI_DIR="$QSIPREP/$SUBJ/ses-01/dwi"
    local CONCAT="$DWI_DIR/${SUBJ}_ses-01_space-ACPC_desc-brain_mask.nii.gz"
    local AP="$DWI_DIR/${SUBJ}_ses-01_acq-AP_space-ACPC_desc-brain_mask.nii.gz"
    if [[ -f "$CONCAT" ]]; then echo "$CONCAT"
    elif [[ -f "$AP" ]]; then echo "$AP"
    else echo "ERROR: No mask for $SUBJ" >&2; return 1; fi
}

# ==========================================================================
# Shared steps: convert DWI, extract RISH, run GLM, verify
# ==========================================================================
step_convert_dwi() {
    local OUT=$1
    echo ""
    echo "================================================================"
    echo "  Converting DWI to .mif"
    echo "================================================================"

    DWI_LIST="$OUT/all_dwis.txt"
    > "$DWI_LIST"
    for SUBJ in "${SUBJECTS[@]}"; do
        MIF_DIR="$OUT/mif/$SUBJ"
        mkdir -p "$MIF_DIR"
        if [[ ! -f "$MIF_DIR/dwi.mif" ]]; then
            DWI_NII=$(find_dwi "$SUBJ")
            BVAL="${DWI_NII%.nii.gz}.bval"
            BVEC="${DWI_NII%.nii.gz}.bvec"
            MASK_NII=$(find_mask "$SUBJ")
            mrconvert "$DWI_NII" "$MIF_DIR/dwi.mif" -fslgrad "$BVEC" "$BVAL" -force -quiet
            mrconvert "$MASK_NII" "$MIF_DIR/mask.mif" -force -quiet
            echo "  [$SUBJ] Converted"
        fi
        echo "$MIF_DIR/dwi.mif" >> "$DWI_LIST"
    done
    echo "  Done. DWI list: $(wc -l < "$DWI_LIST") entries"
}

step_extract_rish() {
    local OUT=$1
    echo ""
    echo "================================================================"
    echo "  Extracting native RISH features"
    echo "================================================================"

    DWI_LIST="$OUT/all_dwis.txt"
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
}

step_run_glm() {
    local OUT=$1
    local MASK=$2
    echo ""
    echo "================================================================"
    echo "  Running RISH-GLM"
    echo "================================================================"

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
        --mask "$MASK" \
        -o "$OUT/glm_output/"
}

step_verify() {
    local OUT=$1
    local MASK=$2
    echo ""
    echo "================================================================"
    echo "  Verifying outputs"
    echo "================================================================"

    GLM_DIR="$OUT/glm_output"

    if [[ -f "$GLM_DIR/glm/shell_lmax.json" ]]; then
        echo "  [OK] shell_lmax.json:"
        cat "$GLM_DIR/glm/shell_lmax.json"
    else
        echo "  [FAIL] shell_lmax.json missing"
    fi

    for SITE_DIR in "$GLM_DIR"/scale_maps/*/; do
        SITE=$(basename "$SITE_DIR")
        echo ""
        echo "  Scale maps for site $SITE:"
        if [[ -f "$SITE_DIR/scale_maps_meta.json" ]]; then
            echo "  [OK] scale_maps_meta.json exists"
            for SCALE_L0 in "$SITE_DIR"/b*/scale_l0_*.mif; do
                if [[ -f "$SCALE_L0" ]]; then
                    BSHELL=$(basename "$(dirname "$SCALE_L0")")
                    read -r MEAN MEDIAN STD <<< $(mrstats "$SCALE_L0" -mask "$MASK" -output mean,median,std 2>/dev/null)
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
}

step_site_effect_comparison() {
    local OUT=$1
    local MASK=$2
    local NPERM=${3:-1000}
    echo ""
    echo "================================================================"
    echo "  Site Effect Comparison (Pre vs Post Harmonization)"
    echo "================================================================"

    GLM_DIR="$OUT/glm_output"
    EFFECT_DIR="$OUT/site_effect_comparison"
    mkdir -p "$EFFECT_DIR"

    # Detect available b-shells from the first subject's template RISH
    FIRST_SUBJ="${SUBJECTS[0]}"
    for BVAL_DIR in "$OUT/template_rish/$FIRST_SUBJ"/b*/; do
        [[ ! -d "$BVAL_DIR" ]] && continue
        BVAL=$(basename "$BVAL_DIR")
        echo ""
        echo "  --- $BVAL ---"

        SHELL_DIR="$EFFECT_DIR/$BVAL"
        POST_DIR="$SHELL_DIR/post_harmonized_l0"
        mkdir -p "$SHELL_DIR" "$POST_DIR"

        # Create pre-harmonization site-list CSV
        PRE_CSV="$SHELL_DIR/pre_site_list.csv"
        echo "site,image_path" > "$PRE_CSV"
        for SUBJ in "${SUBJECTS[@]}"; do
            SITE="${SITE_MAP[$SUBJ]}"
            L0="$OUT/template_rish/$SUBJ/$BVAL/rish/rish_l0.mif"
            if [[ -f "$L0" ]]; then
                echo "${SITE},${L0}" >> "$PRE_CSV"
            else
                echo "  WARNING: Missing $L0"
            fi
        done

        # Create post-harmonization l0 images
        # Reference site subjects: unchanged
        # Target site subjects: multiply RISH l0 by scale map
        POST_CSV="$SHELL_DIR/post_site_list.csv"
        echo "site,image_path" > "$POST_CSV"
        for SUBJ in "${SUBJECTS[@]}"; do
            SITE="${SITE_MAP[$SUBJ]}"
            L0="$OUT/template_rish/$SUBJ/$BVAL/rish/rish_l0.mif"

            if [[ "$SITE" == "$REF_SITE" ]]; then
                POST_L0="$L0"
            else
                SCALE_MAP="$GLM_DIR/scale_maps/$SITE/$BVAL/scale_l0_${SITE}.mif"
                POST_L0="$POST_DIR/${SUBJ}_l0_harmonized.mif"
                if [[ ! -f "$POST_L0" && -f "$SCALE_MAP" && -f "$L0" ]]; then
                    mrcalc "$L0" "$SCALE_MAP" -mult "$POST_L0" -force -quiet
                fi
            fi

            if [[ -f "$POST_L0" ]]; then
                echo "${SITE},${POST_L0}" >> "$POST_CSV"
            fi
        done

        # Run site-effect tests
        echo "  Pre-harmonization ($NPERM permutations):"
        rish-harmonize site-effect \
            --site-list "$PRE_CSV" \
            --mask "$MASK" \
            -o "$SHELL_DIR/pre/" \
            --n-permutations "$NPERM"

        echo "  Post-harmonization ($NPERM permutations):"
        rish-harmonize site-effect \
            --site-list "$POST_CSV" \
            --mask "$MASK" \
            -o "$SHELL_DIR/post/" \
            --n-permutations "$NPERM"

        # Print comparison from saved JSON summaries
        PRE_JSON="$SHELL_DIR/pre/summary.json"
        POST_JSON="$SHELL_DIR/post/summary.json"
        if [[ -f "$PRE_JSON" && -f "$POST_JSON" ]]; then
            echo ""
            python3 -c "
import json, sys
with open('$PRE_JSON') as f: pre = json.load(f)
with open('$POST_JSON') as f: post = json.load(f)
print(f'  Comparison ($BVAL):')
print(f'    Pre:  {pre[\"percent_significant_permutation\"]:.1f}% significant, '
      f'mean eta²={pre[\"mean_effect_size\"]:.4f}')
print(f'    Post: {post[\"percent_significant_permutation\"]:.1f}% significant, '
      f'mean eta²={post[\"mean_effect_size\"]:.4f}')
pre_pct = pre['percent_significant_permutation']
post_pct = post['percent_significant_permutation']
if pre_pct > 0:
    red = (pre_pct - post_pct) / pre_pct * 100
    print(f'    Significant voxel reduction: {red:.1f}%')
pre_es = pre['mean_effect_size']
post_es = post['mean_effect_size']
if pre_es > 0:
    red_es = (pre_es - post_es) / pre_es * 100
    print(f'    Effect size reduction: {red_es:.1f}%')
"
        fi
    done

    echo ""
    echo "  Results saved to: $EFFECT_DIR"
}
