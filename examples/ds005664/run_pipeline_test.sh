#!/bin/bash
# Full signal-level SH RISH harmonization pipeline test on SDSU-TS data
#
# Sites: ses-sdsu1 (b=1500/3000) and ses-cfmri1 (b=750/3000)
# Only b=3000 is common between sites.
#
# Reference site: ses-sdsu1
# Target site:    ses-cfmri1

set -euo pipefail
export PATH="/home/uqahonne/anaconda3/envs/mrtrix3_env/bin:$PATH"

PROC="/home/uqahonne/uq/nif/mrtrix-rish/examples/ds005664/processing"
# Note: DWI data and transforms live in the mrtrix-rish processing directory
OUT="/home/uqahonne/uq/rish-harmonize/examples/ds005664/pipeline_test"
TEMPLATE_GRID="$PROC/sub-ts001/ses-sdsu1/mask.mif"

SUBJECTS=(sub-ts001 sub-ts002 sub-ts003 sub-ts004 sub-ts005 sub-ts007 sub-ts008 sub-ts009 sub-ts010)
REF_SITE="ses-sdsu1"
TARGET_SITE="ses-cfmri1"

mkdir -p "$OUT"

# Allow running individual steps: ./run_pipeline_test.sh [step]
STEP="${1:-all}"

# ======================================================================
# Step 1: Create DWI list for consistent lmax
# ======================================================================
if [[ "$STEP" == "all" || "$STEP" == "1" ]]; then
echo ""
echo "================================================================"
echo "  Step 1: Creating DWI list for consistent lmax"
echo "================================================================"

DWI_LIST="$OUT/all_dwis.txt"
> "$DWI_LIST"
for SUBJ in "${SUBJECTS[@]}"; do
    for SITE in "$REF_SITE" "$TARGET_SITE"; do
        echo "$PROC/$SUBJ/$SITE/dwi.mif" >> "$DWI_LIST"
    done
done
echo "  DWI list: $DWI_LIST ($(wc -l < "$DWI_LIST") entries)"
fi

# ======================================================================
# Step 2: Extract native RISH for all subjects, both sites
# ======================================================================
if [[ "$STEP" == "all" || "$STEP" == "2" ]]; then
echo ""
echo "================================================================"
echo "  Step 2: Extracting native RISH features"
echo "================================================================"

for SUBJ in "${SUBJECTS[@]}"; do
    for SITE in "$REF_SITE" "$TARGET_SITE"; do
        RISH_DIR="$OUT/native_rish/$SUBJ/$SITE"
        if [[ -f "$RISH_DIR/shell_meta.json" ]]; then
            echo "  [$SUBJ/$SITE] Already extracted, skipping"
            continue
        fi
        echo "  [$SUBJ/$SITE] Extracting..."
        rish-harmonize extract-native-rish \
            "$PROC/$SUBJ/$SITE/dwi.mif" \
            -o "$RISH_DIR" \
            --mask "$PROC/$SUBJ/$SITE/mask.mif" \
            --consistent-with "$OUT/all_dwis.txt"
    done
done
echo "  Done."
fi

# ======================================================================
# Step 3: Warp RISH to template space (using affine transforms)
# ======================================================================
if [[ "$STEP" == "all" || "$STEP" == "3" ]]; then
echo ""
echo "================================================================"
echo "  Step 3: Warping RISH to template space"
echo "================================================================"

for SUBJ in "${SUBJECTS[@]}"; do
    for SITE in "$REF_SITE" "$TARGET_SITE"; do
        AFFINE="$PROC/$SUBJ/$SITE/affine.txt"
        for BDIR in "$OUT/native_rish/$SUBJ/$SITE"/b*/; do
            BVAL=$(basename "$BDIR")
            TEMPLATE_RISH_DIR="$OUT/template_rish/$SUBJ/$SITE/$BVAL/rish"
            mkdir -p "$TEMPLATE_RISH_DIR"
            for RISH_FILE in "$BDIR/rish"/rish_l*.mif; do
                FNAME=$(basename "$RISH_FILE")
                OUTFILE="$TEMPLATE_RISH_DIR/$FNAME"
                if [[ -f "$OUTFILE" ]]; then
                    continue
                fi
                mrtransform "$RISH_FILE" \
                    -linear "$AFFINE" \
                    -template "$TEMPLATE_GRID" \
                    -interp linear \
                    "$OUTFILE" \
                    -force -quiet
            done
        done
        echo "  [$SUBJ/$SITE] Warped"
    done
done
echo "  Done."
fi

# ======================================================================
# Step 4: Show pre-harmonization RISH l0 comparison (template space, b=3000)
# ======================================================================
if [[ "$STEP" == "all" || "$STEP" == "4" ]]; then
echo ""
echo "================================================================"
echo "  Step 4: Pre-harmonization RISH l0 comparison (b=3000)"
echo "================================================================"

TEMPLATE_MASK="$PROC/sub-ts001/ses-sdsu1/mask.mif"

echo ""
echo "  Subject        ses-sdsu1 mean    ses-cfmri1 mean    ratio"
echo "  ----------     --------------    ---------------    -----"
for SUBJ in "${SUBJECTS[@]}"; do
    SDSU_MEAN=$(mrstats "$OUT/template_rish/$SUBJ/$REF_SITE/b3000/rish/rish_l0.mif" \
        -mask "$TEMPLATE_MASK" -output mean 2>/dev/null | tr -d ' ')
    CFMRI_MEAN=$(mrstats "$OUT/template_rish/$SUBJ/$TARGET_SITE/b3000/rish/rish_l0.mif" \
        -mask "$TEMPLATE_MASK" -output mean 2>/dev/null | tr -d ' ')
    RATIO=$(python3 -c "print(f'{float('$CFMRI_MEAN') / float('$SDSU_MEAN'):.2f}')")
    printf "  %-14s  %14.2f    %15.2f    %s\n" "$SUBJ" "$SDSU_MEAN" "$CFMRI_MEAN" "$RATIO"
done
fi

# ======================================================================
# Step 5: Create reference template from ses-sdsu1 (b=3000 only)
# ======================================================================
if [[ "$STEP" == "all" || "$STEP" == "5" ]]; then
echo ""
echo "================================================================"
echo "  Step 5: Creating reference template"
echo "================================================================"

# Write list of ref-site RISH directories (template space)
REF_RISH_LIST="$OUT/ref_rish_dirs.txt"
> "$REF_RISH_LIST"
for SUBJ in "${SUBJECTS[@]}"; do
    echo "$OUT/template_rish/$SUBJ/$REF_SITE" >> "$REF_RISH_LIST"
done

rish-harmonize create-template --mode signal \
    --rish-list "$REF_RISH_LIST" \
    -o "$OUT/template/"

echo "  Done."
fi

# ======================================================================
# Step 6: Compute scale maps for each target-site subject
# ======================================================================
if [[ "$STEP" == "all" || "$STEP" == "6" ]]; then
echo ""
echo "================================================================"
echo "  Step 6: Computing scale maps (template space)"
echo "================================================================"

TEMPLATE_MASK="$PROC/sub-ts001/ses-sdsu1/mask.mif"

for SUBJ in "${SUBJECTS[@]}"; do
    echo "  [$SUBJ] Computing scale maps..."
    rish-harmonize compute-scale-maps \
        --ref-rish "$OUT/template/" \
        --target-rish "$OUT/template_rish/$SUBJ/$TARGET_SITE/" \
        -o "$OUT/scale_maps_template/$SUBJ/" \
        --mask "$TEMPLATE_MASK"
done
echo "  Done."
fi

# ======================================================================
# Step 7: Warp scale maps back to native space
# ======================================================================
if [[ "$STEP" == "all" || "$STEP" == "7" ]]; then
echo ""
echo "================================================================"
echo "  Step 7: Warping scale maps to native space"
echo "================================================================"

for SUBJ in "${SUBJECTS[@]}"; do
    AFFINE="$PROC/$SUBJ/$TARGET_SITE/affine.txt"
    NATIVE_MASK="$PROC/$SUBJ/$TARGET_SITE/mask.mif"
    for BDIR in "$OUT/scale_maps_template/$SUBJ"/b*/; do
        BVAL=$(basename "$BDIR")
        NATIVE_SCALE_DIR="$OUT/scale_maps_native/$SUBJ/$BVAL"
        mkdir -p "$NATIVE_SCALE_DIR"
        for SCALE_FILE in "$BDIR"/scale_l*.mif; do
            FNAME=$(basename "$SCALE_FILE")
            OUTFILE="$NATIVE_SCALE_DIR/$FNAME"
            # Inverse warp scale map to native space
            mrtransform "$SCALE_FILE" \
                -linear "$AFFINE" -inverse \
                -template "$NATIVE_MASK" \
                -interp linear \
                "$OUTFILE" \
                -force -quiet
            # Re-mask: set brain voxels to warped value, background to 1.0
            # scale_native = warped * mask + 1.0 * (1 - mask)
            mrcalc "$OUTFILE" "$NATIVE_MASK" -mult \
                "$NATIVE_MASK" 1 -sub -neg -add \
                "$OUTFILE" -force -quiet
        done
    done
    echo "  [$SUBJ] Warped + re-masked"
done

# Create scale_maps_meta.json for each subject's native scale maps
for SUBJ in "${SUBJECTS[@]}"; do
    python3 -c "
import json, os
from pathlib import Path
base = Path('$OUT/scale_maps_native/$SUBJ')
scale_maps = {}
for shell_dir in sorted(base.iterdir()):
    if not shell_dir.is_dir() or not shell_dir.name.startswith('b'):
        continue
    b = int(shell_dir.name[1:])
    orders = {}
    for f in sorted(shell_dir.iterdir()):
        if f.name.startswith('scale_l') and f.suffix == '.mif':
            order = int(f.stem.split('_l')[1])
            orders[order] = str(f)
    if orders:
        scale_maps[b] = orders
meta = {'scale_maps': {str(b): {str(o): p for o, p in ords.items()} for b, ords in scale_maps.items()}}
with open(base / 'scale_maps_meta.json', 'w') as f:
    json.dump(meta, f, indent=2)
"
done
echo "  Done."
fi

# ======================================================================
# Step 8: Show scale map statistics
# ======================================================================
if [[ "$STEP" == "all" || "$STEP" == "8" ]]; then
echo ""
echo "================================================================"
echo "  Step 8: Scale map statistics (b=3000, l=0)"
echo "================================================================"

TEMPLATE_MASK="$PROC/sub-ts001/ses-sdsu1/mask.mif"

echo ""
echo "  Subject        mean scale    median scale"
echo "  ----------     ----------    ------------"
for SUBJ in "${SUBJECTS[@]}"; do
    MEAN=$(mrstats "$OUT/scale_maps_template/$SUBJ/b3000/scale_l0.mif" \
        -mask "$TEMPLATE_MASK" -output mean 2>/dev/null | tr -d ' ')
    MEDIAN=$(mrstats "$OUT/scale_maps_template/$SUBJ/b3000/scale_l0.mif" \
        -mask "$TEMPLATE_MASK" -output median 2>/dev/null | tr -d ' ')
    printf "  %-14s  %10.4f    %12.4f\n" "$SUBJ" "$MEAN" "$MEDIAN"
done
fi

echo ""
echo "================================================================"
echo "  Pipeline test complete!"
echo "  Output: $OUT"
echo "================================================================"
