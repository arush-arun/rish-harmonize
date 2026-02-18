#!/bin/bash
# Dry-run verification: RISH-GLM in signal_rish mode on ds005664 template-space RISH
#
# Sites: ses-sdsu1 (GE) and ses-cfmri1 (Siemens), common shell b=3000
# Reference site: ses-sdsu1

set -euo pipefail
export PATH="/home/uqahonne/anaconda3/envs/mrtrix3_env/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE_RISH="$SCRIPT_DIR/../template_rish"
TEMPLATE_MASK="/home/uqahonne/uq/nif/mrtrix-rish/examples/ds005664/processing/sub-ts001/ses-sdsu1/mask.mif"

SUBJECTS=(sub-ts001 sub-ts002 sub-ts003 sub-ts004 sub-ts005 sub-ts007 sub-ts008 sub-ts009 sub-ts010)

# Step 1: Create symlinked directories with only b3000 (common shell)
echo "Creating b3000-only RISH directories..."
for SUBJ in "${SUBJECTS[@]}"; do
    for SITE in ses-sdsu1 ses-cfmri1; do
        LINK_DIR="$SCRIPT_DIR/rish_b3000_only/$SUBJ/$SITE/b3000"
        mkdir -p "$LINK_DIR"
        ln -sfn "$TEMPLATE_RISH/$SUBJ/$SITE/b3000/rish" "$LINK_DIR/rish"
    done
done

# Step 2: Create manifest CSV
echo "Creating manifest..."
MANIFEST="$SCRIPT_DIR/manifest_rish.csv"
echo "subject,site,rish_dir" > "$MANIFEST"
for SUBJ in "${SUBJECTS[@]}"; do
    echo "${SUBJ}_sdsu1,ses-sdsu1,$SCRIPT_DIR/rish_b3000_only/$SUBJ/ses-sdsu1/" >> "$MANIFEST"
    echo "${SUBJ}_cfmri1,ses-cfmri1,$SCRIPT_DIR/rish_b3000_only/$SUBJ/ses-cfmri1/" >> "$MANIFEST"
done

# Step 3: Run RISH-GLM
echo ""
echo "Running rish-harmonize rish-glm (signal_rish mode)..."
rish-harmonize rish-glm \
    --manifest "$MANIFEST" \
    --reference-site ses-sdsu1 \
    --mask "$TEMPLATE_MASK" \
    -o "$SCRIPT_DIR/output/"

# Step 4: Verify outputs
echo ""
echo "================================================================"
echo "  Verifying outputs"
echo "================================================================"

# Check GLM model
if [[ -f "$SCRIPT_DIR/output/glm/shell_lmax.json" ]]; then
    echo "  [OK] shell_lmax.json exists"
    cat "$SCRIPT_DIR/output/glm/shell_lmax.json"
else
    echo "  [FAIL] shell_lmax.json missing"
fi

# Check scale maps
SCALE_DIR="$SCRIPT_DIR/output/scale_maps/ses-cfmri1"
if [[ -d "$SCALE_DIR" ]]; then
    echo "  [OK] Scale maps directory exists: $SCALE_DIR"
    echo "  Contents:"
    find "$SCALE_DIR" -type f | sort
else
    echo "  [FAIL] Scale maps directory missing"
fi

if [[ -f "$SCALE_DIR/scale_maps_meta.json" ]]; then
    echo "  [OK] scale_maps_meta.json exists"
    cat "$SCALE_DIR/scale_maps_meta.json"
else
    echo "  [FAIL] scale_maps_meta.json missing"
fi

# Check scale map stats
echo ""
echo "  Scale map statistics (b=3000, l=0):"
SCALE_L0=$(ls "$SCALE_DIR"/b3000/scale_l0_*.mif 2>/dev/null | head -1)
if [[ -n "$SCALE_L0" ]]; then
    mrstats "$SCALE_L0" -mask "$TEMPLATE_MASK"
else
    echo "  Could not find scale_l0 file"
fi

echo ""
echo "Done."
