#!/bin/bash
# Test signal-level SH RISH extraction in native space
# Uses SDSU-TS dataset (ds005664)

set -euo pipefail

# MRtrix3 from conda env
export PATH="/home/uqahonne/anaconda3/envs/mrtrix3_env/bin:$PATH"

PROC="/home/uqahonne/uq/nif/mrtrix-rish/examples/ds005664/processing"
OUT="/home/uqahonne/uq/rish-harmonize/examples/ds005664/test_native"
mkdir -p "$OUT"

# Pick one subject, both sites
SUBJ="sub-ts001"
SITES=("ses-sdsu1" "ses-cfmri1")

for SITE in "${SITES[@]}"; do
    DWI="$PROC/$SUBJ/$SITE/dwi.mif"
    MASK="$PROC/$SUBJ/$SITE/mask.mif"
    SITE_OUT="$OUT/$SUBJ/$SITE"
    mkdir -p "$SITE_OUT"

    echo "=== $SUBJ / $SITE ==="
    echo "1. Detecting shells..."
    rish-harmonize detect-shells "$DWI"

    echo ""
    echo "2. Separating b=1500 shell (DW-only, no b=0)..."
    mrconvert "$DWI" -coord 3 $(python3 -c "
from rish_harmonize.core.shells import detect_shells
info = detect_shells('$DWI')
for b in info.b_values:
    indices = info.shell_indices[b]
    print(','.join(str(i) for i in indices))
    break  # just first shell
") "$SITE_OUT/dwi_b1500_dw.mif" -force
    echo "   Separated: $(mrinfo "$SITE_OUT/dwi_b1500_dw.mif" -size)"

    echo ""
    echo "3. Fitting SH (amp2sh) to b=1500 shell..."
    # 47 directions -> lmax=8 (45 coeffs)
    amp2sh "$SITE_OUT/dwi_b1500_dw.mif" "$SITE_OUT/sh_b1500.mif" -lmax 8 -force
    echo "   SH image: $(mrinfo "$SITE_OUT/sh_b1500.mif" -size)"

    echo ""
    echo "4. Extracting RISH features..."
    rish-harmonize extract-rish "$SITE_OUT/sh_b1500.mif" -o "$SITE_OUT/rish_b1500" --lmax 8 --mask "$MASK"
    echo ""

    echo "5. Checking RISH l0 stats..."
    mrstats "$SITE_OUT/rish_b1500/rish_l0.mif" -mask "$MASK"
    echo ""
done

echo "=== Comparing RISH l0 across sites ==="
echo "SDSU:"
mrstats "$OUT/$SUBJ/ses-sdsu1/rish_b1500/rish_l0.mif" -mask "$PROC/$SUBJ/ses-sdsu1/mask.mif"
echo "CFMRI:"
mrstats "$OUT/$SUBJ/ses-cfmri1/rish_b1500/rish_l0.mif" -mask "$PROC/$SUBJ/ses-cfmri1/mask.mif"
