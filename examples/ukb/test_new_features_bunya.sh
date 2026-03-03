#!/bin/bash
# ==========================================================================
# Test new rish-harmonize features on Bunya using existing UKB pipeline output
#
# Tests:
#   1. Update repo and reinstall rish-harmonize
#   2. Re-run RISH-GLM (step 7) to test scale map diagnostics, provenance,
#      and per-subject QC
#   3. Re-run site-effect comparison with --seed to test deterministic
#      permutation testing
#   4. Verify all new output files exist and contain expected fields
#
# Usage:
#   bash test_new_features_bunya.sh          # Run interactively
#   sbatch test_new_features_bunya.sh        # Or submit as SLURM job
#
#SBATCH --job-name=rish_test
#SBATCH --account=a_aibn
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/user/uqahonne/ukb/bed_analysis/rish_harmonize/logs/test_features_%j.out
#SBATCH --error=/scratch/user/uqahonne/ukb/bed_analysis/rish_harmonize/logs/test_features_%j.err
# ==========================================================================

set -euo pipefail

PROJECT="/scratch/user/uqahonne/ukb/bed_analysis/rish_harmonize"
RISH_REPO="$PROJECT/repo"
OUT="$PROJECT/pipeline_output_fod"
MASK="$OUT/template_masks/group_mask.mif"
NTHREADS=${SLURM_CPUS_PER_TASK:-8}
PASS=0
FAIL=0
TOTAL=0

# Color output helpers
green() { echo -e "\033[32m$*\033[0m"; }
red()   { echo -e "\033[31m$*\033[0m"; }

check() {
    ((TOTAL++)) || true
    if "$@"; then
        ((PASS++)) || true
        green "  [PASS] $*"
    else
        ((FAIL++)) || true
        red "  [FAIL] $*"
    fi
}

check_file() {
    ((TOTAL++)) || true
    if [[ -f "$1" ]]; then
        ((PASS++)) || true
        green "  [PASS] File exists: $(basename "$1")"
    else
        ((FAIL++)) || true
        red "  [FAIL] File missing: $1"
    fi
}

check_json_key() {
    local FILE=$1 KEY=$2
    ((TOTAL++)) || true
    if python3 -c "import json; d=json.load(open('$FILE')); assert '$KEY' in d, f'key $KEY not found'" 2>/dev/null; then
        ((PASS++)) || true
        green "  [PASS] $KEY in $(basename "$FILE")"
    else
        ((FAIL++)) || true
        red "  [FAIL] $KEY not in $(basename "$FILE")"
    fi
}

echo "================================================================"
echo "  Testing New rish-harmonize Features on Bunya"
echo "  Job ID: ${SLURM_JOB_ID:-interactive}"
echo "  Node:   $(hostname)"
echo "  Date:   $(date)"
echo "================================================================"

# ==========================================================================
# Step 0: Load environment and update rish-harmonize
# ==========================================================================
echo ""
echo "--- Step 0: Environment setup ---"

# Load Neurodesk modules
module purge
module use /sw/local/rocky8/noarch/neuro/software/neurocommand/local/containers/modules/
export APPTAINER_BINDPATH=/scratch,/QRISdata
ml mrtrix3
ml ants

# Activate venv
source "$PROJECT/venv/bin/activate"

# Update rish-harmonize from repo
echo "Updating rish-harmonize..."
cd "$RISH_REPO"
git pull origin master
pip install -e "$RISH_REPO" --quiet

echo "  rish-harmonize version: $(python3 -c 'from rish_harmonize import __version__; print(__version__)')"
echo "  MRtrix3: $(mrinfo --version 2>&1 | head -1)"

# Verify existing pipeline output exists
if [[ ! -d "$OUT/template_rish" ]]; then
    echo "ERROR: Pipeline output not found at $OUT/template_rish"
    echo "  Run the full FOD pipeline first"
    exit 1
fi
if [[ ! -f "$MASK" ]]; then
    echo "ERROR: Group mask not found at $MASK"
    exit 1
fi

# Source config for SITE_MAP and other helpers
source "$PROJECT/config_bunya.sh"

# ==========================================================================
# Step 1: Re-run RISH-GLM to test diagnostics + provenance + subject QC
# ==========================================================================
echo ""
echo "================================================================"
echo "  Step 1: Re-running RISH-GLM (testing new features)"
echo "================================================================"

# Clean previous GLM output to force re-computation
GLM_OUT="$OUT/glm_output_test"
rm -rf "$GLM_OUT"
mkdir -p "$GLM_OUT"

# Build manifest
MANIFEST="$GLM_OUT/manifest_rish.csv"
echo "subject,site,rish_dir" > "$MANIFEST"
for SUBJ in "${SUBJECTS[@]}"; do
    SITE="${SITE_MAP[$SUBJ]}"
    echo "${SUBJ},${SITE},$OUT/template_rish/$SUBJ/" >> "$MANIFEST"
done

echo "  Manifest: $(wc -l < "$MANIFEST") entries (incl. header)"
echo "  Reference site: $REF_SITE"
echo ""

rish-harmonize rish-glm \
    --manifest "$MANIFEST" \
    --reference-site "$REF_SITE" \
    --mask "$MASK" \
    -o "$GLM_OUT/"

echo ""
echo "--- Verifying RISH-GLM outputs ---"

# Check provenance in shell_lmax.json
SHELL_LMAX="$GLM_OUT/glm/shell_lmax.json"
check_file "$SHELL_LMAX"
if [[ -f "$SHELL_LMAX" ]]; then
    check_json_key "$SHELL_LMAX" "provenance"
    echo "  shell_lmax.json provenance:"
    python3 -c "import json; d=json.load(open('$SHELL_LMAX')); print(json.dumps(d.get('provenance',{}), indent=2))"
fi

# Check provenance in scale_maps_meta.json (for each target site)
for SITE_DIR in "$GLM_OUT"/scale_maps/*/; do
    SITE=$(basename "$SITE_DIR")
    META="$SITE_DIR/scale_maps_meta.json"
    check_file "$META"
    if [[ -f "$META" ]]; then
        check_json_key "$META" "provenance"
    fi

    # Check scale map diagnostics JSON
    DIAG="$SITE_DIR/scale_map_diagnostics.json"
    check_file "$DIAG"
    if [[ -f "$DIAG" ]]; then
        echo "  Scale map diagnostics for site $SITE:"
        python3 -c "
import json
with open('$DIAG') as f:
    diag = json.load(f)
for order, stats in diag.items():
    mean = stats.get('mean', 'N/A')
    pct_clip = stats.get('pct_clipped_total', 'N/A')
    print(f'    {order}: mean={mean:.3f}, clipped={pct_clip:.1f}%')
"
    fi
done

# Check per-subject QC files (only produced in 'signal' mode, not 'signal_rish')
echo ""
echo "--- Checking per-subject QC ---"
if ls "$GLM_OUT"/qc_subjects_b*.json &>/dev/null; then
    for QC_FILE in "$GLM_OUT"/qc_subjects_b*.json; do
        check_file "$QC_FILE"
        echo "  $(basename "$QC_FILE"):"
        python3 -c "
import json
with open('$QC_FILE') as f:
    qc = json.load(f)
print(f'    Subjects: {qc[\"n_subjects\"]}, Flagged: {qc[\"n_flagged\"]}')
if qc.get('flagged_subjects'):
    for s in qc['flagged_subjects']:
        print(f'    WARNING: {s}')
else:
    print('    No outliers flagged')
"
    done
else
    echo "  [SKIP] No qc_subjects_*.json (expected: signal_rish mode does not produce per-subject QC)"
fi

# ==========================================================================
# Step 2: Test deterministic permutation testing (site-effect)
# ==========================================================================
echo ""
echo "================================================================"
echo "  Step 2: Testing deterministic permutation testing"
echo "================================================================"

EFFECT_DIR="$OUT/site_effect_test"
rm -rf "$EFFECT_DIR"
mkdir -p "$EFFECT_DIR"

# Use the first available b-shell
FIRST_SUBJ="${SUBJECTS[0]}"
FIRST_BVAL_DIR=$(ls -d "$OUT/template_rish/$FIRST_SUBJ"/b*/ | head -1)
BVAL=$(basename "$FIRST_BVAL_DIR")

echo "  Testing with shell: $BVAL"

# Create site-list CSV for pre-harmonization RISH l0
SITE_CSV_TEST="$EFFECT_DIR/site_list.csv"
echo "site,image_path" > "$SITE_CSV_TEST"
for SUBJ in "${SUBJECTS[@]}"; do
    SITE="${SITE_MAP[$SUBJ]}"
    L0="$OUT/template_rish/$SUBJ/$BVAL/rish/rish_l0.mif"
    if [[ -f "$L0" ]]; then
        echo "${SITE},${L0}" >> "$SITE_CSV_TEST"
    fi
done

# Run 1: seed=42
echo ""
echo "  Run 1 (seed=42):"
rish-harmonize site-effect \
    --site-list "$SITE_CSV_TEST" \
    --mask "$MASK" \
    -o "$EFFECT_DIR/run1/" \
    --n-permutations 500 \
    --seed 42

# Run 2: same seed=42 — should produce identical results
echo ""
echo "  Run 2 (seed=42, should match run 1):"
rish-harmonize site-effect \
    --site-list "$SITE_CSV_TEST" \
    --mask "$MASK" \
    -o "$EFFECT_DIR/run2/" \
    --n-permutations 500 \
    --seed 42

# Compare
echo ""
echo "--- Comparing deterministic runs ---"
RUN1_JSON="$EFFECT_DIR/run1/summary.json"
RUN2_JSON="$EFFECT_DIR/run2/summary.json"

check_file "$RUN1_JSON"
check_file "$RUN2_JSON"

if [[ -f "$RUN1_JSON" && -f "$RUN2_JSON" ]]; then
    # Check seed is recorded
    check_json_key "$RUN1_JSON" "seed"
    check_json_key "$RUN1_JSON" "provenance"

    # Compare key statistics
    ((TOTAL++)) || true
    MATCH=$(python3 -c "
import json
with open('$RUN1_JSON') as f: r1 = json.load(f)
with open('$RUN2_JSON') as f: r2 = json.load(f)
match = True
for key in ['mean_effect_size', 'median_effect_size', 'percent_significant_permutation',
            'percent_significant_fdr', 'f_statistic', 'seed']:
    v1 = r1.get(key)
    v2 = r2.get(key)
    if v1 != v2:
        print(f'  MISMATCH: {key}: {v1} != {v2}')
        match = False
    else:
        print(f'  {key}: {v1} (identical)')
print('MATCH' if match else 'MISMATCH')
" 2>&1)
    echo "$MATCH"
    if echo "$MATCH" | grep -q "^MATCH$"; then
        ((PASS++)) || true
        green "  [PASS] Deterministic: two runs with seed=42 are identical"
    else
        ((FAIL++)) || true
        red "  [FAIL] Deterministic: runs differ despite same seed"
    fi

    # Run 3 with different seed — should differ
    echo ""
    echo "  Run 3 (seed=123, should differ):"
    rish-harmonize site-effect \
        --site-list "$SITE_CSV_TEST" \
        --mask "$MASK" \
        -o "$EFFECT_DIR/run3/" \
        --n-permutations 500 \
        --seed 123

    RUN3_JSON="$EFFECT_DIR/run3/summary.json"
    if [[ -f "$RUN3_JSON" ]]; then
        ((TOTAL++)) || true
        DIFF=$(python3 -c "
import json
with open('$RUN1_JSON') as f: r1 = json.load(f)
with open('$RUN3_JSON') as f: r3 = json.load(f)
# Effect sizes should be identical (not permutation-dependent)
# But p-values should differ
perm1 = r1.get('percent_significant_permutation')
perm3 = r3.get('percent_significant_permutation')
seed1 = r1.get('seed')
seed3 = r3.get('seed')
print(f'  seed=42: pct_signif={perm1}, seed={seed1}')
print(f'  seed=123: pct_signif={perm3}, seed={seed3}')
if seed1 != seed3:
    print('DIFFERENT_SEEDS')
else:
    print('SAME_SEED')
")
        echo "$DIFF"
        if echo "$DIFF" | grep -q "DIFFERENT_SEEDS"; then
            ((PASS++)) || true
            green "  [PASS] Different seeds recorded correctly"
        else
            ((FAIL++)) || true
            red "  [FAIL] Seeds not recorded differently"
        fi
    fi
fi

# ==========================================================================
# Summary
# ==========================================================================
echo ""
echo "================================================================"
echo "  Test Results Summary"
echo "================================================================"
echo "  Total:  $TOTAL"
echo "  Passed: $PASS"
echo "  Failed: $FAIL"
echo ""

if [[ $FAIL -eq 0 ]]; then
    green "  ALL TESTS PASSED"
    echo ""
    echo "  New output files to inspect:"
    echo "    $GLM_OUT/glm/shell_lmax.json  (provenance)"
    echo "    $GLM_OUT/scale_maps/*/scale_map_diagnostics.json"
    echo "    $GLM_OUT/qc_subjects_b*.json"
    echo "    $EFFECT_DIR/run1/summary.json  (seed + provenance)"
else
    red "  $FAIL TESTS FAILED"
fi

echo ""
echo "  Completed: $(date)"
echo "================================================================"
