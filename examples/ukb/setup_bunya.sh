#!/bin/bash
# ==========================================================================
# One-time setup for rish-harmonize on Bunya
#
# Run this interactively (not via SLURM) before submitting pipeline jobs:
#   bash setup_bunya.sh
#
# Creates:
#   /scratch/user/uqahonne/ukb/bed_analysis/rish_harmonize/
#   /scratch/user/uqahonne/ukb/bed_analysis/rish_harmonize/venv/  (Python venv)
#   /scratch/user/uqahonne/ukb/bed_analysis/rish_harmonize/logs/
# ==========================================================================

set -euo pipefail

PROJECT="/scratch/user/uqahonne/ukb/bed_analysis/rish_harmonize"

echo "================================================================"
echo "  Setting up rish-harmonize on Bunya"
echo "================================================================"

# 1. Create project directories
echo "Creating project directories..."
mkdir -p "$PROJECT"/{logs,pipeline_output_fod,pipeline_output_fa}

# 2. Copy site info CSV
SITE_CSV_SRC="/home/uqahonne/uq/ukb/bids_data_bed_controls_recovered/ukb_site_info.csv"
if [[ -f "$SITE_CSV_SRC" ]]; then
    cp "$SITE_CSV_SRC" "$PROJECT/ukb_site_info.csv"
    echo "  Copied ukb_site_info.csv"
elif [[ -f "$PROJECT/ukb_site_info.csv" ]]; then
    echo "  ukb_site_info.csv already exists"
else
    echo "  WARNING: ukb_site_info.csv not found at $SITE_CSV_SRC"
    echo "  Copy it manually to $PROJECT/ukb_site_info.csv"
fi

# 3. Load MRtrix3 via Neurodesk
echo ""
echo "Loading Neurodesk MRtrix3..."
module purge
module use /sw/local/rocky8/noarch/neuro/software/neurocommand/local/containers/modules/
export APPTAINER_BINDPATH=/scratch,/QRISdata
ml mrtrix3
ml ants

echo "  MRtrix3: $(mrinfo --version 2>&1 | head -1)"
echo "  ANTs:    $(antsRegistration --version 2>&1 | head -1 || echo 'loaded')"

# 4. Create Python venv and install rish-harmonize
echo ""
echo "Setting up Python virtual environment..."
VENV="$PROJECT/venv"

if [[ -f "$VENV/bin/activate" ]]; then
    echo "  Venv already exists at $VENV"
    source "$VENV/bin/activate"
else
    python3 -m venv "$VENV"
    source "$VENV/bin/activate"
    pip install --upgrade pip
    echo "  Venv created at $VENV"
fi

# 5. Install/update rish-harmonize
echo ""
echo "Installing rish-harmonize..."
RISH_REPO="/scratch/user/uqahonne/ukb/bed_analysis/rish_harmonize/repo"
if [[ -d "$RISH_REPO" ]]; then
    echo "  Pulling latest changes..."
    git -C "$RISH_REPO" pull origin master
    pip install -e "$RISH_REPO"
    echo "  Installed from $RISH_REPO (editable)"
else
    echo "  WARNING: rish-harmonize repo not found at $RISH_REPO"
    echo "  Clone it and run: pip install -e /path/to/rish-harmonize"
fi

# 5b. Copy pipeline scripts to project directory
echo ""
echo "Copying pipeline scripts..."
cp "$RISH_REPO/examples/ukb/"*.sh "$PROJECT/"
cp "$RISH_REPO/examples/ukb/"*.slurm "$PROJECT/"
echo "  Scripts copied to $PROJECT"

# 6. Verify installation
echo ""
echo "Verifying installation..."
if command -v rish-harmonize &>/dev/null; then
    echo "  rish-harmonize: $(rish-harmonize --version 2>&1 || echo 'OK')"
else
    echo "  FAIL: rish-harmonize not found in PATH"
    exit 1
fi

if command -v mrconvert &>/dev/null; then
    echo "  mrconvert: OK"
else
    echo "  FAIL: mrconvert not found"
    exit 1
fi

if command -v antsApplyTransforms &>/dev/null; then
    echo "  antsApplyTransforms: OK"
else
    echo "  WARNING: ANTs not available (only needed for FA template approach)"
fi

# 7. Check QSIPrep subjects exist
echo ""
echo "Checking QSIPrep subjects..."
QSIPREP="/scratch/user/uqahonne/ukb/bed_analysis/qsiprep_output_bed_controls_recovered"
FOUND=0 MISSING=0
for SUBJ in sub-1022877 sub-1028566 sub-1526049 sub-1538152 sub-1556312 \
            sub-1563465 sub-1573362 sub-1591722 sub-1595060 sub-1598122 \
            sub-1598698 sub-1818155 sub-3291205 sub-3579967 sub-4124624; do
    if [[ -d "$QSIPREP/$SUBJ/ses-01/dwi" ]]; then
        ((FOUND++)) || true
    else
        echo "  MISSING: $SUBJ"
        ((MISSING++)) || true
    fi
done
echo "  Found: $FOUND/14, Missing: $MISSING"

echo ""
echo "================================================================"
echo "  Setup complete!"
echo ""
echo "  Submit FOD pipeline:  sbatch submit_fod_pipeline.slurm"
echo "  Submit FA pipeline:   sbatch submit_fa_pipeline.slurm"
echo "  Or run a single step: sbatch submit_fod_pipeline.slurm 3"
echo "================================================================"
