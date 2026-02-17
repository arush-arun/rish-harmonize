#!/usr/bin/env python3
"""Test signal-level SH RISH extraction in native space on SDSU-TS data."""

import os
import subprocess
from pathlib import Path

# Ensure MRtrix3 commands are available
os.environ["PATH"] = "/home/uqahonne/anaconda3/envs/mrtrix3_env/bin:" + os.environ["PATH"]

from rish_harmonize.core.shells import detect_shells, separate_shell, extract_b0
from rish_harmonize.core.sh_fitting import fit_sh, determine_lmax
from rish_harmonize.core.rish import extract_rish_features

PROC = Path("/home/uqahonne/uq/nif/mrtrix-rish/examples/ds005664/processing")
OUT = Path("/home/uqahonne/uq/rish-harmonize/examples/ds005664/test_native")
OUT.mkdir(parents=True, exist_ok=True)

SUBJ = "sub-ts001"
SITES = ["ses-sdsu1", "ses-cfmri1"]


def mrstats(image, mask):
    """Run mrstats and return output."""
    result = subprocess.run(
        ["mrstats", image, "-mask", mask],
        capture_output=True, text=True,
    )
    return result.stdout.strip()


for site in SITES:
    dwi = str(PROC / SUBJ / site / "dwi.mif")
    mask = str(PROC / SUBJ / site / "mask.mif")
    site_out = OUT / SUBJ / site
    site_out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {SUBJ} / {site}")
    print(f"{'='*60}")

    # 1. Detect shells
    print("\n1. Detecting shells...")
    shell_info = detect_shells(dwi)
    print(f"   b=0: {len(shell_info.b0_indices)} volumes")
    for b in shell_info.b_values:
        n = shell_info.n_directions(b)
        lmax = determine_lmax(n)
        print(f"   b={b}: {n} directions -> lmax={lmax}")

    # 2. Compute consistent lmax (single subject, so just auto)
    # Process each shell
    for b_value in shell_info.b_values:
        n_dirs = shell_info.n_directions(b_value)
        lmax = determine_lmax(n_dirs)
        shell_dir = site_out / f"b{b_value}"
        shell_dir.mkdir(exist_ok=True)

        # 3. Separate DW-only (no b=0)
        print(f"\n2. Separating b={b_value} shell (DW-only)...")
        shell_dwi = str(shell_dir / "dwi_dw_only.mif")
        separate_shell(dwi, shell_info, b_value, shell_dwi, include_b0=False)

        result = subprocess.run(
            ["mrinfo", shell_dwi, "-size"],
            capture_output=True, text=True,
        )
        print(f"   Size: {result.stdout.strip()}")

        # 4. Fit SH
        print(f"3. Fitting SH (amp2sh, lmax={lmax})...")
        sh_path = str(shell_dir / "sh.mif")
        fit_sh(shell_dwi, sh_path, lmax=lmax)

        result = subprocess.run(
            ["mrinfo", sh_path, "-size"],
            capture_output=True, text=True,
        )
        print(f"   SH image size: {result.stdout.strip()}")

        # 5. Extract RISH
        print("4. Extracting RISH features...")
        rish_dir = str(shell_dir / "rish")
        rish = extract_rish_features(sh_path, rish_dir, lmax=lmax, mask=mask)
        for order, path in sorted(rish.items()):
            print(f"   l={order}: {path}")

        # 6. Show RISH l0 stats
        print(f"5. RISH l0 stats (b={b_value}):")
        stats = mrstats(rish[0], mask)
        print(f"   {stats}")

# Compare RISH l0 across sites for b=1500
print(f"\n{'='*60}")
print("  Cross-site comparison: RISH l0 (b=1500)")
print(f"{'='*60}")
for site in SITES:
    rish_l0 = str(OUT / SUBJ / site / "b1500" / "rish" / "rish_l0.mif")
    mask = str(PROC / SUBJ / site / "mask.mif")
    print(f"\n{site}:")
    print(f"   {mrstats(rish_l0, mask)}")
