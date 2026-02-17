"""
Spherical Harmonic Fitting Module

Wrappers for MRtrix3's amp2sh (DWI → SH) and sh2amp (SH → DWI)
for per-shell signal-level SH processing.
"""

import math
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np


def _run_cmd(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command."""
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\nstderr: {result.stderr.strip()}"
        )
    return result


def determine_lmax(n_directions: int) -> int:
    """Determine maximum SH order from number of gradient directions.

    The number of SH coefficients must not exceed the number of
    directions: n_coeffs = (lmax+1)(lmax+2)/2 <= n_directions.

    Parameters
    ----------
    n_directions : int
        Number of diffusion-weighted directions (excluding b=0).

    Returns
    -------
    int
        Maximum even SH order.
    """
    # Solve (lmax+1)(lmax+2)/2 <= n_directions
    # lmax^2 + 3*lmax + 2 <= 2*n_directions
    # lmax <= (-3 + sqrt(9 - 8 + 8*n_directions)) / 2
    lmax = int((-3 + math.sqrt(1 + 8 * n_directions)) / 2)
    # Ensure even
    if lmax % 2 != 0:
        lmax -= 1
    return max(lmax, 0)


def fit_sh(
    dwi_shell: str,
    output: str,
    lmax: Optional[int] = None,
    mask: Optional[str] = None,
    normalise: bool = False,
    n_threads: int = 1,
) -> str:
    """Fit spherical harmonics to single-shell DWI signal.

    Wrapper for MRtrix3's `amp2sh`.

    Parameters
    ----------
    dwi_shell : str
        Input single-shell DWI (b=0 + DW volumes).
    output : str
        Output path for SH coefficient image.
    lmax : int, optional
        Maximum SH order. If None, auto-determined from
        number of directions.
    mask : str, optional
        Brain mask.
    normalise : bool
        Normalise signal to b=0 before fitting.
    n_threads : int
        Number of threads.

    Returns
    -------
    str
        Path to SH coefficient image.
    """
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    cmd = ["amp2sh", dwi_shell, output, "-force"]

    if lmax is not None:
        cmd.extend(["-lmax", str(lmax)])
    if mask:
        cmd.extend(["-mask", mask])
    if normalise:
        cmd.append("-normalise")
    if n_threads > 1:
        cmd.extend(["-nthreads", str(n_threads)])

    _run_cmd(cmd)
    return output


def reconstruct_signal(
    sh_image: str,
    directions: str,
    output: str,
    n_threads: int = 1,
) -> str:
    """Reconstruct DWI signal from SH coefficients.

    Wrapper for MRtrix3's `sh2amp`.

    Parameters
    ----------
    sh_image : str
        Input SH coefficient image.
    directions : str
        Gradient directions file (MRtrix format or spherical coords).
    output : str
        Output path for reconstructed DWI.
    n_threads : int
        Number of threads.

    Returns
    -------
    str
        Path to reconstructed DWI image.
    """
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    cmd = ["sh2amp", sh_image, directions, output, "-force"]
    if n_threads > 1:
        cmd.extend(["-nthreads", str(n_threads)])

    _run_cmd(cmd)
    return output


def get_directions(
    dwi_image: str,
    output: str,
) -> str:
    """Extract gradient directions from a DWI image header.

    Exports the gradient table in MRtrix format for use with sh2amp.

    Parameters
    ----------
    dwi_image : str
        Input DWI image.
    output : str
        Output path for gradient directions file.

    Returns
    -------
    str
        Path to directions file.
    """
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    _run_cmd([
        "mrinfo", dwi_image,
        "-export_grad_mrtrix", output,
    ])

    return output


def get_n_directions(dwi_image: str, b0_threshold: float = 50.0) -> int:
    """Count number of diffusion-weighted directions (excluding b=0).

    Parameters
    ----------
    dwi_image : str
        Input DWI image.
    b0_threshold : float
        B-values below this are considered b=0.

    Returns
    -------
    int
        Number of DW directions.
    """
    result = _run_cmd(["mrinfo", "-dwgrad", dwi_image])
    lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]

    n_dw = 0
    for line in lines:
        parts = line.split()
        if len(parts) >= 4:
            b = float(parts[3])
            if b >= b0_threshold:
                n_dw += 1

    return n_dw
