"""
RISH Feature Extraction Module

Extracts Rotationally Invariant Spherical Harmonic (RISH) features
from MRtrix3 SH coefficient images.

The RISH feature for order l is:
    θ_l = sqrt(sum_m |c_lm|^2)  for m in [-l, +l]

This is the "power" in each SH order, which is invariant to rotation.
"""

import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SHInfo:
    """Information about a spherical harmonics image."""
    lmax: int
    n_volumes: int
    n_coeffs_per_order: Dict[int, int]  # l -> number of coefficients (2l+1)
    volume_indices: Dict[int, Tuple[int, int]]  # l -> (start, end) indices


def get_sh_indices(lmax: int) -> SHInfo:
    """
    Get volume indices for each SH order.
    
    MRtrix3 stores only even-order SH coefficients (antipodal symmetry).
    Volume index for coefficient c_lm is: V_lm = l(l+1)/2 + m
    
    Parameters
    ----------
    lmax : int
        Maximum SH order (must be even)
        
    Returns
    -------
    SHInfo
        Dataclass with SH image information
        
    Examples
    --------
    >>> info = get_sh_indices(8)
    >>> info.volume_indices
    {0: (0, 1), 2: (1, 6), 4: (6, 15), 6: (15, 28), 8: (28, 45)}
    """
    if lmax % 2 != 0:
        raise ValueError(f"lmax must be even, got {lmax}")
    
    n_volumes = (lmax + 1) * (lmax + 2) // 2
    
    n_coeffs = {}
    indices = {}
    current_idx = 0
    
    for l in range(0, lmax + 1, 2):
        n_m = 2 * l + 1  # Number of m values: -l to +l
        n_coeffs[l] = n_m
        indices[l] = (current_idx, current_idx + n_m)
        current_idx += n_m
    
    return SHInfo(
        lmax=lmax,
        n_volumes=n_volumes,
        n_coeffs_per_order=n_coeffs,
        volume_indices=indices
    )


def run_mrtrix_cmd(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    """
    Run an MRtrix3 command.
    
    Parameters
    ----------
    cmd : list of str
        Command and arguments
    check : bool
        Raise exception on non-zero exit code
        
    Returns
    -------
    CompletedProcess
        Completed process result
    """
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=check
    )
    return result


def get_image_lmax(sh_image: str) -> int:
    """
    Determine lmax from an SH image.
    
    Parameters
    ----------
    sh_image : str
        Path to SH coefficient image
        
    Returns
    -------
    int
        Maximum SH order
    """
    result = run_mrtrix_cmd(["mrinfo", "-size", sh_image])
    sizes = result.stdout.strip().split()
    n_vols = int(sizes[3]) if len(sizes) >= 4 else int(sizes[-1])
    
    # Solve: n_vols = (lmax+1)(lmax+2)/2 for lmax
    # lmax^2 + 3*lmax + 2 - 2*n_vols = 0
    # lmax = (-3 + sqrt(9 - 8 + 8*n_vols)) / 2 = (-3 + sqrt(1 + 8*n_vols)) / 2
    import math
    lmax = int((-3 + math.sqrt(1 + 8 * n_vols)) / 2)
    
    return lmax


def extract_rish_features(
    sh_image: str,
    output_dir: str,
    lmax: Optional[int] = None,
    mask: Optional[str] = None,
    prefix: str = "rish",
    keep_intermediate: bool = False,
    n_threads: int = 1
) -> Dict[int, str]:
    """
    Extract RISH features from an SH coefficient image.
    
    For each SH order l, computes:
        θ_l = sqrt(sum_m |c_lm|^2)
    
    Parameters
    ----------
    sh_image : str
        Path to input SH coefficient image (from amp2sh)
    output_dir : str
        Directory to save output RISH feature images
    lmax : int, optional
        Maximum SH order. If None, determined from image.
    mask : str, optional
        Brain mask to apply
    prefix : str
        Prefix for output filenames
    keep_intermediate : bool
        Keep intermediate files (squared coefficients, etc.)
    n_threads : int
        Number of threads for MRtrix3 commands
        
    Returns
    -------
    dict
        Mapping from SH order (l) to output RISH feature image path
        
    Examples
    --------
    >>> rish_images = extract_rish_features(
    ...     "sh_coeffs.mif",
    ...     "output/",
    ...     lmax=8
    ... )
    >>> rish_images
    {0: 'output/rish_l0.mif', 2: 'output/rish_l2.mif', ...}
    """
    sh_image = Path(sh_image)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine lmax if not provided
    if lmax is None:
        lmax = get_image_lmax(str(sh_image))
    
    sh_info = get_sh_indices(lmax)
    
    # Thread option
    thread_opt = ["-nthreads", str(n_threads)] if n_threads > 1 else []
    
    rish_outputs = {}
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        for l, (start_idx, end_idx) in sh_info.volume_indices.items():
            output_file = output_dir / f"{prefix}_l{l}.mif"
            
            if l == 0:
                # θ_0 is just the DC component (volume 0)
                # But we take absolute value for consistency
                run_mrtrix_cmd([
                    "mrconvert", str(sh_image),
                    "-coord", "3", "0",
                    str(output_file),
                    "-force"
                ] + thread_opt)
                
            else:
                # For l > 0:
                # 1. Extract volumes for this order
                # 2. Square each coefficient
                # 3. Sum across m
                # 4. Take square root

                # MRtrix uses inclusive:inclusive indexing, so use end_idx-1
                vol_range = f"{start_idx}:{end_idx - 1}"
                extracted = tmpdir / f"l{l}_coeffs.mif"
                squared = tmpdir / f"l{l}_squared.mif"
                summed = tmpdir / f"l{l}_summed.mif"
                
                # Extract volumes
                run_mrtrix_cmd([
                    "mrconvert", str(sh_image),
                    "-coord", "3", vol_range,
                    str(extracted),
                    "-force"
                ] + thread_opt)
                
                # Square coefficients
                run_mrtrix_cmd([
                    "mrcalc", str(extracted), "2", "-pow",
                    str(squared),
                    "-force"
                ] + thread_opt)
                
                # Sum across axis 3 (m dimension)
                run_mrtrix_cmd([
                    "mrmath", str(squared),
                    "sum", "-axis", "3",
                    str(summed),
                    "-force"
                ] + thread_opt)
                
                # Square root
                run_mrtrix_cmd([
                    "mrcalc", str(summed), "0.5", "-pow",
                    str(output_file),
                    "-force"
                ] + thread_opt)
                
                # Keep intermediate files if requested
                if keep_intermediate:
                    for f in [extracted, squared, summed]:
                        shutil.copy(f, output_dir / f.name)
            
            # Apply mask if provided
            if mask:
                masked = tmpdir / f"l{l}_masked.mif"
                run_mrtrix_cmd([
                    "mrcalc", str(output_file), str(mask), "-mult",
                    str(masked),
                    "-force"
                ] + thread_opt)
                shutil.move(masked, output_file)
            
            rish_outputs[l] = str(output_file)
    
    return rish_outputs


def extract_rish_features_batch(
    sh_images: List[str],
    output_dir: str,
    lmax: Optional[int] = None,
    masks: Optional[List[str]] = None,
    n_threads: int = 4
) -> List[Dict[int, str]]:
    """
    Extract RISH features from multiple SH images.
    
    Parameters
    ----------
    sh_images : list of str
        Paths to input SH coefficient images
    output_dir : str
        Base directory for outputs
    lmax : int, optional
        Maximum SH order
    masks : list of str, optional
        Brain masks for each image
    n_threads : int
        Number of threads
        
    Returns
    -------
    list of dict
        RISH outputs for each input image
    """
    output_dir = Path(output_dir)
    
    if masks is None:
        masks = [None] * len(sh_images)
    
    results = []
    for i, (sh_img, mask) in enumerate(zip(sh_images, masks)):
        subj_dir = output_dir / f"sub-{i:03d}"
        rish = extract_rish_features(
            sh_img,
            str(subj_dir),
            lmax=lmax,
            mask=mask,
            n_threads=n_threads
        )
        results.append(rish)
    
    return results


if __name__ == "__main__":
    # Quick test
    import sys
    if len(sys.argv) > 1:
        sh_image = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "rish_output"
        
        print(f"Extracting RISH features from {sh_image}...")
        rish = extract_rish_features(sh_image, output_dir)
        print(f"Output files: {rish}")
