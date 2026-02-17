"""
Scale Map Computation Module

Computes per-voxel, per-order scaling factors for RISH harmonization.

The scaling factor for order l is:
    scale_l(x) = θ_l_reference(x) / θ_l_target(x)
"""

import subprocess
from pathlib import Path
from typing import Dict, Optional, List
import tempfile
import shutil


def run_mrtrix_cmd(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run an MRtrix3 command."""
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def compute_scale_maps(
    reference_rish: Dict[int, str],
    target_rish: Dict[int, str],
    output_dir: str,
    mask: Optional[str] = None,
    smoothing_fwhm: float = 3.0,
    clip_range: tuple = (0.5, 2.0),
    min_signal: float = 0.01,
    prefix: str = "scale",
    n_threads: int = 1
) -> Dict[int, str]:
    """
    Compute scale maps for harmonization.
    
    For each SH order l:
        scale_l = θ_l_reference / θ_l_target
    
    Parameters
    ----------
    reference_rish : dict
        Mapping of order l -> reference RISH image path
    target_rish : dict
        Mapping of order l -> target RISH image path
    output_dir : str
        Output directory for scale maps
    mask : str, optional
        Brain mask
    smoothing_fwhm : float
        Gaussian smoothing FWHM in mm (0 to disable)
    clip_range : tuple
        (min, max) values to clip scale factors
    min_signal : float
        Minimum signal threshold (avoid division by near-zero)
    prefix : str
        Output filename prefix
    n_threads : int
        Number of threads
        
    Returns
    -------
    dict
        Mapping of order l -> scale map image path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    thread_opt = ["-nthreads", str(n_threads)] if n_threads > 1 else []
    
    scale_maps = {}
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        for l in reference_rish.keys():
            if l not in target_rish:
                raise ValueError(f"Order {l} present in reference but not target")
            
            ref_img = reference_rish[l]
            tar_img = target_rish[l]
            output_file = output_dir / f"{prefix}_l{l}.mif"
            
            # Step 1: Threshold target to avoid division by zero
            tar_thresh = tmpdir / f"tar_l{l}_thresh.mif"
            run_mrtrix_cmd([
                "mrcalc", tar_img, str(min_signal), "-max",
                str(tar_thresh),
                "-force"
            ] + thread_opt)
            
            # Step 2: Compute ratio
            ratio = tmpdir / f"ratio_l{l}.mif"
            run_mrtrix_cmd([
                "mrcalc", ref_img, str(tar_thresh), "-div",
                str(ratio),
                "-force"
            ] + thread_opt)
            
            # Step 3: Apply mask if provided
            if mask:
                masked = tmpdir / f"ratio_l{l}_masked.mif"
                run_mrtrix_cmd([
                    "mrcalc", str(ratio), mask, "-mult",
                    str(masked),
                    "-force"
                ] + thread_opt)
                ratio = masked

            # Step 4: Smooth if requested
            # Use smooth(ratio*mask) / smooth(mask) to correct for boundary effects
            if smoothing_fwhm > 0:
                smoothed = tmpdir / f"ratio_l{l}_smooth.mif"
                # Convert FWHM to stdev: sigma = FWHM / (2 * sqrt(2 * ln(2))) ≈ FWHM / 2.355
                sigma = smoothing_fwhm / 2.355
                run_mrtrix_cmd([
                    "mrfilter", str(ratio),
                    "smooth", "-stdev", str(sigma),
                    str(smoothed),
                    "-force"
                ] + thread_opt)

                if mask:
                    # Correct for boundary effects: divide by smoothed mask
                    smoothed_mask = tmpdir / f"mask_l{l}_smooth.mif"
                    run_mrtrix_cmd([
                        "mrfilter", mask,
                        "smooth", "-stdev", str(sigma),
                        str(smoothed_mask),
                        "-force"
                    ] + thread_opt)

                    corrected = tmpdir / f"ratio_l{l}_corrected.mif"
                    # Where smoothed_mask > 0.1, divide; else set to 0
                    run_mrtrix_cmd([
                        "mrcalc", str(smoothed), str(smoothed_mask),
                        "0.1", "-max", "-div",
                        str(mask), "-mult",
                        str(corrected),
                        "-force"
                    ] + thread_opt)
                    ratio = corrected
                else:
                    ratio = smoothed
            
            # Step 5: Clip to reasonable range
            clipped = tmpdir / f"ratio_l{l}_clipped.mif"
            min_val, max_val = clip_range
            run_mrtrix_cmd([
                "mrcalc", str(ratio),
                str(min_val), "-max",
                str(max_val), "-min",
                str(clipped),
                "-force"
            ] + thread_opt)
            
            # Step 6: Set non-brain regions to 1.0 (no scaling)
            if mask:
                # Where mask is 0, set scale to 1.0
                final = output_file
                run_mrtrix_cmd([
                    "mrcalc", mask, str(clipped), "-mult",
                    mask, "1", "-sub", "-neg", "-add",  # Add (1-mask) to set background to 1
                    str(final),
                    "-force"
                ] + thread_opt)
            else:
                shutil.copy(clipped, output_file)
            
            scale_maps[l] = str(output_file)
    
    return scale_maps


def compute_scale_maps_from_groups(
    reference_rish_list: List[Dict[int, str]],
    target_rish_list: List[Dict[int, str]],
    output_dir: str,
    mask: Optional[str] = None,
    smoothing_fwhm: float = 3.0,
    clip_range: tuple = (0.5, 2.0),
    n_threads: int = 1
) -> Dict[int, str]:
    """
    Compute scale maps from groups of subjects.
    
    First averages RISH features within each group, then computes scale maps.
    
    Parameters
    ----------
    reference_rish_list : list of dict
        List of RISH outputs for reference site subjects
    target_rish_list : list of dict
        List of RISH outputs for target site subjects
    output_dir : str
        Output directory
    mask : str, optional
        Brain mask (in template space)
    smoothing_fwhm : float
        Smoothing FWHM in mm
    clip_range : tuple
        Scale factor clipping range
    n_threads : int
        Number of threads
        
    Returns
    -------
    dict
        Scale maps for harmonization
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    thread_opt = ["-nthreads", str(n_threads)] if n_threads > 1 else []
    
    # Get orders from first subject
    orders = list(reference_rish_list[0].keys())
    
    # Average RISH features within each group
    ref_avg = {}
    tar_avg = {}
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        for l in orders:
            # Collect reference images for this order
            ref_images = [rish[l] for rish in reference_rish_list]
            tar_images = [rish[l] for rish in target_rish_list]
            
            # Average reference
            ref_avg_file = tmpdir / f"ref_avg_l{l}.mif"
            run_mrtrix_cmd([
                "mrmath", *ref_images,
                "mean", str(ref_avg_file),
                "-force"
            ] + thread_opt)
            ref_avg[l] = str(ref_avg_file)
            
            # Average target
            tar_avg_file = tmpdir / f"tar_avg_l{l}.mif"
            run_mrtrix_cmd([
                "mrmath", *tar_images,
                "mean", str(tar_avg_file),
                "-force"
            ] + thread_opt)
            tar_avg[l] = str(tar_avg_file)
        
        # Now compute scale maps from averages
        scale_maps = compute_scale_maps(
            ref_avg,
            tar_avg,
            str(output_dir),
            mask=mask,
            smoothing_fwhm=smoothing_fwhm,
            clip_range=clip_range,
            n_threads=n_threads
        )
    
    return scale_maps


if __name__ == "__main__":
    import sys
    print("Scale map computation module. Import and use compute_scale_maps().")
