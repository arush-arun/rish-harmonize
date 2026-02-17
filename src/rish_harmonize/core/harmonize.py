"""
Harmonization Module

Applies RISH scale maps to spherical harmonic coefficients.
Supports both signal-level SH (per b-shell) and FOD-level harmonization.
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, Optional, List
import tempfile

from .rish import get_sh_indices, get_image_lmax, extract_rish_features
from .scale_maps import compute_scale_maps
from .shells import ShellInfo, detect_shells, separate_shell, extract_b0, rejoin_shells
from .sh_fitting import fit_sh, reconstruct_signal, get_directions, determine_lmax


def compute_consistent_lmax(
    dwi_list: List[str],
    lmax: Optional[int] = None,
) -> Dict[int, int]:
    """Determine a consistent lmax per b-shell across multiple DWI images.

    For each shell, takes the minimum lmax that all subjects can support.
    This ensures all subjects use the same SH basis, which is required
    for comparable RISH features.

    Parameters
    ----------
    dwi_list : list of str
        DWI images to scan
    lmax : int, optional
        User-specified lmax override (applied to all shells, clamped
        to each shell's maximum)

    Returns
    -------
    dict
        {b_value: lmax} — consistent lmax for each shell
    """
    # Collect n_directions per shell per subject
    # shell_dirs[b_value] = [n_dirs_sub0, n_dirs_sub1, ...]
    shell_dirs: Dict[int, List[int]] = {}

    for dwi in dwi_list:
        info = detect_shells(dwi)
        for b_value in info.b_values:
            if b_value not in shell_dirs:
                shell_dirs[b_value] = []
            shell_dirs[b_value].append(info.n_directions(b_value))

    # For each shell, compute the lmax that all subjects can support
    consistent = {}
    for b_value, dirs_list in shell_dirs.items():
        min_dirs = min(dirs_list)
        auto_lmax = determine_lmax(min_dirs)
        if lmax is not None:
            # User override, but don't exceed what the data supports
            consistent[b_value] = min(lmax, auto_lmax)
        else:
            consistent[b_value] = auto_lmax

    return consistent


def run_mrtrix_cmd(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run an MRtrix3 command."""
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\nstderr: {result.stderr.strip()}"
        )
    return result


def harmonize_sh(
    sh_image: str,
    scale_maps: Dict[int, str],
    output: str,
    lmax: Optional[int] = None,
    n_threads: int = 1
) -> str:
    """
    Apply RISH scale maps to SH coefficients.

    For each order l, multiply all m coefficients by scale_l.
    This is the core operation shared by both signal-level and FOD-level
    harmonization.

    Parameters
    ----------
    sh_image : str
        Input SH coefficient image
    scale_maps : dict
        Mapping of order l -> scale map image path
    output : str
        Output harmonized SH image path
    lmax : int, optional
        Maximum SH order
    n_threads : int
        Number of threads

    Returns
    -------
    str
        Path to harmonized SH image
    """
    sh_image = Path(sh_image)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if lmax is None:
        lmax = get_image_lmax(str(sh_image))

    sh_info = get_sh_indices(lmax)
    thread_opt = ["-nthreads", str(n_threads)] if n_threads > 1 else []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        scaled_parts = []

        for l, (start_idx, end_idx) in sh_info.volume_indices.items():
            if l not in scale_maps:
                raise ValueError(f"No scale map for order {l}")

            scale_map = scale_maps[l]
            vol_range = f"{start_idx}:{end_idx - 1}"

            # Extract coefficients for this order
            extracted = tmpdir / f"l{l}_coeffs.mif"
            run_mrtrix_cmd([
                "mrconvert", str(sh_image),
                "-coord", "3", vol_range,
                str(extracted),
                "-force"
            ] + thread_opt)

            # Apply scaling (broadcast scale map across m dimension)
            scaled = tmpdir / f"l{l}_scaled.mif"
            run_mrtrix_cmd([
                "mrcalc", str(extracted), scale_map, "-mult",
                str(scaled),
                "-force"
            ] + thread_opt)

            scaled_parts.append(str(scaled))

        # Concatenate all scaled parts
        run_mrtrix_cmd([
            "mrcat", *scaled_parts,
            "-axis", "3",
            str(output),
            "-force"
        ] + thread_opt)

    return str(output)


# ---------------------------------------------------------------------------
# Signal-level SH harmonization (per b-shell)
# ---------------------------------------------------------------------------

def harmonize_signal_sh(
    dwi_image: str,
    reference_rish: Dict[int, Dict[int, str]],
    output: str,
    mask: Optional[str] = None,
    shell_info: Optional[ShellInfo] = None,
    shell_lmax: Optional[Dict[int, int]] = None,
    smoothing_fwhm: float = 3.0,
    clip_range: tuple = (0.5, 2.0),
    n_threads: int = 1,
) -> str:
    """
    Full signal-level SH harmonization pipeline.

    Separate shells -> amp2sh per shell -> extract RISH -> compute scale
    maps -> apply to SH -> sh2amp -> rejoin shells.

    Parameters
    ----------
    dwi_image : str
        Input multi-shell DWI image (registered to template space)
    reference_rish : dict
        {b_value: {order: path}} — per-shell, per-order reference RISH
        template images
    output : str
        Output path for harmonized DWI
    mask : str, optional
        Brain mask
    shell_info : ShellInfo, optional
        Pre-computed shell info (auto-detected if None)
    shell_lmax : dict, optional
        {b_value: lmax} — per-shell lmax from template metadata.
        Must match the lmax used to create the reference template.
        If None, auto-determined from this subject's direction count
        (only safe if all subjects have matching directions).
    smoothing_fwhm : float
        Scale map smoothing FWHM in mm
    clip_range : tuple
        (min, max) clipping range for scale factors
    n_threads : int
        Number of threads

    Returns
    -------
    str
        Path to harmonized DWI image
    """
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    if shell_info is None:
        shell_info = detect_shells(dwi_image)

    thread_opt = ["-nthreads", str(n_threads)] if n_threads > 1 else []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Extract b=0 image (not harmonized)
        b0_path = str(tmpdir / "b0.mif")
        extract_b0(dwi_image, b0_path, n_threads=n_threads)

        harmonized_shells = {}

        for b_value in shell_info.b_values:
            if b_value not in reference_rish:
                raise ValueError(
                    f"No reference RISH for b={b_value}. "
                    f"Available: {list(reference_rish.keys())}"
                )

            shell_dir = tmpdir / f"b{b_value}"
            shell_dir.mkdir()

            # 1. Separate DW volumes only (no b=0) — b=0 has no angular
            #    dependence and should not enter the SH fitting pipeline
            shell_dwi = str(shell_dir / "dwi_dw_only.mif")
            separate_shell(
                dwi_image, shell_info, b_value, shell_dwi,
                include_b0=False, n_threads=n_threads,
            )

            # 2. Get lmax for this shell (from template or auto-detect)
            if shell_lmax is not None:
                this_lmax = shell_lmax[b_value]
            else:
                n_dirs = shell_info.n_directions(b_value)
                this_lmax = determine_lmax(n_dirs)

            # 3. Fit SH to this shell's DW signal (no b=0)
            sh_path = str(shell_dir / "sh.mif")
            fit_sh(
                shell_dwi, sh_path,
                lmax=this_lmax, mask=mask,
                n_threads=n_threads,
            )

            # 4. Extract RISH features from fitted SH
            rish_dir = str(shell_dir / "rish")
            target_rish = extract_rish_features(
                sh_path, rish_dir,
                lmax=this_lmax, mask=mask,
                n_threads=n_threads,
            )

            # 5. Compute scale maps for this shell
            ref_rish_for_shell = reference_rish[b_value]
            scale_dir = str(shell_dir / "scale_maps")
            scale_maps = compute_scale_maps(
                ref_rish_for_shell,
                target_rish,
                scale_dir,
                mask=mask,
                smoothing_fwhm=smoothing_fwhm,
                clip_range=clip_range,
                n_threads=n_threads,
            )

            # 6. Apply scale maps to SH coefficients
            sh_harmonized = str(shell_dir / "sh_harmonized.mif")
            harmonize_sh(
                sh_path, scale_maps, sh_harmonized,
                lmax=this_lmax, n_threads=n_threads,
            )

            # 7. Reconstruct DW signal from harmonized SH
            #    Output has same number of volumes as DW-only input
            directions_path = str(shell_dir / "directions.txt")
            get_directions(shell_dwi, directions_path)

            dwi_harmonized = str(shell_dir / "dwi_harmonized.mif")
            reconstruct_signal(
                sh_harmonized, directions_path, dwi_harmonized,
                n_threads=n_threads,
            )

            harmonized_shells[b_value] = dwi_harmonized

        # 8. Rejoin shells into original volume order
        rejoin_shells(
            harmonized_shells, b0_path, output,
            shell_info, n_threads=n_threads,
        )

    return output


def harmonize_signal_sh_with_maps(
    dwi_image: str,
    scale_maps: Dict[int, Dict[int, str]],
    output: str,
    mask: Optional[str] = None,
    shell_info: Optional[ShellInfo] = None,
    shell_lmax: Optional[Dict[int, int]] = None,
    n_threads: int = 1,
) -> str:
    """
    Apply pre-computed per-shell scale maps to a DWI image.

    Like harmonize_signal_sh() but skips RISH extraction and scale map
    computation — useful when scale maps are already available (e.g.
    from RISH-GLM fitting).

    Parameters
    ----------
    dwi_image : str
        Input multi-shell DWI image
    scale_maps : dict
        {b_value: {order: path}} — pre-computed per-shell, per-order
        scale maps
    output : str
        Output path for harmonized DWI
    mask : str, optional
        Brain mask
    shell_info : ShellInfo, optional
        Pre-computed shell info
    shell_lmax : dict, optional
        {b_value: lmax} — per-shell lmax. Must match the lmax used
        when computing the scale maps.
    n_threads : int
        Number of threads

    Returns
    -------
    str
        Path to harmonized DWI image
    """
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    if shell_info is None:
        shell_info = detect_shells(dwi_image)

    thread_opt = ["-nthreads", str(n_threads)] if n_threads > 1 else []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        b0_path = str(tmpdir / "b0.mif")
        extract_b0(dwi_image, b0_path, n_threads=n_threads)

        harmonized_shells = {}

        for b_value in shell_info.b_values:
            if b_value not in scale_maps:
                raise ValueError(
                    f"No scale maps for b={b_value}. "
                    f"Available: {list(scale_maps.keys())}"
                )

            shell_dir = tmpdir / f"b{b_value}"
            shell_dir.mkdir()

            # Separate DW volumes only (no b=0)
            shell_dwi = str(shell_dir / "dwi_dw_only.mif")
            separate_shell(
                dwi_image, shell_info, b_value, shell_dwi,
                include_b0=False, n_threads=n_threads,
            )

            # Get lmax for this shell
            if shell_lmax is not None:
                this_lmax = shell_lmax[b_value]
            else:
                n_dirs = shell_info.n_directions(b_value)
                this_lmax = determine_lmax(n_dirs)

            # Fit SH
            sh_path = str(shell_dir / "sh.mif")
            fit_sh(
                shell_dwi, sh_path,
                lmax=this_lmax, mask=mask,
                n_threads=n_threads,
            )

            # Apply scale maps to SH
            sh_harmonized = str(shell_dir / "sh_harmonized.mif")
            harmonize_sh(
                sh_path, scale_maps[b_value], sh_harmonized,
                lmax=this_lmax, n_threads=n_threads,
            )

            # Reconstruct DW signal from harmonized SH
            directions_path = str(shell_dir / "directions.txt")
            get_directions(shell_dwi, directions_path)

            dwi_harmonized = str(shell_dir / "dwi_harmonized.mif")
            reconstruct_signal(
                sh_harmonized, directions_path, dwi_harmonized,
                n_threads=n_threads,
            )

            harmonized_shells[b_value] = dwi_harmonized

        rejoin_shells(
            harmonized_shells, b0_path, output,
            shell_info, n_threads=n_threads,
        )

    return output


# ---------------------------------------------------------------------------
# FOD-level harmonization
# ---------------------------------------------------------------------------

def harmonize_fod(
    fod_image: str,
    reference_rish: Dict[int, str],
    output: str,
    mask: Optional[str] = None,
    lmax: Optional[int] = None,
    smoothing_fwhm: float = 3.0,
    clip_range: tuple = (0.5, 2.0),
    n_threads: int = 1,
) -> str:
    """
    FOD-level RISH harmonization.

    Operates on a single FOD image (from CSD). Extracts RISH features,
    computes scale maps against the reference, and applies them.

    Parameters
    ----------
    fod_image : str
        Input FOD SH coefficient image (registered to template space)
    reference_rish : dict
        {order: path} — per-order reference RISH template images
    output : str
        Output path for harmonized FOD
    mask : str, optional
        Brain mask
    lmax : int, optional
        Maximum SH order (auto-detected if None)
    smoothing_fwhm : float
        Scale map smoothing FWHM in mm
    clip_range : tuple
        (min, max) clipping range for scale factors
    n_threads : int
        Number of threads

    Returns
    -------
    str
        Path to harmonized FOD image
    """
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    if lmax is None:
        lmax = get_image_lmax(fod_image)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Extract RISH from target FOD
        rish_dir = str(tmpdir / "rish")
        target_rish = extract_rish_features(
            fod_image, rish_dir,
            lmax=lmax, mask=mask,
            n_threads=n_threads,
        )

        # Compute scale maps
        scale_dir = str(tmpdir / "scale_maps")
        scale_maps = compute_scale_maps(
            reference_rish,
            target_rish,
            scale_dir,
            mask=mask,
            smoothing_fwhm=smoothing_fwhm,
            clip_range=clip_range,
            n_threads=n_threads,
        )

        # Apply scale maps to FOD SH coefficients
        harmonize_sh(
            fod_image, scale_maps, output,
            lmax=lmax, n_threads=n_threads,
        )

    return output


# ---------------------------------------------------------------------------
# Reference template creation
# ---------------------------------------------------------------------------

def create_reference_template_signal(
    dwi_list: List[str],
    mask_list: Optional[List[str]],
    output_dir: str,
    lmax: Optional[int] = None,
    n_threads: int = 1,
) -> Dict[int, Dict[int, str]]:
    """
    Create per-shell RISH reference template from a set of DWI images.

    For each subject and each b-shell, fits SH and extracts RISH features.
    Then averages across subjects to produce the template.

    Uses consistent lmax per shell (minimum across all subjects) so that
    all subjects contribute the same SH orders.

    Parameters
    ----------
    dwi_list : list of str
        DWI images from the reference site (registered to template space)
    mask_list : list of str, optional
        Brain masks (one per subject)
    output_dir : str
        Output directory for template
    lmax : int, optional
        Maximum SH order (clamped to per-shell minimum if provided)
    n_threads : int
        Number of threads

    Returns
    -------
    dict
        {b_value: {order: template_rish_path}} — averaged RISH per shell
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_subjects = len(dwi_list)

    # Compute consistent lmax across all subjects
    consistent_lmax = compute_consistent_lmax(dwi_list, lmax=lmax)

    # Per-shell, per-order RISH images for all subjects
    # rish_images[b_value][order] = [sub0_path, sub1_path, ...]
    rish_images: Dict[int, Dict[int, List[str]]] = {}

    for i, dwi in enumerate(dwi_list):
        mask = mask_list[i] if mask_list else None
        subj_dir = output_dir / "subjects" / f"sub-{i:03d}"
        subj_dir.mkdir(parents=True, exist_ok=True)

        subj_shell_info = detect_shells(dwi)

        for b_value in subj_shell_info.b_values:
            shell_dir = subj_dir / f"b{b_value}"
            shell_dir.mkdir(exist_ok=True)

            this_lmax = consistent_lmax[b_value]

            # Separate DW volumes only (no b=0 for SH fitting)
            shell_dwi = str(shell_dir / "dwi_dw_only.mif")
            separate_shell(
                dwi, subj_shell_info, b_value, shell_dwi,
                include_b0=False, n_threads=n_threads,
            )

            # Fit SH
            sh_path = str(shell_dir / "sh.mif")
            fit_sh(
                shell_dwi, sh_path,
                lmax=this_lmax, mask=mask,
                n_threads=n_threads,
            )

            # Extract RISH
            rish_dir = str(shell_dir / "rish")
            rish = extract_rish_features(
                sh_path, rish_dir,
                lmax=this_lmax, mask=mask,
                n_threads=n_threads,
            )

            if b_value not in rish_images:
                rish_images[b_value] = {}
            for order, path in rish.items():
                if order not in rish_images[b_value]:
                    rish_images[b_value][order] = []
                rish_images[b_value][order].append(path)

    # Average across subjects
    template: Dict[int, Dict[int, str]] = {}
    for b_value in rish_images:
        template[b_value] = {}
        template_dir = output_dir / f"template_b{b_value}"
        template_dir.mkdir(exist_ok=True)

        for order, paths in rish_images[b_value].items():
            if len(paths) != n_subjects:
                raise ValueError(
                    f"b={b_value}, order={order}: expected {n_subjects} images, "
                    f"got {len(paths)}"
                )

            avg_path = str(template_dir / f"rish_l{order}_mean.mif")
            run_mrtrix_cmd([
                "mrmath", *paths,
                "mean", avg_path,
                "-force",
                "-nthreads", str(n_threads),
            ])
            template[b_value][order] = avg_path

    # Save template metadata (including lmax per shell for harmonization)
    meta = {
        "mode": "signal",
        "n_subjects": n_subjects,
        "shell_lmax": {str(b): l for b, l in consistent_lmax.items()},
        "reference_rish": {
            str(b): {str(o): p for o, p in orders.items()}
            for b, orders in template.items()
        },
    }
    with open(output_dir / "template_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return template


def create_reference_template_fod(
    fod_list: List[str],
    mask_list: Optional[List[str]],
    output_dir: str,
    lmax: Optional[int] = None,
    n_threads: int = 1,
) -> Dict[int, str]:
    """
    Create RISH reference template from FOD images.

    Parameters
    ----------
    fod_list : list of str
        FOD SH images from the reference site
    mask_list : list of str, optional
        Brain masks (one per subject)
    output_dir : str
        Output directory for template
    lmax : int, optional
        Maximum SH order (auto-detected if None)
    n_threads : int
        Number of threads

    Returns
    -------
    dict
        {order: template_rish_path} — averaged RISH features
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_subjects = len(fod_list)

    # Extract RISH for each subject
    rish_images: Dict[int, List[str]] = {}

    for i, fod in enumerate(fod_list):
        mask = mask_list[i] if mask_list else None
        subj_dir = output_dir / "subjects" / f"sub-{i:03d}"

        if lmax is None:
            subj_lmax = get_image_lmax(fod)
        else:
            subj_lmax = lmax

        rish = extract_rish_features(
            fod, str(subj_dir),
            lmax=subj_lmax, mask=mask,
            n_threads=n_threads,
        )

        for order, path in rish.items():
            if order not in rish_images:
                rish_images[order] = []
            rish_images[order].append(path)

    # Average across subjects
    template: Dict[int, str] = {}
    for order, paths in rish_images.items():
        if len(paths) != n_subjects:
            raise ValueError(
                f"Order {order}: expected {n_subjects} images, got {len(paths)}"
            )

        avg_path = str(output_dir / f"template_rish_l{order}.mif")
        run_mrtrix_cmd([
            "mrmath", *paths,
            "mean", avg_path,
            "-force",
            "-nthreads", str(n_threads),
        ])
        template[order] = avg_path

    # Save template metadata
    used_lmax = max(template.keys())  # highest order present
    meta = {
        "mode": "fod",
        "n_subjects": n_subjects,
        "lmax": used_lmax,
        "reference_rish": {str(o): p for o, p in template.items()},
    }
    with open(output_dir / "template_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return template
