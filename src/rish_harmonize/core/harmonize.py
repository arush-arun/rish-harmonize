"""
Harmonization Module

Applies RISH scale maps to spherical harmonic coefficients.
Supports both signal-level SH (per b-shell) and FOD-level harmonization.

Signal-level SH workflow (native <-> template space):

  1. extract_native_rish()       — Native DWI -> per-shell SH -> RISH features
  2. (user warps RISH to template space, e.g. mrtransform)
  3. create_reference_template_signal() — Average template-space RISH
  4. compute_scale_maps()         — Template-space ref/target RISH -> scale maps
  5. (user warps scale maps back to native space)
  6. apply_harmonization()        — Native DWI + native scale maps -> harmonized DWI

FOD-level workflow (all in template space):

  1. extract_rish_features()     — FOD -> RISH
  2. create_reference_template_fod() — Average RISH
  3. harmonize_fod()             — Compute scale maps + apply to FOD
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


# ---------------------------------------------------------------------------
# RISH extraction (native space)
# ---------------------------------------------------------------------------

def extract_native_rish(
    dwi_image: str,
    output_dir: str,
    mask: Optional[str] = None,
    shell_lmax: Optional[Dict[int, int]] = None,
    n_threads: int = 1,
) -> Dict[int, Dict[int, str]]:
    """Extract RISH features from a native-space DWI image.

    Performs: detect shells -> separate DW-only -> amp2sh -> extract RISH.
    Also saves intermediate SH images and directions for later use
    in apply_harmonization().

    Output directory structure::

        output_dir/
            shell_meta.json      # shell info + lmax metadata
            b{value}/
                dwi_dw_only.mif  # separated DW volumes
                sh.mif           # SH coefficients
                directions.txt   # gradient directions (for sh2amp)
                rish/
                    rish_l0.mif
                    rish_l2.mif
                    ...

    Parameters
    ----------
    dwi_image : str
        Input multi-shell DWI image in native space.
    output_dir : str
        Output directory for extracted features.
    mask : str, optional
        Brain mask (for RISH extraction only — amp2sh does not support masking).
    shell_lmax : dict, optional
        {b_value: lmax}. If None, auto-determined from direction count.
    n_threads : int
        Number of threads.

    Returns
    -------
    dict
        {b_value: {order: rish_path}} — RISH features per shell per order.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shell_info = detect_shells(dwi_image)
    all_rish: Dict[int, Dict[int, str]] = {}

    # Determine lmax per shell
    used_lmax = {}
    for b_value in shell_info.b_values:
        if shell_lmax is not None and b_value in shell_lmax:
            used_lmax[b_value] = shell_lmax[b_value]
        else:
            n_dirs = shell_info.n_directions(b_value)
            used_lmax[b_value] = determine_lmax(n_dirs)

    for b_value in shell_info.b_values:
        this_lmax = used_lmax[b_value]
        shell_dir = output_dir / f"b{b_value}"
        shell_dir.mkdir(exist_ok=True)

        # Check if RISH extraction is already complete for this shell
        rish_subdir = shell_dir / "rish"
        expected_orders = list(range(0, this_lmax + 1, 2))
        if rish_subdir.is_dir():
            existing = {
                int(f.stem.split("_l")[1]): str(f)
                for f in rish_subdir.iterdir()
                if f.name.startswith("rish_l") and f.suffix == ".mif"
            }
            if all(o in existing for o in expected_orders):
                all_rish[b_value] = existing
                continue

        # 1. Separate DW volumes only (no b=0)
        shell_dwi = str(shell_dir / "dwi_dw_only.mif")
        separate_shell(
            dwi_image, shell_info, b_value, shell_dwi,
            include_b0=False, n_threads=n_threads,
        )

        # 2. Fit SH (amp2sh does not support masking)
        sh_path = str(shell_dir / "sh.mif")
        fit_sh(
            shell_dwi, sh_path,
            lmax=this_lmax,
            n_threads=n_threads,
        )

        # 3. Save gradient directions (needed for sh2amp in apply step)
        directions_path = str(shell_dir / "directions.txt")
        get_directions(shell_dwi, directions_path)

        # 4. Extract RISH features
        rish_dir = str(shell_dir / "rish")
        rish = extract_rish_features(
            sh_path, rish_dir,
            lmax=this_lmax, mask=mask,
            n_threads=n_threads,
        )

        all_rish[b_value] = rish

    # Save metadata
    meta = {
        "dwi_image": str(dwi_image),
        "shell_lmax": {str(b): l for b, l in used_lmax.items()},
        "b_values": shell_info.b_values,
        "rish": {
            str(b): {str(o): p for o, p in orders.items()}
            for b, orders in all_rish.items()
        },
    }
    with open(output_dir / "shell_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return all_rish


def load_rish_dir(rish_dir: str) -> Dict[int, Dict[int, str]]:
    """Load RISH features from a standard directory structure.

    Scans for directories named ``b{value}/rish/rish_l{order}.mif``
    (as produced by extract_native_rish or after warping to template).

    Parameters
    ----------
    rish_dir : str
        Directory containing per-shell RISH features.

    Returns
    -------
    dict
        {b_value: {order: rish_path}}
    """
    rish_dir = Path(rish_dir)
    result: Dict[int, Dict[int, str]] = {}

    # Try loading from metadata JSON first
    for meta_name in ("shell_meta.json", "template_meta.json"):
        meta_path = rish_dir / meta_name
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        # shell_meta.json uses "rish", template_meta.json uses "reference_rish"
        rish_key = "reference_rish" if "reference_rish" in meta else "rish"
        if rish_key not in meta:
            continue
        for b_str, orders in meta[rish_key].items():
            result[int(b_str)] = {int(o): p for o, p in orders.items()}
        return result

    # Fall back to directory scanning
    for shell_path in sorted(rish_dir.iterdir()):
        if not shell_path.is_dir() or not shell_path.name.startswith("b"):
            continue
        try:
            b_value = int(shell_path.name[1:])
        except ValueError:
            continue

        rish_subdir = shell_path / "rish"
        if not rish_subdir.is_dir():
            continue

        orders = {}
        for rish_file in sorted(rish_subdir.iterdir()):
            if rish_file.name.startswith("rish_l") and rish_file.suffix == ".mif":
                try:
                    order = int(rish_file.stem.split("_l")[1])
                    orders[order] = str(rish_file)
                except (ValueError, IndexError):
                    continue

        if orders:
            result[b_value] = orders

    if not result:
        raise FileNotFoundError(
            f"No RISH features found in {rish_dir}. "
            "Expected b*/rish/rish_l*.mif structure."
        )

    return result


# ---------------------------------------------------------------------------
# Core SH harmonization operation
# ---------------------------------------------------------------------------

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

def apply_harmonization(
    dwi_image: str,
    scale_maps: Dict[int, Dict[int, str]],
    output: str,
    shell_lmax: Optional[Dict[int, int]] = None,
    n_threads: int = 1,
) -> str:
    """
    Apply per-shell scale maps to a native-space DWI image.

    Performs: detect shells -> separate DW-only -> amp2sh -> apply scale
    maps -> sh2amp -> rejoin with b=0.

    Scale maps must be in native space (warped back from template
    by the user before calling this function).

    Parameters
    ----------
    dwi_image : str
        Input multi-shell DWI image in native space.
    scale_maps : dict
        {b_value: {order: scale_map_path}} — per-shell, per-order
        scale maps in native space.
    output : str
        Output path for harmonized DWI.
    shell_lmax : dict, optional
        {b_value: lmax}. Must match the lmax used when computing
        the scale maps. If None, auto-determined.
    n_threads : int
        Number of threads.

    Returns
    -------
    str
        Path to harmonized DWI image.
    """
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    shell_info = detect_shells(dwi_image)

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
                lmax=this_lmax,
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
    rish_by_subject: List[Dict[int, Dict[int, str]]],
    output_dir: str,
    shell_lmax: Optional[Dict[int, int]] = None,
    n_threads: int = 1,
) -> Dict[int, Dict[int, str]]:
    """
    Create per-shell RISH reference template by averaging across subjects.

    All input RISH images must already be in template space (warped by
    the user before calling this function). Use extract_native_rish()
    to extract RISH in native space, then warp with mrtransform.

    Parameters
    ----------
    rish_by_subject : list of dict
        Each element is {b_value: {order: rish_path}} for one subject,
        already in template space.
    output_dir : str
        Output directory for template.
    shell_lmax : dict, optional
        {b_value: lmax} — stored in metadata for consistency during
        harmonization. If None, inferred from highest RISH order present.
    n_threads : int
        Number of threads.

    Returns
    -------
    dict
        {b_value: {order: template_rish_path}} — averaged RISH per shell.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_subjects = len(rish_by_subject)

    # Collect all RISH paths: rish_images[b_value][order] = [path0, path1, ...]
    rish_images: Dict[int, Dict[int, List[str]]] = {}
    for subj_rish in rish_by_subject:
        for b_value, orders in subj_rish.items():
            if b_value not in rish_images:
                rish_images[b_value] = {}
            for order, path in orders.items():
                if order not in rish_images[b_value]:
                    rish_images[b_value][order] = []
                rish_images[b_value][order].append(path)

    # Average across subjects
    template: Dict[int, Dict[int, str]] = {}
    for b_value in sorted(rish_images.keys()):
        template[b_value] = {}
        template_dir = output_dir / f"template_b{b_value}"
        template_dir.mkdir(exist_ok=True)

        for order, paths in sorted(rish_images[b_value].items()):
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

    # Infer shell_lmax from RISH orders if not provided
    if shell_lmax is None:
        shell_lmax = {}
        for b_value, orders in template.items():
            shell_lmax[b_value] = max(orders.keys())

    # Save template metadata
    meta = {
        "mode": "signal",
        "n_subjects": n_subjects,
        "shell_lmax": {str(b): l for b, l in shell_lmax.items()},
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
