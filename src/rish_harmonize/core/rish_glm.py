"""
RISH-GLM Harmonization Module

Joint GLM-based estimation of site effects and covariates for RISH
harmonization. Implements the approach from:

    De Luca et al., "RISH-GLM: Rotationally invariant spherical harmonic
    general linear model for quantitative dMRI harmonization"
    Magnetic Resonance in Medicine, 2025.

Key idea: Instead of separately adjusting for covariates then computing
scale maps from averaged RISH features, RISH-GLM fits a single joint model:

    RISH_l(v) = β_R(v) * S_R + β_T(v) * S_T + ... + β_cov(v) * cov + ε

where S_R, S_T are site indicator columns (no intercept). The scale factor
for harmonizing site T to match site R is:

    θ_l(v) = β_R(v) / β_T(v)

Advantages over the two-step approach:
- No matched training subjects required
- Multi-site simultaneous harmonization
- Joint covariate-site estimation
- Preserves biological variability with unmatched groups
"""

import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class RISHGLMResult:
    """Stores fitted RISH-GLM model.

    Attributes
    ----------
    site_names : list of str
        Site names in column order
    covariate_names : list of str
        Covariate names in column order
    orders : list of int
        SH orders in model (e.g., [0, 2, 4, 6, 8])
    beta_paths : dict
        (order, column_name) -> path to beta map
    scale_map_paths : dict
        (order, target_site) -> path to scale map
    reference_site : str
        Name of the reference site
    n_subjects : int
        Total number of subjects used in fitting
    n_per_site : dict
        site_name -> number of subjects
    cov_means : dict
        Covariate name -> population mean (for z-scoring)
    cov_stds : dict
        Covariate name -> population std (for z-scoring)
    design_columns : list of str
        Column names in design matrix order
    mask_path : str or None
        Path to mask used during fitting
    output_dir : str
        Directory where model files are stored
    """
    site_names: List[str] = field(default_factory=list)
    covariate_names: List[str] = field(default_factory=list)
    orders: List[int] = field(default_factory=list)
    beta_paths: Dict[str, str] = field(default_factory=dict)
    scale_map_paths: Dict[str, str] = field(default_factory=dict)
    reference_site: str = ""
    n_subjects: int = 0
    n_per_site: Dict[str, int] = field(default_factory=dict)
    cov_means: Dict[str, float] = field(default_factory=dict)
    cov_stds: Dict[str, float] = field(default_factory=dict)
    design_columns: List[str] = field(default_factory=list)
    mask_path: Optional[str] = None
    output_dir: str = ""


def build_rish_glm_design(
    site_labels: List[str],
    covariates: Optional[Dict[str, List[float]]] = None,
) -> Tuple[np.ndarray, List[str], Dict[str, int], Dict[str, float], Dict[str, float]]:
    """Build design matrix for RISH-GLM.

    Uses one indicator column per site (no intercept), so that each
    site's beta directly estimates its mean RISH value. Covariates
    are z-scored and appended.

    Parameters
    ----------
    site_labels : list of str
        Site label for each subject (length N)
    covariates : dict, optional
        Covariate name -> list of float values (length N each)

    Returns
    -------
    design : ndarray, shape (N, n_sites + n_covariates)
        Design matrix
    column_names : list of str
        Name for each column
    site_col_map : dict
        site_name -> column index
    cov_means : dict
        Covariate name -> mean used for z-scoring
    cov_stds : dict
        Covariate name -> std used for z-scoring
    """
    n = len(site_labels)
    unique_sites = sorted(set(site_labels))

    columns = []
    col_names = []
    site_col_map = {}

    # Site indicator columns (no intercept)
    for i, site in enumerate(unique_sites):
        indicator = np.array(
            [1.0 if s == site else 0.0 for s in site_labels],
            dtype=np.float64,
        )
        columns.append(indicator)
        col_names.append(f"site_{site}")
        site_col_map[site] = i

    # Covariate columns (z-scored)
    cov_means = {}
    cov_stds = {}
    if covariates:
        for name in sorted(covariates.keys()):
            values = np.array(covariates[name], dtype=np.float64)
            if len(values) != n:
                raise ValueError(
                    f"Covariate '{name}' has {len(values)} values, expected {n}"
                )
            m = float(values.mean())
            s = float(values.std())
            if s < 1e-10:
                s = 1.0
            cov_means[name] = m
            cov_stds[name] = s
            columns.append((values - m) / s)
            col_names.append(name)

    design = np.column_stack(columns)
    return design, col_names, site_col_map, cov_means, cov_stds


def fit_rish_glm(
    rish_image_paths: Dict[int, List[str]],
    site_labels: List[str],
    mask_path: Optional[str],
    output_dir: str,
    reference_site: str,
    covariates: Optional[Dict[str, List[float]]] = None,
) -> RISHGLMResult:
    """Fit voxel-wise RISH-GLM across all sites.

    For each SH order l and each voxel v, fits:
        RISH_l(v) = sum_s β_s(v) * S_s + sum_k β_k(v) * z_k + ε

    Parameters
    ----------
    rish_image_paths : dict
        order -> list of RISH image paths (one per subject, same order
        as site_labels). E.g., {0: [sub0.mif, sub1.mif, ...]}
    site_labels : list of str
        Site label for each subject
    mask_path : str or None
        Brain mask for restricting computation
    output_dir : str
        Directory for saving beta maps and model
    reference_site : str
        Name of the reference site
    covariates : dict, optional
        covariate_name -> list of float values (one per subject)

    Returns
    -------
    RISHGLMResult
        Fitted model with beta maps saved to disk
    """
    from ..qc.glm import load_image_to_matrix, save_vector_to_image

    output_dir = Path(output_dir)
    model_dir = output_dir / "rish_glm_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    orders = sorted(rish_image_paths.keys())
    n_subjects = len(site_labels)
    unique_sites = sorted(set(site_labels))

    if reference_site not in unique_sites:
        raise ValueError(
            f"Reference site '{reference_site}' not found. "
            f"Available: {unique_sites}"
        )

    # Build design matrix
    design, col_names, site_col_map, cov_means, cov_stds = build_rish_glm_design(
        site_labels, covariates
    )

    n_per_site = {
        site: sum(1 for s in site_labels if s == site) for site in unique_sites
    }

    result = RISHGLMResult(
        site_names=unique_sites,
        covariate_names=sorted(covariates.keys()) if covariates else [],
        orders=orders,
        reference_site=reference_site,
        n_subjects=n_subjects,
        n_per_site=n_per_site,
        cov_means=cov_means,
        cov_stds=cov_stds,
        design_columns=col_names,
        mask_path=mask_path,
        output_dir=str(model_dir),
    )

    for l in orders:
        paths = rish_image_paths[l]
        if len(paths) != n_subjects:
            raise ValueError(
                f"Order {l}: expected {n_subjects} images, got {len(paths)}"
            )

        # Load RISH images into matrix (n_subjects x n_voxels)
        data, mask_indices, image_shape = load_image_to_matrix(paths, mask_path)

        # Fit GLM: design @ betas = data
        betas, _, _, _ = np.linalg.lstsq(design, data, rcond=None)
        # betas shape: (n_columns, n_voxels)

        # Save beta maps
        for j, col_name in enumerate(col_names):
            beta_path = str(model_dir / f"beta_{col_name}_l{l}.mif")
            save_vector_to_image(
                betas[j], mask_indices, image_shape, paths[0], beta_path
            )
            result.beta_paths[f"{l}_{col_name}"] = beta_path

    # Save model metadata
    save_rish_glm_model(result, str(model_dir / "rish_glm_model.json"))

    return result


def compute_glm_scale_maps(
    result: RISHGLMResult,
    target_site: str,
    output_dir: str,
    mask: Optional[str] = None,
    smoothing_fwhm: float = 3.0,
    clip_range: Tuple[float, float] = (0.5, 2.0),
    min_signal: float = 0.01,
    n_threads: int = 1,
) -> Dict[int, str]:
    """Compute scale maps from fitted RISH-GLM betas.

    For each order l:
        scale_l(v) = β_reference(v) / β_target(v)

    Parameters
    ----------
    result : RISHGLMResult
        Fitted RISH-GLM model
    target_site : str
        Site to harmonize (compute scale map for)
    output_dir : str
        Output directory for scale maps
    mask : str, optional
        Brain mask (defaults to model's mask)
    smoothing_fwhm : float
        Gaussian smoothing FWHM in mm (0 to disable)
    clip_range : tuple
        (min, max) values to clip scale factors
    min_signal : float
        Minimum denominator threshold
    n_threads : int
        Number of threads

    Returns
    -------
    dict
        order -> scale map image path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if mask is None:
        mask = result.mask_path

    ref_site = result.reference_site
    if target_site not in result.site_names:
        raise ValueError(
            f"Target site '{target_site}' not in model. "
            f"Available: {result.site_names}"
        )

    thread_opt = ["-nthreads", str(n_threads)] if n_threads > 1 else []
    scale_maps = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        for l in result.orders:
            ref_key = f"{l}_site_{ref_site}"
            tar_key = f"{l}_site_{target_site}"

            ref_beta = result.beta_paths[ref_key]
            tar_beta = result.beta_paths[tar_key]

            output_file = output_dir / f"scale_l{l}_{target_site}.mif"

            # Step 1: Threshold target beta to avoid division by zero
            tar_thresh = tmpdir / f"tar_l{l}_thresh.mif"
            _run_cmd([
                "mrcalc", tar_beta, str(min_signal), "-max",
                str(tar_thresh), "-force",
            ] + thread_opt)

            # Step 2: Compute ratio β_ref / β_target
            ratio = tmpdir / f"ratio_l{l}.mif"
            _run_cmd([
                "mrcalc", ref_beta, str(tar_thresh), "-div",
                str(ratio), "-force",
            ] + thread_opt)

            # Step 3: Apply mask if provided
            if mask:
                masked = tmpdir / f"ratio_l{l}_masked.mif"
                _run_cmd([
                    "mrcalc", str(ratio), mask, "-mult",
                    str(masked), "-force",
                ] + thread_opt)
                ratio = masked

            # Step 4: Smooth with boundary correction
            if smoothing_fwhm > 0:
                sigma = smoothing_fwhm / 2.355
                smoothed = tmpdir / f"ratio_l{l}_smooth.mif"
                _run_cmd([
                    "mrfilter", str(ratio),
                    "smooth", "-stdev", str(sigma),
                    str(smoothed), "-force",
                ] + thread_opt)

                if mask:
                    smoothed_mask = tmpdir / f"mask_l{l}_smooth.mif"
                    _run_cmd([
                        "mrfilter", mask,
                        "smooth", "-stdev", str(sigma),
                        str(smoothed_mask), "-force",
                    ] + thread_opt)

                    corrected = tmpdir / f"ratio_l{l}_corrected.mif"
                    _run_cmd([
                        "mrcalc", str(smoothed), str(smoothed_mask),
                        "0.1", "-max", "-div",
                        str(mask), "-mult",
                        str(corrected), "-force",
                    ] + thread_opt)
                    ratio = corrected
                else:
                    ratio = smoothed

            # Step 5: Clip to range
            min_val, max_val = clip_range
            clipped = tmpdir / f"ratio_l{l}_clipped.mif"
            _run_cmd([
                "mrcalc", str(ratio),
                str(min_val), "-max",
                str(max_val), "-min",
                str(clipped), "-force",
            ] + thread_opt)

            # Step 6: Set non-brain to 1.0
            if mask:
                _run_cmd([
                    "mrcalc", mask, str(clipped), "-mult",
                    mask, "1", "-sub", "-neg", "-add",
                    str(output_file), "-force",
                ] + thread_opt)
            else:
                import shutil
                shutil.copy(clipped, output_file)

            scale_maps[l] = str(output_file)
            result.scale_map_paths[f"{l}_{target_site}"] = str(output_file)

    return scale_maps


def save_rish_glm_model(model: RISHGLMResult, path: str) -> None:
    """Save RISH-GLM model to JSON.

    Paths are stored relative to the JSON file's directory.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    base_dir = path.parent

    def _rel(p: Optional[str]) -> Optional[str]:
        if p is None:
            return None
        try:
            return str(Path(p).relative_to(base_dir))
        except ValueError:
            return p

    data = {
        "site_names": model.site_names,
        "covariate_names": model.covariate_names,
        "orders": model.orders,
        "beta_paths": {k: _rel(v) for k, v in model.beta_paths.items()},
        "scale_map_paths": {k: _rel(v) for k, v in model.scale_map_paths.items()},
        "reference_site": model.reference_site,
        "n_subjects": model.n_subjects,
        "n_per_site": model.n_per_site,
        "cov_means": model.cov_means,
        "cov_stds": model.cov_stds,
        "design_columns": model.design_columns,
        "mask_path": _rel(model.mask_path),
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_rish_glm_model(path: str) -> RISHGLMResult:
    """Load RISH-GLM model from JSON.

    Relative paths are resolved against the JSON file's directory.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"RISH-GLM model not found: {path}")

    base_dir = path.parent

    with open(path) as f:
        data = json.load(f)

    def _abs(p: Optional[str]) -> Optional[str]:
        if p is None:
            return None
        resolved = base_dir / p
        if resolved.exists():
            return str(resolved)
        return p

    return RISHGLMResult(
        site_names=data["site_names"],
        covariate_names=data.get("covariate_names", []),
        orders=data["orders"],
        beta_paths={k: _abs(v) for k, v in data["beta_paths"].items()},
        scale_map_paths={k: _abs(v) for k, v in data.get("scale_map_paths", {}).items()},
        reference_site=data["reference_site"],
        n_subjects=data["n_subjects"],
        n_per_site=data.get("n_per_site", {}),
        cov_means=data.get("cov_means", {}),
        cov_stds=data.get("cov_stds", {}),
        design_columns=data.get("design_columns", []),
        mask_path=_abs(data.get("mask_path")),
        output_dir=str(base_dir),
    )


def _run_cmd(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run an MRtrix3 command."""
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


# ---------------------------------------------------------------------------
# Per-shell RISH-GLM (signal-level SH mode)
# ---------------------------------------------------------------------------

@dataclass
class RISHGLMPerShellResult:
    """Stores per-shell RISH-GLM results.

    Attributes
    ----------
    b_values : list of int
        B-value shells in model
    per_shell : dict
        b_value -> RISHGLMResult for that shell
    output_dir : str
        Top-level output directory
    """
    b_values: List[int] = field(default_factory=list)
    per_shell: Dict[int, RISHGLMResult] = field(default_factory=dict)
    output_dir: str = ""


def fit_rish_glm_per_shell(
    rish_image_paths: Dict[int, Dict[int, List[str]]],
    site_labels: List[str],
    mask_path: Optional[str],
    output_dir: str,
    reference_site: str,
    covariates: Optional[Dict[str, List[float]]] = None,
) -> RISHGLMPerShellResult:
    """Fit RISH-GLM independently for each b-shell.

    Parameters
    ----------
    rish_image_paths : dict
        {b_value: {order: [sub0.mif, sub1.mif, ...]}}
        Per-shell, per-order RISH image paths.
    site_labels : list of str
        Site label for each subject
    mask_path : str or None
        Brain mask
    output_dir : str
        Output directory
    reference_site : str
        Reference site name
    covariates : dict, optional
        covariate_name -> list of float values

    Returns
    -------
    RISHGLMPerShellResult
        Container with per-shell GLM results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    b_values = sorted(rish_image_paths.keys())
    result = RISHGLMPerShellResult(
        b_values=b_values,
        output_dir=str(output_dir),
    )

    for b_value in b_values:
        shell_dir = str(output_dir / f"b{b_value}")
        shell_result = fit_rish_glm(
            rish_image_paths=rish_image_paths[b_value],
            site_labels=site_labels,
            mask_path=mask_path,
            output_dir=shell_dir,
            reference_site=reference_site,
            covariates=covariates,
        )
        result.per_shell[b_value] = shell_result

    # Save per-shell model index
    index = {
        "b_values": b_values,
        "reference_site": reference_site,
        "per_shell_models": {
            b: str(Path(output_dir) / f"b{b}" / "rish_glm_model" / "rish_glm_model.json")
            for b in b_values
        },
    }
    index_path = output_dir / "rish_glm_per_shell.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    return result


def compute_glm_scale_maps_per_shell(
    result: RISHGLMPerShellResult,
    target_site: str,
    output_dir: str,
    mask: Optional[str] = None,
    smoothing_fwhm: float = 3.0,
    clip_range: Tuple[float, float] = (0.5, 2.0),
    min_signal: float = 0.01,
    n_threads: int = 1,
) -> Dict[int, Dict[int, str]]:
    """Compute scale maps from per-shell RISH-GLM results.

    Parameters
    ----------
    result : RISHGLMPerShellResult
        Fitted per-shell RISH-GLM model
    target_site : str
        Site to harmonize
    output_dir : str
        Output directory for scale maps
    mask : str, optional
        Brain mask
    smoothing_fwhm : float
        Gaussian smoothing FWHM in mm
    clip_range : tuple
        (min, max) scale factor clipping
    min_signal : float
        Minimum denominator threshold
    n_threads : int
        Number of threads

    Returns
    -------
    dict
        {b_value: {order: scale_map_path}}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scale_maps = {}
    for b_value in result.b_values:
        shell_dir = str(output_dir / f"b{b_value}")
        shell_scales = compute_glm_scale_maps(
            result=result.per_shell[b_value],
            target_site=target_site,
            output_dir=shell_dir,
            mask=mask,
            smoothing_fwhm=smoothing_fwhm,
            clip_range=clip_range,
            min_signal=min_signal,
            n_threads=n_threads,
        )
        scale_maps[b_value] = shell_scales

    return scale_maps


def load_rish_glm_per_shell(path: str) -> RISHGLMPerShellResult:
    """Load per-shell RISH-GLM model from index JSON.

    Parameters
    ----------
    path : str
        Path to rish_glm_per_shell.json

    Returns
    -------
    RISHGLMPerShellResult
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Per-shell RISH-GLM index not found: {path}")

    with open(path) as f:
        index = json.load(f)

    result = RISHGLMPerShellResult(
        b_values=index["b_values"],
        output_dir=str(path.parent),
    )

    for b_str, model_path in index["per_shell_models"].items():
        b_value = int(b_str)
        result.per_shell[b_value] = load_rish_glm_model(model_path)

    return result
