"""
Covariate Regression Module

Fits voxel-wise regression to RISH features to remove demographic
confounds, producing covariate-free templates and adjusted features
for harmonization.
"""

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import subprocess


def run_mrtrix_cmd(cmd, check=True):
    """Run an MRtrix3 command."""
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


@dataclass
class CovariateModel:
    """Stores fitted covariate regression model.

    Attributes
    ----------
    covariate_names : list of str
        Names of covariates in model
    orders : list of int
        SH orders in model (e.g., [0, 2, 4, 6, 8])
    means : dict
        Covariate name -> population mean (for z-scoring)
    stds : dict
        Covariate name -> population std (for z-scoring)
    beta_paths : dict
        (order, covariate_name) -> path to beta map (MIF)
    intercept_paths : dict
        order -> path to intercept map (MIF)
    mask_path : str or None
        Path to mask used during fitting
    n_subjects : int
        Number of subjects used in fit
    output_dir : str
        Directory where model files are stored
    """
    covariate_names: List[str] = field(default_factory=list)
    orders: List[int] = field(default_factory=list)
    means: Dict[str, float] = field(default_factory=dict)
    stds: Dict[str, float] = field(default_factory=dict)
    beta_paths: Dict[str, str] = field(default_factory=dict)
    intercept_paths: Dict[int, str] = field(default_factory=dict)
    mask_path: Optional[str] = None
    n_subjects: int = 0
    output_dir: str = ""


def save_covariate_model(model: CovariateModel, path: str) -> None:
    """Save covariate model to JSON.

    Paths are stored relative to the JSON file's directory.

    Parameters
    ----------
    model : CovariateModel
        Fitted model to save
    path : str
        Output JSON path
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
        "covariate_names": model.covariate_names,
        "orders": model.orders,
        "means": model.means,
        "stds": model.stds,
        "beta_paths": {k: _rel(v) for k, v in model.beta_paths.items()},
        "intercept_paths": {str(k): _rel(v) for k, v in model.intercept_paths.items()},
        "mask_path": _rel(model.mask_path),
        "n_subjects": model.n_subjects,
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_covariate_model(path: str) -> CovariateModel:
    """Load covariate model from JSON.

    Relative paths are resolved against the JSON file's directory.

    Parameters
    ----------
    path : str
        Path to covariate_model.json

    Returns
    -------
    CovariateModel
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Covariate model not found: {path}")

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

    model = CovariateModel(
        covariate_names=data["covariate_names"],
        orders=data["orders"],
        means=data["means"],
        stds=data["stds"],
        beta_paths={k: _abs(v) for k, v in data["beta_paths"].items()},
        intercept_paths={int(k): _abs(v) for k, v in data["intercept_paths"].items()},
        mask_path=_abs(data.get("mask_path")),
        n_subjects=data.get("n_subjects", 0),
        output_dir=str(base_dir),
    )

    return model


def _standardize_covariates(
    covariates: Dict[str, List[float]]
) -> Tuple[Dict[str, List[float]], Dict[str, float], Dict[str, float]]:
    """Z-score covariates and return standardization parameters.

    Returns
    -------
    z_covariates : dict
        Standardized covariate values
    means : dict
        Per-covariate means
    stds : dict
        Per-covariate stds
    """
    z_covariates = {}
    means = {}
    stds = {}

    for name, values in covariates.items():
        arr = np.array(values, dtype=np.float64)
        m = float(arr.mean())
        s = float(arr.std())
        if s < 1e-10:
            s = 1.0  # Avoid division by zero for constant covariates
        means[name] = m
        stds[name] = s
        z_covariates[name] = ((arr - m) / s).tolist()

    return z_covariates, means, stds


def fit_covariate_model(
    rish_image_paths: Dict[int, List[str]],
    covariates: Dict[str, List[float]],
    mask_path: Optional[str],
    output_dir: str,
) -> CovariateModel:
    """Fit voxel-wise regression of covariates on RISH features.

    For each SH order l and each voxel v:
        rish_l(v) = intercept(v) + sum_k beta_k(v) * z_k + noise

    Parameters
    ----------
    rish_image_paths : dict
        order -> list of RISH image paths (one per subject).
        E.g., {0: [sub0_l0.mif, sub1_l0.mif], 2: [sub0_l2.mif, ...]}
    covariates : dict
        covariate_name -> list of values (one per subject, same order as images)
    mask_path : str or None
        Brain mask for restricting computation
    output_dir : str
        Directory for saving beta maps and model

    Returns
    -------
    CovariateModel
        Fitted model with beta maps saved to disk
    """
    from ..qc.glm import load_image_to_matrix, save_vector_to_image

    output_dir = Path(output_dir)
    model_dir = output_dir / "covariate_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    orders = sorted(rish_image_paths.keys())
    n_subjects = len(next(iter(rish_image_paths.values())))
    cov_names = sorted(covariates.keys())

    # Standardize covariates
    z_covariates, means, stds = _standardize_covariates(covariates)

    # Build design matrix: [intercept, z_cov1, z_cov2, ...]
    design = np.ones((n_subjects, 1 + len(cov_names)), dtype=np.float64)
    for j, name in enumerate(cov_names):
        design[:, 1 + j] = z_covariates[name]

    model = CovariateModel(
        covariate_names=cov_names,
        orders=orders,
        means=means,
        stds=stds,
        n_subjects=n_subjects,
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

        # Solve: design @ betas = data  =>  betas = lstsq(design, data)
        betas, _, _, _ = np.linalg.lstsq(design, data, rcond=None)
        # betas shape: (1 + n_covariates, n_voxels)

        # Save intercept
        intercept_path = str(model_dir / f"intercept_l{l}.mif")
        save_vector_to_image(
            betas[0], mask_indices, image_shape, paths[0], intercept_path
        )
        model.intercept_paths[l] = intercept_path

        # Save beta maps for each covariate
        for j, name in enumerate(cov_names):
            beta_path = str(model_dir / f"beta_{name}_l{l}.mif")
            save_vector_to_image(
                betas[1 + j], mask_indices, image_shape, paths[0], beta_path
            )
            key = f"{l}_{name}"
            model.beta_paths[key] = beta_path

    # Save model metadata
    save_covariate_model(model, str(model_dir / "covariate_model.json"))

    return model


def adjust_rish_features(
    rish_paths: Dict[int, str],
    subject_covariates: Dict[str, float],
    model: CovariateModel,
    output_dir: str,
    mask_path: Optional[str] = None,
) -> Dict[int, str]:
    """Adjust a single subject's RISH features by removing covariate effects.

    adjusted_l = rish_l - sum_k beta_k * z_k

    Uses MRtrix3 mrcalc chains for memory-efficient computation.

    Parameters
    ----------
    rish_paths : dict
        order -> path to subject's RISH image
    subject_covariates : dict
        covariate_name -> raw value for this subject
    model : CovariateModel
        Fitted covariate model
    output_dir : str
        Output directory for adjusted RISH images
    mask_path : str, optional
        Brain mask (if None, uses model's mask)

    Returns
    -------
    dict
        order -> path to adjusted RISH image
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if mask_path is None:
        mask_path = model.mask_path

    # Z-score subject covariates using reference population params
    z_values = {}
    for name in model.covariate_names:
        if name not in subject_covariates:
            raise ValueError(
                f"Missing covariate '{name}' for subject. "
                f"Required: {model.covariate_names}"
            )
        raw = float(subject_covariates[name])
        z_values[name] = (raw - model.means[name]) / model.stds[name]

    adjusted = {}
    for l in model.orders:
        if l not in rish_paths:
            raise ValueError(f"Missing RISH image for order {l}")

        rish_path = rish_paths[l]
        out_path = str(output_dir / f"rish_l{l}_adjusted.mif")

        # Build mrcalc chain: rish - beta1*z1 - beta2*z2 - ...
        # Start with the original RISH image
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            current = rish_path

            for name in model.covariate_names:
                key = f"{l}_{name}"
                beta_path = model.beta_paths[key]
                z_val = z_values[name]

                # tmp = beta * z_value
                tmp_scaled = str(tmpdir / f"beta_{name}_scaled.mif")
                run_mrtrix_cmd([
                    "mrcalc", beta_path, str(z_val), "-mult",
                    tmp_scaled, "-force"
                ])

                # current = current - tmp_scaled
                tmp_sub = str(tmpdir / f"after_{name}.mif")
                run_mrtrix_cmd([
                    "mrcalc", current, tmp_scaled, "-sub",
                    tmp_sub, "-force"
                ])
                current = tmp_sub

            # Clamp to small positive floor (RISH is non-negative)
            floor_val = "1e-6"
            run_mrtrix_cmd([
                "mrcalc", current, floor_val, "-max",
                out_path, "-force"
            ])

        adjusted[l] = out_path

    return adjusted


def adjust_rish_features_batch(
    rish_paths_list: List[Dict[int, str]],
    covariates: Dict[str, List[float]],
    model: CovariateModel,
    output_dir: str,
    mask_path: Optional[str] = None,
) -> List[Dict[int, str]]:
    """Adjust RISH features for multiple subjects.

    Parameters
    ----------
    rish_paths_list : list of dict
        Per-subject RISH paths: [sub0: {0: path, 2: path, ...}, ...]
    covariates : dict
        covariate_name -> list of values (same order as rish_paths_list)
    model : CovariateModel
        Fitted covariate model
    output_dir : str
        Output directory
    mask_path : str, optional
        Brain mask

    Returns
    -------
    list of dict
        Per-subject adjusted RISH paths
    """
    output_dir = Path(output_dir)
    n_subjects = len(rish_paths_list)

    adjusted_list = []
    for i in range(n_subjects):
        subj_covs = {name: vals[i] for name, vals in covariates.items()}
        subj_dir = output_dir / f"sub-{i:03d}"

        adj = adjust_rish_features(
            rish_paths_list[i], subj_covs, model, str(subj_dir), mask_path
        )
        adjusted_list.append(adj)

    return adjusted_list
