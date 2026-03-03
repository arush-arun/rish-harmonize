"""Synthetic data fixtures for integration tests.

Creates small (10x10x10) multi-shell DWI images with known properties
for end-to-end pipeline testing. Requires MRtrix3 to be installed.

All tests in this directory are marked with ``pytest.mark.integration``
and skipped by default. Run with::

    pytest -m integration
"""

import json
import shutil
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


def _mrtrix_available() -> bool:
    """Check if MRtrix3 is installed."""
    try:
        subprocess.run(
            ["mrinfo", "--version"],
            capture_output=True, check=False,
        )
        return True
    except FileNotFoundError:
        return False


# Skip all integration tests if MRtrix3 is not available
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _mrtrix_available(),
        reason="MRtrix3 not installed",
    ),
]


def _generate_gradient_directions(n_dirs: int, seed: int = 0) -> np.ndarray:
    """Generate roughly uniform gradient directions on unit sphere.

    Uses a Fibonacci spiral for repeatability.

    Returns
    -------
    ndarray, shape (n_dirs, 3)
        Unit vectors on the sphere.
    """
    indices = np.arange(n_dirs, dtype=np.float64)
    golden_ratio = (1 + np.sqrt(5)) / 2
    theta = np.arccos(1 - 2 * (indices + 0.5) / n_dirs)
    phi = 2 * np.pi * indices / golden_ratio

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return np.column_stack([x, y, z])


def _create_synthetic_dwi(
    output_path: str,
    shape: tuple = (10, 10, 10),
    b_values: list = None,
    n_dirs_per_shell: int = 30,
    n_b0: int = 3,
    base_signal: float = 1000.0,
    site_effect: dict = None,
    mask: np.ndarray = None,
    seed: int = 42,
) -> str:
    """Create a synthetic multi-shell DWI image.

    Parameters
    ----------
    output_path : str
        Output .mif path.
    shape : tuple
        Spatial dimensions.
    b_values : list of int
        B-value shells (default: [1000, 2000]).
    n_dirs_per_shell : int
        Number of gradient directions per shell.
    n_b0 : int
        Number of b=0 volumes.
    base_signal : float
        Baseline signal intensity.
    site_effect : dict, optional
        {b_value: {order: multiplier}} — multiplicative site effect
        applied to the RISH features. e.g., {1000: {0: 1.2, 2: 1.1}}.
    mask : ndarray, optional
        Brain mask. If None, uses a sphere.
    seed : int
        Random seed.

    Returns
    -------
    str
        Path to created .mif file.
    """
    rng = np.random.RandomState(seed)

    if b_values is None:
        b_values = [1000, 2000]

    if mask is None:
        # Create spherical mask
        coords = np.mgrid[:shape[0], :shape[1], :shape[2]].astype(float)
        center = np.array(shape) / 2
        dist = np.sqrt(sum((c - ctr) ** 2 for c, ctr in zip(coords, center)))
        mask = dist < (min(shape) / 2 - 1)

    # Generate gradient table
    grad_lines = []
    all_volumes = []

    # b=0 volumes
    for _ in range(n_b0):
        vol = np.zeros(shape, dtype=np.float32)
        vol[mask] = base_signal + rng.normal(0, base_signal * 0.01, mask.sum())
        all_volumes.append(vol)
        grad_lines.append("0 0 1 0")

    # DW volumes per shell
    for b in b_values:
        dirs = _generate_gradient_directions(n_dirs_per_shell, seed=b)
        for d in range(n_dirs_per_shell):
            # Simple mono-exponential signal decay
            adc = 0.7e-3  # apparent diffusion coefficient
            attenuation = np.exp(-b * adc)
            signal = base_signal * attenuation

            # Add some spatial variation
            vol = np.zeros(shape, dtype=np.float32)
            x_coords = np.arange(shape[0])[:, None, None] * np.ones(shape)
            spatial_var = 1.0 + 0.1 * np.sin(2 * np.pi * x_coords / shape[0])
            vol[mask] = (signal * spatial_var[mask]
                         + rng.normal(0, signal * 0.02, mask.sum()))

            # Apply site effect if specified
            if site_effect and b in site_effect:
                # Simple multiplicative effect on overall signal
                for order, mult in site_effect[b].items():
                    if order == 0:
                        vol[mask] *= mult

            all_volumes.append(vol)
            x, y, z = dirs[d]
            grad_lines.append(f"{x:.6f} {y:.6f} {z:.6f} {b}")

    # Stack into 4D image
    data_4d = np.stack(all_volumes, axis=-1)

    # Save as NIfTI
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nii_path = str(output_path.with_suffix(".nii"))
    affine = np.eye(4) * 2.0
    affine[3, 3] = 1.0
    img = nib.Nifti1Image(data_4d, affine)
    nib.save(img, nii_path)

    # Save gradient table
    grad_path = str(output_path.with_suffix(".txt"))
    with open(grad_path, "w") as f:
        f.write("\n".join(grad_lines) + "\n")

    # Convert to MRtrix format with gradient table
    subprocess.run(
        ["mrconvert", nii_path, str(output_path),
         "-grad", grad_path, "-force"],
        capture_output=True, check=True,
    )

    # Clean up intermediate files
    Path(nii_path).unlink(missing_ok=True)
    Path(grad_path).unlink(missing_ok=True)

    return str(output_path)


def _create_mask(shape: tuple, output_path: str) -> str:
    """Create a spherical brain mask."""
    coords = np.mgrid[:shape[0], :shape[1], :shape[2]].astype(float)
    center = np.array(shape) / 2
    dist = np.sqrt(sum((c - ctr) ** 2 for c, ctr in zip(coords, center)))
    mask = (dist < (min(shape) / 2 - 1)).astype(np.float32)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nii_path = str(output_path.with_suffix(".nii"))
    affine = np.eye(4) * 2.0
    affine[3, 3] = 1.0
    img = nib.Nifti1Image(mask, affine)
    nib.save(img, nii_path)

    subprocess.run(
        ["mrconvert", nii_path, str(output_path), "-force"],
        capture_output=True, check=True,
    )
    Path(nii_path).unlink(missing_ok=True)

    return str(output_path)


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory):
    """Session-scoped directory for synthetic test data."""
    return tmp_path_factory.mktemp("synth")


@pytest.fixture(scope="session")
def synth_shape():
    """Spatial dimensions for synthetic data."""
    return (10, 10, 10)


@pytest.fixture(scope="session")
def synth_mask(synth_dir, synth_shape):
    """Shared brain mask for synthetic data."""
    return _create_mask(synth_shape, str(synth_dir / "mask.mif"))


@pytest.fixture(scope="session")
def synth_site_a(synth_dir, synth_shape):
    """Synthetic DWI for site A (reference site)."""
    subjects = []
    for i in range(3):
        path = _create_synthetic_dwi(
            str(synth_dir / f"site_a_sub{i}.mif"),
            shape=synth_shape,
            b_values=[1000, 2000],
            n_dirs_per_shell=15,
            seed=100 + i,
        )
        subjects.append(path)
    return subjects


@pytest.fixture(scope="session")
def synth_site_b(synth_dir, synth_shape):
    """Synthetic DWI for site B (target site with known site effect)."""
    subjects = []
    site_effect = {1000: {0: 1.3}, 2000: {0: 1.3}}
    for i in range(3):
        path = _create_synthetic_dwi(
            str(synth_dir / f"site_b_sub{i}.mif"),
            shape=synth_shape,
            b_values=[1000, 2000],
            n_dirs_per_shell=15,
            site_effect=site_effect,
            seed=200 + i,
        )
        subjects.append(path)
    return subjects
