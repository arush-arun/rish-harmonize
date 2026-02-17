"""
B-Shell Detection and Separation Module

Detects b-value shells in multi-shell DWI data and provides
utilities for separating/rejoining per-shell images.
"""

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _run_cmd(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command."""
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\nstderr: {result.stderr.strip()}"
        )
    return result


@dataclass
class ShellInfo:
    """Information about b-value shells in a DWI image.

    Attributes
    ----------
    b_values : list of int
        Unique non-zero b-values (rounded to nearest 50).
    shell_indices : dict
        b_value -> list of volume indices for that shell.
    b0_indices : list of int
        Volume indices for b=0 images.
    gradient_table : ndarray
        Full gradient table (N x 4: x, y, z, b).
    raw_b_values : ndarray
        Per-volume b-values before rounding.
    """
    b_values: List[int] = field(default_factory=list)
    shell_indices: Dict[int, List[int]] = field(default_factory=dict)
    b0_indices: List[int] = field(default_factory=list)
    gradient_table: Optional[np.ndarray] = None
    raw_b_values: Optional[np.ndarray] = None

    @property
    def n_shells(self) -> int:
        """Number of non-zero b-value shells."""
        return len(self.b_values)

    @property
    def n_volumes(self) -> int:
        """Total number of DWI volumes."""
        total = len(self.b0_indices)
        for indices in self.shell_indices.values():
            total += len(indices)
        return total

    def n_directions(self, b_value: int) -> int:
        """Number of diffusion directions for a given shell."""
        return len(self.shell_indices.get(b_value, []))


def detect_shells(
    dwi_image: str,
    b0_threshold: float = 50.0,
    shell_tolerance: float = 100.0,
) -> ShellInfo:
    """Detect b-value shells from a DWI image.

    Parameters
    ----------
    dwi_image : str
        Path to DWI image with gradient information.
    b0_threshold : float
        B-values below this are considered b=0.
    shell_tolerance : float
        B-values within this tolerance are grouped into the same shell.

    Returns
    -------
    ShellInfo
        Detected shell information.
    """
    # Export gradient table
    result = _run_cmd(["mrinfo", "-dwgrad", dwi_image])
    lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]

    if not lines:
        raise ValueError(f"No gradient information found in {dwi_image}")

    grad_table = np.array(
        [[float(x) for x in line.split()] for line in lines]
    )

    if grad_table.shape[1] < 4:
        raise ValueError(
            f"Expected 4-column gradient table, got {grad_table.shape[1]} columns"
        )

    raw_b_values = grad_table[:, 3]

    # Identify b=0 volumes
    b0_indices = list(np.where(raw_b_values < b0_threshold)[0])

    # Cluster non-zero b-values into shells
    dw_indices = np.where(raw_b_values >= b0_threshold)[0]
    dw_b_values = raw_b_values[dw_indices]

    if len(dw_b_values) == 0:
        return ShellInfo(
            b_values=[],
            shell_indices={},
            b0_indices=b0_indices,
            gradient_table=grad_table,
            raw_b_values=raw_b_values,
        )

    # Sort unique b-values and cluster
    sorted_unique = np.sort(np.unique(np.round(dw_b_values / 50) * 50))
    shells = []
    current_cluster = [sorted_unique[0]]

    for i in range(1, len(sorted_unique)):
        if sorted_unique[i] - current_cluster[-1] <= shell_tolerance:
            current_cluster.append(sorted_unique[i])
        else:
            shells.append(int(np.mean(current_cluster)))
            current_cluster = [sorted_unique[i]]
    shells.append(int(np.mean(current_cluster)))

    # Assign volumes to shells
    shell_indices = {b: [] for b in shells}
    for idx in dw_indices:
        b = raw_b_values[idx]
        # Find closest shell
        best_shell = min(shells, key=lambda s: abs(s - b))
        if abs(best_shell - b) <= shell_tolerance:
            shell_indices[best_shell].append(int(idx))

    return ShellInfo(
        b_values=shells,
        shell_indices=shell_indices,
        b0_indices=b0_indices,
        gradient_table=grad_table,
        raw_b_values=raw_b_values,
    )


def separate_shell(
    dwi_image: str,
    shell_info: ShellInfo,
    b_value: int,
    output: str,
    include_b0: bool = True,
    n_threads: int = 1,
) -> str:
    """Extract a single b-shell from multi-shell DWI.

    Parameters
    ----------
    dwi_image : str
        Input multi-shell DWI.
    shell_info : ShellInfo
        Shell information from detect_shells().
    b_value : int
        Target b-value shell.
    output : str
        Output path for single-shell DWI.
    include_b0 : bool
        Whether to include b=0 volumes.
    n_threads : int
        Number of threads.

    Returns
    -------
    str
        Path to extracted single-shell image.
    """
    if b_value not in shell_info.shell_indices:
        raise ValueError(
            f"Shell b={b_value} not found. Available: {shell_info.b_values}"
        )

    Path(output).parent.mkdir(parents=True, exist_ok=True)

    # Build volume index list
    indices = []
    if include_b0:
        indices.extend(shell_info.b0_indices)
    indices.extend(shell_info.shell_indices[b_value])
    indices.sort()

    coord_str = ",".join(str(i) for i in indices)
    thread_opt = ["-nthreads", str(n_threads)] if n_threads > 1 else []

    _run_cmd([
        "mrconvert", dwi_image,
        "-coord", "3", coord_str,
        output, "-force",
    ] + thread_opt)

    return output


def separate_all_shells(
    dwi_image: str,
    output_dir: str,
    shell_info: Optional[ShellInfo] = None,
    include_b0: bool = True,
    n_threads: int = 1,
) -> Dict[int, str]:
    """Separate DWI into per-shell images.

    Parameters
    ----------
    dwi_image : str
        Input multi-shell DWI.
    output_dir : str
        Output directory.
    shell_info : ShellInfo, optional
        Pre-computed shell info (detected if None).
    include_b0 : bool
        Include b=0 volumes in each shell image.
    n_threads : int
        Number of threads.

    Returns
    -------
    dict
        b_value -> path to single-shell image.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if shell_info is None:
        shell_info = detect_shells(dwi_image)

    shell_images = {}
    for b_value in shell_info.b_values:
        output_path = str(output_dir / f"dwi_b{b_value}.mif")
        separate_shell(
            dwi_image, shell_info, b_value, output_path,
            include_b0=include_b0, n_threads=n_threads,
        )
        shell_images[b_value] = output_path

    return shell_images


def extract_b0(
    dwi_image: str,
    output: str,
    n_threads: int = 1,
) -> str:
    """Extract b=0 volumes from DWI using dwiextract.

    Parameters
    ----------
    dwi_image : str
        Input DWI.
    output : str
        Output path.
    n_threads : int
        Number of threads.

    Returns
    -------
    str
        Path to b=0 image.
    """
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    thread_opt = ["-nthreads", str(n_threads)] if n_threads > 1 else []

    _run_cmd([
        "dwiextract", dwi_image, "-bzero",
        output, "-force",
    ] + thread_opt)

    return output


def rejoin_shells(
    harmonized_shell_images: Dict[int, str],
    b0_image: str,
    output: str,
    shell_info: ShellInfo,
    n_threads: int = 1,
) -> str:
    """Reassemble per-shell harmonized DWI into original volume order.

    Parameters
    ----------
    harmonized_shell_images : dict
        b_value -> path to harmonized single-shell DWI.
    b0_image : str
        Path to b=0 image (unharmonized).
    output : str
        Output path for reassembled DWI.
    shell_info : ShellInfo
        Original shell information (for volume ordering).
    n_threads : int
        Number of threads.

    Returns
    -------
    str
        Path to reassembled DWI.
    """
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    thread_opt = ["-nthreads", str(n_threads)] if n_threads > 1 else []

    # Build list of (original_index, source_image, index_within_source)
    # for each volume in the original DWI
    volume_map = []  # (original_idx, source_path, source_idx)

    # Track indices within each source
    b0_counter = 0
    shell_counters = {b: 0 for b in shell_info.b_values}

    # Map each original volume to its source
    for orig_idx in range(shell_info.n_volumes):
        if orig_idx in shell_info.b0_indices:
            volume_map.append((orig_idx, b0_image, b0_counter))
            b0_counter += 1
        else:
            for b_value in shell_info.b_values:
                if orig_idx in shell_info.shell_indices[b_value]:
                    # In the separated shell image, b0 volumes come first
                    # then DW volumes. We need to find the index within
                    # the harmonized image (which has no b0).
                    # The harmonized shell images have only DW volumes
                    # (no b0), because harmonization operates on SH which
                    # is derived from DW directions only.
                    src_idx = shell_counters[b_value]
                    volume_map.append((orig_idx, harmonized_shell_images[b_value], src_idx))
                    shell_counters[b_value] += 1
                    break

    # Concatenate in original order using mrcat
    # For simplicity, extract each volume individually and concatenate
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        vol_paths = []
        for orig_idx, src_path, src_idx in volume_map:
            vol_path = str(Path(tmpdir) / f"vol_{orig_idx:04d}.mif")
            _run_cmd([
                "mrconvert", src_path,
                "-coord", "3", str(src_idx),
                vol_path, "-force",
            ] + thread_opt)
            vol_paths.append(vol_path)

        _run_cmd([
            "mrcat", *vol_paths,
            "-axis", "3",
            output, "-force",
        ] + thread_opt)

    return output
