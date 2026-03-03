"""
Per-Subject QC Metrics and Outlier Flagging

Computes summary statistics for each subject's RISH features and scale maps,
flags outliers that may indicate bad registration, motion artifacts, or
acquisition errors.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SubjectQCMetrics:
    """QC metrics for a single subject."""
    subject_id: str
    site: str
    metrics: Dict[str, float] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "subject_id": self.subject_id,
            "site": self.site,
            "metrics": self.metrics,
            "flags": self.flags,
        }


def compute_rish_subject_stats(
    rish_image_paths: Dict[int, List[str]],
    subject_ids: List[str],
    site_labels: List[str],
    mask_path: Optional[str] = None,
) -> List[SubjectQCMetrics]:
    """Compute per-subject RISH summary statistics.

    For each subject, computes mean RISH l0 (and other orders) within
    the mask. Flags subjects whose values deviate >2 SD from their
    site's mean.

    Parameters
    ----------
    rish_image_paths : dict
        {order: [path_sub0, path_sub1, ...]} — RISH images per order.
    subject_ids : list of str
        Subject identifiers.
    site_labels : list of str
        Site label per subject.
    mask_path : str, optional
        Brain mask path.

    Returns
    -------
    list of SubjectQCMetrics
        Per-subject metrics and flags.
    """
    from .glm import load_image_to_matrix

    n_subjects = len(subject_ids)
    qc_list = [
        SubjectQCMetrics(subject_id=subject_ids[i], site=site_labels[i])
        for i in range(n_subjects)
    ]

    # Compute stats per SH order
    for order in sorted(rish_image_paths.keys()):
        paths = rish_image_paths[order]
        if len(paths) != n_subjects:
            continue

        data, mask_indices, image_shape = load_image_to_matrix(paths, mask_path)

        # Per-subject: mean RISH value within mask
        subject_means = data.mean(axis=1)

        for i in range(n_subjects):
            qc_list[i].metrics[f"mean_rish_l{order}"] = float(subject_means[i])

        # Flag outliers per site: >2 SD from site mean
        unique_sites = sorted(set(site_labels))
        for site in unique_sites:
            site_idx = [j for j, s in enumerate(site_labels) if s == site]
            if len(site_idx) < 3:
                continue  # Need >=3 subjects for meaningful outlier detection
            site_means = subject_means[site_idx]
            mu = site_means.mean()
            sigma = site_means.std()
            if sigma < 1e-10:
                continue
            for j in site_idx:
                z = abs(subject_means[j] - mu) / sigma
                if z > 2.0:
                    qc_list[j].flags.append(
                        f"l{order}_outlier (z={z:.2f}, "
                        f"mean={subject_means[j]:.4f}, "
                        f"site_mean={mu:.4f})"
                    )

    return qc_list


def compute_scale_map_subject_stats(
    scale_map_paths: Dict[int, str],
    subject_rish_paths: Dict[int, str],
    subject_id: str,
    mask_path: Optional[str] = None,
) -> Dict[str, float]:
    """Compute per-subject scale map application statistics.

    Checks whether the scale maps applied to this subject would result
    in extreme corrections.

    Parameters
    ----------
    scale_map_paths : dict
        {order: scale_map_path} for this subject's site.
    subject_rish_paths : dict
        {order: rish_path} for this subject.
    mask_path : str, optional
        Brain mask.

    Returns
    -------
    dict
        Per-order statistics of the effective correction.
    """
    from ..core.scale_maps import _load_mif_as_array, _load_mask_as_array

    stats = {}
    mask = _load_mask_as_array(mask_path) if mask_path else None

    for order in sorted(scale_map_paths.keys()):
        scale_data = _load_mif_as_array(scale_map_paths[order])
        if mask is not None:
            values = scale_data[mask]
        else:
            values = scale_data[scale_data != 0]

        if len(values) == 0:
            continue

        stats[f"scale_l{order}_mean"] = float(np.mean(values))
        stats[f"scale_l{order}_std"] = float(np.std(values))
        stats[f"scale_l{order}_pct_extreme"] = float(
            np.mean((values < 0.6) | (values > 1.8)) * 100
        )

    return stats


def save_subject_qc(
    qc_list: List[SubjectQCMetrics],
    output_path: str,
) -> str:
    """Save per-subject QC metrics to JSON.

    Parameters
    ----------
    qc_list : list of SubjectQCMetrics
        Per-subject QC results.
    output_path : str
        Output JSON path.

    Returns
    -------
    str
        Path to saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "n_subjects": len(qc_list),
        "n_flagged": sum(1 for q in qc_list if q.flags),
        "subjects": [q.to_dict() for q in qc_list],
    }

    # Summary of flagged subjects
    flagged = [q for q in qc_list if q.flags]
    if flagged:
        data["flagged_subjects"] = [q.subject_id for q in flagged]

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    return str(output_path)
