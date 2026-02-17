"""
Site Effect Statistical Testing Module

Implements statistical tests to quantify and validate removal of site effects
in multi-site neuroimaging studies.

Design follows MRtrix3 core/math/stats/shuffle.{h,cpp} for permutation framework
and core/math/stats/glm.{h,cpp} for statistical testing.

Key features:
- Freedman-Lane permutation testing
- Exchangeability block support
- FDR and cluster-based multiple comparison correction
- Effect size quantification

Reference: https://github.com/MRtrix3/mrtrix3/blob/master/core/math/stats/
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

from .glm import (
    GLMResult,
    Hypothesis,
    TestFixedHomoscedastic,
    TestFixedHeteroscedastic,
    TestOutput,
    check_design,
    create_design_matrix,
    create_site_contrast,
    index_array_type,
    load_image_to_matrix,
    matrix_type,
    save_vector_to_image,
    value_type,
    vector_type,
)


# -----------------------------------------------------------------------------
# Shuffle Class (following MRtrix3 shuffle.h)
# -----------------------------------------------------------------------------

@dataclass
class Shuffle:
    """
    Single shuffle/permutation result.

    Following MRtrix3 Shuffle structure.
    """
    index: int
    data: index_array_type


class Shuffler:
    """
    Permutation generator for statistical testing.

    Supports:
    - Random permutations
    - Exhaustive enumeration (for small n)
    - Exchangeability blocks
    - Sign-flipping (for within-subject designs)

    Following MRtrix3 Shuffler class design.
    """

    def __init__(
        self,
        n_subjects: int,
        n_permutations: int = 5000,
        exchangeability_blocks: Optional[index_array_type] = None,
        seed: Optional[int] = None,
        is_sign_flip: bool = False
    ):
        """
        Initialize shuffler.

        Parameters
        ----------
        n_subjects : int
            Number of observations
        n_permutations : int
            Number of permutations to generate
        exchangeability_blocks : array, optional
            Block indices constraining permutations
        seed : int, optional
            Random seed for reproducibility
        is_sign_flip : bool
            Use sign-flipping instead of permutation
        """
        self._n = n_subjects
        self._n_perms = n_permutations
        self._eb = exchangeability_blocks
        self._is_sign_flip = is_sign_flip

        if seed is not None:
            np.random.seed(seed)

        # Pre-generate permutations
        self._shuffles = self._generate_permutations()
        self._current_idx = 0

    def _generate_permutations(self) -> List[Shuffle]:
        """Generate all permutation indices."""
        shuffles = []

        # First "permutation" is identity (unpermuted)
        shuffles.append(Shuffle(
            index=0,
            data=np.arange(self._n, dtype=np.int64)
        ))

        if self._is_sign_flip:
            shuffles.extend(self._generate_sign_flips())
        else:
            shuffles.extend(self._generate_random_permutations())

        return shuffles

    def _generate_random_permutations(self) -> List[Shuffle]:
        """Generate random permutations, respecting exchangeability blocks."""
        shuffles = []
        seen = set()
        seen.add(tuple(range(self._n)))  # Identity already added

        max_attempts = self._n_perms * 10

        for attempt in range(max_attempts):
            if len(shuffles) >= self._n_perms - 1:  # -1 for identity
                break

            if self._eb is not None:
                # Permute within blocks
                perm = self._permute_within_blocks()
            else:
                perm = np.random.permutation(self._n)

            perm_tuple = tuple(perm)
            if perm_tuple not in seen:
                seen.add(perm_tuple)
                shuffles.append(Shuffle(
                    index=len(shuffles) + 1,
                    data=perm.astype(np.int64)
                ))

        return shuffles

    def _permute_within_blocks(self) -> np.ndarray:
        """Generate permutation respecting exchangeability blocks."""
        perm = np.arange(self._n)
        unique_blocks = np.unique(self._eb)

        for block in unique_blocks:
            block_indices = np.where(self._eb == block)[0]
            # Shuffle within block
            shuffled = np.random.permutation(block_indices)
            perm[block_indices] = shuffled

        return perm

    def _generate_sign_flips(self) -> List[Shuffle]:
        """Generate random sign-flip patterns."""
        shuffles = []

        # For sign-flipping, we store signs as negative indices
        # Positive index = keep sign, negative = flip
        for i in range(self._n_perms - 1):
            signs = np.random.choice([-1, 1], size=self._n)
            # Store as indices with sign encoding
            data = np.arange(self._n) * signs
            shuffles.append(Shuffle(index=i + 1, data=data.astype(np.int64)))

        return shuffles

    def __iter__(self) -> Iterator[Shuffle]:
        """Iterate over permutations."""
        self._current_idx = 0
        return self

    def __next__(self) -> Shuffle:
        """Get next permutation."""
        if self._current_idx >= len(self._shuffles):
            raise StopIteration
        shuffle = self._shuffles[self._current_idx]
        self._current_idx += 1
        return shuffle

    def __len__(self) -> int:
        """Number of permutations."""
        return len(self._shuffles)

    def reset(self) -> None:
        """Reset iterator to beginning."""
        self._current_idx = 0

    @property
    def size(self) -> int:
        """Total number of permutations."""
        return len(self._shuffles)


# -----------------------------------------------------------------------------
# Site Effect Result Classes
# -----------------------------------------------------------------------------

@dataclass
class SiteEffectResult:
    """
    Complete results from site effect testing.

    Stores statistics, p-values, and summary metrics.
    """
    # Test configuration
    n_subjects: int
    n_sites: int
    n_voxels: int
    n_permutations: int
    site_labels: List[str]
    unique_sites: List[str]

    # Primary statistics (observed)
    f_statistic: vector_type
    p_values_parametric: vector_type

    # Permutation results
    p_values_permutation: Optional[vector_type] = None
    null_max_distribution: Optional[vector_type] = None

    # Corrected p-values
    p_values_fdr: Optional[vector_type] = None
    fdr_threshold: Optional[float] = None

    # Effect sizes
    effect_size: Optional[vector_type] = None  # Partial eta-squared
    cohens_f: Optional[vector_type] = None

    # Summary statistics
    percent_significant_uncorrected: Optional[float] = None
    percent_significant_fdr: Optional[float] = None
    percent_significant_permutation: Optional[float] = None
    mean_effect_size: Optional[float] = None
    median_effect_size: Optional[float] = None

    # Spatial info for reconstruction
    mask_indices: Optional[index_array_type] = None
    image_shape: Optional[Tuple[int, ...]] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "n_subjects": self.n_subjects,
            "n_sites": self.n_sites,
            "n_voxels": self.n_voxels,
            "n_permutations": self.n_permutations,
            "unique_sites": self.unique_sites,
            "percent_significant_uncorrected": self.percent_significant_uncorrected,
            "percent_significant_fdr": self.percent_significant_fdr,
            "percent_significant_permutation": self.percent_significant_permutation,
            "mean_effect_size": float(self.mean_effect_size) if self.mean_effect_size else None,
            "median_effect_size": float(self.median_effect_size) if self.median_effect_size else None,
            "fdr_threshold": float(self.fdr_threshold) if self.fdr_threshold else None,
        }


# -----------------------------------------------------------------------------
# Multiple Comparison Correction
# -----------------------------------------------------------------------------

def fdr_correction(
    p_values: vector_type,
    alpha: float = 0.05,
    method: str = "bh"
) -> Tuple[vector_type, float, np.ndarray]:
    """
    Apply FDR correction (Benjamini-Hochberg or Benjamini-Yekutieli).

    Parameters
    ----------
    p_values : vector_type
        Raw p-values
    alpha : float
        Desired FDR level
    method : str
        "bh" for Benjamini-Hochberg, "by" for Benjamini-Yekutieli

    Returns
    -------
    q_values : vector_type
        FDR-adjusted p-values
    threshold : float
        P-value threshold for significance
    significant : ndarray
        Boolean mask of significant voxels
    """
    valid_mask = ~np.isnan(p_values) & (p_values >= 0) & (p_values <= 1)
    valid_p = p_values[valid_mask]
    n = len(valid_p)

    if n == 0:
        return p_values.copy(), 0.0, np.zeros_like(p_values, dtype=bool)

    # Sort p-values
    sorted_idx = np.argsort(valid_p)
    sorted_p = valid_p[sorted_idx]

    # Compute critical values
    ranks = np.arange(1, n + 1)
    if method == "by":
        # Benjamini-Yekutieli (more conservative)
        c_m = np.sum(1.0 / ranks)
        critical = (ranks / (n * c_m)) * alpha
    else:
        # Benjamini-Hochberg
        critical = (ranks / n) * alpha

    # Find threshold
    below_critical = sorted_p <= critical
    if not below_critical.any():
        threshold = 0.0
    else:
        max_idx = np.where(below_critical)[0][-1]
        threshold = sorted_p[max_idx]

    # Compute q-values (adjusted p-values)
    q_values = np.full_like(p_values, np.nan)
    q_valid = np.minimum.accumulate(
        (sorted_p * n / ranks)[::-1]
    )[::-1]
    q_valid = np.minimum(q_valid, 1.0)

    # Unsort
    q_unsorted = np.zeros(n)
    q_unsorted[sorted_idx] = q_valid
    q_values[valid_mask] = q_unsorted

    # Significance mask
    significant = np.zeros_like(p_values, dtype=bool)
    significant[valid_mask] = valid_p <= threshold

    return q_values, threshold, significant


def permutation_p_values(
    observed: vector_type,
    null_distribution: matrix_type,
    tail: str = "right"
) -> vector_type:
    """
    Compute p-values from permutation null distribution.

    Parameters
    ----------
    observed : vector_type
        Observed test statistics
    null_distribution : matrix_type
        Null distribution (n_permutations × n_voxels)
    tail : str
        "right" (F-test), "left", or "two" (t-test)

    Returns
    -------
    p_values : vector_type
        Permutation-based p-values
    """
    n_perms = null_distribution.shape[0]

    if tail == "right":
        # Count permutations >= observed
        count = np.sum(null_distribution >= observed, axis=0)
    elif tail == "left":
        count = np.sum(null_distribution <= observed, axis=0)
    else:  # two-tailed
        count = np.sum(np.abs(null_distribution) >= np.abs(observed), axis=0)

    # Include observed in count (+1 for observed itself)
    p_values = (count + 1) / (n_perms + 1)

    return p_values


def max_statistic_correction(
    observed: vector_type,
    null_distribution: matrix_type
) -> vector_type:
    """
    Apply max-statistic FWER correction.

    Parameters
    ----------
    observed : vector_type
        Observed statistics
    null_distribution : matrix_type
        Null distribution (n_perms × n_voxels)

    Returns
    -------
    p_values_corrected : vector_type
        FWER-corrected p-values
    """
    # Max across voxels for each permutation
    null_max = np.max(null_distribution, axis=1)

    # P-value = proportion of permutation maxima >= observed
    p_corrected = np.array([
        np.mean(null_max >= obs) for obs in observed
    ])

    return p_corrected


# -----------------------------------------------------------------------------
# Effect Size Computation
# -----------------------------------------------------------------------------

def compute_partial_eta_squared(
    data: matrix_type,
    site_labels: List[str]
) -> vector_type:
    """
    Compute partial eta-squared effect size.

    eta_p^2 = SS_between / (SS_between + SS_within)

    Parameters
    ----------
    data : matrix_type
        Data matrix (n_subjects × n_voxels)
    site_labels : list
        Site label per subject

    Returns
    -------
    eta_sq : vector_type
        Partial eta-squared per voxel
    """
    unique_sites = sorted(set(site_labels))
    site_indices = {
        s: np.array([i for i, l in enumerate(site_labels) if l == s])
        for s in unique_sites
    }

    grand_mean = data.mean(axis=0)

    # Between-group SS
    ss_between = np.zeros(data.shape[1])
    for site, indices in site_indices.items():
        n_site = len(indices)
        site_mean = data[indices].mean(axis=0)
        ss_between += n_site * (site_mean - grand_mean) ** 2

    # Within-group SS
    ss_within = np.zeros(data.shape[1])
    for site, indices in site_indices.items():
        site_mean = data[indices].mean(axis=0)
        ss_within += np.sum((data[indices] - site_mean) ** 2, axis=0)

    # Partial eta-squared
    eta_sq = ss_between / (ss_between + ss_within + 1e-10)

    return eta_sq


def compute_cohens_f(eta_sq: vector_type) -> vector_type:
    """
    Convert partial eta-squared to Cohen's f.

    f = sqrt(eta_sq / (1 - eta_sq))

    Parameters
    ----------
    eta_sq : vector_type
        Partial eta-squared values

    Returns
    -------
    cohens_f : vector_type
        Cohen's f effect size
    """
    # Clip to avoid division issues
    eta_clipped = np.clip(eta_sq, 0, 0.9999)
    return np.sqrt(eta_clipped / (1 - eta_clipped))


# -----------------------------------------------------------------------------
# Main Site Effect Test Function
# -----------------------------------------------------------------------------

def test_site_effect(
    image_paths: Dict[str, List[str]],
    mask_path: str,
    output_dir: str,
    covariates: Optional[Dict[str, List[float]]] = None,
    n_permutations: int = 5000,
    alpha: float = 0.05,
    variance_groups: Optional[Dict[str, int]] = None,
    exchangeability_blocks: Optional[List[int]] = None,
    seed: Optional[int] = None,
    save_maps: bool = True,
    verbose: bool = True
) -> SiteEffectResult:
    """
    Test for site effects using GLM with permutation inference.

    Parameters
    ----------
    image_paths : dict
        Dictionary of site_id -> list of image paths
    mask_path : str
        Path to brain mask
    output_dir : str
        Output directory
    covariates : dict, optional
        Covariate name -> values (must match concatenated subject order)
    n_permutations : int
        Number of permutations for inference
    alpha : float
        Significance level
    variance_groups : dict, optional
        Site -> variance group index (for heteroscedastic test)
    exchangeability_blocks : list, optional
        Block indices for constrained permutations
    seed : int, optional
        Random seed
    save_maps : bool
        Whether to save statistical maps
    verbose : bool
        Print progress

    Returns
    -------
    SiteEffectResult
        Complete site effect analysis results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect subjects and labels
    all_images = []
    site_labels = []
    unique_sites = sorted(image_paths.keys())

    for site_id in unique_sites:
        for path in image_paths[site_id]:
            all_images.append(path)
            site_labels.append(site_id)

    n_subjects = len(all_images)
    n_sites = len(unique_sites)

    if verbose:
        print(f"Site effect analysis")
        print(f"  Subjects: {n_subjects}")
        print(f"  Sites: {n_sites} ({', '.join(unique_sites)})")
        print(f"  Permutations: {n_permutations}")

    # Load data
    if verbose:
        print("Loading images...")

    data, mask_indices, image_shape = load_image_to_matrix(
        all_images, mask_path
    )
    n_voxels = data.shape[1]

    if verbose:
        print(f"  Voxels: {n_voxels}")

    # Create design matrix
    design, col_names = create_design_matrix(site_labels, covariates)

    if verbose:
        print(f"  Design: {design.shape} ({', '.join(col_names)})")

    # Check design matrix
    rank, condition = check_design(design)
    if verbose:
        print(f"  Design rank: {rank}, condition: {condition:.1f}")

    if condition > 100:
        print(f"  WARNING: High condition number ({condition:.1f}), results may be unstable")

    # Create site effect hypothesis
    hypothesis = create_site_contrast(n_sites)

    # Choose test type based on variance groups
    if variance_groups is not None:
        vg_array = np.array([variance_groups[s] for s in site_labels])
        test = TestFixedHeteroscedastic(data, design, [hypothesis], vg_array)
        if verbose:
            print("  Using heteroscedastic (G-statistic) test")
    else:
        test = TestFixedHomoscedastic(data, design, [hypothesis])
        if verbose:
            print("  Using homoscedastic (F-statistic) test")

    # Run observed test
    if verbose:
        print("Computing observed statistics...")

    observed_output = test()[0]
    f_observed = observed_output.statistic

    # Parametric p-values (from F-distribution)
    try:
        from scipy import stats as scipy_stats
        df1 = n_sites - 1
        df2 = n_subjects - design.shape[1]
        p_parametric = 1 - scipy_stats.f.cdf(f_observed, df1, df2)
    except ImportError:
        p_parametric = np.full_like(f_observed, np.nan)
        print("  WARNING: scipy not available, skipping parametric p-values")

    # Permutation testing
    if verbose:
        print(f"Running {n_permutations} permutations...")

    eb_array = np.array(exchangeability_blocks) if exchangeability_blocks else None
    shuffler = Shuffler(
        n_subjects,
        n_permutations=n_permutations,
        exchangeability_blocks=eb_array,
        seed=seed
    )

    null_distribution = []
    for i, shuffle in enumerate(shuffler):
        if shuffle.index == 0:
            continue  # Skip identity (already computed)

        perm_output = test(shuffle.data)[0]
        null_distribution.append(perm_output.statistic)

        if verbose and (i + 1) % 500 == 0:
            print(f"  Completed {i + 1}/{n_permutations}")

    null_distribution = np.array(null_distribution)

    # Permutation p-values
    p_permutation = permutation_p_values(f_observed, null_distribution, tail="right")

    # FDR correction
    q_values, fdr_thresh, sig_fdr = fdr_correction(p_permutation, alpha)

    # Effect sizes
    if verbose:
        print("Computing effect sizes...")

    eta_sq = compute_partial_eta_squared(data, site_labels)
    cohens_f = compute_cohens_f(eta_sq)

    # Summary statistics
    sig_uncorr = np.mean(p_parametric < alpha) * 100 if not np.all(np.isnan(p_parametric)) else None
    sig_fdr_pct = np.mean(sig_fdr) * 100
    sig_perm = np.mean(p_permutation < alpha) * 100

    # Create result
    result = SiteEffectResult(
        n_subjects=n_subjects,
        n_sites=n_sites,
        n_voxels=n_voxels,
        n_permutations=n_permutations,
        site_labels=site_labels,
        unique_sites=unique_sites,
        f_statistic=f_observed,
        p_values_parametric=p_parametric,
        p_values_permutation=p_permutation,
        null_max_distribution=np.max(null_distribution, axis=1),
        p_values_fdr=q_values,
        fdr_threshold=fdr_thresh,
        effect_size=eta_sq,
        cohens_f=cohens_f,
        percent_significant_uncorrected=sig_uncorr,
        percent_significant_fdr=sig_fdr_pct,
        percent_significant_permutation=sig_perm,
        mean_effect_size=float(np.nanmean(eta_sq)),
        median_effect_size=float(np.nanmedian(eta_sq)),
        mask_indices=mask_indices,
        image_shape=image_shape
    )

    if verbose:
        print("\nResults:")
        print(f"  Significant voxels (uncorrected p<{alpha}): {sig_uncorr:.1f}%")
        print(f"  Significant voxels (permutation p<{alpha}): {sig_perm:.1f}%")
        print(f"  Significant voxels (FDR q<{alpha}): {sig_fdr_pct:.1f}%")
        print(f"  Mean effect size (eta²): {result.mean_effect_size:.4f}")
        print(f"  Median effect size (eta²): {result.median_effect_size:.4f}")

    # Save outputs
    if save_maps:
        if verbose:
            print("\nSaving statistical maps...")

        ref_image = all_images[0]

        save_vector_to_image(
            f_observed, mask_indices, image_shape, ref_image,
            str(output_dir / "f_statistic.mif")
        )

        save_vector_to_image(
            p_permutation, mask_indices, image_shape, ref_image,
            str(output_dir / "p_permutation.mif")
        )

        save_vector_to_image(
            eta_sq, mask_indices, image_shape, ref_image,
            str(output_dir / "effect_size_eta_sq.mif")
        )

        # -log10(p) for visualization
        neg_log_p = -np.log10(np.maximum(p_permutation, 1e-10))
        save_vector_to_image(
            neg_log_p, mask_indices, image_shape, ref_image,
            str(output_dir / "neg_log10_p.mif")
        )

        # Significance mask
        sig_mask = (p_permutation < alpha).astype(np.float32)
        save_vector_to_image(
            sig_mask, mask_indices, image_shape, ref_image,
            str(output_dir / "significant_mask.mif")
        )

    # Save summary JSON
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    if verbose:
        print(f"\nResults saved to: {output_dir}")

    return result


# -----------------------------------------------------------------------------
# Comparison Function (Pre vs Post Harmonization)
# -----------------------------------------------------------------------------

def compare_site_effects(
    pre_result: SiteEffectResult,
    post_result: SiteEffectResult,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Compare site effects before and after harmonization.

    Parameters
    ----------
    pre_result : SiteEffectResult
        Results before harmonization
    post_result : SiteEffectResult
        Results after harmonization
    output_dir : str, optional
        Directory to save comparison

    Returns
    -------
    dict
        Comparison metrics
    """
    comparison = {
        "pre_harmonization": {
            "percent_significant": pre_result.percent_significant_permutation,
            "mean_effect_size": pre_result.mean_effect_size,
            "median_effect_size": pre_result.median_effect_size,
        },
        "post_harmonization": {
            "percent_significant": post_result.percent_significant_permutation,
            "mean_effect_size": post_result.mean_effect_size,
            "median_effect_size": post_result.median_effect_size,
        },
        "reduction": {
            "percent_significant_reduction": (
                (pre_result.percent_significant_permutation - post_result.percent_significant_permutation)
                / (pre_result.percent_significant_permutation + 1e-10) * 100
            ),
            "effect_size_reduction": (
                (pre_result.mean_effect_size - post_result.mean_effect_size)
                / (pre_result.mean_effect_size + 1e-10) * 100
            ),
        },
        "success_criteria": {
            "significant_voxels_below_5_percent": post_result.percent_significant_permutation < 5.0,
            "effect_size_reduction_above_70_percent": (
                (pre_result.mean_effect_size - post_result.mean_effect_size)
                / (pre_result.mean_effect_size + 1e-10) * 100
            ) > 70,
        }
    }

    # Overall pass/fail
    comparison["harmonization_successful"] = all(
        comparison["success_criteria"].values()
    )

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)

    return comparison
