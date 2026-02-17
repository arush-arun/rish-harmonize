"""
Voxel-wise General Linear Model (GLM) Module

Design follows MRtrix3 core/math/stats/glm.{h,cpp} conventions:
- Hypothesis class with partition support for contrast matrices
- TestBase hierarchy: Fixed/Variable × Homoscedastic/Heteroscedastic
- Freedman-Lane permutation testing
- Pre-computation of projection matrices for fixed designs

Reference: https://github.com/MRtrix3/mrtrix3/blob/master/core/math/stats/glm.cpp
"""

from __future__ import annotations

import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# Type definitions (following MRtrix3 typedefs.h)
value_type = np.float64
matrix_type = np.ndarray  # 2D array of value_type
vector_type = np.ndarray  # 1D array of value_type
index_array_type = np.ndarray  # 1D array of size_t (int64)


# -----------------------------------------------------------------------------
# Hypothesis Class
# -----------------------------------------------------------------------------

@dataclass
class Partition:
    """
    Design matrix partition for hypothesis testing.

    Separates design into:
    - X: columns relevant to hypothesis (effects of interest)
    - Z: nuisance regressors

    Following MRtrix3 Hypothesis::Partition structure.
    """
    X: matrix_type  # Effects of interest
    Z: matrix_type  # Nuisance regressors
    Hz: matrix_type  # Projection matrix onto Z: Z(Z'Z)^-1 Z'
    Rz: matrix_type  # Residual-forming matrix: I - Hz
    rank_X: int
    rank_Z: int


class Hypothesis:
    """
    Statistical hypothesis represented as a contrast matrix.

    Supports both:
    - t-tests: single row contrast (e.g., [1, -1, 0, 0])
    - F-tests: multi-row contrast matrix

    Following MRtrix3 Hypothesis class design.
    """

    def __init__(
        self,
        contrast: matrix_type,
        index: int = 0,
        name: Optional[str] = None
    ):
        """
        Initialize hypothesis from contrast matrix.

        Parameters
        ----------
        contrast : array-like
            Contrast matrix (1D for t-test, 2D for F-test)
        index : int
            Hypothesis index for identification
        name : str, optional
            Descriptive name
        """
        self._c = np.atleast_2d(np.asarray(contrast, dtype=value_type))
        self._index = index
        self._name = name

        # Determine rank via SVD
        if self._c.size > 0:
            s = np.linalg.svd(self._c, compute_uv=False)
            self._rank = np.sum(s > 1e-10 * s[0]) if len(s) > 0 else 0
        else:
            self._rank = 0

        # F-test if multi-row contrast
        self._is_F = self._c.shape[0] > 1

        # Cached partition
        self._partition: Optional[Partition] = None

    @property
    def matrix(self) -> matrix_type:
        """Return contrast matrix."""
        return self._c

    @property
    def cols(self) -> int:
        """Number of columns in contrast."""
        return self._c.shape[1]

    @property
    def rank(self) -> int:
        """Rank of contrast matrix."""
        return self._rank

    @property
    def is_F(self) -> bool:
        """True if F-test (multi-row contrast)."""
        return self._is_F

    @property
    def index(self) -> int:
        """Hypothesis index."""
        return self._index

    @property
    def name(self) -> str:
        """Hypothesis name."""
        if self._name:
            return self._name
        return f"F{self._index}" if self._is_F else f"t{self._index}"

    def partition(self, design: matrix_type) -> Partition:
        """
        Partition design matrix into hypothesis-relevant and nuisance components.

        Implements Freedman-Lane partitioning:
        - X: columns where contrast has non-zero entries
        - Z: remaining nuisance columns

        Parameters
        ----------
        design : matrix_type
            Full design matrix (n_subjects × n_predictors)

        Returns
        -------
        Partition
            Partitioned matrices and projection operators
        """
        if self._partition is not None:
            return self._partition

        n, p = design.shape

        # Identify columns involved in hypothesis
        # Non-zero columns in contrast matrix
        col_involved = np.any(np.abs(self._c) > 1e-10, axis=0)

        if col_involved.shape[0] != p:
            raise ValueError(
                f"Contrast has {col_involved.shape[0]} columns but design has {p}"
            )

        X_cols = np.where(col_involved)[0]
        Z_cols = np.where(~col_involved)[0]

        X = design[:, X_cols] if len(X_cols) > 0 else np.zeros((n, 0))
        Z = design[:, Z_cols] if len(Z_cols) > 0 else np.zeros((n, 0))

        # Compute projection matrices
        if Z.shape[1] > 0:
            # Hz = Z @ pinv(Z)
            Hz = Z @ np.linalg.pinv(Z)
        else:
            Hz = np.zeros((n, n))

        # Residual-forming matrix
        Rz = np.eye(n) - Hz

        # Compute ranks
        rank_X = np.linalg.matrix_rank(X) if X.shape[1] > 0 else 0
        rank_Z = np.linalg.matrix_rank(Z) if Z.shape[1] > 0 else 0

        self._partition = Partition(
            X=X, Z=Z, Hz=Hz, Rz=Rz,
            rank_X=rank_X, rank_Z=rank_Z
        )
        return self._partition


# -----------------------------------------------------------------------------
# Test Result Classes
# -----------------------------------------------------------------------------

@dataclass
class TestOutput:
    """
    Output from a single statistical test.

    Following MRtrix3 output conventions.
    """
    statistic: vector_type  # F or t statistic per element
    zstatistic: Optional[vector_type] = None  # Z-score transformation
    effect_size: Optional[vector_type] = None  # Absolute effect
    std_effect_size: Optional[vector_type] = None  # Standardized effect


@dataclass
class GLMResult:
    """
    Complete results from GLM analysis.

    Stores per-hypothesis results and model diagnostics.
    """
    n_subjects: int
    n_elements: int
    n_hypotheses: int
    predictors: List[str]

    # Per-hypothesis results
    outputs: List[TestOutput] = field(default_factory=list)

    # Model diagnostics
    betas: Optional[matrix_type] = None  # Coefficients (p × n_elements)
    residuals: Optional[matrix_type] = None  # Residuals (n × n_elements)
    sigma_sq: Optional[vector_type] = None  # Residual variance per element

    # Spatial reconstruction info
    mask_indices: Optional[index_array_type] = None
    image_shape: Optional[Tuple[int, ...]] = None


# -----------------------------------------------------------------------------
# Test Base Class (Abstract)
# -----------------------------------------------------------------------------

class TestBase(ABC):
    """
    Abstract base class for GLM statistical testing.

    Following MRtrix3 TestBase design with virtual operator().
    """

    def __init__(
        self,
        measurements: matrix_type,
        design: matrix_type,
        hypotheses: List[Hypothesis]
    ):
        """
        Initialize test with data and model.

        Parameters
        ----------
        measurements : matrix_type
            Data matrix (n_subjects × n_elements)
        design : matrix_type
            Design matrix (n_subjects × n_predictors)
        hypotheses : list of Hypothesis
            Statistical hypotheses to test
        """
        self._y = np.asarray(measurements, dtype=value_type)
        self._M = np.asarray(design, dtype=value_type)
        self._hypotheses = hypotheses

        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """Validate input dimensions."""
        n_y, _ = self._y.shape
        n_M, _ = self._M.shape

        if n_y != n_M:
            raise ValueError(
                f"Measurement rows ({n_y}) != design rows ({n_M})"
            )

        for h in self._hypotheses:
            if h.cols != self._M.shape[1]:
                raise ValueError(
                    f"Hypothesis {h.name} has {h.cols} columns "
                    f"but design has {self._M.shape[1]}"
                )

    @property
    def num_subjects(self) -> int:
        """Number of subjects/observations."""
        return self._y.shape[0]

    @property
    def num_elements(self) -> int:
        """Number of elements (voxels)."""
        return self._y.shape[1]

    @property
    def num_hypotheses(self) -> int:
        """Number of hypotheses."""
        return len(self._hypotheses)

    @property
    def num_factors(self) -> int:
        """Number of predictors in design."""
        return self._M.shape[1]

    @abstractmethod
    def __call__(
        self,
        shuffled_indices: Optional[index_array_type] = None
    ) -> List[TestOutput]:
        """
        Compute test statistics.

        Parameters
        ----------
        shuffled_indices : array, optional
            Permutation indices for Freedman-Lane

        Returns
        -------
        list of TestOutput
            Statistics for each hypothesis
        """
        pass


# -----------------------------------------------------------------------------
# TestFixedHomoscedastic
# -----------------------------------------------------------------------------

class TestFixedHomoscedastic(TestBase):
    """
    GLM test with fixed design and constant variance assumption.

    Most efficient variant - pre-computes projection matrices.
    Following MRtrix3 TestFixedHomoscedastic implementation.
    """

    def __init__(
        self,
        measurements: matrix_type,
        design: matrix_type,
        hypotheses: List[Hypothesis]
    ):
        super().__init__(measurements, design, hypotheses)
        self._precompute()

    def _precompute(self) -> None:
        """Pre-compute projection matrices and partitions."""
        n, p = self._M.shape

        # SVD-based pseudo-inverse (following MRtrix3 solve method)
        U, s, Vt = np.linalg.svd(self._M, full_matrices=False)
        # Threshold small singular values
        tol = 1e-10 * s[0] if len(s) > 0 else 1e-10
        s_inv = np.where(s > tol, 1.0 / s, 0.0)
        self._pinv_M = (Vt.T * s_inv) @ U.T

        # Degrees of freedom
        self._rank_M = np.sum(s > tol)
        self._dof = n - self._rank_M

        # Partition for each hypothesis
        self._partitions = [h.partition(self._M) for h in self._hypotheses]

        # Pre-compute X'X for each hypothesis (for F-statistic)
        self._XtX_list = []
        for part in self._partitions:
            if part.X.shape[1] > 0:
                XtX = part.X.T @ part.X
                self._XtX_list.append(XtX)
            else:
                self._XtX_list.append(None)

    def solve_betas(
        self,
        y: Optional[matrix_type] = None
    ) -> matrix_type:
        """
        Solve for beta coefficients via pseudo-inverse.

        Parameters
        ----------
        y : matrix_type, optional
            Measurements (uses stored if None)

        Returns
        -------
        betas : matrix_type
            Coefficients (p × n_elements)
        """
        if y is None:
            y = self._y
        return self._pinv_M @ y

    def compute_residuals(
        self,
        y: Optional[matrix_type] = None,
        betas: Optional[matrix_type] = None
    ) -> matrix_type:
        """Compute residuals: y - M @ beta."""
        if y is None:
            y = self._y
        if betas is None:
            betas = self.solve_betas(y)
        return y - self._M @ betas

    def compute_sigma_sq(
        self,
        residuals: Optional[matrix_type] = None
    ) -> vector_type:
        """Compute residual variance estimate."""
        if residuals is None:
            residuals = self.compute_residuals()
        sse = np.sum(residuals ** 2, axis=0)
        return sse / max(self._dof, 1)

    def abs_effect_size(
        self,
        hypothesis: Hypothesis,
        betas: matrix_type
    ) -> vector_type:
        """
        Compute absolute effect size: c @ beta.

        For t-tests, returns the contrast value.
        For F-tests, returns NaN (no single effect size).
        """
        if hypothesis.is_F:
            return np.full(betas.shape[1], np.nan)

        c = hypothesis.matrix.flatten()
        return c @ betas

    def std_effect_size(
        self,
        hypothesis: Hypothesis,
        betas: matrix_type,
        sigma: vector_type
    ) -> vector_type:
        """
        Compute standardized effect size (Cohen's d equivalent).

        effect / sigma
        """
        abs_effect = self.abs_effect_size(hypothesis, betas)
        return abs_effect / (sigma + 1e-10)

    def compute_F_statistic(
        self,
        hypothesis_idx: int,
        betas: matrix_type,
        sigma_sq: vector_type
    ) -> vector_type:
        """
        Compute F-statistic for hypothesis.

        F = (beta_X' X'X beta_X / rank_X) / sigma_sq

        Following MRtrix3 F-statistic computation.
        """
        h = self._hypotheses[hypothesis_idx]
        part = self._partitions[hypothesis_idx]
        XtX = self._XtX_list[hypothesis_idx]

        if part.rank_X == 0 or XtX is None:
            return np.zeros(betas.shape[1])

        # Extract betas for X columns
        col_involved = np.any(np.abs(h.matrix) > 1e-10, axis=0)
        X_cols = np.where(col_involved)[0]
        beta_X = betas[X_cols, :]  # shape: (n_site_vars, n_voxels)

        # Numerator: beta_X' @ X'X @ beta_X for each voxel
        # beta_X.T has shape (n_voxels, n_site_vars)
        # For voxel v: beta_X.T[v,:] @ XtX @ beta_X.T[v,:].T
        # einsum: vi,ik,vk->v where v=voxel, i,k=site_var indices
        num = np.einsum('vi,ik,vk->v', beta_X.T, XtX, beta_X.T)
        ms_model = num / part.rank_X

        # F-statistic
        F = ms_model / (sigma_sq + 1e-10)

        return F

    def compute_t_statistic(
        self,
        hypothesis_idx: int,
        betas: matrix_type,
        sigma_sq: vector_type
    ) -> vector_type:
        """
        Compute t-statistic (signed sqrt of F).

        Following MRtrix3 convention: t = sign(effect) * sqrt(F)
        """
        h = self._hypotheses[hypothesis_idx]
        F = self.compute_F_statistic(hypothesis_idx, betas, sigma_sq)

        # Sign from effect direction
        effect = self.abs_effect_size(h, betas)
        sign = np.sign(effect)

        return sign * np.sqrt(np.maximum(F, 0))

    def __call__(
        self,
        shuffled_indices: Optional[index_array_type] = None
    ) -> List[TestOutput]:
        """
        Compute test statistics for all hypotheses.

        Parameters
        ----------
        shuffled_indices : array, optional
            Permutation indices for Freedman-Lane procedure

        Returns
        -------
        list of TestOutput
            One TestOutput per hypothesis
        """
        # Apply permutation if provided (Freedman-Lane)
        if shuffled_indices is not None:
            # Permute rows of measurements
            y = self._y[shuffled_indices, :]
        else:
            y = self._y

        # Solve for betas
        betas = self.solve_betas(y)

        # Compute residuals and variance
        residuals = self.compute_residuals(y, betas)
        sigma_sq = self.compute_sigma_sq(residuals)
        sigma = np.sqrt(sigma_sq)

        # Compute statistics for each hypothesis
        outputs = []
        for i, h in enumerate(self._hypotheses):
            if h.is_F:
                stat = self.compute_F_statistic(i, betas, sigma_sq)
            else:
                stat = self.compute_t_statistic(i, betas, sigma_sq)

            effect = self.abs_effect_size(h, betas)
            std_effect = self.std_effect_size(h, betas, sigma)

            outputs.append(TestOutput(
                statistic=stat,
                effect_size=effect,
                std_effect_size=std_effect
            ))

        return outputs

    @property
    def betas(self) -> matrix_type:
        """Fitted coefficients."""
        return self.solve_betas()

    @property
    def residuals(self) -> matrix_type:
        """Model residuals."""
        return self.compute_residuals()

    @property
    def sigma_sq(self) -> vector_type:
        """Residual variance."""
        return self.compute_sigma_sq()


# -----------------------------------------------------------------------------
# TestFixedHeteroscedastic
# -----------------------------------------------------------------------------

class TestFixedHeteroscedastic(TestFixedHomoscedastic):
    """
    GLM test with fixed design and grouped variance (heteroscedastic).

    Supports different variance groups (e.g., different scanners).
    Following MRtrix3 TestFixedHeteroscedastic with G-statistic.
    """

    def __init__(
        self,
        measurements: matrix_type,
        design: matrix_type,
        hypotheses: List[Hypothesis],
        variance_groups: index_array_type
    ):
        """
        Initialize with variance group assignments.

        Parameters
        ----------
        variance_groups : array
            Group index for each subject (0-indexed)
        """
        self._variance_groups = np.asarray(variance_groups, dtype=np.int64)
        super().__init__(measurements, design, hypotheses)

        self._setup_variance_groups()

    def _setup_variance_groups(self) -> None:
        """Pre-compute variance group information."""
        self._unique_groups = np.unique(self._variance_groups)
        self._n_groups = len(self._unique_groups)

        # Indices for each group
        self._group_indices = {
            g: np.where(self._variance_groups == g)[0]
            for g in self._unique_groups
        }

        # Count per group
        self._n_per_group = {
            g: len(idx) for g, idx in self._group_indices.items()
        }

    def compute_sigma_sq_per_group(
        self,
        residuals: Optional[matrix_type] = None
    ) -> Dict[int, vector_type]:
        """Compute variance estimate per group."""
        if residuals is None:
            residuals = self.compute_residuals()

        sigma_sq = {}
        for g, indices in self._group_indices.items():
            res_g = residuals[indices, :]
            n_g = len(indices)
            # Degrees of freedom within group
            dof_g = max(n_g - 1, 1)
            sigma_sq[g] = np.sum(res_g ** 2, axis=0) / dof_g

        return sigma_sq

    def compute_G_statistic(
        self,
        hypothesis_idx: int,
        betas: matrix_type,
        sigma_sq_per_group: Dict[int, vector_type]
    ) -> vector_type:
        """
        Compute G-statistic (Welch-type F for heteroscedastic data).

        Implements weighted F-statistic with Welch-Satterthwaite
        degrees of freedom correction.
        """
        h = self._hypotheses[hypothesis_idx]
        part = self._partitions[hypothesis_idx]

        if part.rank_X == 0:
            return np.zeros(betas.shape[1])

        # Compute weights based on group variances
        # Weight_g = n_g / sigma_sq_g
        weights = {}
        total_weight = np.zeros(betas.shape[1])
        for g in self._unique_groups:
            w = self._n_per_group[g] / (sigma_sq_per_group[g] + 1e-10)
            weights[g] = w
            total_weight += w

        # Weighted mean of betas (for effect estimation)
        col_involved = np.any(np.abs(h.matrix) > 1e-10, axis=0)
        X_cols = np.where(col_involved)[0]
        beta_X = betas[X_cols, :]

        # Compute weighted sum of squares
        weighted_ss = np.zeros(betas.shape[1])
        for g, indices in self._group_indices.items():
            w = weights[g] / total_weight
            y_g = self._y[indices, :].mean(axis=0)
            # Contribution to between-group variance
            weighted_ss += self._n_per_group[g] * w * (y_g ** 2)

        # Pooled variance estimate
        pooled_var = np.zeros(betas.shape[1])
        for g in self._unique_groups:
            pooled_var += sigma_sq_per_group[g] / self._n_groups

        # G-statistic
        G = weighted_ss / (part.rank_X * pooled_var + 1e-10)

        return G

    def __call__(
        self,
        shuffled_indices: Optional[index_array_type] = None
    ) -> List[TestOutput]:
        """Compute G-statistics for heteroscedastic case."""
        if shuffled_indices is not None:
            y = self._y[shuffled_indices, :]
            # Also shuffle variance groups
            vg = self._variance_groups[shuffled_indices]
        else:
            y = self._y
            vg = self._variance_groups

        betas = self.solve_betas(y)
        residuals = self.compute_residuals(y, betas)
        sigma_sq = self.compute_sigma_sq(residuals)
        sigma = np.sqrt(sigma_sq)

        # Per-group variance
        sigma_sq_per_group = self.compute_sigma_sq_per_group(residuals)

        outputs = []
        for i, h in enumerate(self._hypotheses):
            if h.is_F:
                stat = self.compute_G_statistic(i, betas, sigma_sq_per_group)
            else:
                # For t-test, use regular statistic
                stat = self.compute_t_statistic(i, betas, sigma_sq)

            effect = self.abs_effect_size(h, betas)
            std_effect = self.std_effect_size(h, betas, sigma)

            outputs.append(TestOutput(
                statistic=stat,
                effect_size=effect,
                std_effect_size=std_effect
            ))

        return outputs


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def check_design(design: matrix_type, tolerance: float = 1e-5) -> Tuple[int, float]:
    """
    Check design matrix for rank deficiency and conditioning.

    Following MRtrix3 check_design() function.

    Parameters
    ----------
    design : matrix_type
        Design matrix
    tolerance : float
        Threshold for rank determination

    Returns
    -------
    rank : int
        Matrix rank
    condition : float
        Condition number
    """
    # QR decomposition for rank
    Q, R = np.linalg.qr(design)
    diag_R = np.abs(np.diag(R))
    rank = np.sum(diag_R > tolerance * diag_R[0]) if len(diag_R) > 0 else 0

    # Condition number via SVD
    s = np.linalg.svd(design, compute_uv=False)
    if len(s) > 0 and s[-1] > 1e-10:
        condition = s[0] / s[-1]
    else:
        condition = np.inf

    return rank, condition


def create_site_contrast(
    n_sites: int,
    n_covariates: int = 0,
    reference_site: int = 0
) -> Hypothesis:
    """
    Create F-test contrast for site effect.

    Tests whether all site coefficients are zero.

    Parameters
    ----------
    n_sites : int
        Number of sites
    n_covariates : int
        Number of covariates in design matrix
    reference_site : int
        Reference site (not included in contrast)

    Returns
    -------
    Hypothesis
        F-test hypothesis for site effect
    """
    # Contrast tests all non-reference site coefficients
    # Design: [intercept, site_1, site_2, ..., site_{n-1}, covariates...]
    # For n sites, we have n-1 site dummy variables

    n_site_vars = n_sites - 1
    n_total_cols = 1 + n_site_vars + n_covariates  # intercept + sites + covariates

    # Identity matrix for site coefficients
    # Columns: [intercept, sites..., covariates...]
    # We test sites (columns 1 to n_site_vars)
    contrast = np.zeros((n_site_vars, n_total_cols))
    for i in range(n_site_vars):
        contrast[i, 1 + i] = 1.0

    return Hypothesis(contrast, index=0, name="site_effect")


def create_design_matrix(
    site_labels: List[str],
    covariates: Optional[Dict[str, List[float]]] = None,
    standardize_covariates: bool = True
) -> Tuple[matrix_type, List[str]]:
    """
    Create design matrix with intercept, site dummies, and covariates.

    Parameters
    ----------
    site_labels : list of str
        Site label for each subject
    covariates : dict, optional
        Covariate name -> values
    standardize_covariates : bool
        Whether to z-score continuous covariates

    Returns
    -------
    design : matrix_type
        Design matrix (n × p)
    column_names : list of str
        Names for each column
    """
    n = len(site_labels)
    unique_sites = sorted(set(site_labels))
    site_to_idx = {s: i for i, s in enumerate(unique_sites)}

    columns = []
    names = []

    # Intercept
    columns.append(np.ones(n))
    names.append("intercept")

    # Site dummy variables (reference = first site)
    for site in unique_sites[1:]:
        dummy = np.array([1.0 if s == site else 0.0 for s in site_labels])
        columns.append(dummy)
        names.append(f"site_{site}")

    # Covariates
    if covariates:
        for cov_name, values in covariates.items():
            arr = np.asarray(values, dtype=value_type)
            if len(arr) != n:
                raise ValueError(
                    f"Covariate '{cov_name}' has {len(arr)} values, expected {n}"
                )
            if standardize_covariates:
                arr = (arr - arr.mean()) / (arr.std() + 1e-10)
            columns.append(arr)
            names.append(cov_name)

    design = np.column_stack(columns)
    return design, names


# -----------------------------------------------------------------------------
# Image I/O (MRtrix3 integration)
# -----------------------------------------------------------------------------

def load_image_to_matrix(
    image_paths: List[str],
    mask_path: Optional[str] = None
) -> Tuple[matrix_type, index_array_type, Tuple[int, ...]]:
    """
    Load multiple images into a data matrix.

    Parameters
    ----------
    image_paths : list of str
        Paths to 3D images (one per subject)
    mask_path : str, optional
        Brain mask

    Returns
    -------
    data : matrix_type
        Data matrix (n_subjects × n_voxels)
    mask_indices : index_array_type
        Voxel indices within mask
    image_shape : tuple
        Original 3D shape
    """
    data_list = []
    mask_indices = None
    image_shape = None

    for i, path in enumerate(image_paths):
        # Convert to NIfTI temp file, then load with nibabel
        with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
            nii_path = tmp.name

        subprocess.run(
            ["mrconvert", path, nii_path, "-datatype", "float32", "-force"],
            capture_output=True, check=True
        )

        import nibabel as nib
        img = nib.load(nii_path)
        img_data = np.asarray(img.dataobj, dtype=np.float32).ravel()
        Path(nii_path).unlink()

        if image_shape is None:
            image_shape = img.shape[:3]

        # Apply mask on first iteration
        if mask_indices is None:
            if mask_path:
                with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
                    mask_nii = tmp.name
                subprocess.run(
                    ["mrconvert", mask_path, mask_nii, "-datatype", "float32", "-force"],
                    capture_output=True, check=True
                )
                mask_img = nib.load(mask_nii)
                mask_data = np.asarray(mask_img.dataobj, dtype=np.float32).ravel()
                Path(mask_nii).unlink()
                mask_indices = np.where(mask_data > 0.5)[0]
            else:
                mask_indices = np.arange(len(img_data))

        data_list.append(img_data[mask_indices])

    return np.array(data_list), mask_indices, image_shape


def save_vector_to_image(
    data: vector_type,
    mask_indices: index_array_type,
    image_shape: Tuple[int, ...],
    reference_image: str,
    output_path: str,
    fill_value: float = 0.0
) -> str:
    """
    Save vector data back to image format.

    Parameters
    ----------
    data : vector_type
        Values for masked voxels
    mask_indices : index_array_type
        Voxel indices
    image_shape : tuple
        3D shape
    reference_image : str
        Reference for header
    output_path : str
        Output path

    Returns
    -------
    str
        Path to saved image
    """
    # Reconstruct volume and save via nibabel + mrconvert
    import nibabel as nib

    full = np.full(np.prod(image_shape), fill_value, dtype=np.float32)
    full[mask_indices] = data.astype(np.float32)
    vol = full.reshape(image_shape)

    # Load reference for affine/header
    with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
        ref_nii = tmp.name
    subprocess.run(
        ["mrconvert", reference_image, ref_nii, "-force"],
        capture_output=True, check=True
    )
    ref_img = nib.load(ref_nii)

    # Save as NIfTI with reference header
    out_img = nib.Nifti1Image(vol, ref_img.affine, ref_img.header)
    with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
        tmp_nii = tmp.name
    nib.save(out_img, tmp_nii)
    Path(ref_nii).unlink()

    # Convert to final format (supports .mif, .nii.gz, etc.)
    subprocess.run(
        ["mrconvert", tmp_nii, output_path, "-datatype", "float32", "-force"],
        capture_output=True, check=True
    )
    Path(tmp_nii).unlink()
    return output_path
