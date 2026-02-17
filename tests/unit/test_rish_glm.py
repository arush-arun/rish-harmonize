"""Tests for core/rish_glm.py — RISH-GLM design matrix and per-shell wrappers."""

import numpy as np
import pytest

from rish_harmonize.core.rish_glm import (
    build_rish_glm_design,
    RISHGLMResult,
    RISHGLMPerShellResult,
)


class TestBuildDesignMatrix:
    def test_two_sites_no_covariates(self):
        labels = ["A", "A", "B", "B", "B"]
        design, col_names, site_map, cov_means, cov_stds = \
            build_rish_glm_design(labels)

        assert design.shape == (5, 2)
        assert col_names == ["site_A", "site_B"]
        assert site_map == {"A": 0, "B": 1}

        # Site A indicator
        np.testing.assert_array_equal(design[:, 0], [1, 1, 0, 0, 0])
        # Site B indicator
        np.testing.assert_array_equal(design[:, 1], [0, 0, 1, 1, 1])

    def test_with_covariates(self):
        labels = ["A", "A", "B", "B"]
        covariates = {"age": [20.0, 30.0, 40.0, 50.0]}
        design, col_names, site_map, cov_means, cov_stds = \
            build_rish_glm_design(labels, covariates)

        assert design.shape == (4, 3)  # 2 sites + 1 covariate
        assert "age" in col_names

        # Check z-scoring
        age_col = design[:, col_names.index("age")]
        assert abs(age_col.mean()) < 1e-10  # zero mean
        assert abs(age_col.std() - 1.0) < 0.1  # unit std (approx)

        assert cov_means["age"] == pytest.approx(35.0)

    def test_three_sites(self):
        labels = ["X", "Y", "Z", "X", "Y", "Z"]
        design, col_names, site_map, _, _ = build_rish_glm_design(labels)

        assert design.shape == (6, 3)
        assert len(site_map) == 3

        # Each subject has exactly one site indicator = 1
        for row in design:
            assert row.sum() == 1.0

    def test_covariate_length_mismatch(self):
        labels = ["A", "B"]
        covariates = {"age": [20.0]}  # Wrong length

        with pytest.raises(ValueError, match="has 1 values, expected 2"):
            build_rish_glm_design(labels, covariates)

    def test_no_intercept_property(self):
        """Design matrix has no intercept — site columns span the intercept."""
        labels = ["A", "A", "B", "B"]
        design, _, _, _, _ = build_rish_glm_design(labels)

        # Row sums = 1 (each row has exactly one site indicator)
        np.testing.assert_array_almost_equal(design.sum(axis=1), np.ones(4))

    def test_constant_covariate_no_crash(self):
        """A constant covariate should not cause division by zero."""
        labels = ["A", "B"]
        covariates = {"constant": [5.0, 5.0]}

        design, _, _, _, cov_stds = build_rish_glm_design(labels, covariates)
        # std forced to 1.0 for constant covariates
        assert cov_stds["constant"] == 1.0


class TestRISHGLMPerShellResult:
    def test_basic_structure(self):
        r1 = RISHGLMResult(site_names=["A", "B"], orders=[0, 2])
        r2 = RISHGLMResult(site_names=["A", "B"], orders=[0, 2])

        result = RISHGLMPerShellResult(
            b_values=[1000, 3000],
            per_shell={1000: r1, 3000: r2},
        )

        assert len(result.b_values) == 2
        assert result.per_shell[1000].orders == [0, 2]
