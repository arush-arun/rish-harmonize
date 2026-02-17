"""Tests for core/sh_fitting.py — SH fitting wrappers."""

import pytest
from unittest.mock import patch, MagicMock

from rish_harmonize.core.sh_fitting import (
    determine_lmax,
    fit_sh,
    reconstruct_signal,
    get_directions,
    get_n_directions,
)


class TestDetermineLmax:
    """Test lmax auto-determination from number of directions."""

    def test_exact_fits(self):
        # n_coeffs = (lmax+1)(lmax+2)/2
        # lmax=2: 6 coeffs
        assert determine_lmax(6) == 2
        # lmax=4: 15 coeffs
        assert determine_lmax(15) == 4
        # lmax=6: 28 coeffs
        assert determine_lmax(28) == 6
        # lmax=8: 45 coeffs
        assert determine_lmax(45) == 8

    def test_typical_acquisitions(self):
        # 30 directions -> lmax=6 (28 coeffs fit, 36 don't)
        assert determine_lmax(30) == 6
        # 64 directions -> lmax=8 (45 coeffs)
        assert determine_lmax(64) == 8
        # 10 directions -> lmax=2 (6 coeffs fit, 15 don't)
        assert determine_lmax(10) == 2

    def test_few_directions(self):
        # 3 directions -> lmax=0 (only 1 coeff, but lmax=2 needs 6)
        assert determine_lmax(3) == 0
        # 1 direction -> lmax=0
        assert determine_lmax(1) == 0

    def test_ensures_even(self):
        # Must always return even number
        for n in range(1, 100):
            lmax = determine_lmax(n)
            assert lmax % 2 == 0
            assert lmax >= 0

    def test_coefficients_do_not_exceed_directions(self):
        """Core constraint: n_coeffs <= n_directions."""
        for n_dirs in range(1, 100):
            lmax = determine_lmax(n_dirs)
            n_coeffs = (lmax + 1) * (lmax + 2) // 2
            assert n_coeffs <= n_dirs, (
                f"n_dirs={n_dirs}, lmax={lmax}, n_coeffs={n_coeffs}"
            )


class TestFitSh:
    @patch("rish_harmonize.core.sh_fitting._run_cmd")
    def test_basic_call(self, mock_run, tmp_path):
        output = str(tmp_path / "sh.mif")
        fit_sh("dwi.mif", output)

        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "amp2sh"
        assert "dwi.mif" in call_args
        assert output in call_args
        assert "-force" in call_args

    @patch("rish_harmonize.core.sh_fitting._run_cmd")
    def test_with_options(self, mock_run, tmp_path):
        output = str(tmp_path / "sh.mif")
        fit_sh("dwi.mif", output, lmax=6, mask="mask.mif", normalise=True, n_threads=4)

        call_args = mock_run.call_args[0][0]
        assert "-lmax" in call_args
        assert "6" in call_args
        assert "-mask" in call_args
        assert "-normalise" in call_args
        assert "-nthreads" in call_args
        assert "4" in call_args


class TestReconstructSignal:
    @patch("rish_harmonize.core.sh_fitting._run_cmd")
    def test_basic_call(self, mock_run, tmp_path):
        output = str(tmp_path / "dwi.mif")
        reconstruct_signal("sh.mif", "dirs.txt", output)

        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "sh2amp"
        # sh2amp syntax: sh2amp input directions output
        assert call_args[1] == "sh.mif"
        assert call_args[2] == "dirs.txt"
        assert call_args[3] == output


class TestGetDirections:
    @patch("rish_harmonize.core.sh_fitting._run_cmd")
    def test_exports_grad(self, mock_run, tmp_path):
        output = str(tmp_path / "dirs.txt")
        get_directions("dwi.mif", output)

        call_args = mock_run.call_args[0][0]
        assert "mrinfo" in call_args
        assert "-export_grad_mrtrix" in call_args


class TestGetNDirections:
    @patch("rish_harmonize.core.sh_fitting._run_cmd")
    def test_counts_dw_only(self, mock_run):
        """Should count DW directions, excluding b=0."""
        grad_output = "\n".join([
            "  0  0  0  0",
            "  0  0  0  0",
            "  1  0  0  1000",
            "  0  1  0  1000",
            "  0  0  1  1000",
        ])
        mock_run.return_value = MagicMock(stdout=grad_output, returncode=0)

        n = get_n_directions("fake.mif")
        assert n == 3
