"""Tests for core/harmonize.py — harmonization pipeline logic."""

import pytest
from unittest.mock import patch, MagicMock

from rish_harmonize.core.harmonize import (
    harmonize_sh,
    harmonize_signal_sh,
    harmonize_fod,
    compute_consistent_lmax,
)
from rish_harmonize.core.shells import ShellInfo


class TestComputeConsistentLmax:
    """Test consistent lmax computation across subjects."""

    @patch("rish_harmonize.core.harmonize.detect_shells")
    def test_takes_minimum(self, mock_detect):
        """Should use the minimum lmax across all subjects per shell."""
        # Subject 0: 64 dirs at b=1000 -> lmax=8, 30 dirs at b=3000 -> lmax=6
        # Subject 1: 30 dirs at b=1000 -> lmax=6, 30 dirs at b=3000 -> lmax=6
        mock_detect.side_effect = [
            ShellInfo(
                b_values=[1000, 3000],
                shell_indices={1000: list(range(64)), 3000: list(range(30))},
                b0_indices=[],
            ),
            ShellInfo(
                b_values=[1000, 3000],
                shell_indices={1000: list(range(30)), 3000: list(range(30))},
                b0_indices=[],
            ),
        ]

        result = compute_consistent_lmax(["dwi0.mif", "dwi1.mif"])

        # b=1000: min(64, 30) = 30 dirs -> lmax=6
        assert result[1000] == 6
        # b=3000: min(30, 30) = 30 dirs -> lmax=6
        assert result[3000] == 6

    @patch("rish_harmonize.core.harmonize.detect_shells")
    def test_user_lmax_capped(self, mock_detect):
        """User-specified lmax should be capped by what data supports."""
        mock_detect.return_value = ShellInfo(
            b_values=[1000],
            shell_indices={1000: list(range(15))},  # 15 dirs -> lmax=4
            b0_indices=[],
        )

        # User requests lmax=8, but only 15 dirs -> capped to 4
        result = compute_consistent_lmax(["dwi.mif"], lmax=8)
        assert result[1000] == 4

    @patch("rish_harmonize.core.harmonize.detect_shells")
    def test_user_lmax_used_when_lower(self, mock_detect):
        """User lmax should be used when it's lower than auto-detected."""
        mock_detect.return_value = ShellInfo(
            b_values=[1000],
            shell_indices={1000: list(range(64))},  # 64 dirs -> lmax=8
            b0_indices=[],
        )

        # User requests lmax=4, data supports 8 -> use 4
        result = compute_consistent_lmax(["dwi.mif"], lmax=4)
        assert result[1000] == 4


class TestHarmonizeSignalShValidation:
    """Test input validation in harmonize_signal_sh."""

    @patch("rish_harmonize.core.shells._run_cmd")
    @patch("rish_harmonize.core.harmonize.extract_b0")
    @patch("rish_harmonize.core.harmonize.detect_shells")
    def test_missing_shell_in_reference(self, mock_detect, mock_b0, mock_cmd, tmp_path):
        """Should raise if reference_rish is missing a b-value shell."""
        mock_detect.return_value = ShellInfo(
            b_values=[1000, 2000],
            shell_indices={1000: [1, 2], 2000: [3, 4]},
            b0_indices=[0],
        )
        mock_b0.return_value = str(tmp_path / "b0.mif")

        # Only provide reference for b=2000, not b=1000
        # b=1000 is processed first (iteration order), so it will fail
        reference_rish = {2000: {0: "ref_l0.mif", 2: "ref_l2.mif"}}

        with pytest.raises(ValueError, match="No reference RISH for b=1000"):
            harmonize_signal_sh(
                "dwi.mif",
                reference_rish,
                str(tmp_path / "out.mif"),
            )


class TestHarmonizeFodValidation:
    """Test FOD harmonization input handling."""

    @patch("rish_harmonize.core.harmonize.get_image_lmax")
    @patch("rish_harmonize.core.harmonize.extract_rish_features")
    @patch("rish_harmonize.core.harmonize.compute_scale_maps")
    @patch("rish_harmonize.core.harmonize.harmonize_sh")
    def test_auto_detects_lmax(self, mock_harm, mock_scale, mock_rish, mock_lmax, tmp_path):
        """Should auto-detect lmax when not provided."""
        mock_lmax.return_value = 8
        mock_rish.return_value = {0: "r0.mif", 2: "r2.mif"}
        mock_scale.return_value = {0: "s0.mif", 2: "s2.mif"}

        reference_rish = {0: "ref0.mif", 2: "ref2.mif"}
        output = str(tmp_path / "harm.mif")

        harmonize_fod("fod.mif", reference_rish, output)

        mock_lmax.assert_called_once_with("fod.mif")
