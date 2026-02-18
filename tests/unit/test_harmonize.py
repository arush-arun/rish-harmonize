"""Tests for core/harmonize.py — harmonization pipeline logic."""

import json
import pytest
from unittest.mock import patch, MagicMock

from rish_harmonize.core.harmonize import (
    harmonize_sh,
    apply_harmonization,
    harmonize_fod,
    compute_consistent_lmax,
    create_reference_template_signal,
    extract_native_rish,
    load_rish_dir,
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


class TestCreateTemplateValidation:
    """Test lmax consistency validation in create_reference_template_signal."""

    def test_mismatched_shells_raises(self, tmp_path):
        """Should raise if subjects have different b-value shells."""
        subj0 = {1000: {0: "r0.mif", 2: "r2.mif"}, 3000: {0: "r0.mif"}}
        subj1 = {1000: {0: "r0.mif", 2: "r2.mif"}}  # missing b=3000

        with pytest.raises(ValueError, match="shells"):
            create_reference_template_signal(
                [subj0, subj1], str(tmp_path / "template")
            )

    def test_mismatched_orders_raises(self, tmp_path):
        """Should raise if subjects have different lmax (orders) per shell."""
        subj0 = {1000: {0: "r0.mif", 2: "r2.mif", 4: "r4.mif"}}  # lmax=4
        subj1 = {1000: {0: "r0.mif", 2: "r2.mif"}}  # lmax=2

        with pytest.raises(ValueError, match="orders"):
            create_reference_template_signal(
                [subj0, subj1], str(tmp_path / "template")
            )


class TestApplyHarmonizationValidation:
    """Test input validation in apply_harmonization."""

    @patch("rish_harmonize.core.harmonize.extract_b0")
    @patch("rish_harmonize.core.harmonize.detect_shells")
    def test_missing_shell_in_scale_maps(self, mock_detect, mock_b0, tmp_path):
        """Should raise if scale_maps is missing a b-value shell."""
        mock_detect.return_value = ShellInfo(
            b_values=[1000, 2000],
            shell_indices={1000: [1, 2], 2000: [3, 4]},
            b0_indices=[0],
        )
        mock_b0.return_value = str(tmp_path / "b0.mif")

        # Only provide scale maps for b=2000, not b=1000
        scale_maps = {2000: {0: "scale_l0.mif", 2: "scale_l2.mif"}}

        with pytest.raises(ValueError, match="No scale maps for b=1000"):
            apply_harmonization(
                "dwi.mif",
                scale_maps,
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


class TestLoadRishDir:
    """Test RISH directory loading."""

    def test_loads_from_directory_structure(self, tmp_path):
        """Should find RISH files from b*/rish/rish_l*.mif convention."""
        # Create directory structure
        for b in [1000, 2000]:
            rish_dir = tmp_path / f"b{b}" / "rish"
            rish_dir.mkdir(parents=True)
            for order in [0, 2, 4]:
                (rish_dir / f"rish_l{order}.mif").touch()

        result = load_rish_dir(str(tmp_path))

        assert 1000 in result
        assert 2000 in result
        assert set(result[1000].keys()) == {0, 2, 4}
        assert set(result[2000].keys()) == {0, 2, 4}

    def test_loads_from_metadata_json(self, tmp_path):
        """Should load from shell_meta.json if present."""
        meta = {
            "rish": {
                "1000": {"0": "/path/rish_l0.mif", "2": "/path/rish_l2.mif"},
                "3000": {"0": "/path2/rish_l0.mif"},
            }
        }
        with open(tmp_path / "shell_meta.json", "w") as f:
            json.dump(meta, f)

        result = load_rish_dir(str(tmp_path))

        assert result[1000][0] == "/path/rish_l0.mif"
        assert result[1000][2] == "/path/rish_l2.mif"
        assert result[3000][0] == "/path2/rish_l0.mif"

    def test_loads_from_template_metadata_json(self, tmp_path):
        """Should load from template_meta.json if present."""
        meta = {
            "mode": "signal",
            "reference_rish": {
                "3000": {"0": "/tmpl/rish_l0.mif", "2": "/tmpl/rish_l2.mif"},
            }
        }
        with open(tmp_path / "template_meta.json", "w") as f:
            json.dump(meta, f)

        result = load_rish_dir(str(tmp_path))

        assert result[3000][0] == "/tmpl/rish_l0.mif"
        assert result[3000][2] == "/tmpl/rish_l2.mif"

    def test_raises_on_empty_dir(self, tmp_path):
        """Should raise if no RISH features found."""
        with pytest.raises(FileNotFoundError, match="No RISH features found"):
            load_rish_dir(str(tmp_path))
