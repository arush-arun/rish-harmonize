"""Tests for core/shells.py — b-shell detection and separation."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from rish_harmonize.core.shells import (
    ShellInfo,
    detect_shells,
    separate_shell,
    extract_b0,
)


class TestShellInfo:
    def test_n_shells(self):
        info = ShellInfo(b_values=[1000, 2000, 3000])
        assert info.n_shells == 3

    def test_n_shells_empty(self):
        info = ShellInfo()
        assert info.n_shells == 0

    def test_n_volumes(self):
        info = ShellInfo(
            b_values=[1000, 2000],
            shell_indices={1000: [1, 2, 3], 2000: [4, 5]},
            b0_indices=[0],
        )
        assert info.n_volumes == 6  # 1 b0 + 3 + 2

    def test_n_directions(self):
        info = ShellInfo(
            b_values=[1000],
            shell_indices={1000: [1, 2, 3, 4, 5]},
        )
        assert info.n_directions(1000) == 5
        assert info.n_directions(9999) == 0


class TestDetectShells:
    """Test detect_shells with mocked mrinfo output."""

    def _make_grad_output(self, entries):
        """Build mrinfo -dwgrad output from (x, y, z, b) tuples."""
        lines = []
        for x, y, z, b in entries:
            lines.append(f"  {x:.6f}  {y:.6f}  {z:.6f}  {b:.1f}")
        return "\n".join(lines)

    @patch("rish_harmonize.core.shells._run_cmd")
    def test_single_shell(self, mock_run):
        """Detect a single b=1000 shell with 3 b=0 volumes."""
        entries = [
            (0, 0, 0, 0),    # b0, idx 0
            (0, 0, 0, 0),    # b0, idx 1
            (1, 0, 0, 1000),
            (0, 1, 0, 1000),
            (0, 0, 1, 1000),
            (0, 0, 0, 0),    # b0, idx 5
            (0.577, 0.577, 0.577, 1005),  # slight b-value variation
        ]
        mock_run.return_value = MagicMock(
            stdout=self._make_grad_output(entries),
            returncode=0,
        )

        info = detect_shells("fake.mif")

        assert info.n_shells == 1
        assert len(info.b0_indices) == 3
        assert 0 in info.b0_indices
        assert 1 in info.b0_indices
        assert 5 in info.b0_indices
        # b=1000 and b=1005 should cluster together
        b = info.b_values[0]
        assert info.n_directions(b) == 4

    @patch("rish_harmonize.core.shells._run_cmd")
    def test_multi_shell(self, mock_run):
        """Detect two shells: b=1000 and b=3000."""
        entries = [
            (0, 0, 0, 0),
            (1, 0, 0, 1000),
            (0, 1, 0, 1000),
            (0, 0, 1, 3000),
            (1, 1, 0, 3000),
        ]
        mock_run.return_value = MagicMock(
            stdout=self._make_grad_output(entries),
            returncode=0,
        )

        info = detect_shells("fake.mif")

        assert info.n_shells == 2
        assert len(info.b0_indices) == 1

    @patch("rish_harmonize.core.shells._run_cmd")
    def test_no_dw_volumes(self, mock_run):
        """All b=0 image has no shells."""
        entries = [(0, 0, 0, 0), (0, 0, 0, 0)]
        mock_run.return_value = MagicMock(
            stdout=self._make_grad_output(entries),
            returncode=0,
        )

        info = detect_shells("fake.mif")

        assert info.n_shells == 0
        assert len(info.b0_indices) == 2

    @patch("rish_harmonize.core.shells._run_cmd")
    def test_empty_gradient(self, mock_run):
        """No gradient info should raise ValueError."""
        mock_run.return_value = MagicMock(stdout="", returncode=0)

        with pytest.raises(ValueError, match="No gradient information"):
            detect_shells("fake.mif")


class TestSeparateShell:
    @patch("rish_harmonize.core.shells._run_cmd")
    def test_missing_shell_raises(self, mock_run):
        info = ShellInfo(
            b_values=[1000],
            shell_indices={1000: [1, 2, 3]},
            b0_indices=[0],
        )
        with pytest.raises(ValueError, match="Shell b=9999 not found"):
            separate_shell("fake.mif", info, 9999, "out.mif")

    @patch("rish_harmonize.core.shells._run_cmd")
    def test_include_b0(self, mock_run, tmp_path):
        info = ShellInfo(
            b_values=[1000],
            shell_indices={1000: [2, 3, 4]},
            b0_indices=[0, 1],
        )
        output = str(tmp_path / "shell.mif")
        separate_shell("in.mif", info, 1000, output, include_b0=True)

        # Check mrconvert was called with b0 + shell indices
        call_args = mock_run.call_args[0][0]
        assert "mrconvert" in call_args
        coord_idx = call_args.index("-coord")
        coord_str = call_args[coord_idx + 2]
        indices = [int(x) for x in coord_str.split(",")]
        assert indices == [0, 1, 2, 3, 4]

    @patch("rish_harmonize.core.shells._run_cmd")
    def test_exclude_b0(self, mock_run, tmp_path):
        info = ShellInfo(
            b_values=[1000],
            shell_indices={1000: [2, 3, 4]},
            b0_indices=[0, 1],
        )
        output = str(tmp_path / "shell.mif")
        separate_shell("in.mif", info, 1000, output, include_b0=False)

        call_args = mock_run.call_args[0][0]
        coord_idx = call_args.index("-coord")
        coord_str = call_args[coord_idx + 2]
        indices = [int(x) for x in coord_str.split(",")]
        assert indices == [2, 3, 4]


class TestExtractB0:
    @patch("rish_harmonize.core.shells._run_cmd")
    def test_calls_dwiextract(self, mock_run, tmp_path):
        output = str(tmp_path / "b0.mif")
        extract_b0("in.mif", output)

        call_args = mock_run.call_args[0][0]
        assert "dwiextract" in call_args
        assert "-bzero" in call_args
