"""Integration tests for signal-level SH harmonization pipeline.

End-to-end test: extract-native-rish -> create-template -> compute-scale-maps
-> apply-harmonization.

Requires MRtrix3 installed. Run with: pytest -m integration
"""

import json
from pathlib import Path

import numpy as np
import pytest

from .conftest import pytestmark  # noqa: F401 — applies skip markers


class TestExtractNativeRish:
    """Test RISH extraction from synthetic DWI."""

    def test_extract_produces_rish_files(
        self, synth_dir, synth_site_a, synth_mask
    ):
        from rish_harmonize.core.harmonize import extract_native_rish

        dwi = synth_site_a[0]
        output_dir = str(synth_dir / "rish_extract_test")

        rish = extract_native_rish(dwi, output_dir, mask=synth_mask)

        # Should have entries for both b-values
        assert 1000 in rish
        assert 2000 in rish

        # Each shell should have RISH features (at least l=0)
        for b_value, orders in rish.items():
            assert 0 in orders
            assert Path(orders[0]).exists()

    def test_consistent_lmax(self, synth_dir, synth_site_a):
        from rish_harmonize.core.harmonize import compute_consistent_lmax

        lmax = compute_consistent_lmax(synth_site_a)

        # 15 directions -> lmax should be 4
        for b_value, l in lmax.items():
            assert l == 4


class TestSignalPipeline:
    """End-to-end signal-level harmonization pipeline."""

    def test_full_pipeline(
        self, synth_dir, synth_site_a, synth_site_b, synth_mask
    ):
        from rish_harmonize.core.harmonize import (
            extract_native_rish,
            compute_consistent_lmax,
            create_reference_template_signal,
            apply_harmonization,
            load_rish_dir,
        )
        from rish_harmonize.core.scale_maps import compute_scale_maps

        pipeline_dir = synth_dir / "pipeline"
        pipeline_dir.mkdir(exist_ok=True)

        all_dwis = synth_site_a + synth_site_b

        # Step 1: Compute consistent lmax
        shell_lmax = compute_consistent_lmax(all_dwis)
        assert len(shell_lmax) == 2  # b=1000, b=2000

        # Step 2: Extract RISH for all subjects
        rish_dirs = []
        for i, dwi in enumerate(all_dwis):
            rish_dir = str(pipeline_dir / f"rish_sub{i}")
            extract_native_rish(
                dwi, rish_dir,
                mask=synth_mask,
                shell_lmax=shell_lmax,
            )
            rish_dirs.append(rish_dir)

        # Step 3: Create reference template from site A subjects
        ref_rish = [load_rish_dir(d) for d in rish_dirs[:3]]
        template = create_reference_template_signal(
            ref_rish,
            str(pipeline_dir / "template"),
            shell_lmax=shell_lmax,
        )
        assert 1000 in template
        assert 2000 in template

        # Step 4: Compute scale maps (template ref vs target site B mean)
        tar_rish = [load_rish_dir(d) for d in rish_dirs[3:]]
        # Average target RISH
        template_rish = load_rish_dir(str(pipeline_dir / "template"))

        # For each b-shell, compute mean target RISH then scale maps
        all_scale_maps = {}
        for b_value in sorted(template_rish.keys()):
            # Average target RISH for this shell
            from rish_harmonize.core.scale_maps import (
                compute_scale_maps_from_groups,
            )

            ref_list = [subj[b_value] for subj in ref_rish]
            tar_list = [subj[b_value] for subj in tar_rish]

            scale_dir = str(pipeline_dir / "scale_maps" / f"b{b_value}")
            scale_maps = compute_scale_maps_from_groups(
                ref_list, tar_list, scale_dir,
                mask=synth_mask,
            )
            all_scale_maps[b_value] = scale_maps

        # Verify scale maps exist
        for b_value, orders in all_scale_maps.items():
            for order, path in orders.items():
                assert Path(path).exists(), f"Missing scale map b={b_value} l={order}"

        # Verify diagnostics were saved
        for b_value in all_scale_maps:
            diag_path = pipeline_dir / "scale_maps" / f"b{b_value}" / "scale_map_diagnostics.json"
            assert diag_path.exists()
            with open(diag_path) as f:
                diag = json.load(f)
            assert "l0" in diag
            assert "mean" in diag["l0"]

        # Step 5: Apply harmonization to a site B subject
        harm_output = str(pipeline_dir / "harmonized_sub3.mif")
        apply_harmonization(
            synth_site_b[0],
            all_scale_maps,
            harm_output,
            shell_lmax=shell_lmax,
        )
        assert Path(harm_output).exists()


class TestScaleMapDiagnostics:
    """Test that scale map diagnostics are computed correctly."""

    def test_diagnostics_json_structure(self, synth_dir, synth_site_a, synth_mask):
        from rish_harmonize.core.harmonize import extract_native_rish, load_rish_dir
        from rish_harmonize.core.scale_maps import compute_scale_maps

        diag_dir = synth_dir / "diag_test"
        diag_dir.mkdir(exist_ok=True)

        # Extract RISH for two subjects
        rish0 = extract_native_rish(
            synth_site_a[0], str(diag_dir / "rish0"), mask=synth_mask,
        )
        rish1 = extract_native_rish(
            synth_site_a[1], str(diag_dir / "rish1"), mask=synth_mask,
        )

        # Compute scale maps between subjects (should be ~1.0)
        for b_value in rish0:
            scale_dir = str(diag_dir / "scales" / f"b{b_value}")
            compute_scale_maps(
                rish0[b_value], rish1[b_value], scale_dir,
                mask=synth_mask,
            )

            diag_path = Path(scale_dir) / "scale_map_diagnostics.json"
            assert diag_path.exists()

            with open(diag_path) as f:
                diag = json.load(f)

            for key, stats in diag.items():
                assert "mean" in stats
                assert "pct_clipped_total" in stats
                assert "n_voxels" in stats
                # Verify diagnostics are valid numbers
                assert stats["n_voxels"] > 0
                assert 0.0 < stats["mean"] < 3.0
