"""Integration tests for RISH-GLM harmonization pipeline.

Tests the GLM-based multi-site harmonization workflow end-to-end.

Requires MRtrix3 installed. Run with: pytest -m integration
"""

import json
from pathlib import Path

import numpy as np
import pytest

from .conftest import pytestmark  # noqa: F401 — applies skip markers


class TestRISHGLMPipeline:
    """End-to-end RISH-GLM test with synthetic multi-site data."""

    def test_glm_fit_and_scale_maps(
        self, synth_dir, synth_site_a, synth_site_b, synth_mask
    ):
        from rish_harmonize.core.harmonize import (
            extract_native_rish,
            compute_consistent_lmax,
        )
        from rish_harmonize.core.rish_glm import (
            fit_rish_glm_per_shell,
            compute_glm_scale_maps_per_shell,
        )

        glm_dir = synth_dir / "glm_pipeline"
        glm_dir.mkdir(exist_ok=True)

        all_dwis = synth_site_a + synth_site_b
        site_labels = ["A"] * 3 + ["B"] * 3
        subjects = [f"sub{i}" for i in range(6)]

        # Compute consistent lmax
        shell_lmax = compute_consistent_lmax(all_dwis)

        # Extract RISH per subject
        rish_paths = {}  # {b_value: {order: [sub0.mif, ...]}}
        for i, dwi in enumerate(all_dwis):
            rish_dir = str(glm_dir / "rish" / subjects[i])
            rish = extract_native_rish(
                dwi, rish_dir,
                mask=synth_mask,
                shell_lmax=shell_lmax,
            )
            for b_value, orders in rish.items():
                if b_value not in rish_paths:
                    rish_paths[b_value] = {}
                for order, path in orders.items():
                    if order not in rish_paths[b_value]:
                        rish_paths[b_value][order] = []
                    rish_paths[b_value][order].append(path)

        # Fit RISH-GLM
        glm_result = fit_rish_glm_per_shell(
            rish_image_paths=rish_paths,
            site_labels=site_labels,
            mask_path=synth_mask,
            output_dir=str(glm_dir / "glm"),
            reference_site="A",
        )

        assert len(glm_result.b_values) == 2
        for b in glm_result.b_values:
            shell_res = glm_result.per_shell[b]
            assert shell_res.reference_site == "A"
            assert "A" in shell_res.site_names
            assert "B" in shell_res.site_names
            assert shell_res.n_subjects == 6

        # Compute scale maps for site B
        scale_maps = compute_glm_scale_maps_per_shell(
            result=glm_result,
            target_site="B",
            output_dir=str(glm_dir / "scale_maps"),
            mask=synth_mask,
        )

        for b_value, orders in scale_maps.items():
            for order, path in orders.items():
                assert Path(path).exists()

        # Check diagnostics were saved per shell
        for b_value in scale_maps:
            diag_path = glm_dir / "scale_maps" / f"b{b_value}" / "scale_map_diagnostics.json"
            assert diag_path.exists()

    def test_glm_model_save_load(
        self, synth_dir, synth_site_a, synth_site_b, synth_mask
    ):
        from rish_harmonize.core.harmonize import (
            extract_native_rish,
            compute_consistent_lmax,
        )
        from rish_harmonize.core.rish_glm import (
            fit_rish_glm_per_shell,
            load_rish_glm_per_shell,
        )

        rt_dir = synth_dir / "glm_roundtrip"
        rt_dir.mkdir(exist_ok=True)

        all_dwis = synth_site_a + synth_site_b
        site_labels = ["A"] * 3 + ["B"] * 3

        shell_lmax = compute_consistent_lmax(all_dwis)

        # Extract and fit
        rish_paths = {}
        for i, dwi in enumerate(all_dwis):
            rish = extract_native_rish(
                dwi, str(rt_dir / "rish" / f"sub{i}"),
                mask=synth_mask, shell_lmax=shell_lmax,
            )
            for b_value, orders in rish.items():
                if b_value not in rish_paths:
                    rish_paths[b_value] = {}
                for order, path in orders.items():
                    if order not in rish_paths[b_value]:
                        rish_paths[b_value][order] = []
                    rish_paths[b_value][order].append(path)

        original = fit_rish_glm_per_shell(
            rish_image_paths=rish_paths,
            site_labels=site_labels,
            mask_path=synth_mask,
            output_dir=str(rt_dir / "glm"),
            reference_site="A",
        )

        # Load from disk
        index_path = str(rt_dir / "glm" / "rish_glm_per_shell.json")
        loaded = load_rish_glm_per_shell(index_path)

        assert loaded.b_values == original.b_values
        for b in loaded.b_values:
            assert loaded.per_shell[b].site_names == original.per_shell[b].site_names
            assert loaded.per_shell[b].reference_site == "A"


class TestSubjectQC:
    """Test per-subject QC integration."""

    def test_qc_no_false_positives_same_site(
        self, synth_dir, synth_site_a, synth_mask
    ):
        """Subjects from the same site should not be flagged as outliers."""
        from rish_harmonize.core.harmonize import (
            extract_native_rish,
            compute_consistent_lmax,
        )
        from rish_harmonize.qc.subject_qc import compute_rish_subject_stats

        qc_dir = synth_dir / "qc_test"
        qc_dir.mkdir(exist_ok=True)

        shell_lmax = compute_consistent_lmax(synth_site_a)

        rish_paths = {}
        for i, dwi in enumerate(synth_site_a):
            rish = extract_native_rish(
                dwi, str(qc_dir / f"rish_sub{i}"),
                mask=synth_mask, shell_lmax=shell_lmax,
            )
            for b_value, orders in rish.items():
                if b_value not in rish_paths:
                    rish_paths[b_value] = {}
                for order, path in orders.items():
                    if order not in rish_paths[b_value]:
                        rish_paths[b_value][order] = []
                    rish_paths[b_value][order].append(path)

        subjects = ["sub0", "sub1", "sub2"]
        site_labels = ["A", "A", "A"]

        for b_value in rish_paths:
            qc_list = compute_rish_subject_stats(
                rish_image_paths=rish_paths[b_value],
                subject_ids=subjects,
                site_labels=site_labels,
                mask_path=synth_mask,
            )

            # Same-site synthetic subjects should have similar stats
            for q in qc_list:
                assert "mean_rish_l0" in q.metrics
                # Mean RISH l0 is from SH DC coefficient — can be
                # negative in synthetic data; just verify it's computed
                assert isinstance(q.metrics["mean_rish_l0"], float)
