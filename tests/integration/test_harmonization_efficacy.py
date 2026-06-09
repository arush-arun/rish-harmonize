"""Outcome (property) test: harmonization must actually *reduce* a known site effect.

The other integration tests assert that output files are produced. This one
asserts the science works: given a target site built with a known +30% l0 RISH
inflation relative to the reference site, RISH-GLM harmonization must move the
target's RISH back toward the reference.

Uses the RISH-GLM path (the recommended multi-subject workflow), which estimates
per-site mean RISH and is therefore robust to the per-voxel division noise that a
single-subject ref/target ratio would suffer from.

Marked ``integration`` (requires MRtrix3); skipped by default.
"""

from pathlib import Path

import numpy as np
import pytest

from .conftest import pytestmark  # noqa: F401 — applies integration + skip-if-no-MRtrix markers
from .conftest import _create_synthetic_dwi, _create_mask

# Known multiplicative site effect injected into the target site's l0 RISH.
SITE_EFFECT = 1.3
SHAPE = (12, 12, 12)
N_DIRS = 15
B_VALUES = [1000, 2000]


def _mean_in_mask(image_path: str, mask_path: str) -> float:
    from rish_harmonize.core.scale_maps import load_mif_as_array, load_mask_as_array

    data = load_mif_as_array(image_path)
    mask = load_mask_as_array(mask_path)
    return float(np.mean(data[mask]))


@pytest.fixture(scope="module")
def efficacy_data(tmp_path_factory):
    """Build 3 reference + 3 target subjects; target carries a +30% l0 effect."""
    work = tmp_path_factory.mktemp("efficacy")
    mask = _create_mask(SHAPE, str(work / "mask.mif"))

    ref = [
        _create_synthetic_dwi(
            str(work / f"ref{i}.mif"), shape=SHAPE, b_values=B_VALUES,
            n_dirs_per_shell=N_DIRS, seed=100 + i,
        )
        for i in range(3)
    ]
    target = [
        _create_synthetic_dwi(
            str(work / f"tar{i}.mif"), shape=SHAPE, b_values=B_VALUES,
            n_dirs_per_shell=N_DIRS,
            site_effect={b: {0: SITE_EFFECT} for b in B_VALUES},
            seed=200 + i,
        )
        for i in range(3)
    ]
    return {"work": work, "mask": mask, "ref": ref, "target": target}


class TestHarmonizationEfficacy:
    def test_rish_glm_reduces_known_site_effect(self, efficacy_data):
        from rish_harmonize.core.harmonize import (
            compute_consistent_lmax, extract_native_rish, apply_harmonization,
        )
        from rish_harmonize.core.shells import detect_shells, separate_shell
        from rish_harmonize.core.sh_fitting import fit_sh
        from rish_harmonize.core.rish import extract_rish_features
        from rish_harmonize.core.rish_glm import (
            fit_rish_glm_per_shell, compute_glm_scale_maps_per_shell,
        )

        work = efficacy_data["work"]
        mask = efficacy_data["mask"]
        dwis = efficacy_data["ref"] + efficacy_data["target"]
        sites = ["REF"] * 3 + ["TARGET"] * 3

        shell_lmax = compute_consistent_lmax(dwis)

        # Per-shell, per-order RISH for every subject: {b: {order: [paths]}}
        rish_paths = {}
        for i, dwi in enumerate(dwis):
            info = detect_shells(dwi)
            for b_value in info.b_values:
                sd = work / f"subj{i}" / f"b{b_value}"
                sd.mkdir(parents=True, exist_ok=True)
                dw = str(sd / "dw.mif")
                separate_shell(dwi, info, b_value, dw, include_b0=False)
                sh = str(sd / "sh.mif")
                fit_sh(dw, sh, lmax=shell_lmax[b_value])
                r = extract_rish_features(
                    sh, str(sd / "rish"), lmax=shell_lmax[b_value], mask=mask
                )
                rish_paths.setdefault(b_value, {})
                for order, p in r.items():
                    rish_paths[b_value].setdefault(order, []).append(p)

        # Fit GLM across both sites and derive target-site scale maps
        glm = fit_rish_glm_per_shell(
            rish_paths, sites, mask, str(work / "glm"), reference_site="REF"
        )
        scale_maps = compute_glm_scale_maps_per_shell(
            glm, "TARGET", str(work / "scale"), mask=mask
        )

        # Harmonize one target subject and re-extract its RISH
        target_subj = efficacy_data["target"][0]
        pre = extract_native_rish(
            target_subj, str(work / "pre"), mask=mask, shell_lmax=shell_lmax
        )
        harm = str(work / "harmonized.mif")
        apply_harmonization(target_subj, scale_maps, harm, shell_lmax=shell_lmax)
        post = extract_native_rish(
            harm, str(work / "post"), mask=mask, shell_lmax=shell_lmax
        )

        for b_value in B_VALUES:
            # Reference = mean l0 RISH across the REFERENCE-site subjects only
            # (the first 3 entries; the last 3 are the inflated target site).
            ref_l0 = np.mean(
                [_mean_in_mask(p, mask) for p in rish_paths[b_value][0][:3]]
            )
            ratio_pre = _mean_in_mask(pre[b_value][0], mask) / ref_l0
            ratio_post = _mean_in_mask(post[b_value][0], mask) / ref_l0

            # The injected effect must be present before harmonization...
            assert ratio_pre > 1.15, (
                f"b={b_value}: expected inflated pre ratio, got {ratio_pre:.3f}"
            )
            # ...harmonization must move it toward 1.0...
            assert abs(ratio_post - 1.0) < abs(ratio_pre - 1.0), (
                f"b={b_value}: site effect not reduced "
                f"(pre={ratio_pre:.3f}, post={ratio_post:.3f})"
            )
            # ...by a substantial margin (>=50% of the deviation removed)...
            assert abs(ratio_post - 1.0) < 0.5 * abs(ratio_pre - 1.0), (
                f"b={b_value}: site effect reduced by <50% "
                f"(pre={ratio_pre:.3f}, post={ratio_post:.3f})"
            )
            # ...leaving only a small residual.
            assert abs(ratio_post - 1.0) < 0.15, (
                f"b={b_value}: residual site effect too large "
                f"(post={ratio_post:.3f})"
            )
