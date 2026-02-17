"""
rish-harmonize CLI

Subcommands for RISH harmonization of multi-site diffusion MRI data.
Supports both signal-level SH (per b-shell) and FOD-level modes.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional


def _read_list_file(path: str) -> List[str]:
    """Read a text file with one path per line."""
    lines = Path(path).read_text().strip().splitlines()
    return [l.strip() for l in lines if l.strip() and not l.startswith("#")]


def _load_manifest(path: str):
    """Load a manifest CSV with columns: subject, site, dwi_path/fod_path, [covariates].

    Returns
    -------
    subjects, site_labels, image_paths, covariates
    """
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(f"Empty manifest: {path}")

    subjects = [r["subject"] for r in rows]
    site_labels = [r["site"] for r in rows]

    # Determine image path column
    if "dwi_path" in fieldnames:
        image_paths = [r["dwi_path"] for r in rows]
        mode_hint = "signal"
    elif "fod_path" in fieldnames:
        image_paths = [r["fod_path"] for r in rows]
        mode_hint = "fod"
    else:
        raise ValueError("Manifest must have 'dwi_path' or 'fod_path' column")

    # Extract covariates (everything except subject, site, *_path, mask_path)
    skip = {"subject", "site", "dwi_path", "fod_path", "mask_path"}
    cov_names = [f for f in fieldnames if f not in skip]
    covariates = {}
    for name in cov_names:
        values = []
        for r in rows:
            val = r.get(name, "").strip()
            if val:
                values.append(float(val))
            else:
                values.append(0.0)
        covariates[name] = values

    mask_paths = None
    if "mask_path" in fieldnames:
        mask_paths = [r["mask_path"] for r in rows]

    return subjects, site_labels, image_paths, covariates, mask_paths, mode_hint


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_detect_shells(args):
    """Detect and display b-value shells in a DWI image."""
    from ..core.shells import detect_shells

    info = detect_shells(
        args.dwi,
        b0_threshold=args.b0_threshold,
    )

    print(f"Image: {args.dwi}")
    print(f"Total volumes: {info.n_volumes}")
    print(f"b=0 volumes: {len(info.b0_indices)} (indices: {info.b0_indices})")
    print(f"Shells: {info.n_shells}")
    for b in info.b_values:
        n = info.n_directions(b)
        print(f"  b={b}: {n} directions (indices: {info.shell_indices[b][:5]}{'...' if n > 5 else ''})")


def cmd_extract_rish(args):
    """Extract RISH features from an SH image."""
    from ..core.rish import extract_rish_features

    rish = extract_rish_features(
        args.input,
        args.output,
        lmax=args.lmax,
        mask=args.mask,
        n_threads=args.threads,
    )

    print("RISH features extracted:")
    for order, path in sorted(rish.items()):
        print(f"  l={order}: {path}")


def cmd_create_template(args):
    """Create RISH reference template."""
    if args.mode == "signal":
        from ..core.harmonize import create_reference_template_signal

        dwi_list = _read_list_file(args.image_list)
        mask_list = _read_list_file(args.mask_list) if args.mask_list else None

        template = create_reference_template_signal(
            dwi_list=dwi_list,
            mask_list=mask_list,
            output_dir=args.output,
            lmax=args.lmax,
            n_threads=args.threads,
        )

        print("Signal-level SH template created:")
        for b_value, orders in sorted(template.items()):
            print(f"  b={b_value}:")
            for order, path in sorted(orders.items()):
                print(f"    l={order}: {path}")

    elif args.mode == "fod":
        from ..core.harmonize import create_reference_template_fod

        fod_list = _read_list_file(args.image_list)
        mask_list = _read_list_file(args.mask_list) if args.mask_list else None

        template = create_reference_template_fod(
            fod_list=fod_list,
            mask_list=mask_list,
            output_dir=args.output,
            lmax=args.lmax,
            n_threads=args.threads,
        )

        print("FOD-level template created:")
        for order, path in sorted(template.items()):
            print(f"  l={order}: {path}")


def cmd_harmonize(args):
    """Harmonize a target image against a reference template."""
    # Load template metadata
    template_dir = Path(args.template)

    if args.mode == "signal":
        from ..core.harmonize import harmonize_signal_sh

        # Load per-shell template RISH paths from template metadata
        meta_path = template_dir / "template_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Template metadata not found: {meta_path}\n"
                "Run 'create-template' first, or provide template_meta.json"
            )

        with open(meta_path) as f:
            meta = json.load(f)

        reference_rish = {}
        for b_str, orders in meta["reference_rish"].items():
            reference_rish[int(b_str)] = {int(k): v for k, v in orders.items()}

        # Load per-shell lmax from template (ensures consistency)
        shell_lmax = {int(b): l for b, l in meta["shell_lmax"].items()}

        harmonize_signal_sh(
            dwi_image=args.target,
            reference_rish=reference_rish,
            output=args.output,
            mask=args.mask,
            shell_lmax=shell_lmax,
            smoothing_fwhm=args.smoothing,
            clip_range=(args.clip_min, args.clip_max),
            n_threads=args.threads,
        )

        print(f"Harmonized DWI written to: {args.output}")

    elif args.mode == "fod":
        from ..core.harmonize import harmonize_fod

        meta_path = template_dir / "template_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Template metadata not found: {meta_path}\n"
                "Run 'create-template' first, or provide template_meta.json"
            )

        with open(meta_path) as f:
            meta = json.load(f)

        reference_rish = {int(k): v for k, v in meta["reference_rish"].items()}

        harmonize_fod(
            fod_image=args.target,
            reference_rish=reference_rish,
            output=args.output,
            mask=args.mask,
            lmax=args.lmax,
            smoothing_fwhm=args.smoothing,
            clip_range=(args.clip_min, args.clip_max),
            n_threads=args.threads,
        )

        print(f"Harmonized FOD written to: {args.output}")


def cmd_rish_glm(args):
    """Fit RISH-GLM model and optionally harmonize."""
    subjects, site_labels, image_paths, covariates, mask_paths, mode_hint = \
        _load_manifest(args.manifest)

    mode = args.mode or mode_hint
    mask = args.mask

    if mode == "signal":
        from ..core.shells import detect_shells, separate_shell
        from ..core.sh_fitting import fit_sh
        from ..core.rish import extract_rish_features
        from ..core.rish_glm import (
            fit_rish_glm_per_shell,
            compute_glm_scale_maps_per_shell,
        )
        from ..core.harmonize import (
            harmonize_signal_sh_with_maps,
            compute_consistent_lmax,
        )

        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Compute consistent lmax across all subjects
        print("Computing consistent lmax across all subjects...")
        consistent_lmax = compute_consistent_lmax(image_paths, lmax=args.lmax)
        for b, l in sorted(consistent_lmax.items()):
            print(f"  b={b}: lmax={l}")

        # Step 1: For each subject, detect shells, fit SH, extract RISH
        print("Extracting per-shell RISH features...")
        # rish_paths[b_value][order] = [sub0.mif, sub1.mif, ...]
        rish_paths: Dict[int, Dict[int, List[str]]] = {}

        for i, dwi_path in enumerate(image_paths):
            subj_mask = mask_paths[i] if mask_paths else mask
            subj_dir = output_dir / "subjects" / subjects[i]
            subj_dir.mkdir(parents=True, exist_ok=True)

            shell_info = detect_shells(dwi_path)

            for b_value in shell_info.b_values:
                shell_dir = subj_dir / f"b{b_value}"
                shell_dir.mkdir(exist_ok=True)

                this_lmax = consistent_lmax[b_value]

                # Separate DW-only (no b=0)
                shell_dwi = str(shell_dir / "dwi_dw_only.mif")
                separate_shell(
                    dwi_path, shell_info, b_value, shell_dwi,
                    include_b0=False, n_threads=args.threads,
                )

                sh_path = str(shell_dir / "sh.mif")
                fit_sh(
                    shell_dwi, sh_path,
                    lmax=this_lmax, mask=subj_mask,
                    n_threads=args.threads,
                )

                rish_dir = str(shell_dir / "rish")
                rish = extract_rish_features(
                    sh_path, rish_dir,
                    lmax=this_lmax, mask=subj_mask,
                    n_threads=args.threads,
                )

                if b_value not in rish_paths:
                    rish_paths[b_value] = {}
                for order, path in rish.items():
                    if order not in rish_paths[b_value]:
                        rish_paths[b_value][order] = []
                    rish_paths[b_value][order].append(path)

            print(f"  [{i+1}/{len(image_paths)}] {subjects[i]}")

        # Step 2: Fit RISH-GLM per shell
        print("Fitting RISH-GLM per shell...")
        glm_result = fit_rish_glm_per_shell(
            rish_image_paths=rish_paths,
            site_labels=site_labels,
            mask_path=mask,
            output_dir=str(output_dir / "glm"),
            reference_site=args.reference_site,
            covariates=covariates if covariates else None,
        )

        # Save consistent lmax alongside GLM model
        lmax_meta = {"shell_lmax": {str(b): l for b, l in consistent_lmax.items()}}
        with open(output_dir / "glm" / "shell_lmax.json", "w") as f:
            json.dump(lmax_meta, f, indent=2)

        print(f"RISH-GLM model saved to: {output_dir / 'glm'}")

        # Step 3: Compute scale maps and harmonize if requested
        if args.harmonize:
            unique_sites = sorted(set(site_labels))
            target_sites = [s for s in unique_sites if s != args.reference_site]

            for target_site in target_sites:
                print(f"Computing scale maps for site: {target_site}")
                scale_maps = compute_glm_scale_maps_per_shell(
                    result=glm_result,
                    target_site=target_site,
                    output_dir=str(output_dir / "scale_maps" / target_site),
                    mask=mask,
                    smoothing_fwhm=args.smoothing,
                    clip_range=(args.clip_min, args.clip_max),
                    n_threads=args.threads,
                )

                # Harmonize each subject from this site
                site_indices = [
                    j for j, s in enumerate(site_labels) if s == target_site
                ]

                for j in site_indices:
                    harm_dir = output_dir / "harmonized" / subjects[j]
                    harm_dir.mkdir(parents=True, exist_ok=True)
                    harm_output = str(harm_dir / "dwi_harmonized.mif")

                    harmonize_signal_sh_with_maps(
                        dwi_image=image_paths[j],
                        scale_maps=scale_maps,
                        output=harm_output,
                        mask=mask_paths[j] if mask_paths else mask,
                        shell_lmax=consistent_lmax,
                        n_threads=args.threads,
                    )
                    print(f"  Harmonized: {subjects[j]} -> {harm_output}")

    elif mode == "fod":
        from ..core.rish import extract_rish_features
        from ..core.rish_glm import fit_rish_glm, compute_glm_scale_maps
        from ..core.harmonize import harmonize_sh

        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract RISH per subject
        print("Extracting RISH features from FODs...")
        rish_paths: Dict[int, List[str]] = {}

        for i, fod_path in enumerate(image_paths):
            subj_mask = mask_paths[i] if mask_paths else mask
            subj_dir = output_dir / "subjects" / subjects[i]

            rish = extract_rish_features(
                fod_path, str(subj_dir / "rish"),
                lmax=args.lmax, mask=subj_mask,
                n_threads=args.threads,
            )

            for order, path in rish.items():
                if order not in rish_paths:
                    rish_paths[order] = []
                rish_paths[order].append(path)

            print(f"  [{i+1}/{len(image_paths)}] {subjects[i]}")

        # Fit RISH-GLM
        print("Fitting RISH-GLM...")
        glm_result = fit_rish_glm(
            rish_image_paths=rish_paths,
            site_labels=site_labels,
            mask_path=mask,
            output_dir=str(output_dir / "glm"),
            reference_site=args.reference_site,
            covariates=covariates if covariates else None,
        )

        print(f"RISH-GLM model saved to: {output_dir / 'glm'}")

        if args.harmonize:
            unique_sites = sorted(set(site_labels))
            target_sites = [s for s in unique_sites if s != args.reference_site]

            for target_site in target_sites:
                print(f"Computing scale maps for site: {target_site}")
                scale_maps = compute_glm_scale_maps(
                    result=glm_result,
                    target_site=target_site,
                    output_dir=str(output_dir / "scale_maps" / target_site),
                    mask=mask,
                    smoothing_fwhm=args.smoothing,
                    clip_range=(args.clip_min, args.clip_max),
                    n_threads=args.threads,
                )

                site_indices = [
                    j for j, s in enumerate(site_labels) if s == target_site
                ]

                for j in site_indices:
                    harm_dir = output_dir / "harmonized" / subjects[j]
                    harm_dir.mkdir(parents=True, exist_ok=True)
                    harm_output = str(harm_dir / "fod_harmonized.mif")

                    harmonize_sh(
                        image_paths[j],
                        scale_maps,
                        harm_output,
                        lmax=args.lmax,
                        n_threads=args.threads,
                    )
                    print(f"  Harmonized: {subjects[j]} -> {harm_output}")


def cmd_site_effect(args):
    """Test for site effects using permutation testing."""
    from ..qc.site_effects import test_site_effect

    # Load site assignments from CSV
    site_map = {}
    with open(args.site_list) as f:
        reader = csv.DictReader(f)
        for row in reader:
            site = row["site"]
            path = row.get("rish_path") or row.get("image_path")
            if site not in site_map:
                site_map[site] = []
            site_map[site].append(path)

    result = test_site_effect(
        site_image_paths=site_map,
        mask_path=args.mask,
        output_dir=args.output,
        n_permutations=args.n_permutations,
        n_threads=args.threads,
    )

    print(f"Site effect test complete. Results in: {args.output}")
    if hasattr(result, "significant_fraction"):
        print(f"  Fraction of significant voxels: {result.significant_fraction:.4f}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        prog="rish-harmonize",
        description="RISH harmonization for multi-site diffusion MRI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # -- detect-shells --
    p = subparsers.add_parser("detect-shells", help="Detect b-value shells in a DWI")
    p.add_argument("dwi", help="Input DWI image")
    p.add_argument("--b0-threshold", type=float, default=50.0,
                    help="B-value threshold for b=0 (default: 50)")
    p.set_defaults(func=cmd_detect_shells)

    # -- extract-rish --
    p = subparsers.add_parser("extract-rish", help="Extract RISH features from SH image")
    p.add_argument("input", help="Input SH coefficient image")
    p.add_argument("-o", "--output", required=True, help="Output directory")
    p.add_argument("--lmax", type=int, default=None, help="Maximum SH order")
    p.add_argument("--mask", help="Brain mask")
    p.add_argument("--threads", type=int, default=1, help="Number of threads")
    p.set_defaults(func=cmd_extract_rish)

    # -- create-template --
    p = subparsers.add_parser("create-template", help="Create RISH reference template")
    p.add_argument("--mode", required=True, choices=["signal", "fod"],
                    help="Harmonization mode")
    p.add_argument("--image-list", required=True,
                    help="Text file with one image path per line (DWI for signal, FOD for fod)")
    p.add_argument("--mask-list", help="Text file with one mask path per line")
    p.add_argument("-o", "--output", required=True, help="Output directory")
    p.add_argument("--lmax", type=int, default=None, help="Maximum SH order")
    p.add_argument("--threads", type=int, default=1, help="Number of threads")
    p.set_defaults(func=cmd_create_template)

    # -- harmonize --
    p = subparsers.add_parser("harmonize",
                               help="Harmonize a target against a reference template")
    p.add_argument("--mode", required=True, choices=["signal", "fod"],
                    help="Harmonization mode")
    p.add_argument("--target", required=True, help="Target image (DWI or FOD)")
    p.add_argument("--template", required=True, help="Template directory")
    p.add_argument("--mask", help="Brain mask")
    p.add_argument("-o", "--output", required=True, help="Output path")
    p.add_argument("--lmax", type=int, default=None, help="Maximum SH order")
    p.add_argument("--smoothing", type=float, default=3.0,
                    help="Scale map smoothing FWHM in mm (default: 3.0)")
    p.add_argument("--clip-min", type=float, default=0.5,
                    help="Minimum scale factor (default: 0.5)")
    p.add_argument("--clip-max", type=float, default=2.0,
                    help="Maximum scale factor (default: 2.0)")
    p.add_argument("--threads", type=int, default=1, help="Number of threads")
    p.set_defaults(func=cmd_harmonize)

    # -- rish-glm --
    p = subparsers.add_parser("rish-glm",
                               help="Fit RISH-GLM model (joint site + covariate)")
    p.add_argument("--manifest", required=True,
                    help="CSV with columns: subject, site, dwi_path/fod_path, [covariates]")
    p.add_argument("--reference-site", required=True,
                    help="Name of the reference site")
    p.add_argument("--mode", choices=["signal", "fod"],
                    help="Mode (auto-detected from manifest if not specified)")
    p.add_argument("--mask", help="Group brain mask")
    p.add_argument("-o", "--output", required=True, help="Output directory")
    p.add_argument("--lmax", type=int, default=None, help="Maximum SH order")
    p.add_argument("--harmonize", action="store_true",
                    help="Also harmonize all target-site subjects")
    p.add_argument("--smoothing", type=float, default=3.0,
                    help="Scale map smoothing FWHM in mm (default: 3.0)")
    p.add_argument("--clip-min", type=float, default=0.5,
                    help="Minimum scale factor (default: 0.5)")
    p.add_argument("--clip-max", type=float, default=2.0,
                    help="Maximum scale factor (default: 2.0)")
    p.add_argument("--threads", type=int, default=1, help="Number of threads")
    p.set_defaults(func=cmd_rish_glm)

    # -- site-effect --
    p = subparsers.add_parser("site-effect",
                               help="Test for site effects via permutation testing")
    p.add_argument("--site-list", required=True,
                    help="CSV with columns: site, rish_path/image_path")
    p.add_argument("--mask", required=True, help="Group brain mask")
    p.add_argument("-o", "--output", required=True, help="Output directory")
    p.add_argument("--n-permutations", type=int, default=5000,
                    help="Number of permutations (default: 5000)")
    p.add_argument("--threads", type=int, default=1, help="Number of threads")
    p.set_defaults(func=cmd_site_effect)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
