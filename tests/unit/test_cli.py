"""Tests for CLI argument parsing."""

import pytest

from rish_harmonize.cli.main import build_parser, _read_list_file, _load_manifest


class TestParser:
    def setup_method(self):
        self.parser = build_parser()

    def test_detect_shells(self):
        args = self.parser.parse_args(["detect-shells", "dwi.mif"])
        assert args.command == "detect-shells"
        assert args.dwi == "dwi.mif"
        assert args.b0_threshold == 50.0

    def test_extract_rish(self):
        args = self.parser.parse_args([
            "extract-rish", "sh.mif", "-o", "rish/"
        ])
        assert args.command == "extract-rish"
        assert args.input == "sh.mif"
        assert args.output == "rish/"

    def test_extract_native_rish(self):
        args = self.parser.parse_args([
            "extract-native-rish", "dwi.mif",
            "-o", "native_rish/",
            "--mask", "mask.mif",
            "--lmax", "6",
        ])
        assert args.command == "extract-native-rish"
        assert args.dwi == "dwi.mif"
        assert args.output == "native_rish/"
        assert args.mask == "mask.mif"
        assert args.lmax == 6

    def test_extract_native_rish_consistent(self):
        args = self.parser.parse_args([
            "extract-native-rish", "dwi.mif",
            "-o", "native_rish/",
            "--consistent-with", "all_dwis.txt",
        ])
        assert args.consistent_with == "all_dwis.txt"

    def test_create_template_signal(self):
        args = self.parser.parse_args([
            "create-template",
            "--mode", "signal",
            "--rish-list", "rish_dirs.txt",
            "-o", "template/",
        ])
        assert args.mode == "signal"
        assert args.rish_list == "rish_dirs.txt"

    def test_create_template_fod(self):
        args = self.parser.parse_args([
            "create-template",
            "--mode", "fod",
            "--image-list", "fods.txt",
            "--mask-list", "masks.txt",
            "-o", "template/",
            "--lmax", "8",
        ])
        assert args.mode == "fod"
        assert args.lmax == 8

    def test_compute_scale_maps(self):
        args = self.parser.parse_args([
            "compute-scale-maps",
            "--ref-rish", "ref_rish/",
            "--target-rish", "target_rish/",
            "-o", "scale_maps/",
            "--mask", "mask.mif",
            "--smoothing", "5.0",
        ])
        assert args.command == "compute-scale-maps"
        assert args.ref_rish == "ref_rish/"
        assert args.target_rish == "target_rish/"
        assert args.smoothing == 5.0

    def test_apply_harmonization(self):
        args = self.parser.parse_args([
            "apply-harmonization", "dwi.mif",
            "--scale-maps", "scale_maps/",
            "-o", "harmonized.mif",
            "--lmax-json", "lmax.json",
        ])
        assert args.command == "apply-harmonization"
        assert args.dwi == "dwi.mif"
        assert args.scale_maps == "scale_maps/"
        assert args.lmax_json == "lmax.json"

    def test_harmonize_fod(self):
        args = self.parser.parse_args([
            "harmonize",
            "--target", "fod.mif",
            "--template", "template/",
            "--mask", "mask.mif",
            "-o", "out.mif",
            "--smoothing", "5.0",
            "--clip-min", "0.3",
            "--clip-max", "3.0",
        ])
        assert args.command == "harmonize"
        assert args.smoothing == 5.0
        assert args.clip_min == 0.3
        assert args.clip_max == 3.0

    def test_rish_glm(self):
        args = self.parser.parse_args([
            "rish-glm",
            "--manifest", "manifest.csv",
            "--reference-site", "SiteA",
            "-o", "output/",
            "--harmonize",
            "--threads", "8",
        ])
        assert args.reference_site == "SiteA"
        assert args.harmonize is True
        assert args.threads == 8

    def test_site_effect(self):
        args = self.parser.parse_args([
            "site-effect",
            "--site-list", "sites.csv",
            "--mask", "mask.mif",
            "-o", "results/",
            "--n-permutations", "1000",
        ])
        assert args.n_permutations == 1000


class TestReadListFile:
    def test_reads_lines(self, tmp_path):
        f = tmp_path / "list.txt"
        f.write_text("path1.mif\npath2.mif\n# comment\n\npath3.mif\n")

        result = _read_list_file(str(f))
        assert result == ["path1.mif", "path2.mif", "path3.mif"]


class TestLoadManifest:
    def test_signal_manifest(self, tmp_path):
        f = tmp_path / "manifest.csv"
        f.write_text(
            "subject,site,dwi_path,age\n"
            "sub-01,SiteA,/data/sub01.mif,25.0\n"
            "sub-02,SiteB,/data/sub02.mif,30.0\n"
        )

        subjects, sites, paths, covs, masks, mode = _load_manifest(str(f))
        assert subjects == ["sub-01", "sub-02"]
        assert sites == ["SiteA", "SiteB"]
        assert paths == ["/data/sub01.mif", "/data/sub02.mif"]
        assert "age" in covs
        assert covs["age"] == [25.0, 30.0]
        assert masks is None
        assert mode == "signal"

    def test_fod_manifest(self, tmp_path):
        f = tmp_path / "manifest.csv"
        f.write_text(
            "subject,site,fod_path\n"
            "sub-01,SiteA,/data/sub01_fod.mif\n"
        )

        _, _, _, _, _, mode = _load_manifest(str(f))
        assert mode == "fod"

    def test_manifest_with_masks(self, tmp_path):
        f = tmp_path / "manifest.csv"
        f.write_text(
            "subject,site,dwi_path,mask_path\n"
            "sub-01,A,dwi.mif,mask.mif\n"
        )

        _, _, _, _, masks, _ = _load_manifest(str(f))
        assert masks == ["mask.mif"]

    def test_missing_path_column(self, tmp_path):
        f = tmp_path / "manifest.csv"
        f.write_text("subject,site\nsub-01,A\n")

        with pytest.raises(ValueError, match="must have 'dwi_path' or 'fod_path'"):
            _load_manifest(str(f))
