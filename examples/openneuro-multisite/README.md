# OpenNeuro Multi-Site Example

End-to-end `rish-harmonize` example using a publicly available multi-site diffusion MRI dataset from [OpenNeuro](https://openneuro.org).

## Goal

1. Find a suitable multi-site DWI dataset on OpenNeuro
2. Download a small subset (3-5 subjects per site, 2+ sites)
3. Run the full signal-level SH pipeline and verify results
4. Document the workflow so others can reproduce it

## Dataset requirements

- **Multi-site**: acquired on 2+ different scanners/sites
- **Diffusion MRI**: DWI with gradient tables (bval/bvec)
- **Multi-shell preferred**: multiple non-zero b-values (e.g., b=1000/2000)
- **Manageable size**: a handful of subjects per site is enough
- **BIDS format**: standard OpenNeuro layout

### Candidate datasets to explore

| Accession | Name | Sites | Notes |
|-----------|------|-------|-------|
| ds005664 | SDSU Traveling Subject | 2 (GE/Siemens) | Already used in `examples/ds005664/` — different subset or shells could work |
| ds004215 | CoRR multi-site | multiple | Check if DWI is included |
| ds002080 | ABCD (subset) | multiple | Large study, check subset availability |

Search OpenNeuro for: `multi-site diffusion`, `traveling subject`, `harmonization`, `multi-scanner DWI`.

## Pipeline steps to implement

Follow the pattern in `examples/ds005664/run_pipeline_test.sh`:

```
Step 1: Download & convert to MIF (openneuro-cli or datalad + mrconvert)
Step 2: Create DWI list for consistent lmax
Step 3: Extract native RISH per subject/site
Step 4: Register to common space (affine or template-based)
Step 5: Warp RISH to template space
Step 6: Create reference template (create-template --mode signal)
Step 7: Compute scale maps (compute-scale-maps)
Step 8: Warp scale maps to native, apply harmonization
Step 9: Run RISH-GLM as alternative (rish-glm --mode signal_rish)
Step 10: Site effect comparison pre/post (site-effect)
Step 11: QC report (qc-report)
```

## Expected deliverables

- [ ] `config.sh` — dataset paths, subject list, site mapping
- [ ] `download.sh` — script to fetch data from OpenNeuro (using `openneuro-cli` or `datalad`)
- [ ] `run_pipeline.sh` — step-by-step pipeline (runnable with `./run_pipeline.sh [step]`)
- [ ] `manifest.csv` — RISH-GLM manifest for the dataset
- [ ] Results summary in this README (pre/post site effect stats, scale map diagnostics)

## Useful references

- Existing examples: `examples/ds005664/run_pipeline_test.sh`, `examples/ukb/run_fod_template.sh`
- Pipeline docs: `workflow.md`
- CLI reference: `README.md` (root)
- OpenNeuro CLI: `npm install -g @openneuro/cli` or use `datalad install`

## Notes

Registration (step 4) is dataset-dependent. Options:
- **Affine only** (simplest): `mrregister` or ANTs `antsRegistrationSyN.sh`
- **FOD template** (best for multi-shell): `population_template` from MRtrix3 (see UKB example)
- **FA template** (middle ground): ANTs SyN on FA maps (see `examples/ukb/run_fa_template.sh`)
