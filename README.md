# Beyond Benchmark Agreement: A Three-Axis Audit of No-Reference Underwater Image Quality Metrics

Code and result files for the CVIU submission by Ji Hoon Bang and Suchendra M. Bhandarkar (University of Georgia).

The paper audits 14 no-reference image-quality metrics for underwater image enhancement along three axes:
agreement with two human-rated benchmarks (SAUD, UID2021), behavior under a controlled over-processing stress test
(unsharp masking and dehazing sweeps), and stability of downstream pipeline conclusions in a coral-reef case study.
No image is redistributed here. The repository contains the evaluation code and the CSV files behind every number in
the paper and the supplementary material.

## Layout
```
qcs_eval/          metric implementations, operators, calibration, folds and clusters
scripts/           one script per analysis, in paper order (table below)
scripts/figures/   figure scripts
results/           CSV outputs used in the paper (per-scene, per-image scores, summaries, bootstrap intervals)
```

## Setup
```
pip install -r requirements.txt
```
Set three environment variables (paths are never hard-coded):

| Variable | Points to |
|---|---|
| `CORAL_BASE` | folder that holds the enhanced coral outputs (one sub-folder per configuration) and receives `_resub_v3/` outputs |
| `SAUD_DIR` | the SAUD dataset as distributed by its authors |
| `UID2021_DIR` | the UID2021 dataset as distributed by its authors |

The deep NR-IQA models (MUSIQ, TOPIQ-NR, LIQE, HyperIQA, TReS) and NIQE are loaded through `pyiqa`, which downloads
the official weights on first use.

## Scripts and paper sections
| Script | Paper |
|---|---|
| `scripts/a0_reproduce_table1.py` | QCS-v1 whitening fit and freeze (Sec. 3, S1) |
| `scripts/a0b_resolution_census.py` | raw short-side >= 128 px rule, 118 images removed (Sec. 6.1) |
| `scripts/a0b_image_sizes.py` | image-size audit used by the resolution rule |
| `scripts/a0c_duplicate_audit.py` | near-duplicate groups merged into bootstrap clusters (S7) |
| `scripts/a0d_inspect_groups.py` | inspection of the duplicate groups (diagnostic) |
| `scripts/v2_calibrate.py` | v2 robust calibration, frozen (S1) |
| `scripts/a3a_dataset_audit.py` | label-blind SAUD / UID2021 image audit, pHash/dHash overlap (Sec. 4.1, S2) |
| `scripts/a3b_freeze_human_sets.py` | frozen human-benchmark manifests and clusters (Sec. 4.1, S2) |
| `scripts/h1_score_human_sets.py` | scores every human-benchmark image with all metrics |
| `scripts/h2_human_validation.py` | Table 2, Tables S1-S3 (SROCC, pair accuracy, Holm-adjusted paired tests) |
| `scripts/h2a_inspect_labels.py` | first look at the label files (diagnostic) |
| `scripts/h3_uciqe_audit.py` | UCIQE / UIQM implementation audit (S6) |
| `scripts/h4_family_interaction.py` | post hoc family-level interaction (Sec. 4.2, Table S17) |
| `scripts/a2a_stress.py` | stress test on two samples (Table 3, Tables S4-S5, Fig. 3 data) |
| `scripts/a2a_example_figure.py` | Fig. 4 (USM examples on field frames) |
| `scripts/c1_coral_rankings.py` | downstream rankings and contrasts (Table 4, Fig. 6-8, Tables S6-S14) |
| `scripts/c1_cluster_census.py` | cluster composition of the 2,294 test images (S7) |
| `scripts/s8_posthoc_exploratory.py` | exploratory component- and method-level analysis (S8, Table S16) |
| `scripts/diag_stems.py` | file-name diagnostic (optional) |

| Figure script | Output |
|---|---|
| `scripts/figures/make_fig1_cviu.py` | Fig. 1 (three-axis overview) |
| `scripts/figures/make_figs_cviu_readable.py` | Fig. 2, Fig. 6, Fig. 7, Fig. 8 |
| `scripts/figures/make_fig3_cviu.py` | Fig. 3 (mean stress responses) |
| `scripts/figures/make_fig4_cviu.py` | Fig. 5 (stress turnover vs. human agreement) |

## Typical order
1. `a0b_*`, `a0c_duplicate_audit.py` (coral corpus audits), `a0_reproduce_table1.py`, `v2_calibrate.py`
2. `a3a_dataset_audit.py`, `a3b_freeze_human_sets.py`, `h1_score_human_sets.py`, `h2_human_validation.py`, `h3_uciqe_audit.py`, `h4_family_interaction.py`
3. `a2a_stress.py` (run twice: seeds 0 and 20261019), `a2a_example_figure.py`
4. `c1_coral_rankings.py`, `c1_cluster_census.py`
5. figure scripts

Every script prints `[PASS]` / `[WARN]` / `[FAIL]` debug checks and writes its outputs under `$CORAL_BASE/_resub_v3/`.

## Data
- SAUD and UID2021: obtain from the original authors. Their labels are not included here.
- Coral corpus: public images from ReefNet and iNaturalist remain under the licenses of those sources, and the
  Little Grecian field frames are used with permission and are not redistributed. `results/` holds only derived scores.

## Citation
To be added after publication.
