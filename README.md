# nacc-progression-transport

Code for the manuscript *Why dementia progression models are miscalibrated at new centres: sources of bias and training-time fixes* (X. Tao and K. Chen, HKUST, under review).

The repository rebuilds, from one standard NACC Uniform Data Set (UDS) data request, the two prediction tasks (Task A: cognitively normal to MCI or dementia; Task B: MCI to dementia), all comparison models, the three validation schemes (random 5-fold, leave-one-centre-out over 46 ADRCs, temporal), the competing-risk sequence model, and the local-intercept recalibration experiments reported in the paper.

## Data

NACC data cannot be redistributed. Request the UDS Quick-Access file (this study used the September 2026 freeze, file `investigator_ftldlbd_nacc75.parquet`) from <https://naccdata.org>. Nothing under `data/`, `derived/` or `runs/` is tracked by git.

## Environment

Python 3.11 to 3.13, dependencies pinned in `uv.lock` (PyTorch 2.11 cu128, scikit-learn 1.9, scikit-survival 0.28, lifelines).

```bash
uv sync --frozen
```

## Pipeline

All scripts live in `src/`. Paths are controlled by environment variables with relative defaults.

| Step | Script | Purpose |
|---|---|---|
| 0 | `build_v2.py` | Landmark tables for both tasks (`NACC_DATA` = parquet path, `NACC_DERIVED` = output directory). |
| 1 | `phase1.py` | Cohort flow and descriptive tables (Supplementary Table S1). |
| 2 | `run_fold.py` | Train every model for one (task, scheme, fold) and write held-out predictions. Models and hyperparameters are in `models.py`, features in `features.py`. |
| 3 | `evaluate.py` | IPCW time-dependent AUC, O:E, calibration slope, ICI, Brier, IPA, decision curves, per-centre and REML-pooled estimates (`metrics.py`). |
| 4 | `e4_prep.py`, `e4_idm.py`, `e4_riley_inputs.py`, `e4_ensemble.py`, `e4_combine.py` | Sensitivity label variants, illness-death model, Riley sample-size inputs, multi-seed ensembles. |
| 5 | `e5c_analysis.py` | Local-intercept recalibration versus local sample size, per-centre bootstrap, paired model comparisons (`NACC_RUNS` = directory with `pred_*.parquet`). |
| 6 | `e5d_analysis.py` | Consolidated single-pass evaluation: single-model estimates with CIs and paired differences from the same bootstrap draws (B = 1,000 participant resamples in random/temporal; run with `--Bc 400` for 400 per centre in LOCO). Writes the files in `results/`. |

Example (Task B, leave-one-centre-out, held-out centre 3):

```bash
cd src
python run_fold.py --task B --scheme loco --fold 3 --models all --out ../runs/full_v1
python evaluate.py --runs ../runs/full_v1 --task B --scheme loco --out ../runs/full_v1/eval
```

`tasks_full.txt`, `evals_full.txt` and `worker*.sh` are the Slurm queue files used to run all folds in parallel on the HKUST HPC4 cluster; they are kept for provenance and are not needed on a single machine.

## Results and figures

`results/` holds the aggregate result tables behind every number in the manuscript (no row-level NACC data); `figures/` holds the manuscript and supplement figures (300 dpi).

| Manuscript item | File in `results/` | Notes |
|---|---|---|
| Table I (Task B, 3-year main results), Supplementary Table S2 (Task A) | `sanity_point_estimates.csv` (random / temporal columns), `center_pooled.csv` (LOCO columns) | e5d, B = 1,000 / 400 per centre |
| Table II (weighting and death handling) | `sanity_point_estimates.csv`, `scheme=random` | |
| Table III (paired comparisons) | `paired_loco_pooled.csv`, `model_a=crseq` | |
| τ² and calibration-slope prediction intervals in the text | `center_pooled.csv`, `metric=oe` / `slope` | |
| Random-split ICI differences 0.023, 0.046 | `paired_random_temporal.csv` | |
| Fig. 4, Table S5 (sample size for intercept adjustment) | `recal_curve_rows.csv`, `fig5_recal_events_summary.csv` | E5 |
| Supplement Table S1 (cohort) | `table1_v2.csv` | |
| Table S3 (sensitivity analyses) | `sensitivity_results.csv` | |
| Table S4 (illness-death model) | `idm_vs_aj.csv` | |
| S4 multiple seeds, S11 Riley sample size, S12 training time, subgroups | `multiseed_summary.csv`, `riley_samplesize.csv`, `runtimes_e4.csv`, `subgroups.csv` | |
| Earlier pipeline (comparison only) | `table2_taskA_3y.csv`, `table2_taskB_3y.csv` | `evaluate.py`, ICI pooled on the original scale |

## Citation

See `CITATION.cff`. A Zenodo DOI will be added at the first tagged release.

## Acknowledgement

The NACC database is funded by NIA/NIH Grant U24 AG072122. The full list of contributing ADRC grants is given in the manuscript.
