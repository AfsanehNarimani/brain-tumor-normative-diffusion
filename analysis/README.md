# Frozen baseline analysis

Install the additional figure dependency with:

```bash
python -m pip install -r analysis/requirements.txt
python -m scripts.analyze_baseline --experiment-dir outputs/baseline_unet3d_full
```

Required sources are history.csv, summary.json, evaluation/aggregate_metrics.json,
and evaluation/per_case_metrics.csv. Checkpoints are never loaded. Generated
CSV/JSON/Markdown reports and 300-dpi PNG/vector PDF figures are confined to the
experiment's analysis/ directory. Re-running replaces generated reports only.

Selection uses positive-target regions only; missing positive-target observations
make that epoch's selection unavailable. Ties select the earliest epoch. Minimum
validation loss ignores missing observations. NETC uses tolerance 1e-8: first zero
and first zero after a prior positive result are reported separately. A zero
streak includes the first zero and ends at a nonzero or missing epoch. Dataset
Dice at the best NETC epoch is reported, rather than independently maximizing it.

Statistics exclude absent targets, use population standard deviation (ddof=0),
and linear percentiles. Cases with no applicable regions sort last with an empty
mean. Inconsistencies are recorded in analysis_summary.json; no source data is
corrected. Summary metadata is checked instead of loading checkpoint tensors.
The official expected case count is 270; experimental performance is never
hard-coded. Figures plot raw observations without smoothing.

## Experiment 0.1 evaluator reports

Run `python -m scripts.analyze_brats2024_evaluation --evaluation-dir outputs/baseline_unet3d_full/evaluation/brats2024` after installing `analysis/requirements.txt`. Optional `--top-k` defaults to 10. The CLI requires exactly 270 cases, all six regions per case, and 1,620 rows. It reads only `official_metrics.csv` and `overlap_summary.json`; reports go into their sibling `analysis/` directory. Repeated runs regenerate derived reports, never evaluator inputs.

Statistics summarize the evaluator's original values across all cases, including empty references; this is not Experiment 0's presence-aware aggregation. Finite-only summaries always include total/finite/NaN/positive-infinity/negative-infinity counts. No missing metric is imputed. Unavailable summary values are empty in CSV and null in JSON. Standard deviation is population standard deviation (`ddof=0`); quantiles use linear interpolation. Counts and volumes must be finite and nonnegative, and instance counts must be integers so presence and burden remain well-defined.

Extreme-case selection uses finite observations, ordered by metric and then lexicographic case ID, independently per region and criterion. Fewer than K rows are emitted when fewer finite observations exist. A case can appear under multiple criteria. Pairwise overlap fractions are not exclusive categories and can sum above one; a zero denominator produces null. Input byte hashes identify exactly the files analyzed. Numerical reports are deterministic; the provenance timestamp and Git dirty state describe each invocation and may differ. These reports do not establish split membership beyond the case IDs in the evaluator CSV and do not change or recompute official metric definitions.
