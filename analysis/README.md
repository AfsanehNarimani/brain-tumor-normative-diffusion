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
