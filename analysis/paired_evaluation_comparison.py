"""Post-hoc paired comparison of frozen BraTS-compatible evaluation outputs."""

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import wilcoxon

REGIONS = ('NETC', 'SNFH', 'ET', 'RC', 'WT', 'TC')
METRICS = ('global_bin_dsc', 'global_bin_hd95', 'global_bin_nsd', 'rq',
           'sq_dsc', 'pq_dsc', 'sq_hd95', 'sq_nsd', 'prec', 'rec')
COMPONENTS = ('n_pred_instances', 'tp', 'fp', 'fn')
INVARIANTS = ('n_ref_instances', 'global_bin_volume_ref')
REQUIRED = ('case_id', 'Labels', 'global_bin_volume_pred') + INVARIANTS + COMPONENTS + METRICS
DIRECTIONS = {m: ('lower_is_better' if m in ('global_bin_hd95', 'sq_hd95')
                  else 'higher_is_better') for m in METRICS}
WILCOXON_PARAMETERS = {'zero_method': 'wilcox', 'correction': False,
                       'alternative': 'two-sided', 'method': 'auto', 'axis': 0,
                       'nan_policy': 'raise', 'keepdims': False}
BOOTSTRAP_SEED = 20260917
BOOTSTRAP_RESAMPLES = 10000
ARTIFACTS = ('paired_metric_statistics.csv', 'paired_case_metrics.csv',
             'component_comparison.csv', 'reference_empty_fp_transitions.csv',
             'netc_diagnostics.json', 'overlap_comparison.json',
             'comparison_summary.json', 'comparison_metadata.json')


def load_metrics(path, expected_cases=270):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing official metrics: {path}')
    frame = pd.read_csv(path, dtype={'case_id': str, 'Labels': str})
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f'{path}: missing columns: {sorted(missing)}')
    if frame[['case_id', 'Labels']].isna().any().any() or frame.case_id.str.strip().eq('').any():
        raise ValueError(f'{path}: invalid case_id or Labels')
    if frame.duplicated(['case_id', 'Labels']).any():
        raise ValueError(f'{path}: duplicate (case_id, Labels) rows')
    if set(frame.Labels) != set(REGIONS):
        raise ValueError(f'{path}: unexpected or missing regions: {sorted(set(frame.Labels))}')
    if frame.case_id.nunique() != expected_cases or len(frame) != expected_cases * len(REGIONS):
        raise ValueError(f'{path}: expected {expected_cases} cases and {expected_cases * len(REGIONS)} rows')
    for column in set(REQUIRED) - {'case_id', 'Labels'}:
        try:
            frame[column] = pd.to_numeric(frame[column], errors='raise')
        except (ValueError, TypeError) as exc:
            raise ValueError(f'{path}: invalid numeric column {column}') from exc
    return frame


def validate_pair(e0, e1, expected_cases=270):
    a, b = load_metrics(e0, expected_cases), load_metrics(e1, expected_cases)
    keys = ['case_id', 'Labels']
    ka, kb = set(map(tuple, a[keys].itertuples(index=False, name=None))), set(map(tuple, b[keys].itertuples(index=False, name=None)))
    if ka != kb:
        raise ValueError(f'Mismatched (case_id, Labels) keys: E0-only={len(ka-kb)}, E1-A-only={len(kb-ka)}')
    paired = a.merge(b, on=keys, suffixes=('_e0', '_e1a'), validate='one_to_one')
    for column in INVARIANTS:
        x, y = paired[f'{column}_e0'].to_numpy(), paired[f'{column}_e1a'].to_numpy()
        if not np.all((x == y) | (np.isnan(x) & np.isnan(y))):
            raise ValueError(f'Reference invariant mismatch: {column}')
    paired['region'] = paired.Labels
    paired['region_order'] = pd.Categorical(paired.region, REGIONS, ordered=True)
    return paired.sort_values(['region_order', 'case_id'], kind='stable').drop(columns='region_order').reset_index(drop=True)


def bootstrap_ci(delta, seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_RESAMPLES):
    delta = np.asarray(delta, dtype=float)
    if not len(delta):
        return None, None
    rng = np.random.default_rng(seed)
    # Chunking bounds memory while preserving the exact random stream.
    means = np.empty(resamples)
    for start in range(0, resamples, 512):
        stop = min(start + 512, resamples)
        means[start:stop] = delta[rng.integers(0, len(delta), (stop-start, len(delta)))].mean(axis=1)
    return tuple(float(v) for v in np.percentile(means, [2.5, 97.5]))


def holm_adjust(pvalues):
    """Holm step-down adjustment; undefined tests retain null."""
    result = [None] * len(pvalues)
    valid = sorted(((i, p) for i, p in enumerate(pvalues) if p is not None), key=lambda item: item[1])
    running = 0.0
    for rank, (index, p) in enumerate(valid):
        running = max(running, min(1.0, (len(valid)-rank) * p))
        result[index] = running
    return result


def metric_statistics(paired, seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_RESAMPLES):
    rows = []
    for metric in METRICS:
        for region in REGIONS:
            part = paired[paired.region == region]
            x, y = part[f'{metric}_e0'].to_numpy(dtype=float), part[f'{metric}_e1a'].to_numpy(dtype=float)
            fx, fy = np.isfinite(x), np.isfinite(y)
            delta = y[fx & fy] - x[fx & fy]
            n = len(delta)
            direction = DIRECTIONS[metric]
            improved = int(((delta > 0) if direction == 'higher_is_better' else (delta < 0)).sum())
            worsened = int(((delta < 0) if direction == 'higher_is_better' else (delta > 0)).sum())
            sd = float(np.std(delta, ddof=1)) if n > 1 else None
            dz_reason = None if sd is not None and sd > 0 else ('insufficient_finite_pairs' if n < 2 else 'zero_delta_variance')
            if n == 0:
                p, test_reason = None, 'no_finite_pairs'
            elif not np.any(delta):
                p, test_reason = 1.0, 'all_zero_differences'
            else:
                p = float(wilcoxon(y[fx & fy], x[fx & fy], **WILCOXON_PARAMETERS).pvalue)
                test_reason = None
            lo, hi = bootstrap_ci(delta, seed, resamples)
            stat = lambda f: float(f(delta)) if n else None
            rows.append(dict(region=region, metric=metric, interpretation_direction=direction,
                total_paired_rows=len(part), both_finite=n, e0_finite_e1a_nonfinite=int((fx & ~fy).sum()),
                e0_nonfinite_e1a_finite=int((~fx & fy).sum()), both_nonfinite=int((~fx & ~fy).sum()),
                e0_nan_count=int(np.isnan(x).sum()), e1a_nan_count=int(np.isnan(y).sum()),
                e0_posinf_count=int(np.isposinf(x).sum()), e1a_posinf_count=int(np.isposinf(y).sum()),
                e0_neginf_count=int(np.isneginf(x).sum()), e1a_neginf_count=int(np.isneginf(y).sum()),
                e0_mean=float(x[fx & fy].mean()) if n else None, e1a_mean=float(y[fx & fy].mean()) if n else None,
                mean_delta=stat(np.mean), median_delta=stat(np.median), delta_std=sd,
                delta_q1=stat(lambda z: np.quantile(z, .25)), delta_q3=stat(lambda z: np.quantile(z, .75)),
                min_delta=stat(np.min), max_delta=stat(np.max), delta_gt_zero=int((delta > 0).sum()),
                delta_lt_zero=int((delta < 0).sum()), delta_eq_zero=int((delta == 0).sum()),
                improved_case_count=improved, worsened_case_count=worsened, tied_case_count=int((delta == 0).sum()),
                improved_fraction=improved/n if n else None, wilcoxon_raw_p=p, wilcoxon_reason=test_reason,
                cohens_dz=float(delta.mean()/sd) if dz_reason is None else None, cohens_dz_undefined_reason=dz_reason,
                bootstrap_mean_delta_ci_lower=lo, bootstrap_mean_delta_ci_upper=hi))
    for start in range(0, len(rows), len(REGIONS)):
        adjusted = holm_adjust([r['wilcoxon_raw_p'] for r in rows[start:start+len(REGIONS)]])
        for row, value in zip(rows[start:start+len(REGIONS)], adjusted):
            row['wilcoxon_holm_adjusted_p'] = value
    return pd.DataFrame(rows)


def case_metrics(paired):
    result = paired[['case_id', 'region']].copy()
    for metric in METRICS:
        x, y = paired[f'{metric}_e0'], paired[f'{metric}_e1a']
        finite = np.isfinite(x) & np.isfinite(y)
        result[f'{metric}_e0'], result[f'{metric}_e1a'] = x, y
        # Raw IEEE arithmetic retains NaN and infinity semantics.
        with np.errstate(invalid='ignore'):
            result[f'{metric}_delta'] = y - x
        result[f'{metric}_finite_pair'] = finite
    return result


def component_comparison(paired):
    rows = []
    for region in REGIONS:
        part = paired[paired.region == region]
        for column in COMPONENTS:
            x, y = part[f'{column}_e0'].to_numpy(), part[f'{column}_e1a'].to_numpy()
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError(f'Nonfinite component quantity: {column}')
            a, b = float(x.sum()), float(y.sum())
            rows.append(dict(region=region, quantity=column, e0_total=a, e1a_total=b,
                             absolute_change=b-a, relative_change=(b-a)/a if a else None,
                             e1a_lower_count=int((y < x).sum()), e1a_equal_count=int((y == x).sum()),
                             e1a_higher_count=int((y > x).sum())))
    return pd.DataFrame(rows)


def reference_empty_transitions(paired):
    rows = []
    for region in REGIONS:
        part = paired[(paired.region == region) & (paired.global_bin_volume_ref_e0 == 0)]
        x = part.global_bin_volume_pred_e0.to_numpy()
        y = part.global_bin_volume_pred_e1a.to_numpy()
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Nonfinite predicted volume in reference-empty cases')
        a, b = x > 0, y > 0
        rows.append(dict(region=region, reference_empty_cases=len(part), e0_prediction_positive_count=int(a.sum()),
                         e1a_prediction_positive_count=int(b.sum()), negative_to_negative=int((~a & ~b).sum()),
                         negative_to_positive=int((~a & b).sum()), positive_to_negative=int((a & ~b).sum()),
                         positive_to_positive=int((a & b).sum())))
    return pd.DataFrame(rows)


def compare_overlap(e0, e1, expected_cases=270):
    required = ('cases_with_overlap', 'total_foreground_voxels', 'total_overlapping_voxels',
                'overall_overlap_fraction', 'maximum_case_overlap_fraction', 'maximum_overlap_channels_seen')
    for name, obj in (('E0', e0), ('E1-A', e1)):
        if obj.get('case_count') != expected_cases:
            raise ValueError(f'{name} overlap case_count must be {expected_cases}')
        if any(k not in obj for k in required) or not isinstance(obj.get('pairwise_overlap_counts'), dict):
            raise ValueError(f'{name} overlap summary missing required fields')
    if set(e0['pairwise_overlap_counts']) != set(e1['pairwise_overlap_counts']):
        raise ValueError('Overlap pairwise count keys differ')
    def change(a, b):
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or not np.isfinite([a, b]).all():
            raise ValueError('Invalid overlap quantity')
        return dict(e0=a, e1a=b, absolute_change=b-a, relative_change=(b-a)/a if a else None)
    return dict(case_count=expected_cases, quantities={k: change(e0[k], e1[k]) for k in required},
                pairwise_overlap_counts={k: change(e0['pairwise_overlap_counts'][k], e1['pairwise_overlap_counts'][k])
                                         for k in sorted(e0['pairwise_overlap_counts'])},
                interpretation='Overlap changes are descriptive associations and hypotheses for later investigation; no causal inference.')


def _git(*args):
    try:
        return subprocess.check_output(['git', *args], stderr=subprocess.DEVNULL, timeout=5).decode().strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _json_ready(value):
    """Encode undefined summary statistics as JSON null, never as finite numbers."""
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def compare(e0_csv, e1a_csv, e0_overlap, e1a_overlap, output_dir,
            expected_cases=270, seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_RESAMPLES):
    paths = [Path(p).expanduser().resolve() for p in (e0_csv, e1a_csv, e0_overlap, e1a_overlap)]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f'Required frozen artifact missing: {path}')
    output = Path(output_dir).expanduser().resolve()
    if any(output == p or output in p.parents or p in output.parents for p in paths):
        raise ValueError('Output directory must be separate from frozen inputs')
    paired = validate_pair(paths[0], paths[1], expected_cases)
    overlap = compare_overlap(json.loads(paths[2].read_text()), json.loads(paths[3].read_text()), expected_cases)
    stats = metric_statistics(paired, seed, resamples)
    cases = case_metrics(paired)
    components = component_comparison(paired)
    transitions = reference_empty_transitions(paired)
    netc_stats = stats[(stats.region == 'NETC') & stats.metric.isin(
        ('global_bin_dsc', 'global_bin_hd95', 'global_bin_nsd', 'prec', 'rec'))]
    netc = dict(region='NETC', paired_metrics=netc_stats.astype(object).where(pd.notna(netc_stats), None).to_dict('records'),
        components=components[components.region == 'NETC'].to_dict('records'),
        reference_empty_prediction_positive_transitions=transitions[transitions.region == 'NETC'].iloc[0].to_dict())
    dirty = _git('status', '--porcelain')
    metadata = dict(analysis_timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    input_files={name: {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                                 for name, path in zip(('e0_official_metrics', 'e1a_official_metrics',
                                                        'e0_overlap_summary', 'e1a_overlap_summary'), paths)},
                    git_commit=_git('rev-parse', 'HEAD'), git_dirty=None if dirty is None else bool(dirty),
                    python_version=platform.python_version(), pandas_version=pd.__version__,
                    numpy_version=np.__version__, scipy_version=scipy.__version__, bootstrap_seed=seed,
                    bootstrap_resample_count=resamples, expected_case_count=expected_cases,
                    expected_regions=list(REGIONS), metric_order=list(METRICS), metric_directions=DIRECTIONS,
                    wilcoxon_parameters=WILCOXON_PARAMETERS, all_zero_wilcoxon_policy='raw p = 1.0',
                    multiple_comparison_correction='Holm step-down separately across six regions for each metric',
                    delta_definition='E1-A minus E0; no distance-metric reversal',
                    finite_pair_policy='Both observations finite; all nonfinite categories counted separately')
    summary = dict(case_count=expected_cases, paired_row_count=len(paired), regions=list(REGIONS),
                   metric_order=list(METRICS), artifacts=list(ARTIFACTS),
                   interpretation='Descriptive paired comparison only; no causal claims or p-value thresholds.')
    if output.is_symlink() or any((output / name).is_symlink() for name in ARTIFACTS):
        raise ValueError('Output paths must not be symlinks')
    output.mkdir(parents=True, exist_ok=True)
    for name, table in (('paired_metric_statistics.csv', stats), ('paired_case_metrics.csv', cases),
                        ('component_comparison.csv', components), ('reference_empty_fp_transitions.csv', transitions)):
        table.to_csv(output / name, index=False, lineterminator='\n')
    for name, data in (('netc_diagnostics.json', netc), ('overlap_comparison.json', overlap),
                       ('comparison_summary.json', summary), ('comparison_metadata.json', metadata)):
        (output / name).write_text(json.dumps(_json_ready(data), indent=2, allow_nan=False) + '\n')
    return output
