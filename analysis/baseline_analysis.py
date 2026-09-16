"""Deterministic post-hoc analysis of frozen baseline reporting artifacts.

Never loads checkpoints. Missing observations remain missing; statistics use
population standard deviation and NumPy's linear percentile interpolation.
"""
import csv
import json
import math
from pathlib import Path
import numpy as np

REGIONS = ('NETC', 'SNFH', 'ET', 'RC', 'WT', 'TC')
ZERO_TOLERANCE = 1e-8


def number(value):
    if value is None or str(value).strip().lower() in ('', 'none', 'null', 'nan'):
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'Non-finite numeric observation: {value}')
    return result


def mean_present(row):
    values = []
    for region in REGIONS:
        if (number(row.get(f'{region}_positive_cases')) or 0) > 0:
            value = number(row.get(f'{region}_case_dice'))
            if value is None:
                return None  # Do not invent a selection score from incomplete data.
            values.append(value)
    return float(np.mean(values)) if values else None


def netc_analysis(history, selected_epoch, tolerance=ZERO_TOLERANCE):
    observed = [(int(r['epoch']), number(r.get('NETC_case_dice')), r) for r in history
                if (number(r.get('NETC_positive_cases')) or 0) > 0]
    observed = [item for item in observed if item[1] is not None]
    best = max(observed, key=lambda item: item[1]) if observed else None
    first = next((epoch for epoch, value, _ in observed if abs(value) <= tolerance), None)
    streak = 0
    if first is not None:
        by_epoch = {epoch: value for epoch, value, _ in observed}
        while first + streak in by_epoch and abs(by_epoch[first + streak]) <= tolerance:
            streak += 1
    # Collapse requires a previously positive measured value; initial zeros alone
    # are not evidence of loss of a learned capability.
    collapse = next((epoch for epoch, value, _ in observed if abs(value) <= tolerance
                     and any(e < epoch and v > tolerance for e, v, _ in observed)), None)
    selected = next((r for r in history if int(r['epoch']) == selected_epoch), {})
    return dict(best_NETC_epoch=best[0] if best else None,
                best_NETC_case_dice=best[1] if best else None,
                best_NETC_dataset_dice=number(best[2].get('NETC_dataset_dice')) if best else None,
                first_zero_NETC_epoch=first,
                consecutive_zero_epochs_after_first_zero=streak,
                NETC_case_dice_at_selected_best_epoch=number(selected.get('NETC_case_dice')),
                collapse_epoch=collapse, zero_tolerance=tolerance,
                zero_streak_includes_first_zero=True)


def write_csv(path, rows, fields):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def analyze(experiment_dir):
    root = Path(experiment_dir)
    required = ['history.csv', 'summary.json', 'evaluation/aggregate_metrics.json', 'evaluation/per_case_metrics.csv']
    missing = [str(root / name) for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError('Missing required experiment artifacts: ' + ', '.join(missing))
    with (root / 'history.csv').open(newline='') as stream:
        reader = csv.DictReader(stream)
        original_fields = reader.fieldnames
        history = list(reader)
    if not history:
        raise ValueError('history.csv contains no epoch observations.')
    epochs = [int(r['epoch']) for r in history]
    if epochs != sorted(set(epochs)):
        raise ValueError('History epochs must be unique and ascending.')
    for row in history:
        row['selection_metric'] = mean_present(row)
    eligible = [r for r in history if r['selection_metric'] is not None]
    best = max(eligible, key=lambda r: r['selection_metric']) if eligible else None
    best_epoch = int(best['epoch']) if best else None
    losses = [r for r in history if number(r.get('validation_loss')) is not None]
    minimum = min(losses, key=lambda r: number(r['validation_loss'])) if losses else None
    saved = json.loads((root / 'summary.json').read_text())
    aggregate = json.loads((root / 'evaluation/aggregate_metrics.json').read_text())
    with (root / 'evaluation/per_case_metrics.csv').open(newline='') as stream:
        cases = list(csv.DictReader(stream))
    if not cases or len({r['case_id'] for r in cases}) != len(cases):
        raise ValueError('Per-case metrics must contain nonempty unique case IDs.')
    issues = []
    if saved.get('best_epoch') is not None and saved['best_epoch'] != best_epoch:
        issues.append(f"Saved best epoch {saved['best_epoch']} differs from recomputed {best_epoch}.")
    score = best['selection_metric'] if best else None
    if number(saved.get('best_validation_metric')) is not None and (score is None or not math.isclose(saved['best_validation_metric'], score, abs_tol=1e-7)):
        issues.append('Saved best validation metric differs from recomputed selection metric.')
    if aggregate.get('case_count', len(cases)) != len(cases):
        issues.append('Aggregate case count differs from per-case CSV count.')
    if len(cases) != 270:
        issues.append(f'Observed {len(cases)} validation cases; official expected count is 270.')
    statistics, difficult, distributions = [], [], {}
    for case in cases:
        values, row = [], {'case_id': case['case_id']}
        present_count = 0
        for region in REGIONS:
            present = number(case[f'{region}_target_present'])
            if present not in (0, 1):
                raise ValueError(f"Invalid presence flag for {case['case_id']} {region}")
            value = number(case[f'{region}_dice'])
            if not present:
                if value is not None:
                    issues.append(f"Absent target has recorded Dice: {case['case_id']} {region}; excluded.")
                value = None
            else:
                present_count += 1
                if value is None:
                    raise ValueError(f"Missing target-positive Dice: {case['case_id']} {region}")
                values.append(value)
            row[f'{region}_dice'] = value
        row.update(mean_present_region_dice=float(np.mean(values)) if values else None,
                   number_of_present_regions=present_count)
        difficult.append(row)
    for region in REGIONS:
        values = [r[f'{region}_dice'] for r in difficult if r[f'{region}_dice'] is not None]
        distributions[region] = values
        stats = dict(region=region, count=len(values))
        stats.update(dict(zip(('mean', 'median', 'standard_deviation', '25th_percentile', '75th_percentile', 'minimum', 'maximum'),
                             [float(np.mean(values)), float(np.median(values)), float(np.std(values)),
                              float(np.percentile(values,25)), float(np.percentile(values,75)), min(values), max(values)] if values else [None]*7)))
        statistics.append(stats)
        reported = aggregate['regions'][region]
        if reported['positive_case_count'] != len(values):
            issues.append(f'{region}: aggregate positive count differs from per-case count.')
        reported_mean = number(reported['case_dice'])
        if (reported_mean is None) != (stats['mean'] is None) or (reported_mean is not None and stats['mean'] is not None and not math.isclose(reported_mean, stats['mean'], abs_tol=1e-6)):
            issues.append(f'{region}: aggregate case Dice differs from per-case mean.')
    difficult.sort(key=lambda r: (r['mean_present_region_dice'] is None, r['mean_present_region_dice'] or 0, r['case_id']))
    netc = netc_analysis(history, best_epoch)
    output = root / 'analysis'
    figures = output / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'training_history.csv', history, [*original_fields, 'selection_metric'])
    write_json(output / 'netc_analysis.json', netc)
    table = [dict(region=r, **{key: aggregate['regions'][r][key] for key in ('positive_case_count', 'case_dice', 'dataset_dice')}) for r in REGIONS]
    write_csv(output / 'baseline_metrics.csv', table, ['region', 'positive_case_count', 'case_dice', 'dataset_dice'])
    fmt = lambda v: 'N/A' if number(v) is None else f'{float(v):.4f}'
    lines = ['| Region | Positive cases | Case Dice | Dataset Dice |', '|---|---:|---:|---:|']
    lines += [f"| {r['region']} | {r['positive_case_count']} | {fmt(r['case_dice'])} | {fmt(r['dataset_dice'])} |" for r in table]
    lines += ['', 'Overall mean validation selection metric: ' + fmt(aggregate.get('mean_validation_metric'))]
    (output / 'baseline_metrics.md').write_text('\n'.join(lines) + '\n')
    write_csv(output / 'per_region_statistics.csv', statistics, list(statistics[0]))
    write_csv(output / 'difficult_cases.csv', difficult, ['case_id','mean_present_region_dice','number_of_present_regions', *[f'{r}_dice' for r in REGIONS]])
    plot_figures(figures, history, best_epoch, saved.get('best_epoch', best_epoch), netc, distributions)
    summary = dict(experiment_name=root.name, case_count=len(cases), best_epoch=best_epoch,
                   best_selection_metric=score, minimum_validation_loss_epoch=int(minimum['epoch']) if minimum else None,
                   minimum_validation_loss=number(minimum['validation_loss']) if minimum else None,
                   regions=list(REGIONS), NETC_collapse_detected=netc['collapse_epoch'] is not None,
                   NETC_first_zero_epoch=netc['first_zero_NETC_epoch'], NETC_best_epoch=netc['best_NETC_epoch'],
                   NETC_best_case_dice=netc['best_NETC_case_dice'], inconsistencies=issues,
                   statistics_convention='population std (ddof=0); linear percentiles; absent targets excluded',
                   checkpoint_metadata_checked=False)
    summary['artifact_paths'] = sorted([str(p.relative_to(root)) for p in output.rglob('*') if p.is_file()] + ['analysis/analysis_summary.json'])
    summary['artifact_paths'] = sorted(set(summary['artifact_paths']))
    write_json(output / 'analysis_summary.json', summary)
    return summary


def plot_figures(folder, history, best_epoch, official_epoch, netc, distributions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    epochs = [int(r['epoch']) for r in history]
    def curve(key):
        return [number(r.get(key)) if number(r.get(key)) is not None else np.nan for r in history]
    with plt.rc_context({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42}):
        def save(fig, name):
            fig.tight_layout()
            fig.savefig(folder / f'{name}.png', dpi=300)
            fig.savefig(folder / f'{name}.pdf', metadata={'CreationDate': None, 'ModDate': None})
            plt.close(fig)
        for name, title, keys, ylabel in (
            ('loss_curve', 'Baseline training and validation loss', [('training_loss','Training'),('validation_loss','Validation')], 'Dice + BCE loss'),
            ('region_dice_curves', 'Presence-aware validation Dice', [(f'{r}_case_dice',r) for r in REGIONS], 'Target-positive case Dice'),
            ('selection_metric_curve', 'Checkpoint selection metric', [('selection_metric','Mean eligible-region case Dice')], 'Selection metric'),
            ('netc_dice_curve', 'NETC validation behavior', [('NETC_case_dice','Case Dice'),('NETC_dataset_dice','Dataset Dice')], 'Dice'),
        ):
            fig, ax = plt.subplots(figsize=(8,4.5))
            for key, label in keys:
                ax.plot(epochs, curve(key), label=label, linewidth=1.6, marker='.', markersize=4)
            markers = [(best_epoch, 'Selected best epoch', 'black')]
            if name == 'netc_dice_curve':
                markers = [(official_epoch, 'Saved selected epoch', 'black'), (netc['best_NETC_epoch'], 'Best NETC epoch', 'green'), (netc['first_zero_NETC_epoch'], 'First NETC zero', 'red')]
            for epoch, label, color in markers:
                if epoch is not None: ax.axvline(epoch, linestyle='--', linewidth=1, color=color, label=f'{label}: {epoch}')
            ax.set(xlabel='Epoch', ylabel=ylabel, title=title)
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
            save(fig,name)
        fig, ax = plt.subplots(figsize=(8,4.5))
        ax.boxplot([distributions[r] or [np.nan] for r in REGIONS], tick_labels=REGIONS)
        ax.set(ylabel='Target-positive case Dice', title='Validation Dice distributions', ylim=(-.03,1.03))
        ax.grid(axis='y',alpha=.2)
        save(fig,'case_dice_boxplot')
