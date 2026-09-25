#!/usr/bin/env python3
"""Render compact paper/professor reporting from frozen D0-C3b outputs only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DEFAULT = ROOT / 'outputs/scientific/d0c3b_pathology_association'
OUTPUT_DEFAULT = ROOT / 'outputs/reporting/d0c3b_final'
DATASET_DEFAULT = Path('/home/friend/data/training_data1_v2')
PRIMARY_RUN_DEFAULT = ROOT / 'outputs/scientific/d0c3_deltaz/primary_t495'
REGIONS = ('WT', 'TC', 'ET')
AGG_FILES = ('aggregate_summary.csv', 'aggregate_summary.json', 'per_case_metrics.csv',
             'analysis_integrity.json', 'analysis_provenance.json')
OUTPUT_FILES = ('results_table.csv', 'results_table.md', 'figure_01_auroc_summary.png',
    'figure_01_auroc_summary.pdf', 'figure_02_deltaz_difference.png', 'figure_02_deltaz_difference.pdf',
    'figure_03_auprc_lift.png', 'figure_03_auprc_lift.pdf', 'results_summary.md',
    'professor_summary.md', 'paper_results_draft.md', 'reporting_provenance.json',
    'qualitative_case_selection.json')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def verify_source_manifest(source):
    source = Path(source)
    manifest = source / 'MANIFEST.sha256'
    require(manifest.is_file(), 'D0-C3b source MANIFEST.sha256 is missing')
    entries = {}
    for line in manifest.read_text(encoding='utf-8').splitlines():
        digest, rel = line.split('  ', 1)
        require(rel not in entries, f'Duplicate D0-C3b manifest entry: {rel}')
        path = source / rel
        require(path.is_file() and not path.is_symlink(), f'Missing/unsafe D0-C3b source: {rel}')
        require(sha256_file(path) == digest, f'D0-C3b source hash mismatch: {rel}')
        entries[rel] = digest
    actual = {p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_file() and p.name != 'MANIFEST.sha256'}
    require(actual == set(entries), 'D0-C3b source inventory differs from its MANIFEST')
    require(set(AGG_FILES) <= set(entries), 'D0-C3b source manifest lacks required authoritative files')
    integrity = json.loads((source / 'analysis_integrity.json').read_text(encoding='utf-8'))
    require(integrity.get('case_count') == 270 and integrity.get('case_region_rows') == 810
            and integrity.get('expected_case_region_rows') == 810
            and integrity.get('completed_case_set_match') is True,
            'D0-C3b source integrity summary is invalid')
    provenance = json.loads((source / 'analysis_provenance.json').read_text(encoding='utf-8'))
    require(provenance.get('dataset_role') == 'DEVELOPMENT_VALIDATION'
            and provenance.get('independent_test') is False and provenance.get('gt_opened') is True
            and provenance.get('gt_usage') == 'evaluation only',
            'D0-C3b source provenance role/GT-use mismatch')
    entries['MANIFEST.sha256'] = sha256_file(manifest)
    return entries, integrity, provenance


def as_float(value):
    if value is None or value == '' or str(value).lower() in ('none', 'nan', 'null'):
        return None
    return float(value)


def format_num(value, digits=6):
    x = as_float(value)
    return 'NA' if x is None else f'{x:.{digits}f}'


def format_pvalue(value):
    x = as_float(value)
    if x is None:
        return 'NA'
    require(0.0 <= x <= 1.0, f'Invalid p-value: {x}')
    if 0.0 < x < 0.001:
        return f'{x:.3e}'
    return f'{x:.6f}'


def format_ci(row, metric, median_key):
    return f"{format_num(row[median_key])} [{format_num(row[metric+'_ci95_low'])}, {format_num(row[metric+'_ci95_high'])}]"


def order_and_validate(aggregate_rows, per_case_rows):
    by_region = {row['region']: row for row in aggregate_rows}
    require(set(by_region) == set(REGIONS), 'Aggregate source must contain exactly WT, TC, ET')
    require([row['region'] for row in aggregate_rows] == list(REGIONS), 'Aggregate region ordering must be WT/TC/ET')
    require(all(r['region'] in REGIONS for r in per_case_rows), 'Unexpected region in per-case source')
    for region in REGIONS:
        cases = [r for r in per_case_rows if r['region'] == region]
        require(len(cases) == 270 and len({r['case_id'] for r in cases}) == 270,
                f'Per-case source does not have 270 unique {region} cases')
        require(sum(_truth(r['evaluable']) for r in cases) == int(by_region[region]['evaluable_cases']),
                f'Evaluable case count differs for {region}')
    require(by_region['WT'].get('role', 'PRIMARY') in ('PRIMARY', 'primary'), 'WT must be primary endpoint')
    return by_region


def _truth(value):
    return value is True or str(value).strip().lower() in ('true', '1', 'yes')


def build_results_table(aggregate_rows, per_case_rows):
    by_region = order_and_validate(aggregate_rows, per_case_rows)
    titles = ['Region', 'Role', 'Evaluable cases', 'Median AUROC [95% CI]', 'Median AUPRC [95% CI]',
        'Median AUPRC lift [95% CI]', 'Median DeltaZ difference [95% CI]',
        'Cases with positive DeltaZ difference', 'Holm p: AUROC > reference',
        'Holm p: AUPRC lift', 'Holm p: DeltaZ difference']
    field_map = [('median_AUROC', 'AUROC'), ('median_AUPRC', 'AUPRC'),
                 ('median_AUPRC_lift', 'AUPRC_lift'), ('median_median_difference', 'median_difference')]
    rows = []
    for region in REGIONS:
        agg = by_region[region]
        cases = [r for r in per_case_rows if r['region'] == region and _truth(r['evaluable'])]
        positive = sum(as_float(r['median_difference']) > 0 for r in cases)
        row = dict(Region=region, Role='PRIMARY' if region == 'WT' else 'SECONDARY',
            **{'Evaluable cases': f"{int(agg['evaluable_cases'])} / {int(agg['total_cases'])}"})
        for median_key, metric in field_map:
            label = {'median_AUROC': 'Median AUROC [95% CI]', 'median_AUPRC': 'Median AUPRC [95% CI]',
                     'median_AUPRC_lift': 'Median AUPRC lift [95% CI]',
                     'median_median_difference': 'Median DeltaZ difference [95% CI]'}[median_key]
            row[label] = format_ci(agg, metric, median_key)
        row['Cases with positive DeltaZ difference'] = f'{positive} / {len(cases)}'
        row['Holm p: AUROC > reference'] = format_pvalue(agg['AUROC_vs_0.5_p_holm'])
        row['Holm p: AUPRC lift'] = format_pvalue(agg['AUPRC_lift_vs_zero_p_holm'])
        row['Holm p: DeltaZ difference'] = format_pvalue(agg['median_difference_vs_zero_p_holm'])
        rows.append(row)
    return titles, rows


def _save_plot(fig, output, stem):
    fig.savefig(output / f'{stem}.png', dpi=300, bbox_inches='tight')
    fig.savefig(output / f'{stem}.pdf', bbox_inches='tight')
    plt.close(fig)


def figure_auroc(aggregate, output):
    vals = [as_float(aggregate[r]['median_AUROC']) for r in REGIONS]
    lows = [as_float(aggregate[r]['AUROC_ci95_low']) for r in REGIONS]
    highs = [as_float(aggregate[r]['AUROC_ci95_high']) for r in REGIONS]
    y = np.arange(len(REGIONS))
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    colors = ['#245b8f', '#5f7f99', '#5f7f99']
    for i, (value, low, high) in enumerate(zip(vals, lows, highs)):
        ax.errorbar(value, y[i], xerr=[[value-low], [high-value]], fmt='o', color=colors[i],
                    capsize=4, markersize=7, linewidth=1.8)
        ax.text(high + .012, y[i], f'{value:.3f} [{low:.3f}, {high:.3f}]', va='center', fontsize=9)
    ax.axvline(.5, color='#555555', linestyle='--', linewidth=1.2, label='Reference AUROC = 0.5')
    ax.set_xlim(0, 1)  # full scale avoids a misleading truncated axis
    ax.set_yticks(y, ['WT (primary)', 'TC (secondary)', 'ET (secondary)'])
    ax.invert_yaxis(); ax.set_xlabel('Median case-level AUROC (95% bootstrap CI)')
    ax.set_title('Tumor-region association of diffusion-derived deviation')
    ax.grid(axis='x', alpha=.2); ax.legend(frameon=False, loc='lower right')
    fig.tight_layout(); _save_plot(fig, output, 'figure_01_auroc_summary')


def _plot_distribution(per_case_rows, field, title, ylabel, output, stem):
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    data = [[as_float(r[field]) for r in per_case_rows if r['region'] == region and _truth(r['evaluable'])
             and as_float(r[field]) is not None] for region in REGIONS]
    ax.boxplot(data, positions=np.arange(1, 4), widths=.48, showfliers=True,
               medianprops={'color': '#152d40', 'linewidth': 2},
               boxprops={'color': '#456b86'}, whiskerprops={'color': '#456b86'},
               capprops={'color': '#456b86'}, flierprops={'marker': 'o', 'markersize': 3, 'alpha': .65})
    rng = np.random.default_rng(20260923)
    for i, values in enumerate(data, start=1):
        jitter = rng.uniform(-.13, .13, size=len(values))
        ax.scatter(i+jitter, values, s=10, alpha=.38, color='#2675a8', edgecolors='none', zorder=2)
    ax.axhline(0, color='#555555', linestyle='--', linewidth=1.1)
    ax.set_xticks([1, 2, 3], ['WT (primary)', 'TC (secondary)', 'ET (secondary)'])
    ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(axis='y', alpha=.2)
    fig.tight_layout(); _save_plot(fig, output, stem)


def select_qualitative_cases(per_case_rows):
    wt = [r for r in per_case_rows if r['region'] == 'WT' and _truth(r['evaluable'])
          and as_float(r['AUROC']) is not None]
    require(wt, 'No evaluable WT cases available for deterministic qualitative selection')
    ordered_values = np.asarray([as_float(r['AUROC']) for r in wt], dtype=np.float64)
    selected = []
    for quantile, label in ((.25, '25th_percentile'), (.50, 'median'), (.75, '75th_percentile')):
        target = float(np.quantile(ordered_values, quantile))
        best = min(wt, key=lambda r: (abs(as_float(r['AUROC'])-target), r['case_id']))
        selected.append(dict(selection=label, target_quantile=quantile, target_auroc=target,
                             case_id=best['case_id'], case_auroc=as_float(best['AUROC'])))
    return dict(rule='Among evaluable WT cases, choose the case whose per-case AUROC is nearest the empirical AUROC quantile; ties break by ascending case_id.',
                quantile_method='NumPy linear empirical quantile',
                cases=selected)


def _render_qualitative(selection, primary_run, dataset_root, output):
    if not dataset_root or not Path(dataset_root).is_dir() or not Path(primary_run).is_dir():
        return dict(status='SKIPPED', reason='Primary case tensors or BraTS source images unavailable; no qualitative figure fabricated.')
    records = []
    for choice in selection['cases']:
        cid = choice['case_id']; case_dir = Path(primary_run)/'cases'/cid
        image_path = Path(dataset_root)/cid/f'{cid}-t2f.nii.gz'
        seg_path = Path(dataset_root)/cid/f'{cid}-seg.nii.gz'
        tensor_path = case_dir/'case_deltaz.pt'
        if not all(p.is_file() for p in (image_path, seg_path, tensor_path)):
            return dict(status='SKIPPED', reason=f'Qualitative source image/GT/tensor missing for {cid}; no figure fabricated.')
        image = np.asanyarray(nib.load(str(image_path)).dataobj)
        gt = np.asanyarray(nib.load(str(seg_path)).dataobj)
        payload = torch.load(tensor_path, map_location='cpu', weights_only=True)
        delta = payload['DeltaZ_global'].numpy()[0]
        valid = payload['valid_support_global'].numpy().astype(bool)[0]
        original = tuple(payload['original_shape'].tolist())
        padded = tuple(payload['padded_shape'].tolist())
        require(image.shape == gt.shape == original, f'Qualitative image/GT shape mismatch for {cid}')
        wt = np.isin(gt, [1, 2, 3])
        slice_index = int(np.argmax(wt.sum(axis=(0, 1))))
        delta_vox = np.repeat(np.repeat(np.repeat(delta, 8, axis=0), 8, axis=1), 8, axis=2)
        valid_vox = np.repeat(np.repeat(np.repeat(valid, 8, axis=0), 8, axis=1), 8, axis=2)
        delta_vox = delta_vox[:original[0], :original[1], :original[2]]
        valid_vox = valid_vox[:original[0], :original[1], :original[2]]
        records.append((choice, image[:, :, slice_index], wt[:, :, slice_index],
                        np.ma.masked_where(~valid_vox[:, :, slice_index], delta_vox[:, :, slice_index]), slice_index))
    fig, axes = plt.subplots(3, 4, figsize=(13, 9))
    titles = ('T2-FLAIR', 'T2-FLAIR + WT contour', 'DeltaZ', 'DeltaZ + WT contour')
    for row, (choice, flair, wt_slice, dz, slice_index) in enumerate(records):
        axes[row, 0].imshow(np.rot90(flair), cmap='gray'); axes[row, 0].set_title(f"{choice['selection']}: {titles[0]}\n{choice['case_id']}")
        axes[row, 1].imshow(np.rot90(flair), cmap='gray'); axes[row, 1].contour(np.rot90(wt_slice), levels=[.5], colors=['lime'], linewidths=.8); axes[row, 1].set_title(titles[1])
        im = axes[row, 2].imshow(np.rot90(dz), cmap='magma'); axes[row, 2].set_title(titles[2]); fig.colorbar(im, ax=axes[row, 2], fraction=.045)
        axes[row, 3].imshow(np.rot90(dz), cmap='magma'); axes[row, 3].contour(np.rot90(wt_slice), levels=[.5], colors=['cyan'], linewidths=.8); axes[row, 3].set_title(titles[3])
        for ax in axes[row]: ax.set_axis_off()
    fig.suptitle('Representative WT cases selected by per-case AUROC quantiles (visualization only)')
    fig.tight_layout(); _save_plot(fig, output, 'figure_04_qualitative_cases')
    return dict(status='GENERATED', axial_axis=2, slice_rule='GT-only axial slice with maximum WT voxel area; first maximum on ties.',
                delta_z_mapping='Each latent cell repeated exactly 8x along each axis, then high-end cropped to original shape; invalid support masked.',
                cases=[dict(selection=r[0]['selection'], case_id=r[0]['case_id'], case_auroc=r[0]['case_auroc'], axial_slice_index=r[4]) for r in records])


def _write_csv(path, rows, fields):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def _write_markdown_table(fields, rows):
    lines = ['| ' + ' | '.join(fields) + ' |', '| ' + ' | '.join(['---']*len(fields)) + ' |']
    for row in rows: lines.append('| ' + ' | '.join(str(row[f]) for f in fields) + ' |')
    return '\n'.join(lines)


def _p_statement(agg, key):
    return format_pvalue(agg[key])


def make_narratives(aggregate, table_rows, per_case_rows):
    wt, tc, et = (aggregate[r] for r in REGIONS)
    wt_cases = [r for r in per_case_rows if r['region'] == 'WT' and _truth(r['evaluable'])]
    positive = sum(as_float(r['median_difference']) > 0 for r in wt_cases)
    pct = 100 * positive / len(wt_cases) if wt_cases else 0
    main = (
        '### Main result\n\n'
        f"WT was the predeclared primary endpoint. All {int(wt['evaluable_cases'])}/{int(wt['total_cases'])} WT cases were evaluable. "
        f"Median case-level AUROC was {format_ci(wt, 'AUROC', 'median_AUROC')}; median AUPRC was {format_ci(wt, 'AUPRC', 'median_AUPRC')}; "
        f"median AUPRC lift was {format_ci(wt, 'AUPRC_lift', 'median_AUPRC_lift')}; and median case-level DeltaZ difference was "
        f"{format_ci(wt, 'median_difference', 'median_median_difference')}. {positive}/{len(wt_cases)} ({pct:.1f}%) evaluable WT cases had a positive median difference. "
        f"Holm-adjusted p-values were { _p_statement(wt, 'AUROC_vs_0.5_p_holm')} for AUROC versus 0.5, "
        f"{_p_statement(wt, 'AUPRC_lift_vs_zero_p_holm')} for AUPRC lift versus zero, and "
        f"{_p_statement(wt, 'median_difference_vs_zero_p_holm')} for median DeltaZ difference versus zero.\n\n"
        '### Secondary results\n\n'
        f"TC was evaluable in {int(tc['evaluable_cases'])}/{int(tc['total_cases'])} cases; its median AUROC was {format_num(tc['median_AUROC'])}, "
        f"median AUPRC lift {format_num(tc['median_AUPRC_lift'])}, and median DeltaZ difference {format_num(tc['median_median_difference'])}. "
        f"ET was evaluable in {int(et['evaluable_cases'])}/{int(et['total_cases'])} cases; its median AUROC was {format_num(et['median_AUROC'])}, "
        f"median AUPRC lift {format_num(et['median_AUPRC_lift'])}, and median DeltaZ difference {format_num(et['median_median_difference'])}. "
        'TC and ET had fewer evaluable cases than WT.\n\n'
        '### Interpretation\n\n'
        'The results indicate a modest but consistent spatial association between the diffusion-derived deviation feature and tumor-containing regions. '
        'This association analysis is not evidence of diagnostic performance.\n\n'
        '### Limitations\n\n'
        '- The cohort is development-validation, not an independent test set.\n'
        '- This is an association analysis, not a diagnostic evaluation; AUROC was moderate.\n'
        '- Za is an anatomy/background representation and is not claimed to be healthy or pathology-free.\n'
        '- DeltaZ has not been shown to improve a supervised segmentation model.\n'
        '- Sensitivity timesteps 327 and 663 are deferred experiments.\n')
    professor = (
        '# D0-C3b professor summary\n\n'
        '## Research question\n\nCan a diffusion-derived deviation feature be spatially associated with tumor regions in the BraTS 2024 development-validation cohort?\n\n'
        '## Method\n\nA frozen primary timestep-495 DeltaZ map was compared with WT, TC and ET labels after exact 8×8×8 block mapping. Only MRI-supported cells were used; positive cells had any region voxel, while the comparator consisted of blocks containing only label 0. The patient, not the latent cell, was the unit for aggregate statistics.\n\n'
        '## Dataset/evaluation\n\nThe analysis used 270 development-validation cases. WT was primary; TC and ET were secondary. Cases without both positive and clean-background supported cells were retained as non-evaluable.\n\n'
        '## Key result\n\n'
        f"WT was evaluable in 270/270 cases, with median AUROC {format_num(wt['median_AUROC'])} (95% CI {format_num(wt['AUROC_ci95_low'])}–{format_num(wt['AUROC_ci95_high'])}); median AUPRC lift {format_num(wt['median_AUPRC_lift'])}; and median DeltaZ difference {format_num(wt['median_median_difference'])}. "
        f"A positive median difference occurred in {positive}/{len(wt_cases)} evaluable WT cases ({pct:.1f}%). These results suggest a modest spatial association, not diagnostic performance.\n\n"
        '## Why it matters\n\nThe result motivates testing whether the diffusion-derived feature contributes information that is complementary to the supervised model. Association alone does not establish clinical or segmentation benefit.\n\n'
        '## Next research step\n\nTest whether DeltaZ provides complementary information when fused into a supervised segmentation model, with a separately specified evaluation design and independent testing.\n')
    paper = (
        '# Results\n\n'
        f"WT was the predeclared primary endpoint and was evaluable in {int(wt['evaluable_cases'])}/{int(wt['total_cases'])} cases. The median case-level AUROC was {format_ci(wt, 'AUROC', 'median_AUROC')}, median AUPRC was {format_ci(wt, 'AUPRC', 'median_AUPRC')}, median AUPRC lift was {format_ci(wt, 'AUPRC_lift', 'median_AUPRC_lift')}, and median case-level DeltaZ difference was {format_ci(wt, 'median_difference', 'median_median_difference')}. {positive}/{len(wt_cases)} evaluable cases ({pct:.1f}%) had positive median DeltaZ difference. Holm-adjusted p-values were { _p_statement(wt, 'AUROC_vs_0.5_p_holm')}, {_p_statement(wt, 'AUPRC_lift_vs_zero_p_holm')}, and {_p_statement(wt, 'median_difference_vs_zero_p_holm')} for the three predeclared test families.\n\n"
        f"TC was evaluable in {int(tc['evaluable_cases'])}/{int(tc['total_cases'])} cases and ET in {int(et['evaluable_cases'])}/{int(et['total_cases'])}. Their median AUROCs were {format_num(tc['median_AUROC'])} and {format_num(et['median_AUROC'])}; median AUPRC lifts were {format_num(tc['median_AUPRC_lift'])} and {format_num(et['median_AUPRC_lift'])}; median DeltaZ differences were {format_num(tc['median_median_difference'])} and {format_num(et['median_median_difference'])}, respectively.\n\n"
        '# Discussion\n\nThe results indicate a modest but consistent spatial association between the diffusion-derived deviation feature and tumor-containing regions in the development-validation cohort. The association is descriptive and does not establish diagnostic or clinical performance. The result motivates evaluation of whether DeltaZ adds complementary information when fused into a supervised segmentation model.\n\n'
        '# Limitations\n\nThe cohort is development-validation and is not an independent test set. The AUROC values indicate moderate association. Za is described as an anatomy/background representation, not as healthy or pathology-free. DeltaZ has not yet been shown to improve a supervised segmentation model. Sensitivity timesteps 327 and 663 remain deferred experiments.\n')
    return main, professor, paper


def render(source=SOURCE_DEFAULT, output=OUTPUT_DEFAULT, *, dataset_root=DATASET_DEFAULT, primary_run=PRIMARY_RUN_DEFAULT):
    source, output = Path(source), Path(output)
    require(not output.exists(), f'Refusing to overwrite reporting output: {output}')
    input_hashes, integrity, analysis_provenance = verify_source_manifest(source)
    aggregate_rows = read_csv(source / 'aggregate_summary.csv')
    per_case_rows = read_csv(source / 'per_case_metrics.csv')
    aggregate = order_and_validate(aggregate_rows, per_case_rows)
    titles, table_rows = build_results_table(aggregate_rows, per_case_rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'.{output.name}.', dir=output.parent) as tmp:
        stage = Path(tmp) / output.name; stage.mkdir()
        _write_csv(stage / 'results_table.csv', table_rows, titles)
        (stage / 'results_table.md').write_text(_write_markdown_table(titles, table_rows)+'\n', encoding='utf-8')
        figure_auroc(aggregate, stage)
        _plot_distribution(per_case_rows, 'median_difference', 'Per-case median DeltaZ difference',
                           'Positive minus clean-background median DeltaZ', stage, 'figure_02_deltaz_difference')
        _plot_distribution(per_case_rows, 'AUPRC_lift', 'Per-case AUPRC lift over prevalence baseline',
                           'AUPRC minus comparison-set prevalence', stage, 'figure_03_auprc_lift')
        selection = select_qualitative_cases(per_case_rows)
        qualitative = _render_qualitative(selection, primary_run, dataset_root, stage)
        selection.update(qualitative)
        (stage / 'qualitative_case_selection.json').write_text(json.dumps(selection, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')
        results, professor, paper = make_narratives(aggregate, table_rows, per_case_rows)
        (stage / 'results_summary.md').write_text(results, encoding='utf-8')
        (stage / 'professor_summary.md').write_text(professor, encoding='utf-8')
        (stage / 'paper_results_draft.md').write_text(paper, encoding='utf-8')
        def git(*args): return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
        prov = dict(source_directory=str(source.resolve()), source_sha256=input_hashes,
            analysis_contract_sha256=analysis_provenance.get('analysis_contract_sha256'),
            source_analysis_integrity=integrity,
            runtime_git_commit=git('rev-parse', 'HEAD'), runtime_git_branch=git('rev-parse', '--abbrev-ref', 'HEAD'),
            git_status=git('status', '--short').splitlines(), python_version=platform.python_version(),
            matplotlib_version=matplotlib.__version__, reporting_script='scripts/render_d0c3b_final_results.py',
            reporting_script_sha256=sha256_file(Path(__file__)), qualitative_status=qualitative['status'])
        (stage / 'reporting_provenance.json').write_text(json.dumps(prov, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')
        manifest_entries = {p.name: sha256_file(p) for p in sorted(stage.iterdir()) if p.is_file()}
        (stage / 'MANIFEST.sha256').write_text(''.join(f'{digest}  {name}\n' for name, digest in sorted(manifest_entries.items())), encoding='utf-8')
        os.replace(stage, output)
    return table_rows, selection


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=SOURCE_DEFAULT)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument('--dataset-root', type=Path, default=DATASET_DEFAULT)
    parser.add_argument('--primary-run', type=Path, default=PRIMARY_RUN_DEFAULT)
    args = parser.parse_args(argv)
    render(args.source_dir, args.output_dir, dataset_root=args.dataset_root, primary_run=args.primary_run)


if __name__ == '__main__':
    main()
