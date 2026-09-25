import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/render_d0c3b_final_results.py'
spec = importlib.util.spec_from_file_location('render_d0c3b_final_results', SCRIPT)
render_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(render_mod)


def write_csv(path, fields, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def make_source(tmp_path):
    source = tmp_path / 'source'; source.mkdir()
    aggs = []
    for region, evaluable, auc, aprc, lift, diff, positive in (
        ('WT', 270, .6141304701697484, .12899662157523856, .030395627911162543, .01092936098575592, .9333333333333333),
        ('TC', 200, .6191827746449456, .1, .009065179241381349, .010577797889709473, .77),
        ('ET', 198, .6193594018631803, .09, .008520194638620754, .010650858283042908, .7727272727272727)):
        aggs.append(dict(region=region, total_cases=270, evaluable_cases=evaluable,
            median_AUROC=auc, AUROC_ci95_low=auc-.01, AUROC_ci95_high=auc+.01,
            median_AUPRC=aprc, AUPRC_ci95_low=aprc-.01, AUPRC_ci95_high=aprc+.01,
            median_AUPRC_lift=lift, AUPRC_lift_ci95_low=lift-.002, AUPRC_lift_ci95_high=lift+.002,
            median_mean_difference=diff+.001, mean_difference_ci95_low=diff-.001,
            mean_difference_ci95_high=diff+.002, median_median_difference=diff,
            median_difference_ci95_low=diff-.001, median_difference_ci95_high=diff+.001,
            proportion_median_difference_gt_zero=positive))
        aggs[-1].update({'AUROC_vs_0.5_p_holm': 6.06e-41 if region=='WT' else .05,
            'AUPRC_lift_vs_zero_p_holm': 3.05e-12 if region=='WT' else .06,
            'median_difference_vs_zero_p_holm': 1.0e-250 if region=='WT' else .07})
    agg_fields = list(aggs[0])
    write_csv(source/'aggregate_summary.csv', agg_fields, aggs)
    (source/'aggregate_summary.json').write_text(json.dumps({'regions': aggs}), encoding='utf-8')
    case_fields = ['case_id','region','evaluable','AUROC','AUPRC','AUPRC_lift','median_difference']
    per_case=[]
    for region, n in (('WT',270),('TC',200),('ET',198)):
        for i in range(270):
            evaluable = i < n
            per_case.append(dict(case_id=f'case-{i:03}', region=region, evaluable=evaluable,
                AUROC=.4+(i%100)/200, AUPRC=.05+(i%20)/500,
                AUPRC_lift=(i%20-10)/1000, median_difference=(i%9-4)/1000))
    write_csv(source/'per_case_metrics.csv', case_fields, per_case)
    (source/'analysis_integrity.json').write_text(json.dumps(dict(case_count=270, case_region_rows=810,
        expected_case_region_rows=810, completed_case_set_match=True)), encoding='utf-8')
    (source/'analysis_provenance.json').write_text(json.dumps(dict(dataset_role='DEVELOPMENT_VALIDATION',
        independent_test=False, gt_opened=True, gt_usage='evaluation only', analysis_contract_sha256='contract-sha')),
        encoding='utf-8')
    names=['aggregate_summary.csv','aggregate_summary.json','per_case_metrics.csv','analysis_integrity.json','analysis_provenance.json']
    (source/'MANIFEST.sha256').write_text(''.join(f'{hashlib.sha256((source/n).read_bytes()).hexdigest()}  {n}\n' for n in names), encoding='utf-8')
    return source, aggs, per_case


def test_order_roles_values_ci_and_counts_are_source_driven():
    aggregate=[]
    cases=[]
    for region, n, value in [('WT',270,.6141304701697484),('TC',200,.6191827746449456),('ET',198,.6193594018631803)]:
        aggregate.append(dict(region=region,total_cases=270,evaluable_cases=n,
            median_AUROC=value,AUROC_ci95_low=value-.01,AUROC_ci95_high=value+.01,
            median_AUPRC=.1,AUPRC_ci95_low=.08,AUPRC_ci95_high=.12,
            median_AUPRC_lift=.02,AUPRC_lift_ci95_low=.01,AUPRC_lift_ci95_high=.03,
            median_median_difference=.012,median_difference_ci95_low=.01,median_difference_ci95_high=.014,
            **{'AUROC_vs_0.5_p_holm':.05,'AUPRC_lift_vs_zero_p_holm':.06,'median_difference_vs_zero_p_holm':.07}))
        cases.extend(dict(case_id=f'{region}-{i}',region=region,evaluable=i<n,median_difference=.1 if i%2==0 else -.1)
                      for i in range(270))
    columns, rows=render_mod.build_results_table(aggregate,cases)
    assert [r['Region'] for r in rows] == ['WT','TC','ET']
    assert rows[0]['Role']=='PRIMARY' and rows[1]['Role']=='SECONDARY'
    assert rows[0]['Evaluable cases']=='270 / 270'
    assert rows[1]['Evaluable cases']=='200 / 270'
    assert rows[0]['Median AUROC [95% CI]']=='0.614130 [0.604130, 0.624130]'
    assert rows[0]['Median AUPRC lift [95% CI]']=='0.020000 [0.010000, 0.030000]'
    assert rows[0]['Cases with positive DeltaZ difference']=='135 / 270'
    assert rows[0]['Holm p: AUROC > reference']=='0.050000'
    assert columns[0:3]==['Region','Role','Evaluable cases']


def test_qualitative_selection_is_deterministic_with_ties():
    rows=[]
    for case_id, auc in [('zeta',.2),('beta',.4),('alpha',.4),('mid',.6),('high',.8)]:
        rows.append(dict(case_id=case_id,region='WT',evaluable='True',AUROC=auc))
    first=render_mod.select_qualitative_cases(rows)
    second=render_mod.select_qualitative_cases(list(reversed(rows)))
    assert first==second
    assert [item['selection'] for item in first['cases']]==['25th_percentile','median','75th_percentile']
    assert first['cases'][1]['case_id']=='alpha'  # tie breaks lexicographically


def test_renderer_reads_manifest_emits_complete_output_manifest_and_preserves_source(tmp_path):
    source, aggs, per_case=make_source(tmp_path)
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    output=tmp_path/'report'
    table, selection=render_mod.render(source,output,dataset_root=tmp_path/'no-images',primary_run=tmp_path/'no-run')
    assert [r['Region'] for r in table]==['WT','TC','ET']
    assert selection['status']=='SKIPPED'
    table_md=(output/'results_table.md').read_text()
    results_md=(output/'results_summary.md').read_text()
    paper_md=(output/'paper_results_draft.md').read_text()
    for text in (table_md,results_md,paper_md):
        assert '6.060e-41' in text
        assert '3.050e-12' in text
        assert '1.000e-250' in text
        assert '0.000000' not in text
    assert not (output/'figure_04_qualitative_cases.png').exists()
    assert (output/'figure_01_auroc_summary.png').is_file() and (output/'figure_01_auroc_summary.pdf').is_file()
    entries={}
    for line in (output/'MANIFEST.sha256').read_text().splitlines():
        digest,name=line.split('  ',1); entries[name]=digest
    assert set(entries)=={p.name for p in output.iterdir() if p.name!='MANIFEST.sha256'}
    assert all(hashlib.sha256((output/name).read_bytes()).hexdigest()==digest for name,digest in entries.items())
    after={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    assert before==after
    provenance=json.loads((output/'reporting_provenance.json').read_text())
    assert provenance['source_sha256']['MANIFEST.sha256']==before['MANIFEST.sha256']
    assert provenance['source_sha256']['aggregate_summary.csv']==before['aggregate_summary.csv']


def test_source_manifest_mismatch_fails_before_reporting(tmp_path):
    source, _, _=make_source(tmp_path)
    with (source/'aggregate_summary.csv').open('a') as stream: stream.write('\n')
    with pytest.raises(ValueError,match='source hash mismatch'):
        render_mod.render(source,tmp_path/'report',dataset_root=tmp_path/'none',primary_run=tmp_path/'none')


def test_nonzero_tiny_pvalues_never_render_as_zero():
    for pvalue in (6.06e-41, 3.05e-12, 1.0e-250, 9.99e-4):
        rendered=render_mod.format_pvalue(pvalue)
        assert 'e-' in rendered
        assert float(rendered) == pytest.approx(pvalue, rel=1e-3)
    assert render_mod.format_pvalue(.001)=='0.001000'


def test_quantitative_summary_propagates_source_values_without_metric_recalculation(tmp_path):
    source, aggs, per_case=make_source(tmp_path)
    by_region={r['region']:r for r in aggs}
    main, professor, paper=render_mod.make_narratives(by_region, [], per_case)
    assert '0.614130 [0.604130, 0.624130]' in main
    assert '0.030396' in main
    assert '270/270' in main
    assert 'not an independent test set' in main
    assert 'complementary information' in professor
    assert '# Results' in paper and '# Discussion' in paper and '# Limitations' in paper
