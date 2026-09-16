"""Official installed-package adapter contract; no real images or GPU required."""
from types import SimpleNamespace
import pytest
from evaluation import brats_official as adapter


@pytest.fixture
def external(monkeypatch, tmp_path):
    config = tmp_path / 'gli.yaml'
    config.write_text('expected_input: !InputType SEMANTIC\nsegmentation_class_groups: !SegmentationClassGroups\n  groups:\n' + ''.join(
        f'    {name}: !LabelGroup {{value_labels: {values}}}\n' for name, values in
        dict(netc=[1],snfh=[2],et=[3],rc=[4],wt=[1,2,3],tc=[1,3]).items()))
    calls = []
    package = SimpleNamespace(config_path=lambda name: calls.append(name) or config,
                              evaluate_single_exam=lambda **kwargs: {}, __file__=str(tmp_path/'package.py'))
    panoptica = SimpleNamespace(Panoptica_Evaluator=SimpleNamespace(load_from_config=lambda path: calls.append(path) or object()))
    monkeypatch.setattr(adapter.importlib, 'import_module', lambda name: package if name == 'brats_evaluation' else panoptica)
    monkeypatch.setattr(adapter, 'version', lambda name: '0.0.10' if name == 'BraTS-evaluation' else '2.1.7')
    monkeypatch.setattr(adapter, 'source_git_metadata', lambda package: dict(git_revision='abc123', git_dirty_status=False, git_remote=adapter.REPOSITORY, git_metadata_reason='mock source'))
    return package, panoptica, config, calls


def test_metadata_and_config(external):
    evaluator = adapter.OfficialEvaluator()
    package, _, config, calls = external
    assert calls == ['gli', str(config)]
    metadata = evaluator.metadata
    assert metadata['evaluator_configuration'] == 'gli'
    assert metadata['evaluator_package'] == 'BraTS-evaluation'
    assert metadata['package_version'] == metadata['evaluator_version'] == '0.0.10'
    assert metadata['panoptica_version'] == '2.1.7'
    assert metadata['source_repository'] == 'https://github.com/BraTS/BraTS_evaluation'
    assert metadata['evaluator_git_revision'] == 'abc123'
    assert not metadata['is_official_hidden_challenge_scoring']
    assert 'numpy' in metadata['dependency_versions']
    assert evaluator.audit['audit_passed']
    assert evaluator.audit['evaluator_configuration_sha256'] == metadata['bundled_config_sha256']
    assert evaluator.audit['configuration_semantics']['label_groups']['tc'] == [1,3]


def test_missing_gli(external):
    external[2].unlink()
    with pytest.raises(RuntimeError, match='gli configuration unavailable'):
        adapter.OfficialEvaluator()


def test_initialization_failure(external):
    def fail(path): raise ValueError('bad config')
    external[1].Panoptica_Evaluator.load_from_config = fail
    with pytest.raises(RuntimeError, match='initialization failed'):
        adapter.OfficialEvaluator()


@pytest.mark.parametrize('result', [{}, {'subject_name':'case','error':'failed'},
                                    {'subject_name':'wrong'}, {'subject_name':'case', 'netc':{}}])
def test_reject_errors_and_incompatible_schema(external, tmp_path, result):
    external[0].evaluate_single_exam = lambda **kwargs: result
    evaluator = adapter.OfficialEvaluator()
    with pytest.raises(RuntimeError):
        evaluator.evaluate(tmp_path/'case.nii.gz',tmp_path/'ref.nii.gz',tmp_path)
    assert evaluator.raw_results == []


def test_official_api_arguments_and_no_score_substitution(external, tmp_path):
    calls = []
    metrics = dict(tp=1,fp=0,fn=0,sq_dsc=1.,sq_hd95=float('inf'),global_bin_dsc=1.,global_bin_hd95=0.,sq_dsc_std=float('nan'))
    result = {'subject_name':'case', **{r.lower(): dict(metrics) for r in adapter.REGIONS}}
    def evaluate(**kwargs):
        calls.append(kwargs)
        return result
    external[0].evaluate_single_exam = evaluate
    evaluator = adapter.OfficialEvaluator()
    rows = evaluator.evaluate(tmp_path/'case.nii.gz',tmp_path/'ref.nii.gz',tmp_path)
    assert calls[0]['subject_identifier'] == 'case'
    assert calls[0]['evaluator'] is evaluator.evaluator
    assert calls[0]['prediction_filepath'] == str(tmp_path/'case.nii.gz')
    assert rows[0]['sq_hd95'] == 'Infinity'
    assert rows[0]['sq_dsc_std'] == 'NaN'
    assert rows[0]['sq_dsc'] == 1.
    assert len(rows) == 6 and len(evaluator.raw_results) == 1


def test_wrong_gli_label_semantics(external):
    path = external[2]
    path.write_text(path.read_text().replace('value_labels: [4]', 'value_labels: [3]'))
    with pytest.raises(RuntimeError, match='memberships'):
        adapter.OfficialEvaluator()


def test_installed_official_package_on_synthetic_nifti(tmp_path):
    pytest.importorskip('brats_evaluation')
    pytest.importorskip('panoptica')
    import nibabel as nib
    import numpy as np
    evaluator = adapter.OfficialEvaluator()
    data = np.zeros((16,16,16),np.uint8)
    for index in range(4):
        data[2+index*3:4+index*3,2:5,2:5] = index+1
    prediction, reference = tmp_path/'case.nii.gz', tmp_path/'reference.nii.gz'
    for path in (prediction,reference):
        nib.save(nib.Nifti1Image(data,np.eye(4)),path)
    rows = evaluator.evaluate(prediction,reference,tmp_path)
    assert {row['Labels'] for row in rows} == set(adapter.REGIONS)
    assert all(row['sq_dsc'] == 1 and row['sq_hd95'] == 0 for row in rows)
    for path in (prediction,reference):
        nib.save(nib.Nifti1Image(np.zeros_like(data),np.eye(4)),path)
    empty = evaluator.evaluate(prediction,reference,tmp_path)
    assert all(row['sq_dsc'] == 'NaN' for row in empty)
