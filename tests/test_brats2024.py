"""CPU-only Experiment 0.1 conversion, geometry, and orchestration checks."""
import json
import nibabel as nib
import numpy as np
import pytest
import torch
from evaluation.brats2024 import (probabilities_to_labels, overlap_audit, summarize_overlap,
                                  export_labelmap, validate_labelmap)
from evaluation.brats_official import OfficialEvaluator
from scripts.evaluate_brats2024 import select_cases, new_output_directory


def probabilities(values):
    return torch.tensor(values, dtype=torch.float32).reshape(4,1,1,1)


@pytest.mark.parametrize('values,label', [
    ([0,0,0,0],0), ([.49,.4,.3,.2],0),
    ([.5,0,0,0],1), ([0,.5,0,0],2), ([0,0,.5,0],3), ([0,0,0,.5],4),
    ([.6,.8,.4,0],2), ([.4,.3,.2,.5],4),
    ([.8,.8,0,0],1), ([0,.9,.9,.9],2),
    ([.6,.7,.8,0],3), ([.6,.7,.8,.9],4),
])
def test_conversion(values,label):
    result = probabilities_to_labels(probabilities(values))
    assert result.shape == (1,1,1) and result.dtype == torch.uint8
    assert result.item() == label
    assert set(result.unique().tolist()) <= {0,1,2,3,4}


@pytest.mark.parametrize('channels,pairs', [(0,0),(1,0),(2,1),(3,3),(4,6)])
def test_overlap_counts(channels,pairs):
    tensor = torch.zeros(4,1,1,2)
    tensor[:channels,0,0,0] = .7
    row = overlap_audit(tensor,'case')
    assert row['foreground_voxels'] == int(channels > 0)
    assert row['overlapping_voxels'] == int(channels >= 2)
    assert row['max_overlap_channels'] == channels
    assert row['overlap_fraction'] == float(channels >= 2)
    summary = summarize_overlap([row])
    assert sum(summary['pairwise_overlap_counts'].values()) == pairs
    assert summary['maximum_overlap_channels_seen'] == channels
    assert summary['total_overlapping_voxels'] == int(channels >= 2)


def test_fraction():
    tensor = torch.zeros(4,1,1,3)
    tensor[0] = .9
    tensor[1,0,0,0] = .8
    assert overlap_audit(tensor,'a')['overlap_fraction'] == pytest.approx(1/3)


@pytest.mark.parametrize('tensor', [torch.zeros(3,1,1,1), torch.zeros(4,1,1),
                                    torch.full((4,1,1,1),float('nan')), torch.full((4,1,1,1),1.1)])
def test_invalid_probabilities(tensor):
    with pytest.raises(ValueError): probabilities_to_labels(tensor)


def test_export_geometry_and_determinism(tmp_path):
    affine = np.diag([1.,2.,3.,1.]); affine[0,3] = 8
    reference = tmp_path / 'seg.nii.gz'
    nib.save(nib.Nifti1Image(np.zeros((2,3,4),np.uint8),affine),reference)
    labels = np.resize(np.arange(5,dtype=np.uint8),(2,3,4))
    outputs = [tmp_path / name / 'case.nii.gz' for name in ('a','b')]
    for output in outputs:
        export_labelmap(labels,reference,output,'case')
        image = nib.load(output)
        np.testing.assert_array_equal(image.affine,affine)
        np.testing.assert_array_equal(np.asanyarray(image.dataobj),labels)
    assert outputs[0].read_bytes() == outputs[1].read_bytes()
    with pytest.raises(FileExistsError): export_labelmap(labels,reference,outputs[0],'case')
    with pytest.raises(ValueError, match='filename'): export_labelmap(labels,reference,tmp_path/'wrong.nii.gz','case')
    with pytest.raises(ValueError, match='case ID'): export_labelmap(labels,reference,tmp_path/'bad.nii.gz','../bad')
    ref = nib.load(reference)
    for bad in (np.zeros((2,3,3),np.uint8), np.full((2,3,4),5,np.uint8), labels.astype(float)):
        with pytest.raises(ValueError): validate_labelmap(bad,ref)


def test_missing_evaluator(monkeypatch):
    def unavailable(name):
        raise ModuleNotFoundError(name)
    monkeypatch.setattr('evaluation.brats_official.importlib.import_module', unavailable)
    with pytest.raises(RuntimeError, match='package unavailable'):
        OfficialEvaluator()


def test_limits_and_additive_output(tmp_path):
    records = list(range(270))
    assert select_cases(records,5) == list(range(5)) and len(records) == 270
    with pytest.raises(ValueError): select_cases(records,0)
    old = tmp_path/'evaluation'; old.mkdir()
    custom = old/'aggregate_metrics.json'; custom.write_text('frozen')
    smoke = new_output_directory(tmp_path,5)
    (smoke/'run_summary.json').write_text('{}')
    full = new_output_directory(tmp_path,None)
    assert full == smoke.parent
    assert custom.read_text() == 'frozen'
    with pytest.raises(FileExistsError): new_output_directory(tmp_path,5)


@pytest.mark.parametrize('audit_only', [False, True])
@pytest.mark.parametrize('max_cases', [5, None])
def test_smoke_orchestration(tmp_path,monkeypatch,max_cases,audit_only):
    from scripts import evaluate_brats2024 as script
    from models.baseline_unet3d import BaselineUNet3D
    root = tmp_path/'experiment'; root.mkdir()
    old = root/'evaluation'; old.mkdir()
    for name in ('aggregate_metrics.json','per_case_metrics.csv'):
        (old/name).write_text('frozen')
    reference = tmp_path/'ref.nii.gz'
    nib.save(nib.Nifti1Image(np.zeros((8,8,8),np.uint8),np.eye(4)),reference)
    split = tmp_path/'val.txt'; split.write_text('\n'.join(f'case-{i}' for i in range(270)))
    model = BaselineUNet3D(base_channels=2)
    config = {'data':{'val_cases':str(split),'patch_size':[8]*3},'model':{'base_channels':2},'train':{'threshold':.5}}
    torch.save({'config':config,'model':model.state_dict(),'epoch':10},root/'best.pt')
    frozen = (root/'best.pt').read_bytes()
    class Dataset:
        def __init__(self):
            self.records = [{'subject_id':f'case-{i}','paths':{'mask':reference}} for i in range(270)]
        def __len__(self): return len(self.records)
        def __getitem__(self,i): return {'case_id':self.records[i]['subject_id'],'image':torch.zeros(4,8,8,8)}
    class Evaluator:
        metadata = {'evaluator_package':'test-double','evaluator_version':'test','evaluator_configuration':'gli'}
        raw_results = []
        audit = {'audit_passed':True}
        def __init__(self,*args): pass
        def evaluate(self,*args): return [{'Labels':'NETC','LesionWise_Score_Dice':1.,'LesionWise_Score_HD95':0.}]
    monkeypatch.setattr(script,'OfficialEvaluator',Evaluator)
    monkeypatch.setattr(script,'make_dataset',lambda *args:Dataset())
    from evaluation.provenance import sha256_file
    audit_calls = []
    def audited(*args):
        audit_calls.append(args)
        result = dict(train_split_path=str(split),validation_split_path=str(split),
                    train_split_sha256=sha256_file(split), validation_split_sha256=sha256_file(split),
                    manifest_sha256='synthetic-manifest', fingerprint_scope='test double')
        (args[3]/'dataset_manifest.csv').write_text('case_id,split\n')
        (args[3]/'dataset_audit.json').write_text(json.dumps(result))
        return result
    monkeypatch.setattr(script,'audit_dataset',audited)
    monkeypatch.setattr(script,'tiled_logits',lambda *args: torch.zeros(1,4,8,8,8))
    if audit_only:
        def forbidden(*args, **kwargs):
            pytest.fail('Audit-only reached model/inference/export/overlap/evaluation')
        for name in ('build_model', 'tiled_logits', 'export_labelmap', 'overlap_audit', 'probabilities_to_labels'):
            monkeypatch.setattr(script, name, forbidden)
        monkeypatch.setattr(Evaluator, 'evaluate', forbidden)
        monkeypatch.setattr(Dataset, '__getitem__', forbidden)
    result = script.run(root,'unused',device='cuda',max_cases=max_cases,audit_only=audit_only) if audit_only else script.run(root,'unused',max_cases=max_cases)
    if audit_only:
        out = old/'brats2024'/'audit'
        assert result['audit_only']
        assert result['dataset_audit_passed'] and result['evaluator_audit_passed']
        assert not result['complete_validation_run']
        assert not result['official_evaluator_completed']
        assert not result['overlap_audit_completed']
        assert result['successfully_exported_cases'] == 0
        assert result['failed_cases'] == []
        assert len(audit_calls) == 1 and len(audit_calls[0]) == 5
        assert set(p.name for p in out.iterdir()) == {
            'dataset_manifest.csv', 'dataset_audit.json', 'evaluator_audit.json',
            'evaluation_config.json', 'run_summary.json'}
        saved = json.loads((out/'evaluation_config.json').read_text())
        assert saved['audit_only'] and not saved['smoke_run']
        assert saved['checkpoint_sha256'] == sha256_file(root/'best.pt')
        assert json.loads((out/'run_summary.json').read_text()) == result
        assert (root/'best.pt').read_bytes() == frozen
        assert all((old/name).read_text() == 'frozen' for name in ('aggregate_metrics.json','per_case_metrics.csv'))
        # An audit neither occupies nor prevents future full/smoke outputs.
        assert script.new_output_directory(root, None) == out.parent
        assert script.new_output_directory(root, 5) == out.parent/'smoke_5'
        with pytest.raises(FileExistsError):
            script.new_output_directory(root, None, audit_only=True)
        return
    assert result['requested_case_count'] == result['successfully_exported_cases'] == (max_cases or 270)
    assert len(audit_calls) == 1
    assert result['dataset_audit_passed'] and result['evaluator_audit_passed'] and result['prediction_integrity_passed']
    assert result['official_evaluator_completed']
    assert result['complete_validation_run'] is (max_cases is None)
    assert (root/'best.pt').read_bytes() == frozen
    assert all((old/name).read_text() == 'frozen' for name in ('aggregate_metrics.json','per_case_metrics.csv'))
    out = old/'brats2024'
    if max_cases is not None: out = out/'smoke_5'
    config = json.loads((out/'evaluation_config.json').read_text())
    assert config['smoke_run'] is (max_cases is not None)
    assert config['checkpoint_sha256'] == sha256_file(root/'best.pt')
    assert config['validation_split_sha256'] == sha256_file(split)
    for name in ('official_metrics.csv','official_evaluation_summary.json','overlap_per_case.csv','overlap_summary.json','run_summary.json'):
        assert (out/name).exists()


def test_official_failure_has_no_fallback(tmp_path):
    evaluator = OfficialEvaluator.__new__(OfficialEvaluator)
    def fail(**kwargs): raise RuntimeError('upstream failure')
    evaluator.evaluate_single_exam = fail
    evaluator.evaluator = object()
    with pytest.raises(RuntimeError, match='no fallback'):
        evaluator.evaluate(tmp_path/'p.nii.gz', tmp_path/'r.nii.gz', tmp_path)


def test_audit_failure_prevents_model_creation(tmp_path,monkeypatch):
    from scripts import evaluate_brats2024 as script
    experiment = tmp_path/'frozen'; experiment.mkdir()
    torch.save({'config':{'train':{'threshold':.5},'data':{}}},experiment/'best.pt')
    def fail(*args): raise ValueError('Dataset audit failed: overlap')
    monkeypatch.setattr(script,'audit_dataset',fail)
    monkeypatch.setattr(script,'build_model',lambda *args: pytest.fail('Inference initialization before audit'))
    with pytest.raises(ValueError,match='overlap'):
        script.run(experiment,'unused',max_cases=5)
    summary = json.loads((experiment/'evaluation/brats2024/smoke_5/run_summary.json').read_text())
    assert not summary['dataset_audit_passed'] and not summary['official_evaluator_completed']
    assert summary['successfully_exported_cases'] == 0


def test_prediction_integrity_affine_mismatch(tmp_path):
    from evaluation.brats2024 import validate_exported_prediction
    reference = tmp_path/'reference.nii.gz'
    prediction = tmp_path/'case.nii.gz'
    array = np.zeros((2,3,4),np.uint8)
    nib.save(nib.Nifti1Image(array,np.eye(4)),reference)
    affine = np.eye(4); affine[0,3] = 10
    nib.save(nib.Nifti1Image(array,affine),prediction)
    with pytest.raises(ValueError,match='affine'):
        validate_exported_prediction(prediction,reference,'case')


def test_audit_only_cli(monkeypatch):
    from scripts import evaluate_brats2024 as script
    calls = []
    monkeypatch.setattr('sys.argv', ['evaluate_brats2024', '--data-root', 'synthetic', '--audit-only'])
    monkeypatch.setattr(script, 'run', lambda *args, **kwargs: calls.append((args, kwargs)))
    script.main()
    assert calls[0][1]['audit_only'] is True
    assert calls[0][0][1] == 'synthetic'
