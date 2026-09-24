"""CPU/synthetic contract tests; no CUDA, BraTS access, or scientific training."""
import csv
import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from scripts import train_d0_latent_diffusion as trainer


def test_startup_verifies_frozen_provenance_and_manifest():
    result = trainer.verify_startup()
    assert result['final_freeze_sha256'] == trainer.FINAL_SHA256
    assert result['b4c_contract_sha256'] == trainer.B4C_SHA256
    assert result['b4b_contract_sha256'] == trainer.B4B_SHA256
    assert result['validation_manifest_sha256'] == trainer.VAL_MANIFEST_SHA256
    assert result['validation_manifest_audit_sha256'] == trainer.VAL_AUDIT_SHA256
    assert result['validation_rows'] == 1080
    assert result['validation_case_count'] == 270
    assert result['validation_patches_per_case'] == 4
    assert result['validation_role'] == 'DEVELOPMENT_VALIDATION'
    assert result['d0b0_artifact_path'] == str(trainer.DEFAULT_D0B0.resolve())
    assert result['d0b0_artifact_sha256'] == trainer.D0B0_SHA256
    assert result['b4a_decision_path'] == str(trainer.DEFAULT_B4A.resolve())
    assert result['b4a_decision_sha256'] == trainer.B4A_SHA256


@pytest.mark.parametrize(('source_name', 'constant', 'label'), [
    ('DEFAULT_D0B0', 'D0B0_SHA256', 'D0-B0'),
    ('DEFAULT_B4A', 'B4A_SHA256', 'D0-B4a'),
])
def test_startup_checks_actual_d0b0_and_b4a_bytes(source_name, constant, label, tmp_path):
    source = getattr(trainer, source_name)
    good = tmp_path / 'good.json'
    good.write_bytes(source.read_bytes())
    assert trainer.read_sha(good, getattr(trainer, constant), label)
    bad = tmp_path / 'bad.json'
    bad.write_bytes(source.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        trainer.read_sha(bad, getattr(trainer, constant), label)


@pytest.mark.parametrize(('source_name', 'argument'), [
    ('DEFAULT_D0B0', 'd0b0'), ('DEFAULT_B4A', 'b4a')
])
def test_startup_rejects_mutated_d0b0_or_b4a_artifact(source_name, argument, tmp_path):
    source = getattr(trainer, source_name)
    bad = tmp_path / source.name
    bad.write_bytes(source.read_bytes() + b'mutated')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        trainer.verify_startup(**{argument: bad})


def test_runtime_provenance_records_verified_d0b0_and_b4a_references(tmp_path):
    verified = trainer.verify_startup()
    provenance = trainer.runtime_provenance({}, verified, tmp_path, 'cuda', trainer.VAL_MANIFEST_SHA256)
    assert provenance['d0b0_artifact_path'] == str(trainer.DEFAULT_D0B0.resolve())
    assert provenance['d0b0_artifact_sha256'] == trainer.D0B0_SHA256
    assert provenance['d0b4a_decision_path'] == str(trainer.DEFAULT_B4A.resolve())
    assert provenance['d0b4a_decision_sha256'] == trainer.B4A_SHA256
    assert provenance['dataset_root'] == str(tmp_path.resolve())


def test_read_sha_fails_closed(tmp_path):
    path = tmp_path / 'wrong'; path.write_text('tampered')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        trainer.read_sha(path, '0' * 64, 'contract')


def test_architecture_parameter_count_guard():
    from models.latent_diffusion_unet3d import LatentDiffusionUNet3D
    model = LatentDiffusionUNet3D()
    assert sum(p.numel() for p in model.parameters()) == 22_825_280


def test_stable_hash_encoding_and_order():
    values = [20260923, 2, 'café', 'd', 'training_crop']
    assert trainer.stable_payload(values) == '[20260923,2,"café","d","training_crop"]'.encode()
    cases = ['b', 'a', 'c']
    assert trainer.deterministic_case_order(cases, 3) == trainer.deterministic_case_order(list(reversed(cases)), 3)
    assert trainer.deterministic_case_order(cases, 3) != trainer.deterministic_case_order(cases, 4)
    assert sorted(trainer.deterministic_case_order(cases, 3)) == sorted(cases)
    with pytest.raises(ValueError, match='unique'):
        trainer.deterministic_case_order(['a', 'a'], 0)


def test_crop_uses_actual_dimensions_identity_and_no_gt():
    dimensions = [111, 122, 103]
    a = trainer.deterministic_crop('case_1', 4, dimensions)
    b = trainer.deterministic_crop('case_1', 4, dimensions)
    assert a == b
    for axis, size in zip('dhw', dimensions):
        start = trainer.hash_integer([20260923, 4, 'case_1', axis, 'training_crop']) % (size - 96 + 1)
        assert a[f'crop_start_{axis}'] == start
        assert a[f'crop_stop_{axis}'] == start + 96 <= size
    assert 'raw_mask' not in a and 'gt' not in a
    assert list(k for k in a if k.startswith('crop_')) == [
        'crop_start_d','crop_stop_d','crop_start_h','crop_stop_h','crop_start_w','crop_stop_w']
    with pytest.raises(ValueError, match='smaller'):
        trainer.deterministic_crop('case_small', 0, [95, 96, 100])


def test_epoch_case_order_contains_every_case_once():
    ids = [f'case_{i}' for i in range(1080)]
    result = trainer.deterministic_case_order(ids, 9)
    assert len(result) == len(set(result)) == 1080
    assert set(result) == set(ids)


def test_training_crop_manifest_rows_are_exact_unique_and_gt_free(tmp_path):
    ids = [f'case_{i:04d}' for i in range(1080)]
    dimensions = {case: (100, 101, 102) for case in ids}
    rows = trainer.training_epoch_crop_rows(ids, 0, dimensions)
    assert len(rows) == len({row['case_id'] for row in rows}) == 1080
    assert all('gt' not in key.lower() and 'mask' not in key.lower() for row in rows for key in row)
    assert tuple(rows[0]) == trainer.CROP_FIELDS
    assert rows == trainer.training_epoch_crop_rows(list(reversed(ids)), 0, dimensions)
    ordered = trainer.deterministic_case_order(ids, 0)
    assert [row['case_id'] for row in rows] == ordered
    for position, row in enumerate(rows):
        assert row['case_order_position'] == position
        assert row['human_epoch'] == 1 and row['epoch_index'] == 0
        assert row['training_global_seed'] == 20260923
        assert row['coordinate_generation_version'] == 1
        assert row['crop_start_d'] == trainer.deterministic_crop(row['case_id'], 0, dimensions[row['case_id']])['crop_start_d']

    path = tmp_path / 'training_crop_manifest.csv'
    sha1, count1 = trainer.ensure_training_crop_epoch(path, rows)
    before = path.read_bytes()
    sha2, count2 = trainer.ensure_training_crop_epoch(path, rows)
    assert (sha1, count1) == (sha2, count2) == (hashlib.sha256(before).hexdigest(), 1080)
    assert path.read_bytes() == before
    with path.open(newline='') as stream:
        loaded = list(csv.DictReader(stream))
    assert len(loaded) == 1080
    assert [(int(row['epoch_index']), int(row['case_order_position'])) for row in loaded] == sorted(
        (int(row['epoch_index']), int(row['case_order_position'])) for row in loaded)
    assert len({(int(row['epoch_index']), row['case_id']) for row in loaded}) == 1080


def test_training_crop_manifest_mismatch_fails_and_prior_epoch_is_preserved(tmp_path):
    ids = [f'case_{i:04d}' for i in range(1080)]
    dimensions = {case: (96, 96, 96) for case in ids}
    path = tmp_path / 'training_crop_manifest.csv'
    first = trainer.training_epoch_crop_rows(ids, 0, dimensions)
    trainer.ensure_training_crop_epoch(path, first)
    epoch0_bytes = path.read_bytes()
    epoch1 = trainer.training_epoch_crop_rows(ids, 1, dimensions)
    trainer.ensure_training_crop_epoch(path, epoch1)
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2160
    assert path.read_bytes().startswith(epoch0_bytes)
    conflicting = [dict(row) for row in epoch1]
    conflicting[0]['crop_start_d'] += 1
    with pytest.raises(ValueError, match='mismatch'):
        trainer.ensure_training_crop_epoch(path, conflicting)
    assert path.read_bytes().startswith(epoch0_bytes)


def test_training_crop_manifest_resume_prefix_sha_and_count(tmp_path):
    ids = [f'case_{i:04d}' for i in range(1080)]
    dimensions = {case: (96, 96, 96) for case in ids}
    path = tmp_path / 'training_crop_manifest.csv'
    epoch0 = trainer.training_epoch_crop_rows(ids, 0, dimensions)
    sha0, count0 = trainer.ensure_training_crop_epoch(path, epoch0)
    epoch1 = trainer.training_epoch_crop_rows(ids, 1, dimensions)
    trainer.ensure_training_crop_epoch(path, epoch1)
    sha, count = trainer.verify_training_crop_resume(path, completed_epoch_index=0,
        checkpoint_sha256=sha0, checkpoint_row_count=count0, expected_next_rows=epoch1)
    assert sha == hashlib.sha256(path.read_bytes()).hexdigest() and count == 2160
    with pytest.raises(ValueError, match='row count'):
        trainer.verify_training_crop_resume(path, completed_epoch_index=0,
            checkpoint_sha256=sha0, checkpoint_row_count=1080-1, expected_next_rows=epoch1)


def sample_row(case='case_a', epoch=4):
    return dict(case_id=case, epoch_index=epoch, crop_start_d=1, crop_start_h=2, crop_start_w=3,
                crop_stop_d=97, crop_stop_h=98, crop_stop_w=99)


def test_identity_timestep_noise_and_batching_independent():
    rows = [sample_row('a'), sample_row('b', 1), sample_row('c', 9)]
    expected = {r['case_id']: trainer.sample_randomness(r) for r in rows}
    assert all(0 <= value[0] < 1000 for value in expected.values())
    for row in reversed(rows):
        timestep, noise, t_seed, e_seed = trainer.sample_randomness(row)
        ref = expected[row['case_id']]
        assert timestep == ref[0] and t_seed == ref[2] and e_seed == ref[3]
        assert torch.equal(noise, ref[1])
        assert noise.shape == (256,12,12,12) and noise.dtype == torch.float32 and noise.device.type == 'cpu'
        assert t_seed != e_seed
    assert trainer.sample_randomness(rows[0])[0:1] == trainer.sample_randomness(rows[0])[0:1]


def test_forward_diffusion_explicit_broadcast_and_mask():
    from models.latent_diffusion_unet3d import cosine_schedule
    z0 = torch.ones(2,256,12,12,12)
    noise = torch.ones_like(z0) * 2
    mask = torch.zeros(2,1,12,12,12); mask[0,...,:6] = 1; mask[1,...,6:] = 1
    t = torch.tensor([0,999])
    zt, target = trainer.forward_diffuse(z0,t,noise,mask,cosine_schedule())
    ab = cosine_schedule()['alpha_bar'][t][:,None,None,None,None]
    expected = (ab.sqrt()*z0 + (1-ab).sqrt()*(noise*mask))*mask
    torch.testing.assert_close(zt, expected)
    assert zt.shape == z0.shape and target.shape == z0.shape
    assert torch.count_nonzero(zt * (1-mask)) == 0
    assert torch.equal(target, noise*mask)


def test_support_exact_mask_and_zero_support_hard_error():
    raw = torch.zeros(4,96,96,96)
    raw[:,:,:,:48] = 1
    mask = trainer.support_for_patch(raw)
    assert mask.shape == (1,12,12,12)
    assert torch.all(mask[:,:,:,:6] == 1) and torch.count_nonzero(mask[:,:,:,6:]) == 0
    with pytest.raises(ValueError, match='Zero valid latent support'):
        trainer.support_for_patch(torch.zeros_like(raw))
    with pytest.raises(ValueError, match='raw patch'):
        trainer.support_for_patch(torch.zeros(4,95,96,96))


def test_training_corruption_uses_frozen_helpers_raw_support_and_epoch_identity(monkeypatch):
    image = torch.ones(4,96,96,96)
    raw_patch = image.clone()
    normalized = torch.zeros_like(image)
    raw_mask = torch.ones(96,96,96)
    observed = {}
    frozen_support, frozen_abnormal = trainer.support_mask, trainer.abnormal_mask
    def support(raw):
        assert raw is raw_patch
        observed['support'] = frozen_support(raw)
        return observed['support']
    monkeypatch.setattr(trainer,'support_mask',support)
    donor = Mock(return_value=(['frozen pools'],[False,True,False,False]))
    corrupt = Mock(return_value=torch.zeros_like(image))
    monkeypatch.setattr(trainer,'donor_pools',donor)
    monkeypatch.setattr(trainer,'corrupt_input',corrupt)
    out, mask, fallback = trainer.corrupt_loaded_patch(raw_patch,raw_mask,normalized,
        case_id='case',patch_index=7,seed=20260923)
    assert out.shape == image.shape and fallback == 1 and mask.shape == (1,12,12,12)
    assert donor.call_args.args[0] is raw_patch and donor.call_args.args[2] is observed['support']
    assert corrupt.call_args.args[1].all()  # GT-derived abnormality is passed only to corruption.
    assert corrupt.call_args.args[3] == 'local_empirical_resample'
    assert corrupt.call_args.kwargs == dict(global_seed=20260923,case_id='case',patch_index=7)


def test_full_batch_loss_uses_valid_scalar_weighting_not_case_mean():
    prediction = torch.tensor([[[[[2.,0.,0.,0.]]]], [[[[1.,1.,1.,1.]]]]])
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[[[[1.,0.,0.,0.]]]], [[[[1.,1.,1.,1.]]]]])
    loss = trainer.weighted_batch_loss(prediction,target,mask)
    assert loss.item() == pytest.approx(8/5)
    per_case = [4.0, 1.0]
    assert loss.item() != pytest.approx(sum(per_case)/2)


def test_zero_support_batch_loss_errors():
    p = torch.zeros(2,256,2,2,2)
    with pytest.raises(ValueError, match='Zero valid latent support'):
        trainer.weighted_batch_loss(p,p,torch.zeros(2,1,2,2,2))


def test_optimizer_and_clip_contract():
    contract = json.loads(trainer.DEFAULT_CONTRACT.read_text())
    optimizer = contract['optimizer']
    assert (optimizer['name'],optimizer['learning_rate'],optimizer['betas'],optimizer['epsilon'],optimizer['weight_decay']) == ('AdamW',1e-4,[.9,.999],1e-8,1e-4)
    assert optimizer['lr_scheduler'] == 'none'
    assert optimizer['gradient_clipping']['global'] is True
    assert optimizer['gradient_clipping']['max_norm'] == 1.0
    assert trainer.optimizer_step_succeeded(8,8) is True
    assert trainer.optimizer_step_succeeded(8,16) is True
    assert trainer.optimizer_step_succeeded(8,4) is False


def test_ema_copy_equation_and_no_skipped_step_update():
    source = torch.nn.Linear(2,1,bias=False)
    with torch.no_grad(): source.weight.fill_(2)
    ema = trainer.make_ema(source)
    assert torch.equal(ema.weight,source.weight) and not ema.weight.requires_grad
    with torch.no_grad(): source.weight.fill_(4)
    before = ema.weight.clone()
    if trainer.optimizer_step_succeeded(8,4):
        trainer.ema_update(ema,source,.999)
    assert torch.equal(before,ema.weight)
    trainer.ema_update(ema,source,.999)
    torch.testing.assert_close(ema.weight, torch.full_like(ema.weight,2.002))
    assert torch.equal(source.weight, torch.full_like(source.weight, 4))


def test_validation_manifest_is_only_frozen_source():
    rows,data = trainer.load_validation_manifest()
    assert hashlib.sha256(data).hexdigest() == trainer.VAL_MANIFEST_SHA256
    assert len(rows) == 1080
    assert len({r['case_id'] for r in rows}) == 270
    assert {sum(r['case_id'] == case for r in rows) for case in {x['case_id'] for x in rows}} == {4}
    assert all(r['corruption_patch_index']==r['validation_patch_index'] for r in rows)
    assert all(r['corruption_operator']=='local_empirical_resample' for r in rows)


def test_validation_aggregation_patch_case_then_equal_case():
    rows=[]
    for i in range(270):
        loss = 10.0 if i == 0 else 1.0
        for p in range(4): rows.append(dict(case_id=f'c{i:03}',loss=loss))
    cases=trainer.validation_case_metrics(rows)
    assert len(cases)==270 and cases['c000']==10 and cases['c001']==1
    assert trainer.primary_validation_metric(cases)==pytest.approx((10+269)/270)
    with pytest.raises(ValueError,match='exactly four'):
        trainer.validation_case_metrics([dict(case_id='one',loss=1)])
    with pytest.raises(ValueError,match='270'):
        trainer.primary_validation_metric({'one':1})


def test_validation_schedule_and_secondary_metric_nonselection():
    contract=json.loads(trainer.DEFAULT_CONTRACT.read_text())
    assert contract['validation']['human_facing_epochs']==list(range(5,101,5))
    assert contract['validation']['secondary_metrics_select_checkpoint'] is False
    assert trainer.checkpoint_improves(1.0,None)
    assert not trainer.checkpoint_improves(1.0-1e-12,1.0)
    assert not trainer.checkpoint_improves(1.0,1.0)
    assert trainer.checkpoint_improves(1.0-1.01e-12,1.0)
    with pytest.raises(ValueError,match='Non-finite'):
        trainer.checkpoint_improves(float('nan'),None)


def test_checkpoint_schema_resume_guards_and_manifest_sha(monkeypatch):
    model=torch.nn.Linear(2,2); ema=trainer.make_ema(model)
    class State:
        def state_dict(self): return {'ok':True}
    monkeypatch.setattr(torch.cuda,'get_rng_state_all',lambda: [torch.tensor([1],dtype=torch.uint8)])
    payload=trainer.state_payload(model,ema,State(),State(),epoch_index=2,successful_steps=3,
        best_metric=1.,best_epoch=1,manifest_sha=trainer.VAL_MANIFEST_SHA256,provenance_sha=trainer.B4C_SHA256,
        crop_manifest_sha='crop-sha',crop_manifest_row_count=3240)
    for key in ('raw_model_state','ema_state','optimizer_state','gradscaler_state','epoch_index','human_epoch',
                'successful_optimizer_step','next_sample_cursor','best_primary_validation_metric','best_epoch',
                'python_rng_state','numpy_rng_state','torch_cpu_rng_state','torch_cuda_rng_state',
                'sampler_state','training_contract_provenance','validation_manifest_sha256'):
        assert key in payload
    assert payload['training_crop_manifest_sha256'] == 'crop-sha'
    assert payload['training_crop_manifest_row_count'] == 3240
    trainer.validate_resume_payload(payload,model=model,provenance_sha=trainer.B4C_SHA256,manifest_sha=trainer.VAL_MANIFEST_SHA256)
    for field,value,match in [('training_contract_provenance','wrong','contract'),
        ('validation_manifest_sha256','wrong','manifest'),('next_sample_cursor',3,'epoch-boundary')]:
        broken=dict(payload); broken[field]=value
        with pytest.raises(ValueError,match=match):
            trainer.validate_resume_payload(broken,model=model,provenance_sha=trainer.B4C_SHA256,manifest_sha=trainer.VAL_MANIFEST_SHA256)


def test_resume_dataset_root_must_match_even_when_split_hashes_match(tmp_path):
    current = dict(dataset_root=str((tmp_path/'dataset_a').resolve()), training_split_sha256=trainer.TRAIN_SHA256,
        validation_split_sha256=trainer.VAL_SHA256, d0b4_final_freeze_sha256=trainer.FINAL_SHA256,
        d0b4c_contract_sha256=trainer.B4C_SHA256, d0b4b_contract_sha256=trainer.B4B_SHA256,
        validation_manifest_sha256=trainer.VAL_MANIFEST_SHA256)
    existing = dict(current)
    trainer.validate_resume_provenance(existing, current)
    existing['dataset_root'] = str((tmp_path/'dataset_b').resolve())
    with pytest.raises(ValueError, match='dataset_root'):
        trainer.validate_resume_provenance(existing, current)


def test_manifest_copy_is_byte_identical_and_mismatch_rejected(tmp_path):
    data=trainer.DEFAULT_VAL_MANIFEST.read_bytes()
    assert trainer.validation_manifest_copy(tmp_path,data)==trainer.VAL_MANIFEST_SHA256
    assert (tmp_path/'validation_manifest.csv').read_bytes()==data
    with pytest.raises(ValueError,match='differs'):
        trainer.validation_manifest_copy(tmp_path,b'wrong')


def test_history_and_per_case_append_across_epochs(tmp_path):
    history=tmp_path/'training_history.csv'
    trainer.append_csv(history,trainer.HISTORY_FIELDS,dict(epoch_index=0,human_epoch=1,samples_presented=1080))
    trainer.append_csv(history,trainer.HISTORY_FIELDS,dict(epoch_index=1,human_epoch=2,samples_presented=1080))
    with history.open(newline='') as f: rows=list(csv.DictReader(f))
    assert len(rows)==2 and [r['human_epoch'] for r in rows]==['1','2']
    cases=tmp_path/'validation_per_case.csv'
    for epoch in (5,10):
        for i in range(270):
            trainer.append_csv(cases,trainer.CASE_FIELDS,dict(human_epoch=epoch,epoch_index=epoch-1,case_id=f'c{i}',patch_count=4,case_mean_epsilon_mse=1.))
    with cases.open(newline='') as f: rows=list(csv.DictReader(f))
    assert len(rows)==540 and {r['human_epoch'] for r in rows}=={'5','10'}
    trainer.upsert_csv(history,trainer.HISTORY_FIELDS,dict(epoch_index=1,human_epoch=2,samples_presented=999),('epoch_index',))
    with history.open(newline='') as f: rows=list(csv.DictReader(f))
    assert len(rows)==2 and next(r for r in rows if r['epoch_index']=='1')['samples_presented']=='999'
    trainer.truncate_csv_after_epoch(cases,trainer.CASE_FIELDS,3)
    with cases.open(newline='') as f: assert list(csv.DictReader(f)) == []


def test_smoke_separation_and_reported_non_scientific(tmp_path):
    scientific = tmp_path/'runs'/'d0'
    with pytest.raises(ValueError,match='separate'):
        trainer.verify_output_dir(scientific/'smoke',smoke=True,scientific_path=scientific)
    with pytest.raises(ValueError,match='separate'):
        trainer.verify_output_dir(tmp_path/'runs',smoke=True,scientific_path=scientific)
    with pytest.raises(ValueError,match='separate'):
        trainer.verify_output_dir(scientific,smoke=True,scientific_path=scientific)
    assert trainer.verify_output_dir(tmp_path/'runs'/'d0-smoke',smoke=True,
        scientific_path=scientific) == (tmp_path/'runs'/'d0-smoke').resolve()
    with pytest.raises(ValueError,match='smoke output'):
        trainer.verify_output_dir(trainer.DEFAULT_SMOKE_OUTPUT,smoke=False)
    target=tmp_path/'test_smoke_output'
    report=trainer.run_smoke(target)
    assert report['scientific_training'] is False
    assert report['synthetic_cpu_only'] is True
    assert report['no_scientific_artifacts_written'] is True
    assert not (target/'best.pt').exists()
    assert json.loads((target/'smoke_summary.json').read_text())==report


def test_scientific_output_refuses_existing_and_frozen_paths(tmp_path):
    existing=tmp_path/'existing'; existing.mkdir()
    with pytest.raises(FileExistsError): trainer.verify_output_dir(existing)
    with pytest.raises(ValueError,match='frozen'):
        trainer.verify_output_dir(trainer.ROOT/'outputs/frozen'/'bad')


def test_zero_support_error_includes_training_identity(monkeypatch):
    case='case_zero'; image=torch.zeros(4,96,96,96); raw_mask=torch.zeros(96,96,96)
    monkeypatch.setattr(trainer,'load_case',lambda *a:(image,raw_mask))
    monkeypatch.setattr(trainer,'normalize_nonzero',lambda x:x)
    with pytest.raises(ValueError) as error:
        trainer._training_batch('unused',[case],7)
    message=str(error.value)
    assert 'case_id=case_zero' in message and 'epoch_index=7' in message and 'crop_coordinates=' in message


def test_gpu_smoke_cli_modes_limits_and_required_paths():
    with pytest.raises(SystemExit):
        trainer.build_arg_parser(['--smoke', '--gpu-smoke'])
    with pytest.raises(SystemExit):
        trainer.build_arg_parser(['--gpu-smoke-train-batches', '1'])
    with pytest.raises(SystemExit):
        trainer.build_arg_parser(['--smoke', '--gpu-smoke-validation-patches', '2'])
    with pytest.raises(SystemExit):
        trainer.build_arg_parser(['--gpu-smoke'])
    args = trainer.build_arg_parser(['--gpu-smoke', '--dataset-root', '/dataset',
        '--output-dir', '/tmp/run-gpu-smoke'])
    assert args.gpu_smoke and not args.smoke
    assert args.gpu_smoke_train_batches == 2
    assert args.gpu_smoke_validation_patches == 8
    args = trainer.build_arg_parser(['--gpu-smoke', '--dataset-root', '/dataset',
        '--output-dir=/tmp/run-gpu-smoke', '--gpu-smoke-train-batches', '3',
        '--gpu-smoke-validation-patches', '5'])
    assert (args.gpu_smoke_train_batches, args.gpu_smoke_validation_patches) == (3, 5)


def test_gpu_smoke_bounded_train_and_validation_inputs_do_not_regenerate(monkeypatch):
    ids = [f'case_{i:04d}' for i in range(1080)]
    cases = trainer.gpu_smoke_train_cases(ids, 2)
    assert len(cases) == 16 and cases == trainer.deterministic_case_order(ids, 0)[:16]
    assert [len(cases[i:i+trainer.GPU_SMOKE_BATCH_SIZE]) for i in range(0, len(cases), 8)] == [8, 8]
    with pytest.raises(ValueError, match='enough cases'):
        trainer.gpu_smoke_train_cases(ids[:10], 2)
    with pytest.raises(ValueError, match='locked at 8'):
        trainer.gpu_smoke_train_cases(ids, 2, batch_size=4)
    rows = [{'case_id': 'c', 'validation_patch_index': i} for i in range(4)] + [
        {'case_id': 'd', 'validation_patch_index': i} for i in range(4)]
    monkeypatch.setattr(trainer, 'deterministic_crop', lambda *a, **k: pytest.fail('must use fixed manifest rows'))
    assert trainer.gpu_smoke_validation_rows(rows, 6) == rows[:6]
    with pytest.raises(ValueError, match=r'\[1,1080\]'):
        trainer.gpu_smoke_validation_rows(rows, 0)


def test_gpu_smoke_summary_and_artifact_inventory_are_non_scientific(tmp_path):
    verified = trainer.verify_startup()
    summary = trainer.gpu_smoke_summary_template(verified, tmp_path, 2, 8)
    assert summary['scientific_training'] is False
    assert summary['scientific_training_started'] is False
    assert summary['run_type'] == 'non_scientific_gpu_smoke'
    assert summary['checkpoint_selection_performed'] is False
    output = tmp_path / 'gpu_smoke'
    output.mkdir()
    (output / 'gpu_smoke_summary.json').write_text('{}')
    (output / 'gpu_smoke.log').write_text('')
    trainer.validate_gpu_smoke_artifact_inventory(output)
    (output / 'best.pt').write_bytes(b'forbidden')
    with pytest.raises(RuntimeError, match='Unexpected gpu-smoke artifacts'):
        trainer.validate_gpu_smoke_artifact_inventory(output)


def test_gpu_smoke_path_isolation_and_output_refusal(tmp_path):
    scientific = tmp_path / 'runs' / 'd0'
    assert trainer.verify_output_dir(tmp_path / 'runs' / 'd0-smoke', gpu_smoke=True,
        scientific_path=scientific) == (tmp_path / 'runs' / 'd0-smoke').resolve()
    with pytest.raises(ValueError, match='separate'):
        trainer.verify_output_dir(scientific / 'gpu-smoke', gpu_smoke=True, scientific_path=scientific)
    with pytest.raises(ValueError, match='separate'):
        trainer.verify_output_dir(tmp_path / 'runs', gpu_smoke=True, scientific_path=scientific)
    existing = tmp_path / 'existing-gpu-smoke'
    existing.mkdir()
    with pytest.raises(FileExistsError):
        trainer.verify_output_dir(existing, gpu_smoke=True, scientific_path=scientific)
    with pytest.raises(ValueError, match='containing "smoke"'):
        trainer.verify_output_dir(tmp_path / 'runs' / 'cpu-test', smoke=True, scientific_path=scientific)


def test_frozen_source_fingerprint_detects_parameter_mutation():
    encoder = torch.nn.Linear(3, 2, bias=False)
    anatomy = torch.nn.Linear(2, 2, bias=False)
    source = SimpleNamespace(encoder=encoder,
        projection=SimpleNamespace(anatomy_projector=anatomy))
    before = trainer.frozen_source_fingerprint(source)
    assert trainer.source_fingerprint_unchanged(before, source)
    with torch.no_grad():
        anatomy.weight[0, 0] += 1
    assert not trainer.source_fingerprint_unchanged(before, source)
