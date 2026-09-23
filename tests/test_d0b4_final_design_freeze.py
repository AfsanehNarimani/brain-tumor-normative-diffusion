"""CPU-only provenance and design consistency checks; no training or inference."""
import hashlib
import json
from pathlib import Path

import pytest
import torch

from scripts.freeze_d0b4d_validation_manifest import verify

ROOT = Path(__file__).resolve().parents[1]
STAGES = {
    'd0b4a': ('6772b023ab9c1d870f258f45d030b958d1f9b19b',
              'outputs/frozen/d0b4a_diffusion_memory_preflight/decision/microbatch_decision.json',
              '07c268d97c7a39f389548344be7245cee4eba3a390a91d29b6533ab466c5254b'),
    'd0b4b': ('cf5985fdee2546a8d609d3abe76121951749f851',
              'configs/d0b4b_inference_contract.json',
              '29cf659528f6257cd9f8f218e9f4eaf8fabdc6d8a0e4905505aa8548ae68bde4'),
    'd0b4c': ('cb4d830a6ce777fab7935ae77eeb75ca321db5de',
              'configs/d0b4c_training_validation_contract.json',
              '861ea070f7bfa2677b9345403905e70400190012e56efac722ccbb892af21f75'),
}
B4D_COMMIT = '6d25105965d4fbb4badca4ed6447926887e20266'
B4D_HASHES = {
    'validation_manifest.csv': '2a3a10d0d7b6b34bc412cfee5e30f45912161528ff509e69193f3516af3baee4',
    'validation_manifest_audit.csv': 'fe9b055b64c51fa1dafddbe6cb260c73310ef7468ef10ccb6e80477a3cdf2257',
    'validation_manifest_summary.json': 'a109cd877adb18cd6a7d4c2741ffa1017df6593b16aa79ac8b3248f8351b1883',
    'provenance.json': '02ba95bb4cd53fc3fa906e1a44c02974ccbf3e5eb45897cf1e5c0260678f370f',
}
ROOT_HASHES = {
    'train_split': '581dc6326a22cb0b9582fa249b8e13fd948597a60c5f454cb5ade0493fa7c18e',
    'validation_split': 'ec5fe6b0bbcf190fe8b7c032f8da4ff1d495969c2d868561e7a3351c572cc5ca',
    'e1c_checkpoint': 'cf5a04a2d845cecb45a0ea87d617abad0afab9ad4264abce8342a6045563ee53',
    'd0a_normalization_contract': 'bca5c0f8f1617d73fc4d0850fe5b6f45e4b2bcbacbb4018f1d5d825de49ad792',
    'd0b0_manifest': '9c4ba67fc195537a5500229722ce55e4772ea5b7d135b351b3a2fc4b857f44b9',
    'd0b3_corruption_decision': '095e1f9671411eca0d12a2f834833d78477a14664504bde3e6e1b46766c6cc2f',
}


@pytest.fixture(scope='module')
def freeze():
    return json.loads((ROOT / 'configs/d0b4_final_design_freeze.json').read_bytes())


def file_sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def source_json(freeze, stage):
    return json.loads((ROOT / freeze[stage]['path']).read_bytes())


def test_status_and_training_start_gate(freeze):
    assert freeze['contract_version'] == 1
    assert freeze['stage'] == 'D0-B4'
    assert freeze['status'] == 'COMPLETE_DESIGN_FROZEN_PRE_TRAINING'
    assert freeze['freeze_branch'] == 'experiment/d0b4-diffusion-design'
    assert freeze['freeze_base_commit'] == B4D_COMMIT
    assert freeze['purpose'] == 'Single authoritative provenance record proving that architecture, training, validation, inference, sampling, and development-validation choices were frozen before scientific diffusion training.'
    for key in ('scientific_training_started', 'training_results_observed', 'validation_results_observed',
                'diffusion_checkpoint_exists'):
        assert freeze[key] is False
    gate = freeze['training_start_gate']
    assert gate['scientific_diffusion_training_allowed'] is True
    assert gate['training_has_not_started_at_freeze_time'] is True
    assert gate['all_prerequisites_frozen_and_verified'] is True
    assert gate['required_frozen_verified_stages'] == ['D0-B4a', 'D0-B4b', 'D0-B4c', 'D0-B4d']
    assert gate['meaning'] == 'Prerequisites are satisfied; this gate does not mean training occurred.'


@pytest.mark.parametrize('stage', STAGES)
def test_exact_stage_commit_and_actual_file_sha(freeze, stage):
    commit, path, digest = STAGES[stage]
    reference = freeze[stage]
    assert reference['commit'] == commit
    assert reference['path'] == path
    assert reference['sha256'] == file_sha(path) == digest
    assert reference['sha256_verified'] is True


@pytest.mark.parametrize('name', B4D_HASHES)
def test_exact_b4d_artifact_commit_and_actual_sha(freeze, name):
    stage = freeze['d0b4d']
    assert stage['commit'] == B4D_COMMIT
    assert stage['directory'] == 'outputs/frozen/d0b4d_validation_manifest'
    reference = stage['artifacts'][name]
    assert reference['path'] == f'{stage["directory"]}/{name}'
    assert reference['commit'] == B4D_COMMIT
    assert reference['sha256'] == file_sha(reference['path']) == B4D_HASHES[name]
    assert reference['sha256_verified'] is True


def test_b4d_manifest_inventory_and_read_only_semantic_verification(freeze):
    stage = freeze['d0b4d']
    folder = ROOT / stage['directory']
    assert set(stage['artifacts']) == set(B4D_HASHES)
    assert {p.name for p in folder.iterdir()} == {*B4D_HASHES, 'MANIFEST.sha256'}
    expected = ''.join(f'{B4D_HASHES[name]}  {name}\n' for name in sorted(B4D_HASHES)).encode()
    assert (folder / 'MANIFEST.sha256').read_bytes() == expected
    assert stage['manifest_sha256'] == file_sha(stage['manifest_path']) == '5dc557a37a0d8d637afee7217ca907cb6fab56c406e55d57fea22f297c8e3a95'
    assert stage['manifest_integrity_verified'] is True
    assert stage['semantic_integrity_verified'] is True
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in folder.iterdir()}
    summary = verify(folder)
    assert before == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in folder.iterdir()}
    assert stage['locked_values'] == {key: summary[key] for key in stage['locked_values']}
    assert stage['locked_values'] == dict(case_count=270, patches_per_case=4, total_patches=1080,
        zero_support_patch_count=0, validation_global_seed=20260923, support_threshold=0.50,
        corruption_operator='local_empirical_resample')


@pytest.mark.parametrize('name', ROOT_HASHES)
def test_upstream_source_sha_checks(freeze, name):
    reference = freeze['root_scientific_provenance'][name]
    assert reference['sha256'] == file_sha(reference['path']) == ROOT_HASHES[name]
    assert reference['sha256_verified'] is True
    assert source_json(freeze, 'd0b4c')['source_provenance'][name + '_sha256'] == ROOT_HASHES[name]


def test_b4a_batch_lock_matches_decision(freeze):
    locked = freeze['d0b4a']['locked_values']
    assert locked == dict(physical_microbatch=8, gradient_accumulation_steps=1, effective_batch_size=8,
        selected_peak_reserved_fraction=0.16406770176197644,
        selection_rule='largest passing physical microbatch with peak_reserved_fraction <= 0.90')
    decision = source_json(freeze, 'd0b4a')
    for key, value in locked.items():
        assert value == decision['selected_physical_microbatch' if key == 'physical_microbatch' else key]


def test_b4b_inference_lock_matches_source(freeze):
    locked = freeze['d0b4b']['locked_values']
    source = source_json(freeze, 'd0b4b')
    assert locked['primary_timestep'] == source['partial_noise']['primary_timestep'] == 495
    assert locked['sensitivity_timesteps'] == [p['timestep'] for p in source['partial_noise']['points'] if p['secondary']] == [327, 663]
    assert locked['ddim_eta'] == source['ddim']['eta'] == 0.0
    assert locked['ddim_model_evaluations'] == source['ddim']['model_evaluations'] == 50
    assert locked['primary_noise_replicates'] == source['randomness']['primary_noise_replicates'] == 4
    assert locked['inference_global_seed'] == source['randomness']['inference_global_seed'] == 20260923
    assert locked['gt_at_inference'] is source['gt_used_at_inference'] is False
    assert locked['pathology_corruption_at_inference'] is source['inference_source']['pathology_corruption_at_inference'] is False
    assert locked['primary_deltaz'] == 'mean_replicates(mean_channels(abs(z0_supported - Zprior_k)))'
    assert locked['source_aggregation'] == source['primary_deviation']['aggregation']
    assert locked['z0_supported'] == 'z0 * support_mask'
    assert locked['final_support_masking'] == source['primary_deviation']['final_support_masking']
    assert locked['zprior_zero_outside_support'] is source['support']['final_prior_zero_outside_support'] is True
    # Algebra check only: the two masked definitions are identical on finite values.
    z0 = torch.tensor([[[2., 500.], [-1., -900.]]])
    support = torch.tensor([1., 0.])
    priors = torch.tensor([0., 1., 3., -2.])[:, None, None, None].expand(4, *z0.shape) * support
    supported = (z0 * support - priors).abs().mean(dim=2).mean(dim=0) * support
    source_formula = ((z0 - priors).abs() * support).mean(dim=2).mean(dim=0) * support
    assert torch.equal(supported, source_formula)


def test_b4c_training_lock_matches_source(freeze):
    locked = freeze['d0b4c']['locked_values']
    assert locked == dict(train_cases=1080, validation_cases=270, max_epochs=100, samples_per_epoch=1080,
        physical_batch_size=8, gradient_accumulation_steps=1, effective_batch_size=8,
        optimizer_steps_per_epoch=135, max_optimizer_steps=13500, optimizer='AdamW', learning_rate=1e-4,
        betas=[0.9, 0.999], epsilon=1e-8, weight_decay=1e-4, lr_scheduler='none', gradient_clip=1.0,
        prediction_type='epsilon', T=1000, cosine_offset=0.008, max_beta=0.999, ema=True, ema_decay=0.999,
        validation_human_facing_epochs=list(range(5, 101, 5)),
        primary_checkpoint_selection_metric='mean case-level validation support-masked epsilon MSE',
        validation_role='DEVELOPMENT_VALIDATION', independent_final_evaluation='separate/TBD')
    source = source_json(freeze, 'd0b4c')
    for key in ('max_epochs', 'samples_per_epoch', 'physical_batch_size', 'gradient_accumulation_steps',
                'effective_batch_size', 'optimizer_steps_per_epoch', 'max_optimizer_steps'):
        assert locked[key] == source['training'][key]
    for key in ('learning_rate', 'betas', 'epsilon', 'weight_decay', 'lr_scheduler'):
        assert locked[key] == source['optimizer'][key]
    assert locked['train_cases'] == source['training']['case_count']
    assert locked['validation_cases'] == source['validation']['case_count']
    assert locked['primary_checkpoint_selection_metric'] == source['checkpoint_selection']['metric']
    assert locked['validation_role'] == source['validation_interpretation']['split_role']
    assert locked['independent_final_evaluation'] == source['validation_interpretation']['independent_final_evaluation']


def test_architecture_and_scientific_interpretation_guards(freeze):
    assert freeze['architecture'] == dict(model='LatentDiffusionUNet3D', parameter_count=22825280,
        latent_shape=[256, 12, 12, 12], base_widths=[64, 128, 256], attention=False,
        dropout=0, conditioning='none', za_source='Frozen E1-C Encoder + P_a provide Za.',
        za_description='anatomy/background representation')
    source = source_json(freeze, 'd0b4c')['model']
    assert freeze['architecture']['parameter_count'] == source['parameter_count']
    assert freeze['architecture']['base_widths'] == source['channel_widths']
    interpretation = freeze['scientific_interpretation']
    assert interpretation['Za_is_proven_healthy'] is False
    assert interpretation['Za_is_proven_pathology_free'] is False
    assert interpretation['proven_healthy_or_normal_brain_prior_claim_allowed'] is False
    assert interpretation['diffusion_prior_terminology'] == 'anatomy/background latent diffusion prior'
    assert interpretation['groupnorm_caveat'] == 'GroupNorm introduces spatial statistical coupling beyond local Conv/Pool receptive fields; this caveat remains applicable to interpretation of E1-C latent representations.'
    assert interpretation['corruption_caveat'] == 'D0-B3 corruption reduces direct pathology information at the MRI input, but does not prove pathology-free latent representations.'
    text = json.dumps(freeze)
    assert 'healthy latent prior' not in text and 'normal-brain latent prior' not in text


def test_all_post_hoc_tuning_guards(freeze):
    guard = freeze['no_post_hoc_tuning']
    assert guard['frozen_before_scientific_training'] is True
    assert guard['locked_items'] == ['architecture', 'source model', 'Za normalization', 'training split',
        'validation split', 'training crop generation', 'training case ordering', 'training corruption',
        'support semantics', 'diffusion schedule', 'prediction type', 'optimizer', 'learning rate',
        'batch size', 'AMP', 'EMA', 'training duration', 'validation manifest', 'validation randomness',
        'validation frequency', 'checkpoint metric', 'checkpoint selection', 'primary inference timestep',
        'sensitivity timesteps', 'DDIM schedule', 'DDIM eta', 'noise replicates', 'seed semantics',
        'DeltaZ aggregation', 'development-validation interpretation']
    assert guard['future_changes_require_new_protocol_version'] is True
    assert guard['future_changes_require_separate_provenance'] is True
    assert guard['silent_replacement_of_frozen_primary_protocol_allowed'] is False


def test_no_scientific_results_or_diffusion_outcomes(freeze):
    forbidden = {'results', 'training_results', 'validation_results', 'validation_scores', 'validation_metric_value',
                 'best_epoch', 'best_loss', 'dice', 'auroc', 'diffusion_checkpoint_path', 'training_history',
                 'support_fraction_statistics', 'candidate_memory_summaries'}
    def check(value):
        if isinstance(value, dict):
            assert not forbidden.intersection(value)
            for item in value.values():
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)
    check(freeze)
