"""CPU checks of the final pre-training design; no trainer or data processing."""
import hashlib
import json
from pathlib import Path

import pytest

from models.latent_diffusion_unet3d import LatentDiffusionUNet3D

ROOT = Path(__file__).resolve().parents[1]
COORDINATES = ['crop_start_d', 'crop_start_h', 'crop_start_w',
               'crop_stop_d', 'crop_stop_h', 'crop_stop_w']


@pytest.fixture(scope='module')
def contract():
    return json.loads((ROOT / 'configs/d0b4c_training_validation_contract.json').read_text())


def test_pretraining_status_and_no_results(contract):
    assert contract['contract_version'] == 1
    assert contract['stage'] == 'D0-B4c'
    assert contract['status'] == 'DESIGN_LOCKED_PRE_TRAINING'
    for key in ('scientific_training_started', 'training_results_observed',
                'validation_results_observed', 'hyperparameters_selected_from_validation_results'):
        assert contract[key] is False
    def check(value):
        if isinstance(value, dict):
            for key, item in value.items():
                assert key not in {'results', 'training_results', 'validation_results', 'validation_scores', 'performance_claims'}
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)
    check(contract)


def test_exact_source_provenance_and_actual_b4b_reference(contract):
    provenance = contract['source_provenance']
    assert provenance == {
        'train_split_sha256': '581dc6326a22cb0b9582fa249b8e13fd948597a60c5f454cb5ade0493fa7c18e',
        'validation_split_sha256': 'ec5fe6b0bbcf190fe8b7c032f8da4ff1d495969c2d868561e7a3351c572cc5ca',
        'e1c_checkpoint_sha256': 'cf5a04a2d845cecb45a0ea87d617abad0afab9ad4264abce8342a6045563ee53',
        'd0a_normalization_contract_sha256': 'bca5c0f8f1617d73fc4d0850fe5b6f45e4b2bcbacbb4018f1d5d825de49ad792',
        'd0b0_manifest_sha256': '9c4ba67fc195537a5500229722ce55e4772ea5b7d135b351b3a2fc4b857f44b9',
        'd0b3_corruption_decision_sha256': '095e1f9671411eca0d12a2f834833d78477a14664504bde3e6e1b46766c6cc2f',
        'd0b4a_microbatch_decision_sha256': '07c268d97c7a39f389548344be7245cee4eba3a390a91d29b6533ab466c5254b',
        'd0b4a_freeze_commit': '6772b023ab9c1d870f258f45d030b958d1f9b19b',
        'd0b4b_contract_path': 'configs/d0b4b_inference_contract.json',
        'd0b4b_contract_sha256': '29cf659528f6257cd9f8f218e9f4eaf8fabdc6d8a0e4905505aa8548ae68bde4',
        # Captured HEAD at design creation, intentionally not a moving HEAD check.
        'd0b4b_design_lock_commit': 'cf5985fdee2546a8d609d3abe76121951749f851',
    }
    assert hashlib.sha256((ROOT / provenance['d0b4b_contract_path']).read_bytes()).hexdigest() == provenance['d0b4b_contract_sha256']


def test_architecture_matches_existing_cpu_model(contract):
    model_contract = contract['model']
    model = LatentDiffusionUNet3D()
    assert all(p.device.type == 'cpu' for p in model.parameters())
    assert model_contract['parameter_count'] == sum(p.numel() for p in model.parameters()) == 22825280
    assert model_contract['latent_input_shape'] == model_contract['latent_output_shape'] == ['B', 256, 12, 12, 12]
    assert model_contract['architecture'] == 'compact 3D residual U-Net'
    assert model_contract['channel_widths'] == [64, 128, 256]
    assert model_contract['resblocks_per_encoder_level'] == model_contract['middle_resblocks'] == model_contract['resblocks_per_decoder_level'] == 2
    assert model_contract['groupnorm_groups'] == 32
    assert model_contract['activation'] == 'SiLU'
    assert model_contract['dropout'] == 0 and model_contract['attention'] == 'none'
    assert model_contract['timestep_embedding'] == dict(sinusoidal_dimension=64, mlp_dimension=256)
    assert model_contract['final_output'] == dict(type='Conv3d', channels=256, zero_initialized=True)
    assert model_contract['prediction_type'] == 'epsilon'
    assert model_contract['conditioning'] == 'none'
    assert model_contract['support_mask_learned_conditioning'] is False


def test_source_path_and_gt_restrictions(contract):
    source = contract['source_path']
    assert source['pipeline'] == ['raw multimodal MRI', 'existing full-volume nonzero normalization',
        'deterministic 96^3 crop', 'exact frozen D0-B3 local_empirical_resample corruption',
        'frozen E1-C Encoder', 'frozen P_a anatomy projector', 'Za', 'fixed D0-A per-channel standardization', 'z0']
    assert source['production_preprocessing_modified'] is False
    assert source['frozen_e1c_and_pa'] == dict(mode='eval', requires_grad=False, precision='FP32', context='torch.no_grad()')
    assert source['diffusion_model_trainable'] is True
    assert source['gt_allowed_use'] == 'training/validation corruption only'
    assert source['gt_permissions'] == dict(condition_diffusion=False, define_mri_support=False,
                                            model_input=False, diffusion_target=False)


def test_support_is_raw_computational_and_zero_support_is_error(contract):
    support = contract['support']
    assert support['source'] == 'raw cropped multimodal MRI before normalization'
    assert support['raw_definition'] == 'any raw modality nonzero'
    assert support['raw_patch_shape'] == [4, 96, 96, 96]
    assert support['latent_mapping'] == 'exact non-overlapping 8x8x8 occupancy'
    assert (support['occupancy_comparison'], support['occupancy_threshold']) == ('>=', 0.50)
    assert support['shape'] == ['B', 1, 12, 12, 12]
    assert support['computational_masking_only'] is True
    assert support['learned_conditioning'] is False
    assert support['zero_valid_latent_support'] == 'HARD_ERROR_PER_SAMPLE'
    assert support['silently_resample_or_reject'] is False
    assert contract['training_loss']['zero_support'] == contract['validation']['zero_support'] == 'HARD_ERROR_PER_SAMPLE'


def test_training_counts_and_epoch_convention(contract):
    train = contract['training']
    assert train['case_count'] == train['samples_per_epoch'] == 1080
    assert train['samples_per_case_per_epoch'] == 1
    assert train['max_epochs'] == 100
    assert train['max_sample_presentations'] == train['samples_per_epoch'] * train['max_epochs'] == 108000
    assert train['physical_batch_size'] == train['effective_batch_size'] == 8
    assert train['gradient_accumulation_steps'] == 1
    assert train['optimizer_steps_per_epoch'] == train['samples_per_epoch'] // train['physical_batch_size'] == 135
    assert train['max_optimizer_steps'] == train['optimizer_steps_per_epoch'] * train['max_epochs'] == 13500
    assert train['samples_per_epoch'] % train['physical_batch_size'] == 0
    assert train['drop_last'] is False
    assert (train['epoch_index_base'], train['epoch_index_min'], train['epoch_index_max']) == (0, 0, 99)
    assert train['human_epoch'] == 'epoch_index + 1'
    assert train['training_global_seed'] == train['model_initialization_seed'] == 20260923


@pytest.mark.parametrize('phase', ['training', 'validation'])
def test_no_augmentation_enrichment_or_rejection(contract, phase):
    for key in ('augmentation', 'tumor_enrichment', 'tumor_positive_rejection', 'support_rejection'):
        assert contract[phase][key] is False


def test_stable_hash_and_training_order(contract):
    hashing = contract['stable_hashing']
    assert hashing['algorithm'] == 'SHA256'
    assert hashing['encoding'] == 'UTF-8'
    assert hashing['serialization'] == "Ordered JSON arrays, separators=(',', ':'), ensure_ascii=false, no whitespace or trailing newline. Seeds, indices and coordinates are integers; case_id, axis and purpose are exact strings."
    assert hashing['hash_integer'] == 'int.from_bytes(SHA256(payload_bytes).digest(), byteorder="big", signed=False)'
    assert hashing['generator_seed'] == 'int.from_bytes(SHA256(payload_bytes).digest()[:8], byteorder="big", signed=False) & ((1 << 63) - 1)'
    assert hashing['python_hash_allowed'] is False
    order = contract['training']['case_order']
    assert order['payload_fields'] == ['training_global_seed', 'epoch_index', 'case_id', 'purpose']
    assert order['purpose'] == 'training_case_order'
    assert order['rule'] == 'Sort all locked training case IDs by (full SHA256 hexadecimal digest of the ordered payload, case_id); case_id is the deterministic collision tie-breaker.'
    assert order['every_case_exactly_once'] is True
    assert order['independent_of'] == ['filesystem ordering', 'DataLoader worker order', 'Python hash randomization', 'batch size']


def test_training_crop_identity_bounds_and_logging(contract):
    crop = contract['training']['crop']
    assert crop['shape'] == [96, 96, 96]
    assert crop['purpose'] == 'training_crop'
    assert crop['payload_fields'] == ['training_global_seed', 'epoch_index', 'case_id', 'axis', 'purpose']
    assert crop['axes'] == ['d', 'h', 'w']
    assert crop['dimension_source'] == 'actual spatial dimensions of the image returned by the existing case loader'
    assert crop['start_axis'] == 'hash_integer % (volume_axis_size - 96 + 1)'
    assert crop['stop_axis'] == 'start_axis + 96'
    assert crop['volume_axis_smaller_than_patch'] == 'HARD_ERROR'
    assert crop['entirely_in_bounds'] is True
    assert crop['gt_used'] is False
    assert crop['independent_of_batch_order'] is True
    assert crop['record_coordinates_for_every_sample'] is True
    assert crop['coordinate_generation_version'] == 1


def test_exact_training_corruption_epoch_mapping(contract):
    corruption = contract['training']['corruption']
    assert corruption['operator'] == 'local_empirical_resample'
    assert corruption['implementation'] == 'scripts.audit_d0b3_corruption_operator.corrupt_input'
    assert corruption['donor_semantics'] == 'reuse exact frozen D0-B3 donor construction and corruption helpers without reimplementation or modification'
    assert corruption['global_seed'] == 'training_global_seed'
    assert corruption['patch_index'] == 'epoch_index'
    assert corruption['identity'] == ['training_global_seed', 'case_id', 'epoch_index']
    assert corruption['varies_across_epochs'] is True
    assert corruption['selected_from_validation_results'] is False


@pytest.mark.parametrize('phase,index', [('training', 'epoch_index'), ('validation', 'validation_patch_index')])
def test_partition_independent_diffusion_randomness(contract, phase, index):
    randomness = contract[phase]['diffusion_randomness']
    assert randomness['sample_identity_fields'] == ['case_id', index, *COORDINATES]
    assert randomness['seed_payload_fields'] == [f'{phase}_global_seed', 'sample_identity', 'purpose']
    assert randomness['sample_identity_encoding'] == 'Ordered JSON array in sample_identity_fields order; use stable_hashing serialization and generator_seed.'
    assert randomness['timestep_purpose'] == f'{phase}_timestep'
    assert randomness['epsilon_purpose'] == f'{phase}_epsilon'
    assert randomness['timestep_distribution'] == 'uniform integer [0,999]'
    assert randomness['timestep_generation'] == 'torch.randint(0, 1000, (1,), generator=separately_seeded_cpu_generator).item()'
    assert randomness['epsilon_distribution'] == 'independent standard Gaussian'
    assert randomness['epsilon_shape'] == [256, 12, 12, 12]
    assert randomness['epsilon_dtype'] == 'FP32'
    assert randomness['epsilon_generation'] == "One separately seeded CPU torch.Generator per sample; torch.randn((256,12,12,12), dtype=torch.float32, device='cpu', generator=generator), then transfer to execution device."
    if phase == 'training':
        assert randomness['independent_of'] == ['batch ordering', 'worker scheduling', 'future batching/partitioning']
    else:
        assert randomness['epoch_in_seed_payload'] is False


def test_forward_diffusion_and_full_batch_loss(contract):
    forward = contract['forward_diffusion']
    assert forward['schedule_implementation'] == 'models.latent_diffusion_unet3d.cosine_schedule'
    assert (forward['T'], forward['cosine_offset'], forward['max_beta']) == (1000, 0.008, 0.999)
    assert forward['alpha'] == '1-beta' and forward['alpha_bar'] == 'cumprod(alpha)'
    assert forward['prediction_type'] == 'epsilon'
    assert forward['epsilon_supported'] == 'epsilon * support_mask'
    assert forward['z_t'] == '(sqrt(alpha_bar_t) * z0 + sqrt(1-alpha_bar_t) * epsilon_supported) * support_mask'
    loss = contract['training_loss']
    assert loss['target'] == 'epsilon_supported'
    assert loss['numerator'] == 'sum((epsilon_hat - epsilon_supported)^2 * support_mask) over the full physical batch'
    assert loss['denominator'] == 'sum(valid spatial support cells across B) * 256'
    assert loss['aggregation'] == 'valid-scalar-weighted over the full physical batch of 8'
    assert loss['equal_weight_per_case'] is False
    assert loss['accumulation_weighting_ambiguity'] is False
    assert loss['physical_batch_below_8_requires_loss_equivalence_reaudit'] is True


def test_optimizer_and_stop_protocol(contract):
    optimizer = contract['optimizer']
    assert optimizer['name'] == 'AdamW'
    assert optimizer['learning_rate'] == 1e-4
    assert optimizer['betas'] == [0.9, 0.999]
    assert optimizer['epsilon'] == 1e-8
    assert optimizer['weight_decay'] == 1e-4
    assert optimizer['lr_scheduler'] == 'none'
    assert optimizer['gradient_clipping'] == {'norm_type': 2, 'global': True, 'max_norm': 1.0, 'after_gradscaler_unscale': True}
    assert optimizer['early_stopping'] is False
    assert optimizer['max_epochs'] == 100
    assert optimizer['after_epoch_100'] == 'Stop. If convergence is insufficient, formally define a new protocol before additional training; never silently extend.'


def test_precision_and_ema(contract):
    precision = contract['precision']
    for key in ('raw_mri_operations_and_corruption', 'frozen_e1c', 'pa', 'd0a_standardization', 'model_master_parameters'):
        assert precision[key] == 'FP32'
    assert precision['diffusion_forward_backward'] == 'CUDA AMP FP16'
    assert precision['gradscaler_enabled'] is True
    assert precision['optimizer_state'] == 'FP32/default AdamW state semantics'
    assert precision['autocast_frozen_source_extraction'] is False
    ema = contract['ema']
    assert ema['enabled'] is True and ema['decay'] == 0.999
    assert ema['initialization'] == 'copy initial diffusion model weights before the first optimizer step'
    assert ema['update'] == 'after every successful optimizer step'
    assert ema['update_on_gradscaler_skipped_step'] is False
    assert ema['warmup'] is False and ema['alternate_decay_schedule'] is False
    assert ema['validation_and_checkpoint_selection_weights'] == 'EMA'
    assert ema['preserve_raw_and_ema_state'] is True


def test_validation_manifest_frozen_before_training(contract):
    validation = contract['validation']
    assert validation['case_count'] == 270
    assert validation['patches_per_case'] == 4
    assert validation['total_patches'] == validation['case_count'] * validation['patches_per_case'] == 1080
    assert validation['validation_patch_indices'] == [0, 1, 2, 3]
    assert validation['validation_global_seed'] == 20260923
    manifest = validation['manifest']
    assert manifest['fixed'] is True and manifest['written_and_frozen_before_training'] is True
    assert manifest['patch_shape'] == [96, 96, 96]
    assert manifest['gt_used_for_coordinates'] is False
    assert manifest['purpose'] == 'validation_crop'
    assert manifest['payload_fields'] == ['validation_global_seed', 'case_id', 'validation_patch_index', 'axis', 'purpose']
    assert manifest['dimension_source'] == 'actual spatial dimensions of the image returned by the existing case loader'
    assert manifest['start_axis'] == 'hash_integer % (volume_axis_size - 96 + 1)'
    assert manifest['stop_axis'] == 'start_axis + 96'
    assert manifest['entirely_in_bounds'] is True
    assert manifest['volume_axis_smaller_than_patch'] == 'HARD_ERROR'
    assert manifest['required_columns'] == ['case_id', 'validation_patch_index', *COORDINATES,
        'coordinate_generation_version', 'seed', 'split_sha256', 'timestep', 'epsilon_seed']
    assert manifest['split_sha256'] == contract['source_provenance']['validation_split_sha256']


def test_validation_fixed_corruption_and_scheduled_epochs(contract):
    validation = contract['validation']
    corruption = validation['corruption']
    assert corruption['operator'] == 'local_empirical_resample'
    assert corruption['implementation'] == contract['training']['corruption']['implementation']
    assert corruption['donor_semantics'] == contract['training']['corruption']['donor_semantics']
    assert corruption['global_seed'] == 'validation_global_seed'
    assert corruption['patch_index'] == 'validation_patch_index'
    assert corruption['gt_used_only_for_corruption'] is True
    assert corruption['identical_at_every_validation_epoch'] is True
    assert validation['identical_across_epochs'] == ['crop', 'corruption', 'timestep', 'epsilon']
    assert validation['human_facing_epochs'] == list(range(5, 101, 5))
    assert validation['validation_based_early_stopping'] is False
    assert validation['model_selection_between_scheduled_epochs'] is False
    assert validation['weights'] == 'EMA'
    assert validation['model_mode'] == 'eval' and validation['context'] == 'torch.no_grad()'


def test_primary_validation_metric_and_checkpoint_selection(contract):
    validation = contract['validation']
    metric = validation['primary_metric']
    assert metric['patch_loss'] == 'sum((epsilon_hat - epsilon_supported)^2 * support_mask) / (sum(patch valid spatial support cells) * 256)'
    assert metric['case_metric'] == 'arithmetic mean of the 4 patch losses for that case'
    assert metric['primary_metric'] == 'arithmetic mean of case_metric over all 270 cases'
    assert metric['aggregation_order'] == ['patch valid-scalar-normalized MSE', '4-patch arithmetic case mean', '270-case arithmetic mean']
    assert metric['equal_case_weight_regardless_of_support_size'] is True
    assert metric['intentionally_differs_from_training_batch_objective'] is True
    assert validation['secondary_metrics_select_checkpoint'] is False
    selection = contract['checkpoint_selection']
    assert selection['weights'] == 'EMA'
    assert selection['metric'] == metric['name'] == 'mean case-level validation support-masked epsilon MSE'
    assert selection['rule'] == 'lowest primary validation metric'
    assert selection['primary_metric_only'] is True
    assert selection['tie_tolerance'] == 1e-12
    assert selection['improvement'] == 'new_metric < best_metric - 1e-12'
    assert selection['tie_winner'] == 'earlier epoch'
    assert selection['forbidden_selection_sources'] == ['secondary metrics', 'visual reconstruction',
        'manual selection', 'DeltaZ quality', 'segmentation results', 'sensitivity timestep results']


def test_checkpoint_artifacts_and_deterministic_resume(contract):
    artifacts = contract['artifacts']
    assert artifacts['required'] == ['best.pt', 'latest.pt', 'training_history.csv', 'validation_per_case.csv',
        'validation_summary.json', 'training_provenance.json', 'validation_manifest.csv']
    assert set(artifacts['best_checkpoint_required_fields']) == {'raw_model_state', 'ema_state', 'optimizer_state',
        'gradscaler_state', 'epoch_index', 'human_epoch', 'optimizer_step', 'primary_validation_metric',
        'architecture_provenance', 'training_contract_provenance'}
    assert {'raw_model_state', 'ema_state', 'optimizer_state', 'gradscaler_state', 'rng_state',
            'sampler_state', 'next_sample_cursor', 'best_primary_validation_metric', 'best_epoch'} <= set(artifacts['latest_checkpoint_required_fields'])
    assert artifacts['latest_role'] == 'resume-only; must not replace best.pt for primary inference'
    assert artifacts['retain_every_epoch_checkpoint'] is False
    resume = contract['resume']
    assert resume['identity_derived_randomness'] is True
    assert resume['depends_on_number_of_prior_random_calls'] is False
    assert resume['same_future_for_epoch_sample_identity'] == ['case order', 'crop coordinates', 'corruption identities', 'timesteps', 'epsilon']
    assert {'raw model', 'EMA', 'optimizer', 'GradScaler', 'sampler epoch and next sample cursor'} <= set(resume['preserve_states'])
    assert resume['silently_restart_optimizer'] is False and resume['silently_restart_ema'] is False


def test_development_validation_is_not_independent_final_test(contract):
    interpretation = contract['validation_interpretation']
    assert interpretation['split_role'] == 'DEVELOPMENT_VALIDATION'
    assert interpretation['used_for_checkpoint_selection'] is True
    assert interpretation['untouched_final_test_set'] is False
    assert interpretation['independent_final_test_set'] is False
    assert interpretation['later_deltaz_or_segmentation_analysis_label'] == 'development/validation analysis, not independent final-test performance'
    assert interpretation['independent_final_evaluation'] == 'separate/TBD'
    assert interpretation['independent_paper_claims_require'] == 'separate held-out, external, challenge, or otherwise independent evaluation source'
    assert interpretation['carry_limitation_into_paper_provenance'] is True
    assert 'development-validation limitation' in contract['artifacts']['training_provenance_required']


def test_complete_pretraining_freeze_list(contract):
    freeze = contract['pretraining_freeze']
    assert freeze['frozen_before_scientific_training'] is True
    assert freeze['items'] == ['architecture', 'source checkpoint', 'Za normalization', 'training split',
        'validation split', 'training crop sampler', 'case ordering', 'training corruption', 'support semantics',
        'diffusion schedule', 'prediction type', 'training loss', 'optimizer', 'learning rate', 'weight decay',
        'batch size', 'AMP', 'EMA', 'epochs', 'validation manifest algorithm', 'validation randomness',
        'validation frequency', 'primary validation metric', 'checkpoint selection', 'inference contract reference']
