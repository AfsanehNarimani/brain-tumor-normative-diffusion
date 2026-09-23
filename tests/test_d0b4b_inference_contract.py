"""CPU-only checks of the pre-training design; no inference runner or training."""
import json
from pathlib import Path

import pytest
import torch

from models.latent_diffusion_unet3d import cosine_schedule


@pytest.fixture(scope='module')
def contract():
    path = Path(__file__).resolve().parents[1] / 'configs/d0b4b_inference_contract.json'
    return json.loads(path.read_text())


@pytest.fixture(scope='module')
def schedule(contract):
    settings = contract['forward_schedule']
    return cosine_schedule(timesteps=settings['T'], offset=settings['cosine_offset'],
                           max_beta=settings['max_beta'])


def test_design_status_and_exact_provenance(contract):
    assert contract['contract_version'] == 1
    assert contract['stage'] == 'D0-B4b'
    assert contract['status'] == 'DESIGN_LOCKED_PRE_TRAINING'
    assert contract['artifact_type'] == 'methodological design, not an experiment result'
    for key in ('scientific_training', 'validation_used_to_choose_inference_hyperparameters',
                'gt_used_at_inference', 'model_selection'):
        assert contract[key] is False
    assert contract['locked_source_provenance'] == {
        'e1c_checkpoint_sha256': 'cf5a04a2d845cecb45a0ea87d617abad0afab9ad4264abce8342a6045563ee53',
        'd0a_normalization_contract_sha256': 'bca5c0f8f1617d73fc4d0850fe5b6f45e4b2bcbacbb4018f1d5d825de49ad792',
        'd0b0_patch_support_manifest_sha256': '9c4ba67fc195537a5500229722ce55e4772ea5b7d135b351b3a2fc4b857f44b9',
        'd0b3_corruption_decision_sha256': '095e1f9671411eca0d12a2f834833d78477a14664504bde3e6e1b46766c6cc2f',
        'd0b4a_microbatch_decision_sha256': '07c268d97c7a39f389548344be7245cee4eba3a390a91d29b6533ab466c5254b',
        'd0b4a_freeze_commit': '6772b023ab9c1d870f258f45d030b958d1f9b19b',
    }
    def check_no_results(value):
        if isinstance(value, dict):
            for key, item in value.items():
                assert key not in {'results', 'performance', 'validation_scores', 'diffusion_checkpoint_path'}
                check_no_results(item)
        elif isinstance(value, list):
            for item in value:
                check_no_results(item)
    check_no_results(contract)


def test_schedule_uses_existing_post_clamp_cumulative_product(contract, schedule):
    settings = contract['forward_schedule']
    assert settings['implementation'] == 'models.latent_diffusion_unet3d.cosine_schedule'
    assert settings['prediction_type'] == 'epsilon'
    assert (settings['T'], settings['cosine_offset'], settings['max_beta']) == (1000, 0.008, 0.999)
    assert (settings['timestep_index_min'], settings['timestep_index_max']) == (0, 999)
    assert settings['semantics'] == [
        'beta is derived from consecutive normalized cosine alpha-bar points, then clamped to max_beta',
        'alpha = 1 - beta', 'alpha_bar = cumulative product(alpha)']
    assert schedule['alpha_bar'].device.type == 'cpu'
    assert torch.equal(schedule['alpha'], 1 - schedule['beta'])
    assert torch.equal(schedule['alpha_bar'], torch.cumprod(schedule['alpha'], dim=0))


@pytest.mark.parametrize('timestep,target,locked', [
    (327, 0.75, 0.7501165866851807),
    (495, 0.50, 0.5000776052474976),
    (663, 0.25, 0.25003883242607117),
])
def test_unique_nearest_schedule_points(contract, schedule, timestep, target, locked):
    points = {p['timestep']: p for p in contract['partial_noise']['points']}
    assert set(points) == {327, 495, 663}
    point = points[timestep]
    assert point['target_alpha_bar'] == target
    assert point['alpha_bar'] == locked
    assert schedule['alpha_bar'][timestep].item() == pytest.approx(locked, rel=0, abs=1e-8)
    distances = (schedule['alpha_bar'].double() - target).abs()
    assert distances.argmin().item() == timestep
    assert (distances == distances.min()).sum().item() == 1


def test_primary_noise_strength(contract, schedule):
    noise = contract['partial_noise']
    assert noise['primary_timestep'] == 495
    assert noise['primary_alpha_bar'] == 0.5000776052474976
    assert noise['primary_noise_variance'] == 0.49992239475250244
    assert noise['primary_noise_variance'] == pytest.approx(1 - schedule['alpha_bar'][495].item(), rel=0, abs=1e-8)
    assert noise['reason'] == 'nearest locked cosine-schedule timestep to alpha_bar = 0.50'
    assert noise['selection_basis'] == 'schedule-defined and pre-specified, not validation-selected'


def test_exact_ddim_timestep_sequence(contract):
    ddim = contract['ddim']
    assert ddim['sampler'] == 'deterministic DDIM'
    assert ddim['eta'] == 0.0
    assert ddim['model_evaluations'] == 50
    assert (ddim['start_timestep'], ddim['end_timestep']) == (495, 0)
    assert ddim['sequence_definition'] == 'round(linspace(495, 0, 50))'
    sequence = ddim['resolved_timesteps']
    expected = torch.linspace(495, 0, 50, dtype=torch.float64).round().long().tolist()
    assert sequence == expected
    assert all(type(t) is int for t in sequence)
    assert len(sequence) == len(set(sequence)) == 50
    assert sequence[0] == 495 and sequence[-1] == 0
    assert all(a > b for a, b in zip(sequence, sequence[1:]))


def test_ddim_equations_and_terminal_evaluation(contract):
    ddim = contract['ddim']
    assert ddim['z0_hat'] == '(z_t - sqrt(1-alpha_bar_t) * epsilon_hat) / sqrt(alpha_bar_t)'
    assert ddim['z_prev'] == 'sqrt(alpha_bar_prev) * z0_hat + sqrt(1-alpha_bar_prev) * epsilon_hat'
    assert ddim['reverse_transition_condition'] == 't_prev >= 0; t_prev is the next entry in resolved_timesteps, not necessarily t-1'
    assert ddim['terminal_model_evaluation'] == 'At t=0: Zprior = z0_hat, followed by the required support masking; do not take a further scheduled transition.'
    assert ddim['stochastic_reverse_noise'] is False


@pytest.mark.parametrize('start', [327, 663])
def test_frozen_sensitivity_ddim_sequences(contract, start):
    policies = {entry['start_timestep']: entry for entry in contract['sensitivity_ddim']}
    assert set(policies) == {327, 663}
    policy = policies[start]
    assert policy['role'] == 'SECONDARY_ONLY'
    assert policy['eligible_for_winner_selection'] is False
    assert policy['sampler'] == 'deterministic DDIM'
    assert policy['eta'] == 0.0
    assert policy['model_evaluations'] == 50
    assert policy['end_timestep'] == 0
    assert policy['sequence_definition'] == 'round(linspace(start_timestep, 0, 50))'
    sequence = policy['resolved_timesteps']
    assert sequence == torch.linspace(start, 0, 50, dtype=torch.float64).round().long().tolist()
    assert len(sequence) == len(set(sequence)) == 50
    assert all(type(t) is int for t in sequence)
    assert sequence[0] == start and sequence[-1] == 0
    assert all(a > b for a, b in zip(sequence, sequence[1:]))
    assert contract['partial_noise']['primary_timestep'] == 495
    assert contract['ddim']['start_timestep'] == 495


def test_original_mri_source_and_gt_exclusion(contract):
    source = contract['inference_source']
    assert source['pipeline'] == ['original unmodified multimodal MRI', 'frozen E1-C Encoder',
        'frozen anatomy projector P_a', 'Za', 'fixed D0-A per-channel standardization', 'z0']
    assert source['d0b3_corruption_role'] == 'TRAINING_ONLY'
    for key in ('pathology_corruption_at_inference', 'd0b3_corruption_performed_at_inference',
                'segmentation_mask_available_to_model', 'segmentation_mask_consumed_by_model'):
        assert source[key] is False
    assert source['gt_permissions'] == dict(construct_z0=False, condition_diffusion=False,
        define_support=False, affect_ddim=False, construct_deviation_maps=False,
        offline_evaluation_of_already_created_deviation_outputs_only=True)


def test_support_masking_at_every_stage(contract):
    support = contract['support']
    assert support['raw_definition'] == 'any raw modality nonzero'
    assert support['raw_patch_shape'] == [4, 96, 96, 96]
    assert support['source'] == 'Raw MRI patch before normalization, never normalized MRI or GT segmentation.'
    assert support['latent_mapping'] == 'Exact non-overlapping 8x8x8 raw-voxel block occupancy'
    assert (support['occupancy_comparison'], support['occupancy_threshold']) == ('>=', 0.50)
    assert support['support_mask_shape'] == ['B', 1, 12, 12, 12]
    assert support['latent_shape'] == ['B', 256, 12, 12, 12]
    assert support['computational_masking_only'] is True
    assert support['learned_conditioning'] is False
    assert support['z0_supported'] == 'z0 * support_mask'
    assert support['epsilon_supported'] == 'epsilon * support_mask'
    assert support['initial_partial_noising'] == '(sqrt(alpha_bar_t) * z0_supported + sqrt(1-alpha_bar_t) * epsilon_supported) * support_mask'
    assert support['before_every_network_evaluation'] == 'z_current = z_current * support_mask'
    assert support['after_every_reverse_latent'] == 'z_prev = z_prev * support_mask before the next network evaluation'
    assert support['final_prior'] == 'Zprior = Zprior * support_mask'
    assert support['final_prior_zero_outside_support'] is True


def test_patch_specific_reproducible_seed_contract(contract):
    randomness = contract['randomness']
    assert randomness['inference_global_seed'] == 20260923
    assert randomness['primary_noise_replicates'] == 4
    assert randomness['replicate_indices'] == [0, 1, 2, 3]
    assert randomness['sample_identity_fields'] == ['case_id', 'crop_start_d', 'crop_start_h',
        'crop_start_w', 'crop_stop_d', 'crop_stop_h', 'crop_stop_w']
    assert randomness['case_id_alone_is_sufficient'] is False
    assert randomness['purpose'] == 'partial_noise'
    assert randomness['seed_payload'] == '[inference_global_seed, sample_identity, replicate_index, purpose]'
    assert randomness['seed_serialization'] == "UTF-8 encoding of compact JSON with separators=(',', ':'), ensure_ascii=false, no whitespace or trailing newline; seed and replicate index are integers."
    assert randomness['seed_hash'] == 'SHA256'
    assert randomness['seed_integer'] == 'int(sha256(payload_bytes).hexdigest()[:16], 16) & ((1 << 63) - 1)'
    assert 'separate CPU torch.Generator' in randomness['noise_generation']
    assert "dtype=torch.float32, device='cpu'" in randomness['noise_generation']
    assert randomness['invariant_to'] == ['batch size', 'batch ordering', 'device batching/partitioning']
    assert randomness['actual_case_specific_seeds_in_contract'] is False


def test_primary_deviation_order_and_invalid_outside_support(contract):
    deviation = contract['primary_deviation']
    assert deviation['aggregation'] == 'mean_replicates(mean_channels(abs(z0-Zprior_k) * support_mask)) * support_mask'
    assert deviation['per_channel_absolute_difference'] == 'R_k(c,x) = abs(z0(c,x) - Zprior_k(c,x)) * support_mask(x)'
    assert deviation['final_support_masking'] == 'DeltaZ = DeltaZ * support_mask'
    assert deviation['per_replicate_map'] == 'D_k(x) = mean_c R_k(c,x)'
    assert deviation['primary_map'] == 'DeltaZ(x) = mean_k D_k(x)'
    assert deviation['absolute_difference_before_replicate_averaging'] is True
    assert deviation['forbidden_primary_aggregation'] == 'abs(z0 - mean_k(Zprior_k))'
    assert deviation['channel_aggregation'] == deviation['replicate_aggregation'] == 'arithmetic mean'
    assert (deviation['channel_count'], deviation['replicate_count']) == (256, 4)
    assert deviation['output_shape'] == ['B', 1, 12, 12, 12]
    assert deviation['outside_support_value'] == 0
    assert deviation['outside_support_is_valid_observation'] is False
    assert deviation['outside_support_in_scientific_statistics'] is False


def test_nonzero_unsupported_z0_cannot_affect_primary_deviation(contract):
    # CPU algebra demonstration of the locked formula, not an inference implementation.
    test_primary_deviation_order_and_invalid_outside_support(contract)
    mask = torch.zeros(1, 1, 12, 12, 12)
    mask[..., :6] = 1
    z0 = torch.full((1, 256, 12, 12, 12), 2.0)
    priors = torch.tensor([-1., 1., -3., 3.]).reshape(4, 1, 1, 1, 1, 1).expand(4, *z0.shape) * mask
    baseline = ((z0 - priors).abs() * mask).mean(dim=2, keepdim=True).mean(dim=0) * mask
    arbitrary = torch.linspace(-1e6, 1e6, z0.numel()).reshape_as(z0)
    changed_z0 = torch.where(mask.bool(), z0, arbitrary)
    assert torch.count_nonzero(changed_z0 * (1 - mask)) > 0
    changed = ((changed_z0 - priors).abs() * mask).mean(dim=2, keepdim=True).mean(dim=0) * mask
    assert torch.equal(changed, baseline)
    assert changed.shape == (1, 1, 12, 12, 12)
    assert torch.equal(changed, 2.5 * mask)
    assert torch.count_nonzero(changed * (1 - mask)) == 0


def test_noise_is_paired_across_sensitivity_timesteps(contract):
    randomness = contract['randomness']
    assert randomness['same_noise_realization_across_sensitivity_timesteps'] is True
    assert randomness['paired_noise_timesteps'] == [327, 495, 663]
    assert randomness['paired_noise_identity'] == 'For a fixed sample identity and replicate index, use the same epsilon realization at t=327, t=495, and t=663.'
    assert randomness['paired_noise_reason'] == 'paired sensitivity comparison isolates partial-noise strength while holding the underlying Gaussian noise realization fixed'
    assert randomness['diffusion_timestep_in_seed_payload'] is False
    assert randomness['seed_timestep_exclusion'] == 'Diffusion timestep is NOT part of the partial-noise seed payload.'
    assert randomness['seed_payload'] == '[inference_global_seed, sample_identity, replicate_index, purpose]'


def test_sensitivity_and_other_outputs_are_secondary_nonselective(contract):
    noise = contract['partial_noise']
    assert noise['sensitivity_may_redefine_primary_retrospectively'] is False
    for point in noise['points']:
        assert point['eligible_for_winner_selection'] is False
        assert point['secondary'] is (point['timestep'] != 495)
        assert point['role'] == {327: 'lower-noise sensitivity point', 495: 'PRIMARY',
                                 663: 'higher-noise sensitivity point'}[point['timestep']]
    secondary = contract['secondary_outputs']
    assert secondary['role'] == 'SECONDARY_ONLY'
    assert secondary['outputs'] == ['per-replicate channel-mean absolute deviation maps',
        'full per-channel absolute difference tensor when storage permits',
        'replicate-wise variance of D_k(x)', 'RMS/L2 channel deviation', 'support mask',
        'sensitivity maps at t=327 and t=663']
    assert secondary['replace_primary_deltaz'] is False
    assert secondary['sensitivity_results_may_select_primary_timestep'] is False


def test_pretraining_lock_and_existing_ema_checkpoint_rule(contract):
    lock = contract['design_lock']
    assert lock['hyperparameters'] == ['primary_timestep', 'sensitivity_timesteps', 'eta',
        'model_evaluations', 'primary_noise_replicates', 'inference_global_seed',
        'seed_derivation', 'paired_noise_across_sensitivity_timesteps',
        'support_masking_semantics', 'DeltaZ_aggregation']
    assert lock['frozen_before_scientific_training'] is True
    assert lock['frozen_before_observing_validation_reconstruction_performance'] is True
    assert lock['selected_from_validation_reconstruction_performance'] is False
    weights = contract['inference_weights']
    assert weights['weight_type'] == 'EMA'
    assert weights['checkpoint_source'] == 'checkpoint selected by the already-locked validation criterion'
    assert weights['checkpoint_selection_criterion'] == 'lowest mean case-level validation support-masked epsilon MSE'
    assert weights['defines_new_checkpoint_selection_rule'] is False
    assert weights['checkpoint_selection_uses_predeclared_validation_epsilon_mse'] is True
    assert weights['inference_hyperparameters_use_validation_reconstruction_performance'] is False
