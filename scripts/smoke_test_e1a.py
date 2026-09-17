"""One synthetic E1-A optimization step and checkpoint round-trip; no real data."""
import argparse
from pathlib import Path
import torch
from models.baseline_unet3d import BaselineUNet3D
from models.latent_split_unet3d import LatentSplitUNet3D
from losses.segmentation_loss import SegmentationLoss
from training.checkpointing import save_checkpoint, load_checkpoint
from training.artifacts import snapshot, write_json


def run(device: str, output_dir: str, patch_size: int = 96, base_channels: int = 32) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Smoke output directory must be empty; refusing to overwrite artifacts.')
    torch.manual_seed(42)
    model = LatentSplitUNet3D(base_channels=base_channels).to(device)
    baseline_count = sum(p.numel() for p in BaselineUNet3D(base_channels=base_channels).parameters())
    count = sum(p.numel() for p in model.parameters())
    config = dict(experiment_name='E1-A synthetic smoke', model=dict(name='latent_split_unet3d',
                  in_channels=4, out_channels=4, base_channels=base_channels, latent_projection='conv1x1x1', latent_channels=8*base_channels),
                  train=dict(seed=42, device=device, learning_rate=1e-3, weight_decay=1e-5, batch_size=1, epochs=1),
                  data=dict(synthetic=True, patch_size=[patch_size]*3), optimizer='AdamW',
                  segmentation_loss=dict(name='Dice+BCE', dice_weight=1., bce_weight=1.))
    snapshot(output, config, count)
    image = torch.randn(1, 4, patch_size, patch_size, patch_size, device=device)
    result = model.forward_with_latents(image)
    target = torch.zeros_like(result.logits)
    target[:, 1, patch_size//4:3*patch_size//4, patch_size//4:3*patch_size//4, patch_size//4:3*patch_size//4] = 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss = SegmentationLoss()(result.logits, target)
    loss.backward()
    assert torch.isfinite(loss)
    for module in (model.encoder, model.decoder, model.projection.disease_projector):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
    assert all(p.grad is None for p in model.projection.anatomy_projector.parameters())
    optimizer.step()
    save_checkpoint(output/'smoke.pt', model, optimizer, 1, config, seed=42)
    restored = LatentSplitUNet3D(base_channels=base_channels).to(device)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3, weight_decay=1e-5)
    load_checkpoint(output/'smoke.pt', restored, restored_optimizer)
    assert all(torch.equal(value, restored.state_dict()[key]) for key, value in model.state_dict().items())
    report = dict(model_parameter_count=count, baseline_parameter_count=baseline_count, parameter_delta=count-baseline_count,
                  logits_shape=list(result.logits.shape), anatomy_shape=list(result.anatomy_latent.shape),
                  disease_shape=list(result.disease_latent.shape), loss=loss.item(), checkpoint_round_trip=True)
    write_json(output/'summary.json', report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output-dir', default='outputs/e1a_latent_split_smoke')
    args = parser.parse_args()
    print(run(args.device, args.output_dir))


if __name__ == '__main__':
    main()
