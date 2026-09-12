"""Synthetic default 96-cubed model check; no real data or optimizer."""
import argparse
import torch
from models.baseline_unet3d import BaselineUNet3D
from losses.segmentation_loss import SegmentationLoss


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backward", action="store_true", help="Also compute synthetic loss and gradients (uses more memory).")
    args = parser.parse_args()
    torch.manual_seed(42)
    model = BaselineUNet3D()
    image = torch.randn(1, 4, 96, 96, 96)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}", flush=True)
    with torch.set_grad_enabled(args.backward):
        output = model(image)
        print(f"Input shape: {tuple(image.shape)}")
        print(f"Output shape: {tuple(output.shape)}")
        print(f"Output min/max: {output.detach().min().item():.6f}/{output.detach().max().item():.6f}", flush=True)
        assert output.shape == image.shape and torch.isfinite(output).all()
        if args.backward:
            target = torch.zeros_like(output)
            target[:, 1, 24:72, 24:72, 24:72] = 1
            loss = SegmentationLoss()(output, target)
            loss.backward()
            assert torch.isfinite(loss)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            print(f"Synthetic loss: {loss.item():.6f}; gradients finite")


if __name__ == "__main__":
    main()
