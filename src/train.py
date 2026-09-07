#!/usr/bin/env python3
# src/train.py

import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import segmentation_models_pytorch as smp
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Import your dataset module
from dataset import get_dataloaders


# ---------- Loss Functions ----------
class DiceLoss(nn.Module):
    """Dice loss for binary segmentation."""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # pred: (B, C, H, W), target: (B, H, W)
        pred = torch.softmax(pred, dim=1)
        pred_flat = pred[:, 1, :, :].reshape(-1)
        target_flat = target.reshape(-1)

        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


def combined_loss(pred, target, class_weights, bce_weight=0.5, dice_weight=0.5):
    """
    Combined BCE + Dice loss.
    Args:
        pred: Model predictions (B, C, H, W)
        target: Ground truth masks (B, H, W)
        class_weights: Tensor of shape (2,) for background and oil
    """
    # Cast target to Long (required for CrossEntropyLoss)
    target = target.long()
    bce = nn.CrossEntropyLoss(weight=class_weights)(pred, target)
    dice = DiceLoss()(pred, target)
    return bce_weight * bce + dice_weight * dice


# ---------- Helpers ----------
def calculate_class_weights(train_loader, device):
    """
    Compute inverse frequency weights for binary segmentation.
    Returns tensor of shape (2,) for background and oil.
    """
    oil_pixels = 0
    total_pixels = 0
    for _, masks in train_loader:
        oil_pixels += (masks == 1).sum().item()
        total_pixels += masks.numel()

    oil_ratio = oil_pixels / total_pixels
    sea_ratio = 1 - oil_ratio

    # Inverse frequency, then normalize so sum = 2
    weights = torch.tensor([1.0, sea_ratio / oil_ratio], dtype=torch.float32)
    weights = weights / weights.sum() * 2
    print(f"📊 Class distribution: Sea={sea_ratio:.4f}, Oil={oil_ratio:.4f}")
    print(f"📊 Class weights: {weights.tolist()}")
    return weights.to(device)


def get_model(model_name, encoder_name, in_channels=3, classes=2):
    """Return the requested segmentation model."""
    model_map = {
        'unet': smp.Unet,
        'deeplabv3': smp.DeepLabV3,
        'fpn': smp.FPN,
    }
    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(model_map.keys())}")

    return model_map[model_name](
        encoder_name=encoder_name,
        encoder_weights='imagenet',
        in_channels=in_channels,
        classes=classes,
    )


# ---------- Training ----------
def train_one_epoch(model, loader, optimizer, scaler, device, class_weights):
    model.train()
    total_loss = 0.0
    for images, masks in tqdm(loader, desc='Training', leave=False):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        # Use new autocast syntax for AMP
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            pred = model(images)
            loss = combined_loss(pred, masks, class_weights)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, device):
    model.eval()
    iou_list = []
    dice_list = []
    with torch.no_grad():
        for images, masks in tqdm(loader, desc='Validating', leave=False):
            images = images.to(device)
            masks = masks.to(device)

            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                pred = model(images)
            pred_mask = torch.argmax(pred, dim=1)

            # IoU for oil class (class 1)
            inter = ((pred_mask == 1) & (masks == 1)).sum().float()
            union = ((pred_mask == 1) | (masks == 1)).sum().float()
            iou = (inter + 1e-6) / (union + 1e-6)
            iou_list.append(iou.item())

            # Dice for oil class
            dice = (2 * inter + 1e-6) / (pred_mask[pred_mask == 1].numel() + masks[masks == 1].numel() + 1e-6)
            dice_list.append(dice.item())

    return np.mean(iou_list), np.mean(dice_list)


def test_model(model, loader, device):
    """Final evaluation on test set."""
    iou, dice = validate(model, loader, device)
    return iou, dice


# ---------- Main ----------
def main(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")

    # Output directories
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # DataLoaders (your dataset.py will handle paths)
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Class weights (computed from training set)
    class_weights = calculate_class_weights(train_loader, device)

    # Model
    model = get_model(args.model, args.encoder).to(device)
    print(f"📊 Model: {args.model} with {args.encoder} backbone")
    print(f"📊 Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # GradScaler for AMP
    scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    # Training state
    best_iou = 0.0
    best_epoch = -1
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\n🔁 Epoch {epoch}/{args.epochs}")

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, class_weights)

        # Validate
        val_iou, val_dice = validate(model, val_loader, device)
        print(f"📈 Train Loss: {train_loss:.4f} | Val IoU: {val_iou:.4f} | Val Dice: {val_dice:.4f}")

        # Scheduler step
        scheduler.step(val_iou)

        # Save best model with model name prefix
        if val_iou > best_iou:
            best_iou = val_iou
            best_epoch = epoch
            patience_counter = 0
            checkpoint_path = checkpoint_dir / f'{args.model}_best_model.pt'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✅ Best model saved to {checkpoint_path} (IoU = {best_iou:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"⏹️ Early stopping at epoch {epoch}")
                break

    print(f"\n🏆 Best IoU: {best_iou:.4f} at epoch {best_epoch}")

    # Load best model and evaluate on test set
    checkpoint_path = checkpoint_dir / f'{args.model}_best_model.pt'
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint {checkpoint_path} not found. Skipping test.")
        return
    model.load_state_dict(torch.load(checkpoint_path))
    test_iou, test_dice = test_model(model, test_loader, device)
    print(f"\n✅ Test IoU: {test_iou:.4f}")
    print(f"✅ Test Dice: {test_dice:.4f}")

    # Save metrics with model name prefix
    metrics_path = checkpoint_dir / f'{args.model}_metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Encoder: {args.encoder}\n")
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Best validation IoU: {best_iou:.4f}\n")
        f.write(f"Test IoU: {test_iou:.4f}\n")
        f.write(f"Test Dice: {test_dice:.4f}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Learning rate: {args.lr}\n")
        f.write(f"Weight decay: {args.weight_decay}\n")
        f.write(f"Patience: {args.patience}\n")

    print(f"\n✅ Training complete! Checkpoints and metrics saved to: {checkpoint_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train oil spill detection model")
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'deeplabv3', 'fpn'],
                        help='Model architecture')
    parser.add_argument('--encoder', type=str, default='resnet34',
                        help='Encoder backbone (e.g., resnet34, resnet50, efficientnet-b3)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Path to processed .npz files')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--num_workers', type=int, default=2, help='DataLoader workers')
    args = parser.parse_args()

    main(args)
