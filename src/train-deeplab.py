#!/usr/bin/env python3
# src/train_deeplab.py
# Standalone DeepLabV3 training with 512×512 input.
# Does NOT modify any existing code.

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

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os as _os

# ---------- Dataset Class (copied from dataset.py) ----------
class OilSpillDataset(Dataset):
    def __init__(self, npz_path, transform=None):
        data = np.load(npz_path)
        self.images = data['images'].astype(np.float32)
        self.masks = data['masks'].astype(np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].copy()
        mask = self.masks[idx].copy()
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask).long()
        return image, mask

# ---------- Transform Functions (with image_size) ----------
def get_transforms_512(train=True, image_size=512):
    """
    Transform pipeline for 512×512 images.
    """
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomGamma(p=0.3, gamma_limit=(80, 120)),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ])

# ---------- DataLoader ----------
def get_dataloaders_512(data_dir=None, batch_size=4, num_workers=2):
    if data_dir is None:
        script_dir = Path(__file__).parent
        data_dir = script_dir.parent / "data" / "processed"
    else:
        data_dir = Path(data_dir)

    train_path = data_dir / "train.npz"
    val_path   = data_dir / "val.npz"
    test_path  = data_dir / "test.npz"

    for p in [train_path, val_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    train_dataset = OilSpillDataset(str(train_path), transform=get_transforms_512(train=True, image_size=512))
    val_dataset   = OilSpillDataset(str(val_path),   transform=get_transforms_512(train=False, image_size=512))
    test_dataset  = OilSpillDataset(str(test_path),  transform=get_transforms_512(train=False, image_size=512))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"📊 DataLoaders (512×512):")
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val:   {len(val_dataset)} samples")
    print(f"   Test:  {len(test_dataset)} samples")
    return train_loader, val_loader, test_loader

# ---------- Loss Functions ----------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        pred_flat = pred[:, 1, :, :].reshape(-1)
        target_flat = target.reshape(-1)
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice

def combined_loss(pred, target, class_weights, bce_weight=0.5, dice_weight=0.5):
    target = target.long()
    bce = nn.CrossEntropyLoss(weight=class_weights)(pred, target)
    dice = DiceLoss()(pred, target)
    return bce_weight * bce + dice_weight * dice

# ---------- Helpers ----------
def calculate_class_weights(train_loader, device):
    oil_pixels = 0
    total_pixels = 0
    for _, masks in train_loader:
        oil_pixels += (masks == 1).sum().item()
        total_pixels += masks.numel()
    oil_ratio = oil_pixels / total_pixels
    sea_ratio = 1 - oil_ratio
    weights = torch.tensor([1.0, sea_ratio / oil_ratio], dtype=torch.float32)
    weights = weights / weights.sum() * 2
    print(f"📊 Class distribution: Sea={sea_ratio:.4f}, Oil={oil_ratio:.4f}")
    print(f"📊 Class weights: {weights.tolist()}")
    return weights.to(device)

# ---------- Training ----------
def train_one_epoch(model, loader, optimizer, scaler, device, class_weights):
    model.train()
    total_loss = 0.0
    for images, masks in tqdm(loader, desc='Training', leave=False):
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
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
    iou_list, dice_list = [], []
    with torch.no_grad():
        for images, masks in tqdm(loader, desc='Validating', leave=False):
            images = images.to(device)
            masks = masks.to(device)
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                pred = model(images)
            pred_mask = torch.argmax(pred, dim=1)
            inter = ((pred_mask == 1) & (masks == 1)).sum().float()
            union = ((pred_mask == 1) | (masks == 1)).sum().float()
            iou = (inter + 1e-6) / (union + 1e-6)
            iou_list.append(iou.item())
            dice = (2 * inter + 1e-6) / (pred_mask[pred_mask == 1].numel() + masks[masks == 1].numel() + 1e-6)
            dice_list.append(dice.item())
    return np.mean(iou_list), np.mean(dice_list)

def test_model(model, loader, device):
    return validate(model, loader, device)

# ---------- Main ----------
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = get_dataloaders_512(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    class_weights = calculate_class_weights(train_loader, device)

    model = smp.DeepLabV3(
        encoder_name=args.encoder,
        encoder_weights='imagenet',
        in_channels=3,
        classes=2,
    ).to(device)
    print(f"📊 Model: DeepLabV3 with {args.encoder} backbone")
    print(f"📊 Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    best_iou = 0.0
    best_epoch = -1
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\n🔁 Epoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, class_weights)
        val_iou, val_dice = validate(model, val_loader, device)
        print(f"📈 Train Loss: {train_loss:.4f} | Val IoU: {val_iou:.4f} | Val Dice: {val_dice:.4f}")
        scheduler.step(val_iou)

        if val_iou > best_iou:
            best_iou = val_iou
            best_epoch = epoch
            patience_counter = 0
            checkpoint_path = checkpoint_dir / 'deeplabv3_best_model.pt'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✅ Best model saved to {checkpoint_path} (IoU = {best_iou:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"⏹️ Early stopping at epoch {epoch}")
                break

    print(f"\n🏆 Best IoU: {best_iou:.4f} at epoch {best_epoch}")

    checkpoint_path = checkpoint_dir / 'deeplabv3_best_model.pt'
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path))
        test_iou, test_dice = test_model(model, test_loader, device)
        print(f"\n✅ Test IoU: {test_iou:.4f}")
        print(f"✅ Test Dice: {test_dice:.4f}")

        with open(checkpoint_dir / 'deeplabv3_metrics.txt', 'w') as f:
            f.write(f"Model: deeplabv3\n")
            f.write(f"Encoder: {args.encoder}\n")
            f.write(f"Best epoch: {best_epoch}\n")
            f.write(f"Best validation IoU: {best_iou:.4f}\n")
            f.write(f"Test IoU: {test_iou:.4f}\n")
            f.write(f"Test Dice: {test_dice:.4f}\n")
            f.write(f"Batch size: {args.batch_size}\n")
            f.write(f"Learning rate: {args.lr}\n")
            f.write(f"Image size: 512\n")
    else:
        print("❌ Best checkpoint not found, skipping test.")

    print(f"\n✅ Training complete! Checkpoints saved to: {checkpoint_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DeepLabV3 for oil spill detection")
    parser.add_argument('--encoder', type=str, default='resnet34',
                        help='Encoder backbone (resnet34, resnet50, mobilenet_v2)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size (512×512 needs less memory)')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Path to processed .npz files')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--num_workers', type=int, default=2, help='DataLoader workers')
    args = parser.parse_args()

    main(args)
