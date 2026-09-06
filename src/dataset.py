# src/dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

class OilSpillDataset(Dataset):
    """PyTorch Dataset for oil spill detection."""
    
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
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            # Convert to tensor manually if no transform
            # (H, W, C) -> (C, H, W)
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask).long()
        
        return image, mask

def get_transforms(train=True):
    """
    Get augmentation transforms.
    
    Args:
        train: If True, apply heavy augmentations
    """
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomGamma(p=0.3, gamma_limit=(80, 120)),
            # Fixed: Use correct syntax for GaussNoise
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            # Normalize images to [0, 1] range first
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ])

def get_dataloaders(data_dir=None, batch_size=8, num_workers=2):
    """
    Create train, val, test dataloaders.
    
    Args:
        data_dir: Path to folder containing .npz files. 
                  If None, uses default relative path.
    """
    if data_dir is None:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level to project root, then into data/processed
        data_dir = os.path.join(os.path.dirname(script_dir), "data", "processed")
    
    train_path = os.path.join(data_dir, "train.npz")
    val_path = os.path.join(data_dir, "val.npz")
    test_path = os.path.join(data_dir, "test.npz")
    
    # Check if files exist
    for path in [train_path, val_path, test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
    
    train_dataset = OilSpillDataset(
        train_path,
        transform=get_transforms(train=True)
    )
    val_dataset = OilSpillDataset(
        val_path,
        transform=get_transforms(train=False)
    )
    test_dataset = OilSpillDataset(
        test_path,
        transform=get_transforms(train=False)
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    print(f"📊 DataLoaders created:")
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val:   {len(val_dataset)} samples")
    print(f"   Test:  {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader

# Quick test
if __name__ == "__main__":
    try:
        train_loader, val_loader, test_loader = get_dataloaders()
        
        # Test one batch
        images, masks = next(iter(train_loader))
        print(f"\n✅ Test batch:")
        print(f"   Images shape: {images.shape}")
        print(f"   Masks shape:  {masks.shape}")
        print(f"   Image min/max: {images.min():.3f} / {images.max():.3f}")
        print(f"   Mask unique values: {torch.unique(masks)}")
        print(f"   Oil pixel count: {(masks == 1).sum().item()}")
    except Exception as e:
        print(f"❌ Error: {e}")
