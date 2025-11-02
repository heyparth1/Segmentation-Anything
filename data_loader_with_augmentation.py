
import os
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json

print("="*60)
print("PYTORCH DATALOADER WITH AUGMENTATION")
print("="*60)

class IDCDataset(Dataset):
    """
    IDC Breast Histopathology Dataset
    Handles loading, preprocessing, and augmentation
    """
    def __init__(self, root_dir, patient_ids, transform=None, target_size=256):
        """
        Args:
            root_dir: Root directory of dataset
            patient_ids: List of patient IDs for this split
            transform: Albumentations transform pipeline
            target_size: Target image size (will upscale from 50x50)
        """
        self.root_dir = Path(root_dir)
        self.patient_ids = patient_ids
        self.transform = transform
        self.target_size = target_size
        
        # Build image list
        self.samples = []
        self._build_dataset()
        
        print(f"Loaded {len(self.samples)} images from {len(patient_ids)} patients")
    
    def _build_dataset(self):
        """Build list of (image_path, label) tuples"""
        for patient_id in self.patient_ids:
            patient_path = self.root_dir / patient_id
            
            # Load negative samples (label=0)
            negative_path = patient_path / "0"
            if negative_path.exists():
                for img_path in negative_path.glob("*.png"):
                    self.samples.append((str(img_path), 0))
            
            # Load positive samples (label=1)
            positive_path = patient_path / "1"
            if positive_path.exists():
                for img_path in positive_path.glob("*.png"):
                    self.samples.append((str(img_path), 1))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Upscale to target size
        image = cv2.resize(image, (self.target_size, self.target_size), 
                          interpolation=cv2.INTER_CUBIC)
        
        # Apply augmentation
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label
    
    def get_class_distribution(self):
        """Get class distribution for weighted sampling"""
        labels = [label for _, label in self.samples]
        return np.bincount(labels)

def get_train_transforms(target_size=256):
    """Augmentation pipeline for training"""
    return A.Compose([
        # Geometric transforms
        A.Rotate(limit=180, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, 
                          rotate_limit=45, p=0.5),
        
        # Color transforms
        A.RandomBrightnessContrast(brightness_limit=0.2, 
                                   contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, 
                            sat_shift_limit=30, 
                            val_shift_limit=20, p=0.5),
        
        # Noise and blur
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        
        # Elastic deformation (simulate tissue variability)
        A.ElasticTransform(alpha=50, sigma=5, p=0.3),
        
        # Normalization
        A.Normalize(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]),
        
        ToTensorV2()
    ])

def get_val_transforms(target_size=256):
    """Minimal transforms for validation/test"""
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def create_weighted_sampler(dataset):
    """
    Create weighted sampler for balanced batches
    Handles class imbalance by oversampling minority class
    """
    class_counts = dataset.get_class_distribution()
    
    # Calculate weights (inverse of class frequency)
    class_weights = 1.0 / class_counts
    
    # Assign weight to each sample
    sample_weights = [class_weights[label] for _, label in dataset.samples]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler

def create_dataloaders(root_dir, splits_path, batch_size=32, 
                      num_workers=4, target_size=256):
    """
    Create train, validation, and test dataloaders
    
    Args:
        root_dir: Root directory of dataset
        splits_path: Path to dataset_splits.json
        batch_size: Batch size for training
        num_workers: Number of worker processes
        target_size: Target image size
    """
    # Load splits
    with open(splits_path, 'r') as f:
        splits = json.load(f)
    
    train_patients = splits['train']
    val_patients = splits['val']
    test_patients = splits['test']
    
    print(f"\n{'='*60}")
    print("CREATING DATALOADERS")
    print("="*60)
    print(f"Train patients: {len(train_patients)}")
    print(f"Val patients: {len(val_patients)}")
    print(f"Test patients: {len(test_patients)}")
    
    # Create datasets
    train_dataset = IDCDataset(
        root_dir=root_dir,
        patient_ids=train_patients,
        transform=get_train_transforms(target_size),
        target_size=target_size
    )
    
    val_dataset = IDCDataset(
        root_dir=root_dir,
        patient_ids=val_patients,
        transform=get_val_transforms(target_size),
        target_size=target_size
    )
    
    test_dataset = IDCDataset(
        root_dir=root_dir,
        patient_ids=test_patients,
        transform=get_val_transforms(target_size),
        target_size=target_size
    )
    
    # Print class distributions
    print(f"\nClass Distribution:")
    print(f"  Train: {train_dataset.get_class_distribution()}")
    print(f"  Val:   {val_dataset.get_class_distribution()}")
    print(f"  Test:  {test_dataset.get_class_distribution()}")
    
    # Create weighted sampler for training
    train_sampler = create_weighted_sampler(train_dataset)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\nDataLoader Info:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print(f"  Batch size: {batch_size}")
    
    return train_loader, val_loader, test_loader

# Test dataloader
if __name__ == "__main__":
    DATASET_PATH = r"D:\projs\segment_anything\data\IDC_regular_ps50_idx5"
    SPLITS_PATH = "dataset_splits.json"
    
    if not os.path.exists(SPLITS_PATH):
        print(f"❌ Please run 03_create_train_val_test_splits.py first!")
        exit(1)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        root_dir=DATASET_PATH,
        splits_path=SPLITS_PATH,
        batch_size=32,
        num_workers=4,
        target_size=256
    )
    
    # Test loading a batch
    print(f"\n{'='*60}")
    print("TESTING DATALOADER")
    print("="*60)
    
    images, labels = next(iter(train_loader))
    
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Labels: {labels[:10].tolist()}")
    print(f"Class distribution in batch: {torch.bincount(labels)}")
    
    print(f"\n{'='*60}")
    print("✅ DataLoader Implementation Complete!")
    print("="*60)
