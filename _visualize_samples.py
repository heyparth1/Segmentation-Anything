"""
Script 2: Visualize Sample Patches
Shows examples from both classes with proper labeling
"""

import os
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

# Set your dataset path
DATASET_PATH = r"D:\projs\segment_anything\data\IDC_regular_ps50_idx5"

def load_sample_patches(root_path, num_samples=20):
    """Load sample patches from both classes"""
    negative_samples = []
    positive_samples = []
    
    patient_folders = list(Path(root_path).glob("*"))
    
    print(f"Loading {num_samples} samples from each class...")
    
    # Iterate through patients to collect samples
    for patient_folder in patient_folders:
        if not patient_folder.is_dir():
            continue
            
        # Load negative samples
        negative_folder = patient_folder / "0"
        if negative_folder.exists():
            neg_images = list(negative_folder.glob("*.png"))
            for img_path in random.sample(neg_images, min(2, len(neg_images))):
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                negative_samples.append((img, img_path.name, patient_folder.name))
                
        # Load positive samples
        positive_folder = patient_folder / "1"
        if positive_folder.exists():
            pos_images = list(positive_folder.glob("*.png"))
            for img_path in random.sample(pos_images, min(2, len(pos_images))):
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                positive_samples.append((img, img_path.name, patient_folder.name))
        
        if len(negative_samples) >= num_samples and len(positive_samples) >= num_samples:
            break
    
    return negative_samples[:num_samples], positive_samples[:num_samples]

def visualize_samples(negative_samples, positive_samples):
    """Create comprehensive sample visualization"""
    n_samples = min(len(negative_samples), len(positive_samples))
    
    # Create grid: 2 rows (negative, positive) x n_samples columns
    fig, axes = plt.subplots(2, n_samples, figsize=(20, 4))
    fig.suptitle('IDC Breast Histopathology Samples (50×50 pixels, 40× magnification)', 
                 fontsize=14, fontweight='bold')
    
    # Plot negative samples (top row)
    for idx in range(n_samples):
        img, filename, patient = negative_samples[idx]
        axes[0, idx].imshow(img)
        axes[0, idx].set_title(f'Negative\nPt:{patient[:6]}', fontsize=8)
        axes[0, idx].axis('off')
    
    # Plot positive samples (bottom row)
    for idx in range(n_samples):
        img, filename, patient = positive_samples[idx]
        axes[1, idx].imshow(img)
        axes[1, idx].set_title(f'Positive\nPt:{patient[:6]}', fontsize=8)
        axes[1, idx].axis('off')
    
    # Add row labels
    fig.text(0.02, 0.75, 'IDC\nNegative', ha='center', va='center', 
             fontsize=12, fontweight='bold', color='green')
    fig.text(0.02, 0.25, 'IDC\nPositive', ha='center', va='center', 
             fontsize=12, fontweight='bold', color='red')
    
    plt.tight_layout()
    plt.savefig('sample_patches.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved sample visualization to: sample_patches.png")
    plt.close()

def analyze_image_properties(negative_samples, positive_samples):
    """Analyze image properties like color distribution"""
    print(f"\n{'='*60}")
    print("IMAGE PROPERTIES ANALYSIS")
    print("="*60)
    
    # Combine all samples
    all_images = [s[0] for s in negative_samples] + [s[0] for s in positive_samples]
    
    # Check dimensions
    dims = [img.shape for img in all_images]
    unique_dims = set(dims)
    print(f"\nImage Dimensions: {unique_dims}")
    print(f"Format: (Height, Width, Channels)")
    
    # Analyze color distribution
    neg_colors = np.array([s[0].mean(axis=(0, 1)) for s in negative_samples])
    pos_colors = np.array([s[0].mean(axis=(0, 1)) for s in positive_samples])
    
    print(f"\n📊 Mean RGB Values:")
    print(f"  Negative: R={neg_colors[:, 0].mean():.1f}, G={neg_colors[:, 1].mean():.1f}, B={neg_colors[:, 2].mean():.1f}")
    print(f"  Positive: R={pos_colors[:, 0].mean():.1f}, G={pos_colors[:, 1].mean():.1f}, B={pos_colors[:, 2].mean():.1f}")
    
    # Intensity analysis
    neg_intensity = np.array([s[0].mean() for s in negative_samples])
    pos_intensity = np.array([s[0].mean() for s in positive_samples])
    
    print(f"\n📊 Mean Intensity:")
    print(f"  Negative: {neg_intensity.mean():.1f} ± {neg_intensity.std():.1f}")
    print(f"  Positive: {pos_intensity.mean():.1f} ± {pos_intensity.std():.1f}")
    
    # Visualize color distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    colors = ['red', 'green', 'blue']
    for i, color in enumerate(colors):
        axes[0].hist(neg_colors[:, i], bins=20, alpha=0.5, label=f'{color.upper()}', color=color)
    axes[0].set_title('Negative Patches - RGB Distribution')
    axes[0].set_xlabel('Pixel Intensity')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    for i, color in enumerate(colors):
        axes[1].hist(pos_colors[:, i], bins=20, alpha=0.5, label=f'{color.upper()}', color=color)
    axes[1].set_title('Positive Patches - RGB Distribution')
    axes[1].set_xlabel('Pixel Intensity')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('color_distribution.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved color distribution to: color_distribution.png")
    plt.close()

if __name__ == "__main__":
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset not found at: {DATASET_PATH}")
        exit(1)
    
    # Load samples
    negative_samples, positive_samples = load_sample_patches(DATASET_PATH, num_samples=10)
    
    print(f"✅ Loaded {len(negative_samples)} negative and {len(positive_samples)} positive samples")
    
    # Visualize
    visualize_samples(negative_samples, positive_samples)
    
    # Analyze properties
    analyze_image_properties(negative_samples, positive_samples)
    
    print(f"\n{'='*60}")
    print("✅ Phase 2 - Step 2 Complete!")
    print("="*60)
