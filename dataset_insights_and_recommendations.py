"""
Script 5: Dataset Insights and Recommendations
Analyzes Phase 2 results and provides preprocessing recommendations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

print("="*60)
print("DATASET INSIGHTS & PREPROCESSING RECOMMENDATIONS")
print("="*60)

# Load statistics
df = pd.read_csv('dataset_statistics.csv')

print("\n📊 KEY INSIGHTS FROM EXPLORATION:\n")

# 1. Class Imbalance Analysis
total_neg = df['negative_count'].sum()
total_pos = df['positive_count'].sum()
imbalance_ratio = total_neg / total_pos

print("1. CLASS IMBALANCE")
print("-" * 60)
print(f"   Imbalance Ratio: {imbalance_ratio:.2f}:1")
print(f"   Impact: Model will bias toward negative class")
print(f"   Solution Strategies:")
print(f"     • Weighted loss functions (weight positive class 2.5x)")
print(f"     • Focal loss (reduces easy negative influence)")
print(f"     • Balanced sampling during training")
print(f"     • Data augmentation for positive class")

# 2. Patient Variability
print(f"\n2. PATIENT-LEVEL VARIABILITY")
print("-" * 60)
print(f"   Min positive ratio: {df['positive_ratio'].min():.1%}")
print(f"   Max positive ratio: {df['positive_ratio'].max():.1%}")
print(f"   Std deviation: {df['positive_ratio'].std():.3f}")
print(f"   Impact: Some patients have very few positive samples")
print(f"   Solution: Stratified patient-level splits")

# 3. Patients with extreme imbalance
extreme_negative = df[df['positive_ratio'] < 0.05]
extreme_positive = df[df['positive_ratio'] > 0.8]

print(f"\n   Extreme Cases:")
print(f"     • Very low positive (<5%): {len(extreme_negative)} patients")
print(f"     • Very high positive (>80%): {len(extreme_positive)} patients")

# 4. Patch count distribution
print(f"\n3. PATCH COUNT DISTRIBUTION")
print("-" * 60)
print(f"   Mean patches per patient: {df['total_patches'].mean():.0f}")
print(f"   Min patches: {df['total_patches'].min()}")
print(f"   Max patches: {df['total_patches'].max()}")
print(f"   Impact: Uneven contribution to training")
print(f"   Solution: Consider patient-weighted sampling")

# 5. Color Analysis Insights
print(f"\n4. COLOR CHARACTERISTICS")
print("-" * 60)
print(f"   From 02_visualize_samples.py results:")
print(f"     • Negative patches: Lighter (mean intensity ~188.6)")
print(f"     • Positive patches: Darker (mean intensity ~149.6)")
print(f"     • RGB differences: Positive has less red/green")
print(f"   Impact: Color is discriminative feature")
print(f"   Recommendation: Stain normalization may help generalization")

# Generate Preprocessing Recommendations
print(f"\n{'='*60}")
print("🔧 PREPROCESSING RECOMMENDATIONS")
print("="*60)

recommendations = {
    "1. Data Augmentation": [
        "Rotation: ±180° (tissue has no preferred orientation)",
        "Horizontal/Vertical Flip",
        "Color Jittering: brightness ±20%, contrast ±20%",
        "Elastic Deformation: simulate tissue variability",
        "Random Crop/Zoom: 0.9-1.1x scale",
        "Gaussian Blur: occasional (simulate focus variations)"
    ],
    
    "2. Class Imbalance Handling": [
        "Weighted Cross-Entropy Loss (positive weight = 2.5)",
        "Focal Loss (α=0.25, γ=2.0)",
        "Balanced Random Sampler (equal positive/negative per batch)",
        "SMOTE or similar oversampling for positive class",
        "Cost-sensitive learning"
    ],
    
    "3. Stain Normalization": [
        "Method: Macenko or Vahadane normalization",
        "When: Optional - test with and without",
        "Benefit: Reduce staining variability across patients",
        "Tools: torchstain, staintools libraries",
        "Note: May be less critical for patches from same institution"
    ],
    
    "4. Patch Resizing for SAM": [
        "Current: 50×50 pixels",
        "SAM optimal: 1024×1024 (can work with smaller)",
        "Options:",
        "  • Upscale to 256×256 using bicubic interpolation",
        "  • Use SAM with smaller input (may reduce performance)",
        "  • Extract larger patches from original slides (if available)"
    ],
    
    "5. Train/Val/Test Strategy": [
        "Split by PATIENT (avoid data leakage) ✓ Already done!",
        "Stratified split by positive_ratio quartiles",
        "Ensure each split has similar class distribution",
        "Cross-validation: 5-fold patient-level CV (optional)"
    ]
}

for category, items in recommendations.items():
    print(f"\n{category}:")
    for item in items:
        print(f"  • {item}")

# Create preprocessing config
preprocessing_config = {
    "class_weights": {
        "negative": 1.0,
        "positive": 2.52  # Inverse of imbalance ratio
    },
    
    "augmentation": {
        "train": {
            "rotation": 180,
            "flip_horizontal": True,
            "flip_vertical": True,
            "brightness_contrast": {"brightness_limit": 0.2, "contrast_limit": 0.2},
            "elastic_transform": {"alpha": 50, "sigma": 5},
            "random_scale": {"scale_limit": 0.1}
        },
        "val_test": {
            "normalize_only": True
        }
    },
    
    "sampling": {
        "strategy": "balanced",  # Equal positives and negatives per batch
        "samples_per_class": 16  # For batch_size=32
    },
    
    "normalization": {
        "mean": [0.485, 0.456, 0.406],  # ImageNet stats (starting point)
        "std": [0.229, 0.224, 0.225]
    },
    
    "resize": {
        "target_size": [256, 256],  # Upscale from 50×50
        "interpolation": "bicubic"
    }
}

# Save config
with open('preprocessing_config.json', 'w') as f:
    json.dump(preprocessing_config, f, indent=2)

print(f"\n✅ Saved preprocessing configuration to: preprocessing_config.json")

# Create visual summary
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Dataset Challenges & Solutions', fontsize=16, fontweight='bold')

# 1. Class imbalance visualization
axes[0, 0].bar(['Negative', 'Positive'], [total_neg, total_pos], 
               color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
axes[0, 0].set_ylabel('Number of Patches')
axes[0, 0].set_title('Challenge: Class Imbalance (2.52:1)')
axes[0, 0].text(0.5, 0.95, 'Solution: Weighted Loss + Balanced Sampling', 
                transform=axes[0, 0].transAxes, ha='center', 
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

# 2. Patient variability
axes[0, 1].hist(df['positive_ratio'], bins=30, color='#3498db', 
                edgecolor='black', alpha=0.7)
axes[0, 1].axvline(0.05, color='red', linestyle='--', label='<5% positive')
axes[0, 1].axvline(0.80, color='red', linestyle='--', label='>80% positive')
axes[0, 1].set_xlabel('Positive Ratio per Patient')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Challenge: High Patient Variability')
axes[0, 1].legend()

# 3. Patch count distribution
axes[1, 0].scatter(df['total_patches'], df['positive_count'], 
                   alpha=0.5, edgecolor='black')
axes[1, 0].set_xlabel('Total Patches per Patient')
axes[1, 0].set_ylabel('Positive Patches')
axes[1, 0].set_title('Challenge: Uneven Patient Contribution')
axes[1, 0].grid(alpha=0.3)

# 4. Recommended workflow
workflow_text = """
RECOMMENDED PREPROCESSING PIPELINE:

1. Load Image (50×50 RGB)
   ↓
2. Resize to 256×256 (bicubic)
   ↓
3. Apply Augmentation (if training)
   ↓
4. Normalize (ImageNet stats)
   ↓
5. Convert to Tensor
   ↓
6. Feed to Model

Class Balancing:
- Weighted Loss (pos: 2.52)
- Balanced Batch Sampling
"""

axes[1, 1].text(0.1, 0.5, workflow_text, transform=axes[1, 1].transAxes,
                fontsize=10, verticalalignment='center', 
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
axes[1, 1].axis('off')
axes[1, 1].set_title('Recommended Preprocessing Pipeline')

plt.tight_layout()
plt.savefig('preprocessing_recommendations.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved visualization to: preprocessing_recommendations.png")

print(f"\n{'='*60}")
print("NEXT STEPS")
print("="*60)
print("1. ✅ Run remaining Phase 2 scripts:")
print("     python 03_create_train_val_test_splits.py")
print("     python 04_create_project_structure.py")
print("\n2. 📚 Move to Phase 3: SAM Architecture Study")
print("     - Understand SAM components")
print("     - Load pretrained SAM checkpoint")
print("     - Test on sample IDC patches")
print("\n3. 🔨 Implement PyTorch DataLoader with:")
print("     - Patient-level splits")
print("     - Augmentation pipeline")
print("     - Balanced sampling")
print("     - Class weighting")

print(f"\n{'='*60}")
print("✅ Phase 2 - Step 5 Complete!")
print("="*60)
