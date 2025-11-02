"""
Script 1: Explore IDC Dataset Structure
Analyzes the dataset structure, counts, and basic statistics
"""

import os
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set your dataset path
DATASET_PATH = r"D:\projs\segment_anything\data\IDC_regular_ps50_idx5"

def explore_dataset_structure(root_path):
    """Comprehensive dataset structure exploration"""
    print("="*60)
    print("IDC BREAST HISTOPATHOLOGY DATASET EXPLORATION")
    print("="*60)
    
    if not os.path.exists(root_path):
        print(f"❌ Dataset not found at: {root_path}")
        print("Please update DATASET_PATH in the script!")
        return None
    
    print(f"\n📂 Dataset Location: {root_path}\n")
    
    # Collect statistics
    stats = {
        'patient_id': [],
        'negative_count': [],
        'positive_count': [],
        'total_patches': [],
        'positive_ratio': []
    }
    
    total_negative = 0
    total_positive = 0
    
    print("📊 Scanning patient folders...")
    for patient_folder in sorted(Path(root_path).glob("*")):
        if patient_folder.is_dir():
            patient_id = patient_folder.name
            
            # Count patches in each class
            negative_folder = patient_folder / "0"
            positive_folder = patient_folder / "1"
            
            neg_count = len(list(negative_folder.glob("*.png"))) if negative_folder.exists() else 0
            pos_count = len(list(positive_folder.glob("*.png"))) if positive_folder.exists() else 0
            
            total_negative += neg_count
            total_positive += pos_count
            
            total = neg_count + pos_count
            pos_ratio = pos_count / total if total > 0 else 0
            
            stats['patient_id'].append(patient_id)
            stats['negative_count'].append(neg_count)
            stats['positive_count'].append(pos_count)
            stats['total_patches'].append(total)
            stats['positive_ratio'].append(pos_ratio)
    
    # Create DataFrame
    df = pd.DataFrame(stats)
    
    # Print summary
    print(f"\n{'='*60}")
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Total Patients: {len(df)}")
    print(f"Total Patches: {total_negative + total_positive:,}")
    print(f"  ├─ IDC Negative (Class 0): {total_negative:,} ({total_negative/(total_negative+total_positive)*100:.1f}%)")
    print(f"  └─ IDC Positive (Class 1): {total_positive:,} ({total_positive/(total_negative+total_positive)*100:.1f}%)")
    print(f"\nClass Imbalance Ratio: {total_negative/total_positive:.2f}:1")
    
    print(f"\n{'='*60}")
    print("PER-PATIENT STATISTICS")
    print("="*60)
    print(df[['negative_count', 'positive_count', 'total_patches', 'positive_ratio']].describe())
    
    # Save statistics
    df.to_csv('dataset_statistics.csv', index=False)
    print(f"\n✅ Saved statistics to: dataset_statistics.csv")
    
    return df

def visualize_distribution(df):
    """Create visualizations of dataset distribution"""
    print(f"\n{'='*60}")
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('IDC Dataset Distribution Analysis', fontsize=16, fontweight='bold')
    
    # 1. Class distribution pie chart
    total_neg = df['negative_count'].sum()
    total_pos = df['positive_count'].sum()
    
    axes[0, 0].pie([total_neg, total_pos], 
                    labels=['IDC Negative', 'IDC Positive'],
                    autopct='%1.1f%%',
                    colors=['#2ecc71', '#e74c3c'],
                    startangle=90)
    axes[0, 0].set_title('Overall Class Distribution')
    
    # 2. Positive ratio distribution
    axes[0, 1].hist(df['positive_ratio'], bins=30, color='#3498db', edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(df['positive_ratio'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {df["positive_ratio"].mean():.3f}')
    axes[0, 1].set_xlabel('Positive Patch Ratio per Patient')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of IDC Positive Ratios')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # 3. Total patches per patient
    axes[1, 0].hist(df['total_patches'], bins=30, color='#9b59b6', edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Total Patches per Patient')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Patches per Patient Distribution')
    axes[1, 0].grid(alpha=0.3)
    
    # 4. Scatter: total patches vs positive ratio
    scatter = axes[1, 1].scatter(df['total_patches'], df['positive_ratio'], 
                                  c=df['positive_count'], cmap='YlOrRd', 
                                  s=50, alpha=0.6, edgecolor='black')
    axes[1, 1].set_xlabel('Total Patches per Patient')
    axes[1, 1].set_ylabel('Positive Patch Ratio')
    axes[1, 1].set_title('Patches vs Positive Ratio')
    axes[1, 1].grid(alpha=0.3)
    cbar = plt.colorbar(scatter, ax=axes[1, 1])
    cbar.set_label('Positive Patches')
    
    plt.tight_layout()
    plt.savefig('dataset_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved visualization to: dataset_distribution.png")
    plt.close()

if __name__ == "__main__":
    df = explore_dataset_structure(DATASET_PATH)
    
    if df is not None:
        visualize_distribution(df)
        print(f"\n{'='*60}")
        print("✅ Phase 2 - Step 1 Complete!")
        print("="*60)
