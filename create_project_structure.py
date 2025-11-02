"""
Script 4: Create Complete Project Structure
Sets up organized folder hierarchy for the project
"""

import os
from pathlib import Path

# Project root
PROJECT_ROOT = r"D:\projs\segment_anything"

def create_project_structure():
    """Create comprehensive project folder structure"""
    
    structure = {
        'data': {
            'raw': {},
            'processed': {},
            'splits': {}
        },
        'models': {
            'sam': {},
            'unet': {},
            'checkpoints': {}
        },
        'src': {
            'data': {},
            'models': {},
            'training': {},
            'evaluation': {},
            'visualization': {},
            'utils': {}
        },
        'notebooks': {},
        'results': {
            'sam': {
                'visualizations': {},
                'metrics': {},
                'predictions': {}
            },
            'unet': {
                'visualizations': {},
                'metrics': {},
                'predictions': {}
            },
            'comparison': {}
        },
        'configs': {},
        'logs': {},
        'reports': {}
    }
    
    def create_nested_dirs(base_path, structure):
        """Recursively create directory structure"""
        for name, subdirs in structure.items():
            dir_path = base_path / name
            dir_path.mkdir(exist_ok=True)
            
            # Create __init__.py for Python packages
            if 'src' in str(base_path) and name not in ['data', 'models', 'training', 'evaluation', 'visualization', 'utils']:
                continue
            if base_path.name == 'src' or (base_path.parent.name == 'src'):
                init_file = dir_path / '__init__.py'
                if not init_file.exists():
                    init_file.touch()
            
            if subdirs:
                create_nested_dirs(dir_path, subdirs)
    
    print(f"\n{'='*60}")
    print("CREATING PROJECT STRUCTURE")
    print("="*60)
    print(f"\nProject Root: {PROJECT_ROOT}\n")
    
    base_path = Path(PROJECT_ROOT)
    create_nested_dirs(base_path, structure)
    
    # Create README files
    readme_content = {
        'data/README.md': """# Data Directory

## Structure
- `raw/`: Original downloaded datasets
- `processed/`: Preprocessed and augmented data
- `splits/`: Train/val/test split information
""",
        'models/README.md': """# Models Directory

## Structure
- `sam/`: SAM model implementations and configs
- `unet/`: U-Net model implementations
- `checkpoints/`: Saved model weights
""",
        'results/README.md': """# Results Directory

## Structure
- `sam/`: SAM experiment results
- `unet/`: U-Net experiment results
- `comparison/`: Comparative analysis
"""
    }
    
    for filepath, content in readme_content.items():
        full_path = base_path / filepath
        with open(full_path, 'w') as f:
            f.write(content)
    
    # Create requirements.txt
    requirements = """# Core Frameworks
torch>=2.5.0
torchvision>=0.20.0
tensorflow>=2.16.0

# Image Processing
opencv-python>=4.10.0
pillow>=10.0.0
scikit-image>=0.22.0
albumentations>=2.0.0

# Segmentation
segment-anything
openslide-bin

# Data Science
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0

# ML Utilities
scikit-learn>=1.3.0
tqdm>=4.65.0
tensorboard>=2.14.0

# Utilities
pyyaml>=6.0
h5py>=3.9.0
"""
    
    with open(base_path / 'requirements.txt', 'w') as f:
        f.write(requirements)
    
    # Create main config file
    config_content = """# IDC Segmentation Project Configuration

## Dataset
dataset:
  name: "IDC_Breast_Histopathology"
  root_path: "data/IDC_regular_ps50_idx5"
  image_size: [50, 50]
  num_classes: 2
  
## Training
training:
  batch_size: 32
  epochs: 50
  learning_rate: 0.001
  device: "cuda"
  
## SAM Configuration
sam:
  model_type: "vit_b"  # vit_b, vit_l, vit_h
  checkpoint: "sam_checkpoints/sam_vit_b_01ec64.pth"
  
## U-Net Configuration  
unet:
  depth: 4
  start_filters: 64
  
## Evaluation
evaluation:
  metrics: ["iou", "dice", "precision", "recall"]
"""
    
    with open(base_path / 'configs' / 'config.yaml', 'w') as f:
        f.write(config_content)
    
    print("✅ Created directory structure")
    print("✅ Created README files")
    print("✅ Created requirements.txt")
    print("✅ Created config.yaml")
    
    print(f"\n📁 Project Structure:")
    print("project_root/")
    print("├── data/")
    print("├── models/")
    print("├── src/")
    print("├── notebooks/")
    print("├── results/")
    print("├── configs/")
    print("├── logs/")
    print("└── reports/")

if __name__ == "__main__":
    create_project_structure()
    
    print(f"\n{'='*60}")
    print("✅ Phase 2 - Step 4 Complete!")
    print("="*60)
