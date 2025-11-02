# Image Segmentation using SAM & U-Net

This repository contains an exploratory implementation and study of image segmentation techniques using Meta's Segment Anything Model (SAM) and U-Net variants, with a focus on medical image analysis.

## Project Overview

This project explores state-of-the-art image segmentation approaches, combining transformer-based architectures (SAM) with traditional convolutional approaches (U-Net) for medical image segmentation tasks. The implementation focuses on understanding and adapting these architectures for histopathological image analysis.

### Key Features

- Implementation and study of Meta's Segment Anything Model (SAM)
- Adaptation of U-Net variants for medical image segmentation
- Patch-based segmentation workflow for large histopathology images
- Comprehensive evaluation metrics implementation (IoU, Dice Score)
- Dataset preprocessing and augmentation pipeline

## Dataset

The project utilizes open-source medical datasets:
- Camelyon16
- Breast Histopathology Dataset

## Project Structure

```
├── configs/            # Configuration files
├── data/              # Dataset storage
├── logs/              # Training and evaluation logs
├── models/            # Model implementations
├── notebooks/         # Jupyter notebooks for analysis
├── reports/           # Generated analysis reports
├── results/          # Output results and visualizations
├── sam_checkpoints/   # SAM model checkpoints
└── src/              # Source code
```

## Scripts

- `01_explore_dataset.py`: Initial dataset exploration and analysis
- `02_visualize_samples.py`: Data visualization utilities
- `03_create_train_val_test_splits.py`: Dataset splitting
- `04_create_project_structure.py`: Project setup and organization
- `05_dataset_insights_and_recommendations.py`: Dataset analysis
- `06_sam_architecture_study.py`: SAM architecture implementation study
- `07_test_sam_on_idc.py`: Model testing and evaluation

## Technologies Used

- PyTorch
- OpenCV
- TensorFlow
- Python
- Segment Anything Model (SAM)
- U-Net

## Key Achievements

- Successfully studied and reimplemented Meta's Segment Anything (SAM) architecture
- Adapted U-Net variants for patch-based segmentation on medical datasets
- Implemented comprehensive evaluation metrics for segmentation quality assessment
- Developed efficient preprocessing pipelines for large-scale histopathology images

## Evaluation Metrics

The project implements and utilizes several evaluation metrics:
- Intersection over Union (IoU)
- Dice Coefficient
- Additional custom metrics for medical image analysis

## Requirements

Dependencies can be installed using:
```bash
pip install -r requirements.txt
```

## Future Work

- Integration of additional segmentation architectures
- Enhancement of preprocessing pipeline
- Implementation of more advanced evaluation metrics
- Optimization for large-scale deployment

## License

[Add your license information here]

## Contact

[Add your contact information here]

## Acknowledgments

- Meta AI for the Segment Anything Model
- Contributors to the open-source medical datasets
- PyTorch and TensorFlow communities