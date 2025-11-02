"""
Script 6: SAM Architecture Study
Understand the three components of SAM and their interactions
"""

import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

print("="*60)
print("SEGMENT ANYTHING MODEL (SAM) ARCHITECTURE STUDY")
print("="*60)

# SAM Checkpoint path (adjust to your download location)
SAM_CHECKPOINT = "sam_checkpoints/sam_vit_b_01ec64.pth"
MODEL_TYPE = "vit_b"  # vit_b (91M), vit_l (308M), vit_h (636M)

print(f"\n📚 SAM ARCHITECTURE OVERVIEW")
print("="*60)

architecture_info = """
SAM consists of THREE main components:

1. IMAGE ENCODER (Heavy - runs once per image)
   ├─ Architecture: Vision Transformer (ViT)
   ├─ Input: RGB image (1024×1024 optimal, accepts various sizes)
   ├─ Output: Image embedding (256×64×64 for ViT-B)
   ├─ Computation: ~150ms on GPU (most expensive component)
   └─ Purpose: Extract rich visual features from entire image

2. PROMPT ENCODER (Lightweight - runs in real-time)
   ├─ Handles THREE types of prompts:
   │   • Sparse Prompts:
   │   │   - Points: (x,y) with positive/negative label
   │   │   - Boxes: (x1,y1,x2,y2) bounding box
   │   │   - Text: CLIP-based text encoding (optional)
   │   └─ Dense Prompts:
   │       - Masks: Low-resolution mask hints
   ├─ Output: Prompt embedding
   ├─ Computation: <5ms (very fast)
   └─ Purpose: Tell SAM WHAT to segment and WHERE

3. MASK DECODER (Lightweight - runs in real-time)
   ├─ Architecture: Modified Transformer decoder
   ├─ Input: Image embedding + Prompt embedding
   ├─ Output: Segmentation mask(s) + IoU confidence scores
   ├─ Computation: ~50ms on GPU
   └─ Purpose: Generate final segmentation mask

TOTAL INFERENCE TIME: ~200-300ms on GPU for full pipeline
"""

print(architecture_info)

print(f"\n{'='*60}")
print("LOADING SAM MODEL")
print("="*60)

# Check if checkpoint exists
if not Path(SAM_CHECKPOINT).exists():
    print(f"❌ Checkpoint not found: {SAM_CHECKPOINT}")
    print("\nDownload SAM checkpoints:")
    print("  mkdir sam_checkpoints")
    print("  cd sam_checkpoints")
    print("  curl -L -o sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
    exit(1)

# Load SAM model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
sam.to(device=device)

print(f"\n✅ Loaded SAM model: {MODEL_TYPE}")

# Analyze model architecture
print(f"\n{'='*60}")
print("MODEL ARCHITECTURE DETAILS")
print("="*60)

# Count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

total_params = count_parameters(sam)
image_encoder_params = count_parameters(sam.image_encoder)
prompt_encoder_params = count_parameters(sam.prompt_encoder)
mask_decoder_params = count_parameters(sam.mask_decoder)

print(f"\nTotal Parameters: {total_params:,}")
print(f"  ├─ Image Encoder: {image_encoder_params:,} ({image_encoder_params/total_params*100:.1f}%)")
print(f"  ├─ Prompt Encoder: {prompt_encoder_params:,} ({prompt_encoder_params/total_params*100:.1f}%)")
print(f"  └─ Mask Decoder: {mask_decoder_params:,} ({mask_decoder_params/total_params*100:.1f}%)")

print(f"\n{'='*60}")
print("IMAGE ENCODER DETAILS")
print("="*60)
print(f"Type: {sam.image_encoder.__class__.__name__}")
print(f"Expected Input Size: 1024×1024 (but flexible)")
print(f"Output Embedding Shape: 256×64×64")
print(f"Note: This is the slowest component (~150ms)")

print(f"\n{'='*60}")
print("PROMPT ENCODER DETAILS")
print("="*60)
print("Supported Prompt Types:")
print("  1. Points: (x, y, label) where label=1 (foreground) or 0 (background)")
print("  2. Boxes: (x1, y1, x2, y2) bounding box coordinates")
print("  3. Masks: Low-resolution mask as hint")
print("  4. Combinations: Mix multiple prompt types")

print(f"\n{'='*60}")
print("MASK DECODER DETAILS")
print("="*60)
print(f"Type: {sam.mask_decoder.__class__.__name__}")
print("Outputs:")
print("  • 3 mask predictions (whole, part, subpart)")
print("  • IoU confidence scores for each mask")
print("  • Typically use the mask with highest IoU score")

# Create predictor (wraps SAM for easier inference)
predictor = SamPredictor(sam)

print(f"\n{'='*60}")
print("SAM PREDICTOR WRAPPER")
print("="*60)
print("The SamPredictor class simplifies inference:")
print("  1. predictor.set_image(image) - Encode image once")
print("  2. predictor.predict(...) - Generate masks with different prompts")
print("     (Can call predict() multiple times with different prompts)")

print(f"\n{'='*60}")
print("KEY DESIGN INSIGHTS")
print("="*60)

insights = """
1. AMORTIZED COMPUTATION:
   • Image encoding is expensive (~150ms) but done ONCE
   • Prompt encoding + Mask decoding is cheap (~50ms)
   • Can generate many masks from same image with different prompts

2. PROMPT FLEXIBILITY:
   • Point prompt: Click on object
   • Box prompt: Draw bounding box
   • Mask prompt: Provide rough segmentation
   • Can combine: e.g., box + negative points for refinement

3. MULTIPLE MASK OUTPUTS:
   • Always returns 3 masks: whole object, part, subpart
   • Each has IoU confidence score
   • Typically use highest IoU mask

4. ZERO-SHOT CAPABILITY:
   • Trained on SA-1B dataset (1 billion masks!)
   • Generalizes to unseen objects and domains
   • No fine-tuning needed for basic segmentation

5. OPTIMIZATION FOR INTERACTIVITY:
   • Design prioritizes real-time feedback
   • Image encoding cached, prompt changes instant
   • Enables interactive annotation tools
"""

print(insights)

print(f"\n{'='*60}")
print("NEXT STEPS")
print("="*60)
print("Now that you understand SAM's architecture:")
print("  1. Test SAM on IDC sample patches")
print("  2. Experiment with different prompt types")
print("  3. Visualize segmentation outputs")
print("  4. Measure inference time on your GPU")
print("\nRun: python 07_test_sam_on_idc.py")

print(f"\n{'='*60}")
print("✅ Phase 3 - Step 1 Complete!")
print("="*60)
