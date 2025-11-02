"""
Script 7: Test SAM on IDC Dataset
Apply SAM to sample IDC patches and evaluate performance
"""

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from pathlib import Path
import time
import json

# Configuration
SAM_CHECKPOINT = "sam_checkpoints/sam_vit_b_01ec64.pth"
MODEL_TYPE = "vit_b"
DATASET_PATH = r"D:\projs\segment_anything\data\IDC_regular_ps50_idx5"

print("="*60)
print("TESTING SAM ON IDC DATASET")
print("="*60)

# Load SAM
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}")

sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
sam.to(device=device)
predictor = SamPredictor(sam)

print(f"✅ Loaded SAM model: {MODEL_TYPE}")

def load_sample_patches(root_path, n_samples=5):
    """Load sample patches from both classes"""
    samples = {'negative': [], 'positive': []}
    
    for patient_folder in Path(root_path).glob("*"):
        if not patient_folder.is_dir():
            continue
        
        # Load negative
        neg_folder = patient_folder / "0"
        if neg_folder.exists():
            neg_images = list(neg_folder.glob("*.png"))[:1]
            for img_path in neg_images:
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                samples['negative'].append((img, str(img_path)))
        
        # Load positive
        pos_folder = patient_folder / "1"
        if pos_folder.exists():
            pos_images = list(pos_folder.glob("*.png"))[:1]
            for img_path in pos_images:
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                samples['positive'].append((img, str(img_path)))
        
        if len(samples['negative']) >= n_samples and len(samples['positive']) >= n_samples:
            break
    
    return samples

def upscale_image(img, target_size=256):
    """Upscale 50×50 patches to larger size for SAM"""
    return cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_CUBIC)

def test_sam_with_prompts(image, predictor):
    """Test SAM with different prompt types"""
    h, w = image.shape[:2]
    
    # Set image (encodes once)
    predictor.set_image(image)
    
    results = {}
    
    # 1. Center Point Prompt
    center_point = np.array([[w//2, h//2]])
    center_label = np.array([1])  # 1 = foreground
    
    start = time.time()
    masks, scores, logits = predictor.predict(
        point_coords=center_point,
        point_labels=center_label,
        multimask_output=True
    )
    point_time = time.time() - start
    
    results['point'] = {
        'masks': masks,
        'scores': scores,
        'time': point_time,
        'prompt': 'Center point'
    }
    
    # 2. Box Prompt (covering most of image)
    margin = int(w * 0.1)
    box = np.array([margin, margin, w-margin, h-margin])
    
    start = time.time()
    masks, scores, logits = predictor.predict(
        box=box,
        multimask_output=True
    )
    box_time = time.time() - start
    
    results['box'] = {
        'masks': masks,
        'scores': scores,
        'time': box_time,
        'prompt': 'Bounding box'
    }
    
    # 3. Multiple Points (corners)
    corner_points = np.array([
        [w//4, h//4],
        [3*w//4, h//4],
        [w//2, h//2],
        [w//4, 3*h//4],
        [3*w//4, 3*h//4]
    ])
    corner_labels = np.array([1, 1, 1, 1, 1])  # All foreground
    
    start = time.time()
    masks, scores, logits = predictor.predict(
        point_coords=corner_points,
        point_labels=corner_labels,
        multimask_output=True
    )
    multipoint_time = time.time() - start
    
    results['multipoint'] = {
        'masks': masks,
        'scores': scores,
        'time': multipoint_time,
        'prompt': '5 points'
    }
    
    return results

def visualize_sam_results(image, results, class_name, save_path):
    """Visualize SAM segmentation results"""
    n_prompts = len(results)
    fig, axes = plt.subplots(n_prompts, 4, figsize=(16, 4*n_prompts))
    
    if n_prompts == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'SAM Results on {class_name} IDC Patch', fontsize=16, fontweight='bold')
    
    for idx, (prompt_type, data) in enumerate(results.items()):
        masks = data['masks']
        scores = data['scores']
        prompt = data['prompt']
        inf_time = data['time']
        
        # Original image
        axes[idx, 0].imshow(image)
        axes[idx, 0].set_title(f'Original\nPrompt: {prompt}')
        axes[idx, 0].axis('off')
        
        # Three mask predictions
        for mask_idx in range(3):
            axes[idx, mask_idx+1].imshow(image)
            axes[idx, mask_idx+1].imshow(masks[mask_idx], alpha=0.5, cmap='jet')
            axes[idx, mask_idx+1].set_title(
                f'Mask {mask_idx+1}\nIoU: {scores[mask_idx]:.3f}\n{inf_time*1000:.1f}ms'
            )
            axes[idx, mask_idx+1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {save_path}")
    plt.close()

def benchmark_sam(predictor, image):
    """Benchmark SAM inference time"""
    h, w = image.shape[:2]
    
    # Warmup
    predictor.set_image(image)
    for _ in range(3):
        predictor.predict(
            point_coords=np.array([[w//2, h//2]]),
            point_labels=np.array([1]),
            multimask_output=False
        )
    
    # Benchmark image encoding
    encoding_times = []
    for _ in range(10):
        start = time.time()
        predictor.set_image(image)
        encoding_times.append(time.time() - start)
    
    # Benchmark prediction (with cached encoding)
    predictor.set_image(image)
    prediction_times = []
    for _ in range(50):
        start = time.time()
        predictor.predict(
            point_coords=np.array([[w//2, h//2]]),
            point_labels=np.array([1]),
            multimask_output=False
        )
        prediction_times.append(time.time() - start)
    
    return {
        'image_encoding': {
            'mean': np.mean(encoding_times) * 1000,
            'std': np.std(encoding_times) * 1000
        },
        'prediction': {
            'mean': np.mean(prediction_times) * 1000,
            'std': np.std(prediction_times) * 1000
        }
    }

# Main testing
print(f"\n{'='*60}")
print("LOADING SAMPLE PATCHES")
print("="*60)

samples = load_sample_patches(DATASET_PATH, n_samples=2)
print(f"Loaded {len(samples['negative'])} negative and {len(samples['positive'])} positive samples")

# Test on negative sample
print(f"\n{'='*60}")
print("TESTING ON NEGATIVE SAMPLE")
print("="*60)

neg_img, neg_path = samples['negative'][0]
print(f"Original size: {neg_img.shape}")

# Upscale for SAM
neg_img_upscaled = upscale_image(neg_img, target_size=256)
print(f"Upscaled size: {neg_img_upscaled.shape}")

neg_results = test_sam_with_prompts(neg_img_upscaled, predictor)
visualize_sam_results(neg_img_upscaled, neg_results, 'Negative', 'sam_results_negative.png')

# Test on positive sample
print(f"\n{'='*60}")
print("TESTING ON POSITIVE SAMPLE")
print("="*60)

pos_img, pos_path = samples['positive'][0]
pos_img_upscaled = upscale_image(pos_img, target_size=256)

pos_results = test_sam_with_prompts(pos_img_upscaled, predictor)
visualize_sam_results(pos_img_upscaled, pos_results, 'Positive', 'sam_results_positive.png')

# Benchmark
print(f"\n{'='*60}")
print("BENCHMARKING SAM INFERENCE TIME")
print("="*60)

benchmark_results = benchmark_sam(predictor, neg_img_upscaled)

print(f"\nImage Encoding (happens once per image):")
print(f"  Mean: {benchmark_results['image_encoding']['mean']:.2f} ms")
print(f"  Std:  {benchmark_results['image_encoding']['std']:.2f} ms")

print(f"\nMask Prediction (per prompt, with cached encoding):")
print(f"  Mean: {benchmark_results['prediction']['mean']:.2f} ms")
print(f"  Std:  {benchmark_results['prediction']['std']:.2f} ms")

print(f"\nTotal Time per Image (encoding + prediction):")
total_time = benchmark_results['image_encoding']['mean'] + benchmark_results['prediction']['mean']
print(f"  {total_time:.2f} ms (~{1000/total_time:.1f} images/sec)")

# Save benchmark results
with open('sam_benchmark_results.json', 'w') as f:
    json.dump(benchmark_results, f, indent=2)
print(f"\n✅ Saved benchmark to: sam_benchmark_results.json")

print(f"\n{'='*60}")
print("KEY OBSERVATIONS")
print("="*60)

observations = """
1. PATCH SIZE IMPACT:
   • Original IDC patches: 50×50 pixels
   • Upscaled to: 256×256 (SAM prefers larger)
   • Note: SAM optimal size is 1024×1024
   • Small patches lack context → may need different approach

2. PROMPT TYPE EFFECTS:
   • Point prompts: Fast, but ambiguous for small patches
   • Box prompts: Better for whole-patch segmentation
   • Multiple points: Provides more context

3. INFERENCE TIME:
   • Image encoding: ~50-150ms (depends on GPU)
   • Mask prediction: ~5-50ms
   • Fast enough for interactive use

4. LIMITATIONS FOR IDC:
   • IDC patches are already small (50×50)
   • SAM designed for larger images with distinct objects
   • For patch classification, U-Net might be more suitable
   • SAM better for whole-slide image segmentation

5. POTENTIAL APPLICATIONS:
   • Use SAM on whole-slide images (not patches)
   • Generate tumor region proposals
   • Interactive annotation tool
   • Then extract patches from SAM masks for training
"""

print(observations)

print(f"\n{'='*60}")
print("✅ Phase 3 - Step 2 Complete!")
print("="*60)
print("\nNext: Implement U-Net and compare with SAM")
