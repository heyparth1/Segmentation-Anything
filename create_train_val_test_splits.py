import openslide
from pathlib import Path
import numpy as np
import cv2

def extract_patches_from_wsi(wsi_path, patch_size=256, stride=256):
    """
    Extract patches from whole-slide image
    """
    # Open WSI
    slide = openslide.OpenSlide(str(wsi_path))
    
    # Get dimensions at desired magnification level
    level = 0  # Highest resolution
    width, height = slide.level_dimensions[level]
    
    print(f"WSI dimensions: {width}x{height}")
    print(f"Extracting {patch_size}x{patch_size} patches...")
    
    patches = []
    coordinates = []
    
    for y in range(0, height - patch_size, stride):
        for x in range(0, width - patch_size, stride):
            # Read patch
            patch = slide.read_region((x, y), level, (patch_size, patch_size))
            patch = np.array(patch.convert('RGB'))
            
            # Simple tissue detection (skip background)
            if is_tissue(patch):
                patches.append(patch)
                coordinates.append((x, y))
    
    slide.close()
    return patches, coordinates

def is_tissue(patch, threshold=0.8):
    """Simple tissue detection - check if patch is not mostly white"""
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    white_ratio = np.sum(gray > 220) / gray.size
    return white_ratio < threshold
