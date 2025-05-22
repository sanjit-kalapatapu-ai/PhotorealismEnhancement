#!/usr/bin/env python3

import os
import argparse
import numpy as np
import cv2
from pathlib import Path
import csv

def create_dummy_rgb(width: int, height: int) -> np.ndarray:
    """Create a dummy RGB image with random patterns."""
    # Create a random noise image
    noise = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    # Add some structure with random shapes
    img = noise.copy()
    
    # Add random rectangles
    for _ in range(3):
        x1, y1 = np.random.randint(0, width-100), np.random.randint(0, height-100)
        x2, y2 = x1 + np.random.randint(50, 100), y1 + np.random.randint(50, 100)
        color = np.random.randint(0, 255, 3).tolist()
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    
    # Add random circles
    for _ in range(3):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        radius = np.random.randint(20, 50)
        color = np.random.randint(0, 255, 3).tolist()
        cv2.circle(img, (x, y), radius, color, -1)
    
    return img

def create_dummy_segmentation(width: int, height: int, num_classes: int = 34) -> np.ndarray:
    """Create a dummy segmentation mask."""
    # Create base segmentation with random classes
    seg = np.random.randint(0, num_classes, (width, height), dtype=np.uint8)
    
    # Add some structure with larger continuous regions
    for _ in range(5):
        x1, y1 = np.random.randint(0, width-100), np.random.randint(0, height-100)
        x2, y2 = x1 + np.random.randint(50, 100), y1 + np.random.randint(50, 100)
        class_id = np.random.randint(0, num_classes)
        seg[x1:x2, y1:y2] = class_id
    
    return seg

def create_dummy_gbuffer(width: int, height: int, num_channels: int) -> np.ndarray:
    """Create dummy g-buffer data with specified number of channels."""
    # Create random g-buffer data with some structure
    gbuffer = np.random.randn(width, height, num_channels).astype(np.float32)
    
    # Add some structure (e.g., depth-like gradients for first channel)
    gbuffer[:, :, 0] = np.linspace(0, 1, height)[None, :] * np.linspace(0, 1, width)[:, None]
    
    # Add some normal-like data for channels 1-3 if they exist
    if num_channels > 3:
        # Normalize channels 1-3 to create plausible normal vectors
        norms = np.random.randn(width, height, 3)
        norms = norms / np.linalg.norm(norms, axis=2, keepdims=True)
        gbuffer[:, :, 1:4] = norms
    
    return gbuffer

def create_dummy_dataset(dataset_dir: str, num_samples: int, width: int, height: int, num_gbuffers: int) -> None:
    """
    Create a dummy dataset matching the format from prepare_synthetic_dataset.py.
    
    Args:
        dataset_dir: Output directory for the dummy dataset
        num_samples: Number of samples to generate
        width: Width of the images
        height: Height of the images
        num_gbuffers: Number of g-buffer channels
    """
    dataset_dir = Path(dataset_dir)
    
    # Create directory structure
    output_dirs = {
        'rgb': dataset_dir / "rgb",
        'gt_seg': dataset_dir / "gt_seg",
        'gbuffer': dataset_dir / "gbuffer",
        'mseg': dataset_dir / "mseg"
    }
    
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # Create dummy g-buffer statistics
    g_means = np.random.randn(num_gbuffers)
    g_stds = np.abs(np.random.randn(num_gbuffers))
    np.savez(
        dataset_dir / 'gbuffer_stats.npz',
        g_m=g_means,
        g_s=g_stds
    )
    
    # Generate samples
    file_mappings = []
    for i in range(num_samples):
        basename = f"sample_{i:06d}.png"
        
        # Create and save RGB image
        rgb = create_dummy_rgb(width, height)
        rgb_path = output_dirs['rgb'] / basename
        cv2.imwrite(str(rgb_path), rgb)
        
        # Create and save ground truth segmentation
        gt_seg = create_dummy_segmentation(width, height)
        gt_seg_path = output_dirs['gt_seg'] / basename.replace('.png', '.npy')
        np.save(str(gt_seg_path), gt_seg)
        
        # Create MSeg universal taxonomy version (just reuse gt_seg for dummy data)
        mseg_path = output_dirs['mseg'] / basename
        cv2.imwrite(str(mseg_path), gt_seg.T)  # Transpose before saving to maintain width x height dims
        
        # Create and save g-buffer data
        gbuffer = create_dummy_gbuffer(width, height, num_gbuffers)
        gbuffer_path = output_dirs['gbuffer'] / basename.replace('.png', '.npy')
        np.save(str(gbuffer_path), gbuffer)
        
        # Store file mappings
        file_mappings.append([
            str(rgb_path.absolute()),
            str(mseg_path.absolute()),
            str(gbuffer_path.absolute()),
            str(gt_seg_path.absolute())
        ])
    
    # Write CSV file
    csv_path = dataset_dir / f"{dataset_dir.name}_files.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(file_mappings)
    
    print(f"Created dummy dataset with {num_samples} samples at {dataset_dir}")
    print(f"Image size: {width}x{height}")
    print(f"Number of g-buffer channels: {num_gbuffers}")
    print(f"CSV file created at: {csv_path}")

def main():
    parser = argparse.ArgumentParser(description='Create a dummy dataset matching the synthetic dataset format')
    parser.add_argument('dataset_dir', type=str, help='Output directory for the dummy dataset')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to generate')
    parser.add_argument('--width', type=int, default=640, help='Width of the images')
    parser.add_argument('--height', type=int, default=480, help='Height of the images')
    parser.add_argument('--num_gbuffers', type=int, default=8, help='Number of g-buffer channels')
    
    args = parser.parse_args()
    create_dummy_dataset(
        args.dataset_dir,
        args.num_samples,
        args.width,
        args.height,
        args.num_gbuffers
    )

if __name__ == '__main__':
    main() 