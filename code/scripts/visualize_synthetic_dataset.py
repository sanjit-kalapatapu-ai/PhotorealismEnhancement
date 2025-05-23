#!/usr/bin/env python3

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import random
import csv

def load_dataset_files(dataset_dir: Path):
    """
    Load the dataset file mappings from the CSV file.
    
    Args:
        dataset_dir: Path to the dataset directory
        
    Returns:
        list: List of file mappings [rgb_path, mseg_path, gbuffer_path, gt_seg_path]
    """
    csv_path = dataset_dir / f"{dataset_dir.name}_files.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        file_mappings = list(reader)
    
    return file_mappings

def find_corresponding_files(rgb_path: str, file_mappings: list) -> tuple:
    """
    Find the corresponding MSEG and gbuffer files for a given RGB file.
    
    Args:
        rgb_path: Path to the RGB image
        file_mappings: List of file mappings from the dataset
        
    Returns:
        tuple: (mseg_path, gbuffer_path) or None if not found
    """
    rgb_path = str(Path(rgb_path).absolute())
    for mapping in file_mappings:
        if mapping[0] == rgb_path:
            return mapping[1], mapping[2]
    raise ValueError(f"Could not find corresponding files for RGB image: {rgb_path}")

def plot_gbuffer_histogram(channel, ax, channel_idx):
    """
    Plot histogram of gbuffer channel values.
    
    Args:
        channel: The gbuffer channel data
        ax: Matplotlib axis to plot on
        channel_idx: Index of the channel for title
    """
    # Compute histogram
    hist, bins = np.histogram(channel.flatten(), bins=50)
    
    # Plot histogram
    ax.hist(channel.flatten(), bins=50, density=True, alpha=0.7)
    ax.set_title(f'GBuffer Channel {channel_idx} Distribution')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    
    # Add statistics
    mean = np.mean(channel)
    std = np.std(channel)
    min_val = np.min(channel)
    max_val = np.max(channel)
    
    stats_text = f'Mean: {mean:.2f}\nStd: {std:.2f}\nMin: {min_val:.2f}\nMax: {max_val:.2f}'
    ax.text(0.95, 0.95, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

def visualize_sample(rgb_path: str, mseg_path: str, gbuffer_path: str, output_path: str):
    """
    Create and save a visualization of a random sample from the dataset.
    
    Args:
        rgb_path: Path to the RGB image
        mseg_path: Path to the MSEG segmentation mask
        gbuffer_path: Path to the gbuffer array
        output_path: Path where to save the visualization
    """
    # Load the data
    rgb_img = cv2.imread(rgb_path)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    
    mseg_mask = cv2.imread(mseg_path, cv2.IMREAD_GRAYSCALE)
    gbuffer_array = np.load(gbuffer_path)
    
    # Transpose the gbuffer array to match RGB orientation
    gbuffer_array = gbuffer_array.transpose(1, 0, 2)
    
    # Get number of gbuffer channels
    num_gbuffer_channels = gbuffer_array.shape[2]
    
    # Calculate number of rows needed for the subplot grid
    # We'll have RGB and MSEG in the first row
    # Each subsequent row will have an image and its histogram side by side
    num_rows = 1 + num_gbuffer_channels  # First row for RGB/MSEG, then one row per channel
    num_cols = 2
    
    # Create figure
    plt.figure(figsize=(15, 5 * num_rows))
    
    # Plot RGB image
    plt.subplot(num_rows, num_cols, 1)
    plt.imshow(rgb_img)
    plt.title('RGB Image')
    plt.axis('off')
    
    # Plot MSEG mask
    plt.subplot(num_rows, num_cols, 2)
    plt.imshow(mseg_mask, cmap='gray')
    plt.title('MSEG Segmentation')
    plt.axis('off')
    
    # Plot gbuffer channels and their histograms
    for i in range(num_gbuffer_channels):
        # Plot the gbuffer channel
        ax_img = plt.subplot(num_rows, num_cols, (i + 1) * 2 + 1)
        channel = gbuffer_array[..., i]
        channel_norm = (channel - channel.min()) / (channel.max() - channel.min())
        plt.imshow(channel_norm, cmap='gray')
        plt.title(f'GBuffer Channel {i}')
        plt.axis('off')
        
        # Plot the histogram
        ax_hist = plt.subplot(num_rows, num_cols, (i + 1) * 2 + 2)
        plot_gbuffer_histogram(channel, ax_hist, i)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Visualization saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Visualize components of the synthetic dataset')
    parser.add_argument('dataset_dir', type=str, help='Path to the synthetic dataset directory')
    parser.add_argument('--rgb', type=str, help='Path to specific RGB image to visualize (optional)', default=None)
    parser.add_argument('--output', type=str, default='visualization.png', help='Path to save the visualization')
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    
    # Load dataset file mappings
    file_mappings = load_dataset_files(dataset_dir)
    
    if args.rgb is not None:
        # Use specified RGB image
        rgb_path = args.rgb
        try:
            mseg_path, gbuffer_path = find_corresponding_files(rgb_path, file_mappings)
        except ValueError as e:
            print(e)
            return
    else:
        # Select a random sample
        sample = random.choice(file_mappings)
        rgb_path, mseg_path, gbuffer_path, _ = sample
    
    # Create visualization
    visualize_sample(rgb_path, mseg_path, gbuffer_path, args.output)

if __name__ == '__main__':
    main() 