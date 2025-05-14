#!/usr/bin/env python3

import os
import argparse
from synthetic_datasets.public.sdk.synthetic_dataset_loader import SyntheticDatasetLoader
import numpy as np
import cv2
from pathlib import Path
import csv

def compute_gbuffer_statistics(gbuffer_arrays):
    """
    Compute mean and standard deviation for each channel of g-buffer data.
    
    Args:
        gbuffer_arrays: List of g-buffer arrays, each of shape (H, W, 4)
        
    Returns:
        means: Array of shape (4,) with mean per channel
        stds: Array of shape (4,) with standard deviation per channel
    """
    # Stack all g-buffers along first dimension
    all_gbuffers = np.stack(gbuffer_arrays, axis=0)  # (N, H, W, 4)
    
    # Compute statistics per channel
    means = np.mean(all_gbuffers, axis=(0,1,2))  # (4,)
    stds = np.std(all_gbuffers, axis=(0,1,2))  # (4,)
    
    return means, stds

def prepare_synthetic_dataset(dataset_dir: str) -> None:
    """
    Prepare synthetic dataset by organizing data into appropriate subdirectories.
    
    Args:
        dataset_dir: Path to the synthetic dataset directory
    """
    dataset_dir = Path(dataset_dir)
    # Setup output directories
    output_dirs = {
        'rgb': dataset_dir / "rgb",
        'gt_seg': dataset_dir / "gt_seg",
        'gbuffer': dataset_dir / "gbuffer"
    }
    
    # Fields to stack into gbuffer
    gbuffer_fields = [
        "GROUND_TRUTH_POSITION_X",      # depth - channel 0
        "GROUND_TRUTH_WORLD_NORMAL_X",  # normal x - channel 1
        "GROUND_TRUTH_WORLD_NORMAL_Y",  # normal y - channel 2
        "GROUND_TRUTH_WORLD_NORMAL_Z"   # normal z - channel 3
    ]

    # Create output directories
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # Store all file paths for CSV
    file_mappings = []
    # Store all g-buffer arrays for computing statistics
    all_gbuffers = []
    
    # Load dataset
    dataset = SyntheticDatasetLoader.load_from_local_dataroot(local_dataroot=str(dataset_dir))
    
    # Process all scenes and samples
    for scene in dataset.list_scenes():
        for sample in dataset.get_samples(scene.token):
            for sample_data in dataset.get_sample_data(sample.token):
                # Get sensor information
                calibrated_sensor = dataset.get('calibrated_sensor', sample_data.calibrated_sensor_token)
                sensor = dataset.get('sensor', calibrated_sensor.sensor_token)
                sensor_modality = sensor.modality

                print(f"Processing sample_data token: {sample_data.token}")

                if sensor_modality == "camera":
                    # Extract RGB image and save as PNG
                    rgb_filename = os.path.join(dataset_dir, sample_data.image_file_path)
                    rgb_basename = os.path.splitext(os.path.basename(rgb_filename))[0] + '.png'
                    rgb_output_path = output_dirs['rgb'] / rgb_basename
                    
                    # Read and save as PNG
                    img = cv2.imread(rgb_filename)
                    cv2.imwrite(str(rgb_output_path), img)

                    # Extract and stack g-buffer data
                    sensor_output = dataset.get_sensor_outputs(sample_data.token)[0]
                    
                    # Save semantic segmentation separately
                    gt_seg = dataset.get_sensor_output_field_data(sensor_output.token, "GROUND_TRUTH_SEMANTIC_CLASS")
                    gt_seg_filename = output_dirs['gt_seg'] / rgb_basename.replace('.png', '.npy')
                    np.save(str(gt_seg_filename), gt_seg)
                    
                    # Stack g-buffer fields
                    gbuffer_arrays = []
                    for field in gbuffer_fields:
                        data = dataset.get_sensor_output_field_data(sensor_output.token, field)
                        # Ensure data is 3D with channel dimension
                        if len(data.shape) == 2:
                            data = data[..., np.newaxis]
                        gbuffer_arrays.append(data)
                    
                    # Stack along the channel dimension
                    gbuffer = np.concatenate(gbuffer_arrays, axis=-1)
                    gbuffer_filename = output_dirs['gbuffer'] / rgb_basename.replace('.png', '.npy')
                    np.save(str(gbuffer_filename), gbuffer)
                    
                    # Store g-buffer for statistics computation
                    all_gbuffers.append(gbuffer)
                    
                    # Store absolute paths for CSV
                    file_mappings.append([
                        str(rgb_output_path.absolute()),
                        str(rgb_output_path.absolute().parent.parent / 'mseg' / 'gray' / rgb_output_path.name), # TODO Sanjit: this assumes we have already run mseg on the synth rgb data, we should incorporate it into this script.
                        str(gbuffer_filename.absolute()),
                        str(gt_seg_filename.absolute())
                    ])
                else:
                    print(f"Skipping non-camera modality: {sensor_modality}")
    
    # Compute and save g-buffer statistics
    print("Computing g-buffer statistics...")
    g_means, g_stds = compute_gbuffer_statistics(all_gbuffers)
    stats_file = dataset_dir / 'gbuffer_stats.npz'
    np.savez(
        stats_file,
        g_m=g_means,  # shape: (4,) - one mean per channel
        g_s=g_stds    # shape: (4,) - one std per channel
    )
    print(f"Saved g-buffer statistics to {stats_file}")
    print(f"G-buffer means: {g_means}")
    print(f"G-buffer stds: {g_stds}")
    
    # Write CSV file
    csv_path = dataset_dir / f"{dataset_dir.name}_files.csv"
    print(f"Writing CSV file to {csv_path}")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write data
        writer.writerows(file_mappings)
    
    print("Processing complete!")
    print(f"Found {len(file_mappings)} images")
    print(f"CSV file created at: {csv_path}")

def main():
    parser = argparse.ArgumentParser(description='Prepare synthetic dataset by organizing data into appropriate subdirectories')
    parser.add_argument('dataset_dir', type=str, help='Path to the synthetic dataset directory')
    args = parser.parse_args()
    
    prepare_synthetic_dataset(args.dataset_dir)

if __name__ == '__main__':
    main()
