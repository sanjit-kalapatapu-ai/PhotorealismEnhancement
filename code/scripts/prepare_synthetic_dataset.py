#!/usr/bin/env python3

import os
import argparse
from synthetic_datasets.public.sdk.synthetic_dataset_loader import SyntheticDatasetLoader
import numpy as np
import cv2
from pathlib import Path
import csv
from multiprocessing import Pool, cpu_count
from functools import partial
import tqdm

# Mapping from Applied semantic IDs to MSeg universal taxonomy IDs

APPLIED_TO_MSEG_UNIVERSAL_TAXONOMY_MAP = {
    0: 194,    # NONE -> unlabeled
    1: 35,     # BUILDINGS -> building
    2: 144,    # FENCES -> fence
    3: 194,    # OTHER -> unlabeled
    4: 125,    # PEDESTRIANS -> person
    5: 143,    # POLES -> pole
    6: 98,     # ROADLINES -> road
    7: 98,     # ROADS -> road
    8: 100,    # SIDEWALKS -> sidewalk_pavement
    9: 174,    # VEGETATION -> vegetation
    10: 176,   # VEHICLES -> car
    11: 191,   # WALLS -> wall
    12: 135,   # TRAFFICSIGNS -> traffic_sign
    13: 194,   # CUSTOM_ASSET_0 -> unlabeled
    14: 194,   # CUSTOM_ASSET_1 -> unlabeled
    15: 194,   # CUSTOM_ASSET_2 -> unlabeled
    16: 194,   # CUSTOM_ASSET_3 -> unlabeled
    17: 194,   # CUSTOM_ASSET_4 -> unlabeled
    18: 194,   # CUSTOM_ASSET_5 -> unlabeled
    19: 194,   # CUSTOM_ASSET_6 -> unlabeled
    20: 194,   # CUSTOM_ASSET_7 -> unlabeled (military equipment)
    21: 194,   # CUSTOM_ASSET_8 -> unlabeled (parking space surfaces)
    22: 142,   # SKY -> sky
    23: 131,   # CONES -> road_barrier
    24: 131,   # BARRIERS -> road_barrier
    25: 16,    # ANIMALS -> animal_other
    26: 108,   # PROPS -> plaything_other
    27: 148,   # ROCKS -> rock
    28: 102,   # TERRAIN -> terrain
    29: 194,   # ROBOTS -> unlabeled
    30: 98,    # SHOULDERS -> road
    31: 143,   # REFLECTORS -> pole
    32: 125,   # PASSENGERS -> person
    33: 136,   # TRAFFIC_LIGHTS -> traffic_light
}

def convert_gt_seg_mask_to_mseg_mask(gt_seg_mask):
    """
    Convert Applied semantic labels to MSeg universal taxonomy labels.
    
    Args:
        gt_seg_mask: numpy array of shape (H, W) containing Applied semantic IDs
        
    Returns:
        grayscale_mseg_mask: numpy array of shape (W, H) with MSeg universal taxonomy IDs, transposed to match RGB orientation
    """
    # Verify input is 2D
    if len(gt_seg_mask.shape) != 2:
        raise ValueError(f"Expected 2D array, got shape {gt_seg_mask.shape}")
    
    # Find any Applied IDs that aren't in our mapping
    unique_ids = np.unique(gt_seg_mask)
    unmapped_ids = [id for id in unique_ids if id not in APPLIED_TO_MSEG_UNIVERSAL_TAXONOMY_MAP]
    if unmapped_ids:
        print(f"Warning: Found Applied IDs without MSeg mappings: {unmapped_ids}")
    
    mseg_mask = np.zeros_like(gt_seg_mask, dtype=np.uint8)
    for applied_id, mseg_id in APPLIED_TO_MSEG_UNIVERSAL_TAXONOMY_MAP.items():
        mseg_mask[gt_seg_mask == applied_id] = mseg_id
    
    # Transpose the mask to match RGB orientation
    return mseg_mask.T

def process_single_sample(sample_data, dataset, output_dirs, gbuffer_fields):
    """
    Process a single sample data entry.
    
    Args:
        sample_data: Sample data entry from dataset
        dataset: SyntheticDatasetLoader instance
        output_dirs: Dictionary of output directories
        gbuffer_fields: List of gbuffer fields to process
        
    Returns:
        list: File mapping [rgb_path, mseg_path, gbuffer_path, gt_seg_path] or None if error
    """
    try:
        # Get sensor information
        calibrated_sensor = dataset.get('calibrated_sensor', sample_data.calibrated_sensor_token)
        sensor = dataset.get('sensor', calibrated_sensor.sensor_token)
        
        if sensor.modality != "camera":
            return None

        # Extract RGB image and save as PNG
        rgb_filename = os.path.join(dataset.data_root, sample_data.image_file_path)
        rgb_basename = os.path.splitext(os.path.basename(rgb_filename))[0] + '.png'
        rgb_output_path = output_dirs['rgb'] / rgb_basename
        
        # Read and save as PNG
        img = cv2.imread(rgb_filename)
        cv2.imwrite(str(rgb_output_path), img)

        # Extract and stack g-buffer data
        sensor_output = dataset.get_sensor_outputs(sample_data.token)[0]
        
        # Save gt semantic segmentation 
        gt_seg = dataset.get_sensor_output_field_data(sensor_output.token, "GROUND_TRUTH_SEMANTIC_CLASS")
        gt_seg_filename = output_dirs['gt_seg'] / rgb_basename.replace('.png', '.npy')
        np.save(str(gt_seg_filename), gt_seg)

        # Compute robust label map from gt semantic segmentation
        mseg_mask = convert_gt_seg_mask_to_mseg_mask(gt_seg)
        mseg_filename = output_dirs['mseg'] / rgb_basename
        cv2.imwrite(str(mseg_filename), mseg_mask)
        
        # Stack g-buffer fields efficiently
        gbuffer_arrays = []
        for field in gbuffer_fields:
            data = dataset.get_sensor_output_field_data(sensor_output.token, field)
            if len(data.shape) == 2:
                data = data[..., np.newaxis]
            gbuffer_arrays.append(data)
        
        # Stack along channel dimension efficiently
        gbuffer = np.concatenate(gbuffer_arrays, axis=-1)
        gbuffer_filename = output_dirs['gbuffer'] / rgb_basename.replace('.png', '.npy')
        np.save(str(gbuffer_filename), gbuffer)
        
        return [
            str(rgb_output_path.absolute()),
            str(rgb_output_path.absolute().parent.parent / 'mseg' / rgb_output_path.name),
            str(gbuffer_filename.absolute()),
            str(gt_seg_filename.absolute())
        ]
    except Exception as e:
        print(f"Error processing sample {sample_data.token}: {str(e)}")
        return None

def compute_gbuffer_statistics(dataset_dir: Path, file_mappings: list) -> None:
    """
    Compute and save g-buffer statistics by sampling a subset of files.
    """
    print("Computing g-buffer statistics...")
    
    # Sample a subset of files
    num_samples = min(1000, int(0.1 * len(file_mappings)))
    sampled_files = np.random.choice([m[2] for m in file_mappings if m], size=num_samples, replace=False)
    
    # Load files in parallel
    with Pool(cpu_count()) as pool:
        gbuffer_arrays = list(tqdm.tqdm(
            pool.imap(np.load, sampled_files),
            total=len(sampled_files),
            desc="Loading gbuffer files"
        ))
    
    # Stack and compute statistics efficiently
    means = np.mean([arr.mean(axis=(0, 1)) for arr in gbuffer_arrays], axis=0)
    stds = np.std([arr.mean(axis=(0, 1)) for arr in gbuffer_arrays], axis=0)
    
    stats_file = dataset_dir / 'gbuffer_stats.npz'
    np.savez(stats_file, g_m=means, g_s=stds)
    print(f"Saved g-buffer statistics to {stats_file}")
    print(f"G-buffer means: {means}")
    print(f"G-buffer stds: {stds}")

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
        'gbuffer': dataset_dir / "gbuffer",
        'mseg': dataset_dir / "mseg"
    }
    
    # Fields to stack into gbuffer
    gbuffer_fields = [
        "GROUND_TRUTH_POSITION_X",      # depth - channel 0
        "GROUND_TRUTH_WORLD_NORMAL_X",  # normal x - channel 1
        "GROUND_TRUTH_WORLD_NORMAL_Y",  # normal y - channel 2
        "GROUND_TRUTH_WORLD_NORMAL_Z",   # normal z - channel 3
        "GROUND_TRUTH_WORLD_POSITION_X", # diffuse r - channel 4
        "GROUND_TRUTH_WORLD_POSITION_Y", # diffuse g - channel 5
        "GROUND_TRUTH_WORLD_POSITION_Z", # diffuse b - channel 6
        "PADDING_ZEROES", # roughness - channel 7
    ]

    # Create output directories
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # Load dataset
    dataset = SyntheticDatasetLoader.load_from_local_dataroot(local_dataroot=str(dataset_dir))
    
    # Collect all sample data entries
    all_sample_data = []
    for scene in dataset.list_scenes():
        for sample in dataset.get_samples(scene.token):
            all_sample_data.extend(dataset.get_sample_data(sample.token))
    
    # Process samples in parallel
    process_func = partial(process_single_sample, dataset=dataset, output_dirs=output_dirs, gbuffer_fields=gbuffer_fields)
    
    print(f"Processing {len(all_sample_data)} samples using {cpu_count()} processes...")
    with Pool(cpu_count()) as pool:
        file_mappings = list(tqdm.tqdm(
            pool.imap(process_func, all_sample_data),
            total=len(all_sample_data),
            desc="Processing samples"
        ))
    
    # Filter out None results from failed processing
    file_mappings = [m for m in file_mappings if m is not None]
    
    # Compute and save g-buffer statistics
    compute_gbuffer_statistics(dataset_dir, file_mappings)
    
    # Write CSV file
    csv_path = dataset_dir / f"{dataset_dir.name}_files.csv"
    print(f"Writing CSV file to {csv_path}")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(file_mappings)
    
    print(f"Found {len(file_mappings)} valid images")
    print(f"CSV file created at: {csv_path}")
    print("Processing complete!")

def main():
    parser = argparse.ArgumentParser(description='Prepare synthetic dataset by organizing data into appropriate subdirectories')
    parser.add_argument('dataset_dir', type=str, help='Path to the synthetic dataset directory')
    args = parser.parse_args()
    
    prepare_synthetic_dataset(args.dataset_dir)

if __name__ == '__main__':
    main()
