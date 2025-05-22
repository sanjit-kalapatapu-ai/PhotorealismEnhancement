#!/usr/bin/env python3

import os
import argparse
from pathlib import Path
import csv

def create_mapping_csv(dataset_dir: Path) -> None:
    """
    Create a CSV file mapping RGB images to their corresponding MSeg outputs.
    The CSV will contain absolute paths for both RGB and MSeg files.
    
    Args:
        dataset_dir: Path to the dataset directory containing 'rgb' and 'mseg' subdirectories
    """
    rgb_dir = dataset_dir / 'rgb'
    mseg_dir = dataset_dir / 'mseg'
    output_csv = dataset_dir / f"{dataset_dir.name}_files.csv"
    
    # Validate directories exist
    if not rgb_dir.exists():
        raise ValueError(f"RGB directory not found: {rgb_dir}")
    if not mseg_dir.exists():
        raise ValueError(f"MSeg directory not found: {mseg_dir}")
    
    # Get all RGB files
    rgb_files = list(rgb_dir.glob('*.jpg')) + list(rgb_dir.glob('*.png'))
    file_pairs = []
    
    print(f"Processing {len(rgb_files)} images...")
    for rgb_path in rgb_files:
        # Expected corresponding MSeg file (with _seg.png suffix)
        mseg_path = mseg_dir / 'gray' / f"{rgb_path.stem}.png"
        
        if not mseg_path.exists():
            print(f"Warning: No matching MSeg file for {rgb_path}")
            continue
            
        # Store absolute paths
        file_pairs.append((str(rgb_path.absolute()), str(mseg_path.absolute())))
    
    # Write CSV file
    print(f"Writing CSV file to {output_csv}")
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(file_pairs)
    
    print("Processing complete!")
    print(f"Found {len(file_pairs)} matching RGB-MSeg pairs")
    print(f"CSV file created at: {output_csv}")

def main():
    parser = argparse.ArgumentParser(description='Create CSV mapping between RGB and MSeg files')
    parser.add_argument('dataset_dir', type=Path, help='Path to dataset directory containing rgb and mseg subdirectories')
    args = parser.parse_args()
    
    create_mapping_csv(args.dataset_dir)

if __name__ == '__main__':
    main()
