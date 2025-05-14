#!/usr/bin/env python3

import os
import argparse
from PIL import Image
from pathlib import Path
from typing import Tuple

def validate_dimensions(img: Image.Image, target_width: int, target_height: int) -> bool:
    """
    Validate that we're only downsampling, not upsampling.
    
    Args:
        img: PIL Image object
        target_width: Desired output width
        target_height: Desired output height
    
    Returns:
        bool: True if dimensions are valid for downsampling
    """
    return img.width >= target_width and img.height >= target_height

def downsample_image(img: Image.Image, target_width: int, target_height: int) -> Image.Image:
    """
    Downsample an image to the target dimensions.
    
    Args:
        img: PIL Image object
        target_width: Desired output width
        target_height: Desired output height
    
    Returns:
        PIL Image object: Downsampled image
    """
    return img.resize((target_width, target_height), Image.Resampling.LANCZOS)

def process_directory(input_dir: str, output_dir: str, target_width: int, target_height: int) -> Tuple[int, int]:
    """
    Process all images in the input directory and save downsampled versions to output directory.
    
    Args:
        input_dir: Path to input directory containing images
        output_dir: Path to output directory for downsampled images
        target_width: Desired output width
        target_height: Desired output height
    
    Returns:
        Tuple[int, int]: Count of (successful, failed) operations
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Track success and failures
    success_count = 0
    failure_count = 0
    
    # Supported image extensions
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    # Process each file in the input directory
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        # Skip if not a file or not an image
        if not os.path.isfile(input_path) or not any(filename.lower().endswith(ext) for ext in valid_extensions):
            continue
            
        try:
            # Open and process image
            with Image.open(input_path) as img:
                # Convert to RGB if necessary (handles RGBA, etc.)
                if img.mode in ('RGBA', 'LA'):
                    img = img.convert('RGB')
                
                # Validate dimensions
                if not validate_dimensions(img, target_width, target_height):
                    print(f"Skipping {filename}: Target dimensions ({target_width}x{target_height}) "
                          f"larger than original ({img.width}x{img.height})")
                    failure_count += 1
                    continue
                
                # Downsample image
                downsampled = downsample_image(img, target_width, target_height)
                
                # Save downsampled image
                downsampled.save(output_path, quality=95, optimize=True)
                success_count += 1
                print(f"Successfully processed: {filename}")
                
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            failure_count += 1
            
    return success_count, failure_count

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Downsample images in a directory')
    parser.add_argument('input_dir', help='Input directory containing images')
    parser.add_argument('output_dir', help='Output directory for downsampled images')
    parser.add_argument('width', type=int, help='Target width in pixels')
    parser.add_argument('height', type=int, help='Target height in pixels')
    
    args = parser.parse_args()
    
    # Validate input directory exists
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        return
    
    # Validate dimensions are positive
    if args.width <= 0 or args.height <= 0:
        print("Error: Width and height must be positive integers")
        return
    
    # Process images
    success, failed = process_directory(args.input_dir, args.output_dir, args.width, args.height)
    
    # Print summary
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {success} images")
    print(f"Failed to process: {failed} images")

if __name__ == '__main__':
    main()
