#! /bin/bash

# Exit on any error
set -e

# Function for logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function for error handling
handle_error() {
    log "ERROR: Step failed with exit code $1"
    exit 1
}

# Set up error handling
trap 'handle_error $?' ERR

# Usage: ./prepare_datasets.sh <real_dataset_dir> <synthetic_dataset_dir> <mseg_dir> <img_height> <img_width> 

if [ $# -ne 5 ]; then
    log "ERROR: Incorrect number of arguments"
    echo "Usage: ./prepare_datasets.sh <real_dataset_dir> <synthetic_dataset_dir> <mseg_dir> <img_height> <img_width>"
    exit 1
fi

real_dataset_dir=$1
synthetic_dataset_dir=$2
mseg_dir=$3
img_height=$4
img_width=$5

log "Starting dataset preparation with parameters:"
log "Real dataset directory: $real_dataset_dir"
log "Synthetic dataset directory: $synthetic_dataset_dir"
log "MSeg directory: $mseg_dir"
log "Image dimensions: ${img_height}x${img_width}"

real_dataset_name=$(basename "$real_dataset_dir")
synthetic_dataset_name=$(basename "$synthetic_dataset_dir")

# Real and synthetic dataset directories should be initially formatted as follows:
# real_dataset_dir/
#   rgb/
#     img_1.jpg
#     img_2.jpg
#     ...
# synthetic_dataset_dir/
#   v1/
#   scenarios/
#   data/
#   ...
#
# The synthetic dataset should contain RGB images, and the following sensor output list data:
# [GROUND_TRUTH_POSITION_X, GROUND_TRUTH_SEMANTIC_CLASS, GROUND_TRUTH_WORLD_NORMAL_X, GROUND_TRUTH_WORLD_NORMAL_Y, GROUND_TRUTH_WORLD_NORMAL_Z]
#
# Note: the model currently takes in images of a fixed resolution denoted by the img_height and img_width.

# Check if required directories exist
if [ ! -d "$real_dataset_dir/rgb" ]; then
    log "ERROR: RGB directory not found in real dataset: $real_dataset_dir/rgb"
    exit 1
fi

if [ ! -d "$synthetic_dataset_dir" ]; then
    log "ERROR: Synthetic dataset directory not found: $synthetic_dataset_dir"
    exit 1
fi

if [ ! -d "$mseg_dir" ]; then
    log "ERROR: MSeg directory not found: $mseg_dir"
    exit 1
fi

# Generate the mseg outputs for real dataset:
# log "Generating MSeg outputs for real dataset..."
# python -u mseg_semantic/tool/universal_demo.py  --config=$mseg_dir/default_config_360_ms.yaml model_name mseg-3m model_path $mseg_dir/mseg-3m.pth input_file $real_dataset_dir/rgb/ save_folder $real_dataset_dir/mseg/
# log "✓ MSeg outputs for real dataset generated"

# Prepare the real dataset
# log "Preparing real dataset..."
# python prepare_real_dataset.py $real_dataset_dir
# log "✓ Real dataset prepared"

# Prepare the synthetic dataset
# log "Preparing synthetic dataset..."
# python /home/sanjitk/repos/PhotorealismEnhancement/code/scripts/prepare_synthetic_dataset.py $synthetic_dataset_dir
# log "✓ Synthetic dataset prepared"

# Generate the mseg outputs for synthetic dataset:
# log "Generating MSeg outputs for synthetic dataset..."
# python -u /home/sanjitk/repos/mseg-semantic/mseg_semantic/tool/universal_demo.py  --config=$mseg_dir/default_config_360_ms.yaml model_name mseg-3m model_path $mseg_dir/mseg-3m.pth input_file $synthetic_dataset_dir/rgb/ save_folder $synthetic_dataset_dir/mseg/
# log "✓ MSeg outputs for synthetic dataset generated"

# Generate crops of real and synthetic datasets:
# log "Generating crops for datasets..."
# python /home/sanjitk/repos/PhotorealismEnhancement/code/epe/matching/feature_based/collect_crops.py "$real_dataset_name" "${real_dataset_dir}/${real_dataset_name}_files.csv" --out_dir "$real_dataset_dir"
# python /home/sanjitk/repos/PhotorealismEnhancement/code/epe/matching/feature_based/collect_crops.py "$synthetic_dataset_name" "${synthetic_dataset_dir}/${synthetic_dataset_name}_files.csv" --out_dir "$synthetic_dataset_dir"
# log "✓ Crops generated"

# # Match crops of real and synthetic datasets:
# log "Matching crops between datasets..."
# python /home/sanjitk/repos/PhotorealismEnhancement/code/epe/matching/feature_based/find_knn.py "${synthetic_dataset_dir}/crop_${synthetic_dataset_name}.npz" "${real_dataset_dir}/crop_${real_dataset_name}.npz" "${synthetic_dataset_dir}/knn_matches.npz" -k 10
# log "✓ Crops matched"

# # Generate visualization of matched crops:
# log "Generating visualizations of matched crops..."
# python /home/sanjitk/repos/PhotorealismEnhancement/code/epe/matching/feature_based/sample_matches.py \
#     "${synthetic_dataset_dir}/${synthetic_dataset_name}_files.csv" \
#     "${synthetic_dataset_dir}/crop_${synthetic_dataset_name}.csv" \
#     "${real_dataset_dir}/${real_dataset_name}_files.csv" \
#     "${real_dataset_dir}/crop_${real_dataset_name}.csv" \
#     "${synthetic_dataset_dir}/knn_matches.npz" \
#     --output_dir "${synthetic_dataset_dir}/knn_match_viz"
# log "✓ Visualizations generated"

# #Filter crops within 1.0 distance threshold in VGG embedding space:
# log "Filtering crops with distance threshold..."
# python /home/sanjitk/repos/PhotorealismEnhancement/code/epe/matching/filter.py \
#     "${synthetic_dataset_dir}/knn_matches.npz" \
#     "${synthetic_dataset_dir}/crop_${synthetic_dataset_name}.csv" \
#     "${real_dataset_dir}/crop_${real_dataset_name}.csv" \
#     1.0 \
#     "${synthetic_dataset_dir}/knn_matches_filtered.csv"
# log "✓ Crops filtered"

# # Create sampling weights for crops:
# log "Creating sampling weights..."
# python /home/sanjitk/repos/PhotorealismEnhancement/code/epe/matching/compute_weights.py \
#     "${synthetic_dataset_dir}/knn_matches_filtered.csv" \
#     "$img_height" \
#     "$img_width" \
#     "${synthetic_dataset_dir}/crop_weights.npz"
# log "✓ Sampling weights created"

# Kick off training:
python /home/sanjitk/repos/PhotorealismEnhancement/code/epe/EPEExperiment.py train ./config/train_synth2cs.yaml --log=info

# Kick off inference using trained model:
# python /home/sanjitk/repos/PhotorealismEnhancement/code/epe/EPEExperiment.py test ./config/test_synth2cs.yaml

log "All steps completed successfully!"