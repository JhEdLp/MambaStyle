#!/bin/bash

# Default directory containing the subfolders
BASE_DIR="./Final_Results/"

# Python scripts you want to execute
PYTHON_SCRIPT="./scripts/inference.py"
PYTHON_SCRIPT_1="./scripts/calculate_metrics.py"

# Starting folder name (if provided as an argument)
START_FROM_FOLDER="$1"

# List of folders to evaluate (if provided as additional arguments)
EVALUATE_FOLDERS=("${@:2}")

# If no specific folders are provided, evaluate all folders
if [ ${#EVALUATE_FOLDERS[@]} -eq 0 ]; then
  EVALUATE_ALL=true
else
  EVALUATE_ALL=false
fi

# If no starting folder is provided, start processing from the first folder
if [ -z "$START_FROM_FOLDER" ]; then
  START_PROCESSING=true
else
  START_PROCESSING=false
fi

# Define a list of worker counts for retry attempts
WORKER_COUNTS=(3 2 4 6)  # Example list of worker counts

# Loop through all subdirectories in the base directory
for folder in "$BASE_DIR"/*/; do
  # Extract the folder name without the path and trailing slash
  folder_name=$(basename "$folder")

  # Check if we should start processing
  if [ -z "$START_FROM_FOLDER" ] || [ "$folder_name" == "$START_FROM_FOLDER" ]; then
    START_PROCESSING=true
  fi

  # Skip folders until the specified start folder is reached
  if [ "$START_PROCESSING" = true ]; then
    # Check if the folder is in the list of folders to evaluate (or evaluate all if no list is provided)
    if [ "$EVALUATE_ALL" = true ] || [[ " ${EVALUATE_FOLDERS[@]} " =~ " ${folder_name} " ]]; then
      clear
      # Define paths for model and experiment directories
      model_path="${folder}/checkpoints/latest_model.pt"
      exp_dir="${folder}/test/"
      imgs_dir="${folder}/test/inference_results/"

      # Print the model path
      echo "Model path: $model_path"

      # Execute the first Python script (inference)
      python "$PYTHON_SCRIPT" --checkpoint_path "$model_path" --exp_dir "$exp_dir"

      # Execute the second Python script (metric calculation) with retry logic
      for attempt in "${WORKER_COUNTS[@]}"; do
        if python "$PYTHON_SCRIPT_1" --reconstr_path "$imgs_dir" --metrics_dir "$exp_dir" --metrics l2 lpips msssim fid --workers "$attempt"; then
            echo "Metric estimation completed successfully with workers=$attempt."
            break
        else
            echo "Error in the metric estimation. Retrying with workers=$attempt..."
        fi
      done
    fi
  fi
done