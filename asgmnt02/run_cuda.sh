#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --reservation=fri
#SBATCH --gpus=1
#SBATCH --job-name=cuda_histeq
#SBATCH --output=cuda_results.log

module load CUDA
nvcc -diag-suppress 550 -O2 -lm cuda_histeq.cu -o cuda_histeq

# Create output directories
mkdir -p cuda_results/outputs
mkdir -p cuda_results/times

# Process test images
for img in test_images/*; do
    filename=$(basename "$img")
    base="${filename%.*}"
    ext="${filename##*.}"
    
    output="cuda_results/outputs/${base}_eq.${ext}"
    time_file="cuda_results/times/${base}_times.txt"
    
    echo "Processing $filename (7 runs)..."
    
    # Run 7 iterations
    for i in {1..7}; do
        srun --gres=gpu:1 --exclusive ./cuda_histeq "$img" "$output" 2>&1 | tee -a "$time_file"
    done
    
    # Process timing data
    awk '/DIMS/ {print $3 " " $5}' "$time_file" > "cuda_results/times/${base}_processed.txt"
done

echo "Processing completed. Results in cuda_results/"