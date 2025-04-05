#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --reservation=fri
#SBATCH --job-name=parallel
#SBATCH --gpus=1
#SBATCH --output=parallel_out.log

module load CUDA
nvcc -diag-suppress 550 -O2 -lm parallel.cu -o parallel

# Create output directories
mkdir -p parallel_results/outputs
mkdir -p parallel_results/times

# Process all images in test_images
for img in test_images/*; do
    # Get filename without path
    filename=$(basename "$img")
    base="${filename%.*}"
    ext="${filename##*.}"
    
    # Create output filename
    output_img="parallel_results/outputs/${base}_eq.${ext}"
    
    # Run 7 iterations
    for i in {1..7}; do
        # Run program and capture output
        result=$(srun ./parallel "$img" "$output_img" 2>&1)
        
        # Extract dimensions and time
        dims=$(echo "$result" | grep "DIMS" | awk '{print $2 "x" $3}')
        time=$(echo "$result" | grep "TIME" | awk '{print $NF}')
        
        # Append to results file
        echo "$dims $time" >> "parallel_results/times/${base}_times.txt"
    done
    
    # Process the times file to have single line with 7 times
    awk 'NR==1 {dims=$1} {times=times $2 " "} END {print dims " " times}' \
        "parallel_results/times/${base}_times.txt" > \
        "parallel_results/times/${base}_processed.txt"
    
    # Cleanup temporary files
    rm "parallel_results/times/${base}_times.txt"
done

echo "All processing completed. Results in parallel_results/"