#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=correct_parallel_test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=benchmark_results_correct.log

export OMP_PLACES=cores
export OMP_PROC_BIND=TRUE

gcc -O2 -lm -fopenmp parallel.c -o parallel

# Image configurations (input -> output)
declare -A image_pairs=(
    ["test_images/720x480.png"]="592x480.png"
    ["test_images/1024x768.png"]="896x768.png"
    ["test_images/1920x1200.png"]="1892x1200.png"
    ["test_images/3840x2160.png"]="3712x2160.png"
    ["test_images/7680x4320.png"]="7552x4320.png"
)

# Thread/core configurations to test
configurations=(1 2 4 8 16)

for cores in "${configurations[@]}"; do
    export OMP_NUM_THREADS=$cores
    echo "---- Testing with $cores cores/threads ----"

    for input_image in "${!image_pairs[@]}"; do
        output_image="${image_pairs[$input_image]}"
        echo "Processing $input_image (128 seams)"
        
        # Run 5 times for stable average
        for ((run=1; run<=5; run++)); do
            srun --cpus-per-task=$cores --ntasks=1 parallel "$input_image" "$output_image"
        done
    done
done
