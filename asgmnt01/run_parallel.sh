#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=parallel_seam_carving
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16  # Maximum number of cores available
#SBATCH --output=parallel_out.log

# Define the combinations of cores and threads to test as a list of tuples
declare -a core_thread_combinations=(
    "16 16"
    "16 32"
)

# Function to run srun for a given number of cores and threads
run_with_cores_and_threads() {
    local cores=$1
    local threads=$2
    export OMP_NUM_THREADS=$threads

    # Set OpenMP thread placement
    export OMP_PLACES="cores"
    export OMP_PROC_BIND="close"

    echo "Running with $cores cores and $threads threads..."

    # Function to run srun 30 times for each image
    run_multiple_times() {
        for i in {1..20}; do
            # Use SLURM's CPU binding
            srun --cpu-bind=cores ./parallel "$1" "$2"
        done
    }

    # Run each test 30 times
    run_multiple_times test_images/720x480.png 592x480.png
    run_multiple_times test_images/1024x768.png 896x768.png
    run_multiple_times test_images/1920x1200.png 1892x1200.png
    run_multiple_times test_images/3840x2160.png 3712x2160.png
    run_multiple_times test_images/7680x4320.png 7552x4320.png
}

# Test all combinations of cores and threads
for combination in "${core_thread_combinations[@]}"; do
    # Split the combination into cores and threads
    cores=$(echo $combination | awk '{print $1}')
    threads=$(echo $combination | awk '{print $2}')
    
    echo "Testing configuration: $cores cores, $threads threads"
    run_with_cores_and_threads $cores $threads
done