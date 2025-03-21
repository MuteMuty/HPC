#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=seam_carving_bonus
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16  # Maximum number of cores available
#SBATCH --output=seam_carving_%j.log
#SBATCH --time=00:30:00     # Maximum runtime (30 minutes)

export OMP_PLACES=cores
export OMP_PROC_BIND=TRUE

# Define the combinations of cores and threads to test
declare -A core_thread_combinations=(
    [1]=1      # 1 core, 1 thread
    [1]=2      # 1 core, 2 threads (oversubscription)
    [2]=2      # 2 cores, 2 threads
    [2]=4      # 2 cores, 4 threads (oversubscription)
    [4]=4      # 4 cores, 4 threads
    [4]=8      # 4 cores, 8 threads (oversubscription)
    [8]=8      # 8 cores, 8 threads
    [8]=16     # 8 cores, 16 threads (oversubscription)
    [16]=16    # 16 cores, 16 threads
    [16]=32    # 16 cores, 32 threads (oversubscription)
)

# Function to run the program with a given number of threads
run_with_threads() {
    local threads=$1
    export OMP_NUM_THREADS=$threads

    echo "Running with $threads threads..."

    # Function to run the program 30 times for each image
    run_multiple_times() {
        for i in {1..30}; do
            srun ./seam_carving_bonus "$1" "$2"
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
for cores in "${!core_thread_combinations[@]}"; do
    threads=${core_thread_combinations[$cores]}
    echo "Testing configuration: $cores cores, $threads threads"
    run_with_threads $threads
done