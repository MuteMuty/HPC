#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=sequential
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=sequential_out.log

gcc -lm sequential.c -o sequential

# Function to run srun 20 times for each image
run_multiple_times() {
    for i in {1..20}; do
        srun sequential "$1" "$2"
    done
}

# Run each test 20 times
run_multiple_times test_images/720x480.png 592x480.png
run_multiple_times test_images/1024x768.png 896x768.png
run_multiple_times test_images/1920x1200.png 1892x1200.png
run_multiple_times test_images/3840x2160.png 3712x2160.png
run_multiple_times test_images/7680x4320.png 7552x4320.png
