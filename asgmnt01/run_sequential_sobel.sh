#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=sequential_sobel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=sequential_sobel_out.log

gcc -lm sequential_sobel.c -o sequential_sobel

# Function to run srun 50 times for each image
run_multiple_times() {
    for i in {1..30}; do
        srun sequential_sobel "$1" "$2"
    done
}

# Run each test 50 times
run_multiple_times test_images/720x480.png 592x480.png
run_multiple_times test_images/1024x768.png 896x768.png
run_multiple_times test_images/1920x1200.png 1892x1200.png
run_multiple_times test_images/3840x2160.png 3712x2160.png
run_multiple_times test_images/7680x4320.png 7552x4320.png
