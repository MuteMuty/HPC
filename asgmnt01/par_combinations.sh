#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=parallel_test
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=8
#SBATCH --output=parellel_out_%j.log

export OMP_PLACES=cores
export OMP_PROC_BIND=TRUE
export OMP_NUM_THREADS=8

gcc -O2 -lm --openmp parallel.c -o parellel

srun parellel test_images/720x480.png 592x480.png
srun parellel test_images/1024x768.png 896x768.png
srun parellel test_images/1920x1200.png 1892x1200.png
srun parellel test_images/3840x2160.png 3712x2160.png
srun parellel test_images/7680x4320.png 7552x4320.png