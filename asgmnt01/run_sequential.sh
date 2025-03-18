#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=sequential
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=sequential_out.log

gcc -lm sequential.c -o sequential

srun sequential test_images/720x480.png 592x480.png
#srun sequential test_images/1024x1200.png 896x1200.png
#srun sequential test_images/1920x1080.png 1892x1080.png
#srun sequential test_images/3840x2160.png 3712x2160.png
#srun sequential test_images/7680x4320.png 7552x4320.png