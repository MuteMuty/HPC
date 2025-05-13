#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --reservation=fri
#SBATCH --job-name=test
#SBATCH --gpus=1
#SBATCH --time=00:00:30
#SBATCH --output=test.log

# Define patterns (Name F K) - Add more as needed
declare -a PATTERNS=(
    "Default 0.060 0.062"
    "Flower 0.055 0.062"
    "Mazes 0.029 0.057"
    "Mitosis 0.028 0.062"
    "Solitons 0.030 0.060"
)

module load CUDA

nvcc -gencode arch=compute_70,code=sm_70 \
     -gencode arch=compute_80,code=sm_80 \
     -O3 -lm gray_scott_cuda.cu -o gray_scott_cuda_test

srun ./gray_scott_cuda_test 4096 10000 0.060 0.062