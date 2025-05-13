#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --reservation=fri
#SBATCH --job-name=gs_patterns
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00 # Increased time for multiple patterns
#SBATCH --output=gs_patterns_%j.log

# --- Configuration ---
MODULE_TO_LOAD="CUDA"
GPU_ARCH="sm_70" # Adjust as needed!
SEQ_EXECUTABLE="gray_scott_seq"
CUDA_EXECUTABLE="gray_scott_cuda"
SEQ_RESULTS_DIR="sequential_results_patterns"
CUDA_RESULTS_DIR="cuda_results_patterns"
GRID_SIZES=(256 512 1024 2048 4096) # Reduced list for quicker testing, expand later
SIMULATION_STEPS=5000     # Keep steps high enough for patterns to emerge
NUM_RUNS=1                # Reduce runs per setting for quicker testing

# Define patterns (Name F K) - Add more as needed
declare -a PATTERNS=(
    "Default 0.060 0.062"
    "Flower 0.055 0.062"
    "Mazes 0.029 0.057"
    "Mitosis 0.028 0.062"
    "Solitons 0.030 0.060"
)
# --- End Configuration ---

echo "Loading module: $MODULE_TO_LOAD"
module load "$MODULE_TO_LOAD"
if [ $? -ne 0 ]; then echo "Failed to load module $MODULE_TO_LOAD"; exit 1; fi

# --- Compilation ---
echo "Compiling Sequential Code..."
gcc gray_scott_seq.c -o "$SEQ_EXECUTABLE" -lm -O3 -Wall
if [ $? -ne 0 ]; then echo "Sequential compilation failed!"; exit 1; fi

echo "Compiling CUDA Code for $GPU_ARCH..."
nvcc -gencode arch=compute_70,code=sm_70 \
     -gencode arch=compute_80,code=sm_80 \
     -O3 -lm gray_scott_cuda.cu -o "$CUDA_EXECUTABLE"
if [ $? -ne 0 ]; then echo "CUDA compilation failed!"; exit 1; fi
echo "Compilation successful."

# Create output directories
mkdir -p "${SEQ_RESULTS_DIR}/outputs" "${SEQ_RESULTS_DIR}/times"
mkdir -p "${CUDA_RESULTS_DIR}/outputs" "${CUDA_RESULTS_DIR}/times"
echo "Created directories"

# Function to run and process results
# Usage: run_benchmark <executable> <N> <steps> <F> <k> <pattern_name> <results_dir>
run_benchmark() {
    local executable=$1
    local N=$2
    local steps=$3
    local F_val=$4
    local k_val=$5
    local pattern_name=$6
    local results_dir=$7
    local exec_type=$(basename "$executable") # seq or cuda

    echo "    Running $exec_type: N=$N, Steps=$steps, Pattern=$pattern_name (F=$F_val, k=$k_val)"

    local TEMP_TIME_FILE="${results_dir}/times/${exec_type}_${N}x${N}_${pattern_name}_temp_times.txt"
    local PROCESSED_TIME_FILE="${results_dir}/times/${exec_type}_${N}x${N}_${pattern_name}_processed.txt"
    # Base name includes pattern details
    local OUTPUT_PGM_BASE=$(printf "%s/outputs/%s_N%d_steps%d_F%.3f_k%.3f_%s_V" \
                               "$results_dir" "$exec_type" "$N" "$steps" "$F_val" "$k_val" "$pattern_name")

    > "$TEMP_TIME_FILE" # Clear temp file

    for i in $(seq 1 $NUM_RUNS); do
        echo "      Run $i/$NUM_RUNS..."
        # Use srun for CUDA, direct execution for sequential is fine too
        if [[ "$exec_type" == "$CUDA_EXECUTABLE" ]]; then
             result=$(srun ./"$executable" "$N" "$steps" "$F_val" "$k_val" 2>&1)
        else
             result=$(./"$executable" "$N" "$steps" "$F_val" "$k_val" 2>&1)
        fi

        # --- CORRECTED Success Check ---
        local ran_ok=false
        local time_line_present=false

        # Check if the TIME line exists first (common requirement)
        if echo "$result" | grep -q "TIME"; then
            time_line_present=true
        fi

        # Check specific success conditions based on type
        if [[ "$exec_type" == "$CUDA_EXECUTABLE" ]]; then
            # For CUDA, check DIMS line AND the final success message exist anywhere in output
            if echo "$result" | grep -q "DIMS ${N}x${N}" && \
               echo "$result" | grep -q "INFO: CUDA simulation finished successfully."; then
                ran_ok=true
            fi
        else # For Sequential, just check DIMS line exists (simpler assumption)
            if echo "$result" | grep -q "DIMS ${N}x${N}"; then
                 ran_ok=true
            fi
        fi
        # --- End CORRECTED Success Check ---


        # Now, proceed only if basic checks passed AND the TIME line was found
        if [[ "$ran_ok" == true ]] && [[ "$time_line_present" == true ]]; then
            # --- SUCCESS BLOCK ---
            dims=$(echo "$result" | grep "DIMS" | awk '{print $2}')
            time=$(echo "$result" | grep "TIME" | awk '{print $NF}')
            echo "$dims $time" >> "$TEMP_TIME_FILE" # Write time to temp file
            echo "        Run $i successful. Time: $time"

            # PGM Handling (for last run) - Should now execute correctly for CUDA too
            if [ $i -eq $NUM_RUNS ]; then
                local generated_pgm_in_cwd=$(printf "%s_N%d_steps%d_F%.3f_k%.3f_V.pgm" "$exec_type" "$N" "$steps" "$F_val" "$k_val")
                # OUTPUT_PGM_BASE uses the printf fix from previous step
                local destination_pgm_path="${OUTPUT_PGM_BASE}_run${i}.pgm"

                echo "        Checking for PGM: './${generated_pgm_in_cwd}'"
                if [ -f "$generated_pgm_in_cwd" ]; then
                     echo "        Found. Moving to '$destination_pgm_path'"
                     mv "$generated_pgm_in_cwd" "$destination_pgm_path"
                     if [ $? -ne 0 ]; then echo "        ERROR: Failed move"; fi
                else
                    echo "        Warning: PGM file '$generated_pgm_in_cwd' not found."
                fi
            fi
            # --- End SUCCESS BLOCK ---

        else
            # --- ERROR BLOCK ---
            echo "      Error running $executable on run $i (Success check failed: ran_ok=$ran_ok, time_line_present=$time_line_present):"
            echo "$result" | sed 's/^/        /'
            echo "${N}x${N} ERROR" >> "$TEMP_TIME_FILE"
            # --- End ERROR BLOCK ---
        fi

    done # End runs loop

    # Process times
    awk -v n=$N -v pattern="$pattern_name" \
        'BEGIN {dims=""; times=""; count=0}
         !/ERROR/ { if (dims=="") dims=$1; times=times $2 " "; count++}
         END { if (dims=="") dims=n"x"n; print dims " " pattern " " times " (Successful runs: " count ")"}' \
        "$TEMP_TIME_FILE" > "$PROCESSED_TIME_FILE"
    echo "    Finished $exec_type Pattern $pattern_name for ${N}x${N}. Results: $PROCESSED_TIME_FILE"
    rm "$TEMP_TIME_FILE"
}

# --- Main Execution Logic ---
for N in "${GRID_SIZES[@]}"; do
    echo "===================================="
    echo "GRID SIZE: ${N}x${N}"
    echo "===================================="
    for p_config in "${PATTERNS[@]}"; do
        # Read pattern name, F, k from the string
        read -r pattern_name F_val k_val <<< "$p_config"

        echo "------------------------------------"
        echo "PATTERN: $pattern_name (F=$F_val, k=$k_val)"
        echo "------------------------------------"

        # Run Sequential Benchmark
        run_benchmark "$SEQ_EXECUTABLE" "$N" "$SIMULATION_STEPS" "$F_val" "$k_val" "$pattern_name" "$SEQ_RESULTS_DIR"

        # Run CUDA Benchmark
        run_benchmark "$CUDA_EXECUTABLE" "$N" "$SIMULATION_STEPS" "$F_val" "$k_val" "$pattern_name" "$CUDA_RESULTS_DIR"

    done # End pattern loop
done # End grid size loop

echo "===================================="
echo "All Benchmarking Completed."
echo "Sequential Results: $SEQ_RESULTS_DIR"
echo "CUDA Results: $CUDA_RESULTS_DIR"
echo "===================================="