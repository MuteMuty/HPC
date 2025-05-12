#!/bin/bash

#SBATCH --job-name=gs_seq_bench   # Job name
#SBATCH --nodes=1                 # Run on a single node
#SBATCH --ntasks=1                # Run a single task
#SBATCH --cpus-per-task=1         # Use 1 CPU core
#SBATCH --output=gs_seq_bench_%j.log # Standard output and error log

# Compile the program (do this once, maybe outside the loop if source doesn't change)
echo "Compiling sequential code..."
gcc sequential.c -o sequential -lm -O3 -Wall
if [ $? -ne 0 ]; then echo "Compilation failed!"; exit 1; fi
echo "Compilation successful."

# Create output directories
mkdir -p sequential_results/outputs
mkdir -p sequential_results/times

# Define grid sizes and steps
GRID_SIZES=(256 512 1024 2048 4096) # Adjust as needed
SIMULATION_STEPS=5000               # As per assignment
NUM_RUNS=1                          # Number of times to run for averaging

# Process each grid size
for N in "${GRID_SIZES[@]}"; do
    echo "Processing grid size ${N}x${N}..."
    TIME_FILE="sequential_results/times/gs_seq_${N}x${N}_times.txt"
    PROCESSED_TIME_FILE="sequential_results/times/gs_seq_${N}x${N}_processed.txt"
    OUTPUT_PGM_BASE="sequential_results/outputs/gs_seq_N${N}_steps${SIMULATION_STEPS}_V"

    # Clear previous time file for this size if it exists
    > "$TIME_FILE"

    # Run NUM_RUNS iterations for averaging
    for i in $(seq 1 $NUM_RUNS); do
        echo "  Run $i/$NUM_RUNS for ${N}x${N}..."
        # Run program and capture output containing DIMS and TIME
        # Use taskset to potentially pin to a specific core if needed, though likely not necessary here
        result=$(srun ./sequential "$N" "$SIMULATION_STEPS" 2>&1)

        # Check if execution was successful (e.g., by checking exit code or output format)
        if [[ $result == *"DIMS"* && $result == *"TIME"* ]]; then
            # Extract dimensions and time
            dims=$(echo "$result" | grep "DIMS" | awk '{print $2}')
            time=$(echo "$result" | grep "TIME" | awk '{print $NF}')

            # Append to results file (format: DIMS TIME)
            echo "$dims $time" >> "$TIME_FILE"

            # Save the PGM from the *last* run only (optional, could save first or none)
            if [ $i -eq $NUM_RUNS ]; then
                # The C code saves the PGM, find the generated file and move/rename it
                generated_pgm="sequential_N${N}_steps${SIMULATION_STEPS}_V.pgm"
                if [ -f "$generated_pgm" ]; then
                    mv "$generated_pgm" "${OUTPUT_PGM_BASE}_run${i}.pgm"
                    echo "    Saved PGM: ${OUTPUT_PGM_BASE}_run${i}.pgm"
                else
                    echo "    Warning: PGM file $generated_pgm not found."
                fi
            fi
        else
            echo "    Error running sequential for ${N}x${N} on run $i:"
            echo "$result"
            # Optionally write an error marker to the time file
            echo "${N}x${N} ERROR" >> "$TIME_FILE"
        fi

    done # End of runs loop

    # Process the times file to have single line with NUM_RUNS times
    awk -v n=$N 'NR==1 {dims=$1} !/ERROR/ {times=times $2 " "} END {if (dims=="") dims=n"x"n; print dims " " times}' \
        "$TIME_FILE" > "$PROCESSED_TIME_FILE"

    echo "Finished processing ${N}x${N}. Results in $PROCESSED_TIME_FILE"
    echo "---"

done # End of grid sizes loop

echo "Sequential benchmarking completed. Results in sequential_results/"