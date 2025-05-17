#!/bin/bash
#SBATCH --job-name=gs_mpi
#SBATCH --reservation=fri
#SBATCH --output=gs_mpi_arr%A_task%a_%j.out # Simplified output name
#SBATCH --error=gs_mpi_arr.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1    # Each MPI rank gets 1 CPU
#SBATCH --time=00:20:00
#SBATCH --mem-per-cpu=2G     # Memory per allocated CPU (task)

# NB: The --ntasks directive should be provided in the sbatch command line
#     or set here to the MAXIMUM number of cores any array task will use.
#     Example: #SBATCH --ntasks=64

# --- Configuration ---
MPI_EXECUTABLE="gray_scott_mpi"
SOURCE_FILE="gray_scott_mpi.c"

SIMULATION_STEPS=5000
F_PARAM=0.060
K_PARAM=0.062
NUM_RUNS=5 # Set to 1 for debugging, increase to 3 later

# --- Environment Setup ---
echo "Job ID: ${SLURM_JOB_ID}, Array Job ID: ${SLURM_ARRAY_JOB_ID}, Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Running on host: $(hostname)"
echo "Allocated nodes: $SLURM_JOB_NODELIST"
echo "Number of tasks allocated by SLURM for this job step: $SLURM_NTASKS" # This should now reflect the --ntasks from sbatch
echo "Number of CPUs on node: $SLURM_CPUS_ON_NODE"
echo "Working directory: $(pwd)"
echo "Start time: $(date)"

echo "Loading MPI module..."
module purge
module load OpenMPI/4.1.6-GCC-13.2.0 # Ensure this is the correct and intended module
module list
echo "MPI module loaded."

# --- Compilation ---
echo "Compiling MPI Code: ${SOURCE_FILE} -> ${MPI_EXECUTABLE}"
which mpicc
mpicc "${SOURCE_FILE}" -o "${MPI_EXECUTABLE}" -lm -O3 -Wall
if [ $? -ne 0 ]; then
    echo "MPI compilation failed!"
    exit 1
fi
echo "Compilation successful."
ls -l "${MPI_EXECUTABLE}"

# --- Determine N and Cores based on SLURM_ARRAY_TASK_ID ---
ALL_GRID_SIZES=(256 512 1024 2048 4096)
ALL_CORE_COUNTS=(1 2 4 16 32 64)

NUM_GRID_SIZES=${#ALL_GRID_SIZES[@]}
NUM_CORE_COUNTS=${#ALL_CORE_COUNTS[@]}

GRID_SIZE_IDX=$((SLURM_ARRAY_TASK_ID / NUM_CORE_COUNTS))
CORE_COUNT_IDX=$((SLURM_ARRAY_TASK_ID % NUM_CORE_COUNTS))

N_GLOBAL=${ALL_GRID_SIZES[$GRID_SIZE_IDX]}
CORES=${ALL_CORE_COUNTS[$CORE_COUNT_IDX]}

if [ -z "$N_GLOBAL" ] || [ -z "$CORES" ]; then
    echo "Error: Invalid N_GLOBAL or CORES from SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

echo "----------------------------------------------------"
echo "Running Configuration:"
echo "  Grid Size (N_GLOBAL): ${N_GLOBAL}x${N_GLOBAL}"
echo "  Cores for mpirun (-np): ${CORES}"
echo "  SLURM_NTASKS available to job: ${SLURM_NTASKS}" # Should be >= CORES
echo "  Simulation Steps: ${SIMULATION_STEPS}"
echo "  F: ${F_PARAM}, k: ${K_PARAM}"
echo "  Number of Runs: ${NUM_RUNS}"
echo "----------------------------------------------------"

# Check if SLURM allocated enough tasks
if [ "${SLURM_NTASKS}" -lt "${CORES}" ]; then
    echo "ERROR: SLURM allocated ${SLURM_NTASKS} tasks, but this run requires ${CORES} cores."
    echo "Please submit the job array with 'sbatch --ntasks=<max_cores_needed_by_any_array_task> ...'"
    echo "For example, if max CORES is 64, use 'sbatch --ntasks=64 ...'"
    exit 1
fi


SUCCESSFUL_RUNS=0
# Unique temp file per array task to avoid clashes if they run on the same node concurrently
TEMP_TIME_FILE="times_N${N_GLOBAL}_cores${CORES}_arr${SLURM_ARRAY_JOB_ID}_task${SLURM_ARRAY_TASK_ID}.log"
> "${TEMP_TIME_FILE}"

for i in $(seq 1 $NUM_RUNS); do
    echo "  Run $i/$NUM_RUNS for N=${N_GLOBAL}, Cores=${CORES}..."
    
    # Use mpirun. It should pick up the SLURM allocation.
    # Add --oversubscribe as a last resort if slot calculation is still tricky,
    # but it's better to get the allocation right.
    # Omitting --hostfile and --host as OpenMPI under SLURM should detect this.
    RUN_OUTPUT=$(mpirun -np "${CORES}" ./"${MPI_EXECUTABLE}" "${N_GLOBAL}" "${SIMULATION_STEPS}" "${F_PARAM}" "${K_PARAM}" 2>&1)
    # If still issues with slots, you could try:
    # RUN_OUTPUT=$(mpirun -np "${CORES}" --mca pml ob1 --mca btl tcp,self ./"${MPI_EXECUTABLE}" "${N_GLOBAL}" "${SIMULATION_STEPS}" "${F_PARAM}" "${K_PARAM}" 2>&1)
    # Or, as a diagnostic:
    # RUN_OUTPUT=$(mpirun -np "${CORES}" --debug-daemons --leave-session-attached ./"${MPI_EXECUTABLE}" "${N_GLOBAL}" "${SIMULATION_STEPS}" "${F_PARAM}" "${K_PARAM}" 2>&1)


    EXEC_STATUS=$?
    CURRENT_TIME_FOUND=""

    # Check for the success message from your C code
    if [ $EXEC_STATUS -eq 0 ] && echo "${RUN_OUTPUT}" | grep -q "INFO: Rank 0 saved V grid"; then
        EXEC_TIME=$(echo "${RUN_OUTPUT}" | grep "TIME" | awk '{print $NF}')
        if [ -n "$EXEC_TIME" ]; then
            echo "    Run $i successful. Time: $EXEC_TIME s"
            echo "$EXEC_TIME" >> "${TEMP_TIME_FILE}"
            SUCCESSFUL_RUNS=$((SUCCESSFUL_RUNS + 1))
            CURRENT_TIME_FOUND="yes"
        fi
    fi
    
    if [ "$CURRENT_TIME_FOUND" != "yes" ]; then
        echo "    Run $i FAILED (mpirun exit status $EXEC_STATUS or PGM/Time message not found)."
        echo "--- Full Output for Run $i (N=${N_GLOBAL}, Cores=${CORES}) ---"
        echo "${RUN_OUTPUT}"
        echo "----------------------------------------------------------"
    fi
done

if [ "$SUCCESSFUL_RUNS" -gt 0 ]; then
    AVG_TIME=$(awk '{ total += $1; count++ } END { if (count > 0) print total/count; else print "N/A" }' "${TEMP_TIME_FILE}")
    echo "----------------------------------------------------"
    echo "SUMMARY: N_GLOBAL=${N_GLOBAL} CORES=${CORES} AVG_TIME=${AVG_TIME} s (from $SUCCESSFUL_RUNS successful runs)"
    echo "----------------------------------------------------"
else
    echo "----------------------------------------------------"
    echo "SUMMARY: No successful runs for N_GLOBAL=${N_GLOBAL}, Cores=${CORES}."
    echo "----------------------------------------------------"
fi

rm -f "${TEMP_TIME_FILE}" 
echo "End time: $(date)"