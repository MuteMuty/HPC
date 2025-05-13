#!/bin/bash

# --- Configuration (Should match run_both.sh) ---
SEQ_RESULTS_DIR="sequential_results_patterns"
CUDA_RESULTS_DIR="cuda_results_patterns"
GRID_SIZES=(256 512 1024 2048 4096) # Match the sizes you benchmarked

# Define patterns (Name F K) - Must match run_both.sh
declare -a PATTERNS=(
    "Default 0.060 0.062"
    "Flower 0.055 0.062"
    "Mazes 0.029 0.057"
    "Mitosis 0.028 0.062"
    "Solitons 0.030 0.060"
)
# --- End Configuration ---

# Function to extract average time from processed file
# Input: filename
# Output: average_time or ERROR
get_average_time() {
    local file=$1
    if [ ! -f "$file" ]; then
        echo "ERROR"
        return
    fi
    # Awk: Sum fields from $3 onwards until '('; divide by count. Handle ERROR lines.
    awk '
    /ERROR/ { print "ERROR"; exit }
    {
        sum = 0;
        count = 0;
        for (i = 3; i <= NF; i++) {
            if ($i == "(Successful") { break; }
            sum += $i;
            count++;
        }
        if (count > 0) {
            printf "%.6f", sum / count;
        } else {
            print "ERROR"; # No time values found?
        }
    }' "$file"
}

echo "Calculating Speedups..."
printf "%-10s | %-10s | %-15s | %-15s | %-10s\n" "Grid Size" "Pattern" "Seq Time (s)" "CUDA Time (s)" "Speedup"
echo "---------------------------------------------------------------------"

for N in "${GRID_SIZES[@]}"; do
    for p_config in "${PATTERNS[@]}"; do
        # Read pattern name, F, k from the string
        read -r pattern_name F_val k_val <<< "$p_config"

        SEQ_TIME_FILE="${SEQ_RESULTS_DIR}/times/gray_scott_seq_${N}x${N}_${pattern_name}_processed.txt"
        CUDA_TIME_FILE="${CUDA_RESULTS_DIR}/times/gray_scott_cuda_${N}x${N}_${pattern_name}_processed.txt"

        ts=$(get_average_time "$SEQ_TIME_FILE")
        tp=$(get_average_time "$CUDA_TIME_FILE")

        speedup="N/A" # Default if error
        if [[ "$ts" != "ERROR" && "$tp" != "ERROR" && $(echo "$tp > 0" | bc -l) -eq 1 ]]; then
             # Use bc for floating point division
            speedup=$(echo "scale=2; $ts / $tp" | bc -l)
        elif [[ "$ts" == "ERROR" ]]; then
            ts="Error"
        elif [[ "$tp" == "ERROR" ]]; then
            tp="Error"
        fi

         printf "%-10s | %-10s | %-15s | %-15s | %-10s\n" "${N}x${N}" "$pattern_name" "$ts" "$tp" "$speedup"

    done # End pattern loop
     echo "---------------------------------------------------------------------"
done # End grid size loop

echo "Speedup calculation finished."