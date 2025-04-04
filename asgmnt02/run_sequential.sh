#!/bin/bash

# Compile the program
gcc sequential.c -o seq_histeq -lm -O3

# Create output directories
mkdir -p sequential_results/outputs
mkdir -p sequential_results/times

# Process all images in test_images
for img in test_images/*; do
    # Get filename without path
    filename=$(basename "$img")
    base="${filename%.*}"
    ext="${filename##*.}"
    
    # Create output filename
    output_img="sequential_results/outputs/${base}_eq.${ext}"
    
    # Run 7 iterations
    for i in {1..7}; do
        # Run program and capture output
        result=$(./seq_histeq "$img" "$output_img" 2>&1)
        
        # Extract dimensions and time
        dims=$(echo "$result" | grep "DIMS" | awk '{print $2 "x" $3}')
        time=$(echo "$result" | grep "TIME" | awk '{print $NF}')
        
        # Append to results file
        echo "$dims $time" >> "sequential_results/times/${base}_times.txt"
    done
    
    # Process the times file to have single line with 7 times
    awk 'NR==1 {dims=$1} {times=times $2 " "} END {print dims " " times}' \
        "sequential_results/times/${base}_times.txt" > \
        "sequential_results/times/${base}_processed.txt"
done

# Cleanup temporary files
rm sequential_results/times/*_times.txt

echo "All processing completed. Results in sequential_results/"