# Example: Compare execution times for 720x480.png across configurations
for file in sequential_results/*.png.txt; do
    echo "Configuration: $file"
    awk '{sum+=$1} END {print "Average time:", sum/NR}' "$file"
done