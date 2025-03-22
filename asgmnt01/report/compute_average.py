from collections import defaultdict

# Read the results file
results_file = "execution_times.txt"  # Update with your file path
with open(results_file, "r") as file:
    lines = file.readlines()

# Dictionary to store times for each combination of cores, threads, and image
results = defaultdict(list)

# Parse the file and group times by cores, threads, and image
for line in lines:
    if "Cores:" in line:
        parts = line.strip().split(", ")
        cores = int(parts[0].split(": ")[1])  # Extract cores
        threads = int(parts[1].split(": ")[1])  # Extract threads
        image = parts[2].split(": ")[1]  # Extract image name
        time = float(parts[3].split(": ")[1].replace(" sec", ""))  # Extract time
        
        # Use a tuple of (cores, threads, image) as the key
        key = (cores, threads, image)
        results[key].append(time)

# Compute averages for each combination
averages = {}
for key, times in results.items():
    avg_time = sum(times) / len(times)
    averages[key] = avg_time

# Print the results in a structured way
print("Average Execution Times:")
print("{:<10} {:<10} {:<15} {:<10}".format("Cores", "Threads", "Image", "Avg Time (sec)"))
print("=" * 50)
for (cores, threads, image), avg_time in sorted(averages.items()):
    print("{:<10} {:<10} {:<15} {:<10.4f}".format(cores, threads, image, avg_time))