import matplotlib.pyplot as plt

# Data
image_sizes = ["592x480", "896x768", "1892x1200", "3712x2160", "7552x4320"]
sequential_times = [1.43286, 3.35978, 2.15124, 28.1146, 112.177]
parallel_times = [0.6307, 1.1608, 0.6170, 3.8650, 14.8920]  # Best parallel times

# Compute speed-up
speed_up = [seq / par for seq, par in zip(sequential_times, parallel_times)]

# Plot
plt.figure(figsize=(8, 5))
plt.bar(image_sizes, speed_up, color='blue')
plt.xlabel("Image Size")
plt.ylabel("Speed-Up (Sequential / Parallel)")
plt.title("Speed-Up Achieved by Parallel Implementation")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("speedup.png", dpi=300, bbox_inches='tight')
plt.show()