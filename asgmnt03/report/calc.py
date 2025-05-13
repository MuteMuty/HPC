import matplotlib.pyplot as plt
import numpy as np

# Data
grid_sizes = ['256x256', '512x512', '1024x1024', '2048x2048', '4096x4096']
patterns = ['Default', 'Flower', 'Mazes', 'Mitosis', 'Solitons']

# Speedup data organized as [pattern][grid_size]
speedups = {
    'Default': [128.56, 202.76, 240.00, 263.16, 221.21],
    'Flower': [135.13, 204.18, 237.06, 267.62, 221.03],
    'Mazes': [130.00, 206.40, 204.55, 271.82, 222.58],
    'Mitosis': [134.16, 207.13, 222.52, 273.69, 229.33],
    'Solitons': [127.27, 199.41, 223.39, 265.58, 223.66]
}

# Plot settings
plt.figure(figsize=(12, 7))
plt.title('Speedup Comparison of Different Patterns Across Grid Sizes', fontsize=14)
plt.xlabel('Grid Size', fontsize=12)
plt.ylabel('Speedup (Sequential Time / CUDA Time)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# X-axis as numeric values for better spacing
x = np.arange(len(grid_sizes))
width = 0.15  # Width of each bar

# Plot each pattern
for i, pattern in enumerate(patterns):
    plt.bar(x + i*width - width*2, speedups[pattern], width=width, label=pattern)

plt.xticks(x, grid_sizes)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Optional: Add value labels on top of bars
ax = plt.gca()
for i, pattern in enumerate(patterns):
    for j, val in enumerate(speedups[pattern]):
        ax.text(x[j] + i*width - width*2, val + 5, f'{val:.1f}', 
                ha='center', va='bottom', fontsize=8)

plt.show()