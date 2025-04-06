import matplotlib.pyplot as plt

# Image sizes (we use the width as a proxy for the resolution)
image_sizes = ['720x480', '1024x768', '1920x1200', '3840x2160', '7680x4320']

# Speed-up factors (sequential time divided by parallel time)
parallel_speedup = [19.7, 29.3, 25.6, 35.4, 31.4]
bonus_speedup    = [31.0, 39.3, 49.9, 67.9, 65.0]

plt.figure(figsize=(8, 5))
plt.plot(image_sizes, parallel_speedup, marker='o', label='Parallel')
plt.plot(image_sizes, bonus_speedup, marker='s', label='Bonus (Shared Memory)')
plt.xlabel('Image Resolution')
plt.ylabel('Speed-up Factor')
plt.title('Speed-up of Parallel Histogram Equalization')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('speedup.png', dpi=300)
plt.show()
