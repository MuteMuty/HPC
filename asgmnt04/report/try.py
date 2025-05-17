import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- YOU MUST PROVIDE YOUR ACTUAL SEQUENTIAL TIMES (ts) HERE ---
# (Matching the F, k values of your MPI runs, e.g., Default pattern)
sequential_times_ts = {
    256: 3.65,    # Placeholder - REPLACE
    512: 12.38,   # Placeholder - REPLACE
    1024: 65.05,  # Placeholder - REPLACE
    2048: 258.33, # Placeholder - REPLACE
    4096: 867.52  # Placeholder - REPLACE
}

# Your MPI parallel times (tp)
# Grid Size label, 1 core, 2 cores, 4 cores, 16 cores, 32 cores, 64 cores
mpi_times_tp_data = {
    'Grid Size Label': ['$256^2$', '$512^2$', '$1024^2$', '$2048^2$', '$4096^2$'],
    'N': [256, 512, 1024, 2048, 4096], # Numeric grid size for lookup
    1:  [3.60096, 11.8407, 62.4339, 239.252, 801.418],
    2:  [2.22889, 6.48054, 32.0298, 121.123, 401.763],
    4:  [1.35506, 3.86730, 18.8706, 62.8751, 219.462],
    16: [0.534491, 1.57683, 6.99392, 20.1589, 109.845],
    32: [0.366059, 1.01023, 4.35307, 11.6466, 84.6429],
    64: [145.375, 0.912569, 2.71798, 11.5991, 115.078]
}
df_mpi_times = pd.DataFrame(mpi_times_tp_data)
# print(df_mpi_times) # For debugging

# Calculate speedup S = ts / tp
cores_columns = [1, 2, 4, 16, 32, 64]
df_speedup = pd.DataFrame(index=df_mpi_times['Grid Size Label'], columns=cores_columns)

for index, row in df_mpi_times.iterrows():
    grid_size_numeric = row['N']
    ts = sequential_times_ts[grid_size_numeric]
    grid_label = row['Grid Size Label']
    for core_count in cores_columns:
        tp = row[core_count]
        if tp > 0: # Avoid division by zero if a time was 0 or invalid
            df_speedup.loc[grid_label, core_count] = ts / tp
        else:
            df_speedup.loc[grid_label, core_count] = np.nan # Or 0, or some indicator of bad data

# print("\nCalculated Speedups (ts/tp):") # For debugging
# print(df_speedup) # For debugging

# Core counts for x-axis
cores = np.array(cores_columns)

# --- Plotting (same as before, but uses the new df_speedup) ---
plt.style.use('seaborn-v0_8-whitegrid') 
plt.figure(figsize=(10, 6))

markers = ['o', 's', '^', 'D', 'X'] 
plot_colors = plt.cm.viridis(np.linspace(0, 0.9, len(df_speedup.index)))

for i, grid_size_label in enumerate(df_speedup.index):
    # Ensure speedup values are numeric for plotting, handle potential NaNs by not plotting them or plotting as 0
    speedup_values = pd.to_numeric(df_speedup.loc[grid_size_label].values, errors='coerce')
    plt.plot(cores, speedup_values,
             label=f'Grid {grid_size_label}',
             marker=markers[i % len(markers)],
             linestyle='-',
             color=plot_colors[i])

# Ideal speedup line (slope=1)
max_cores_for_ideal_line = cores[-1]
ideal_cores = np.array([1, max_cores_for_ideal_line]) # Plot ideal line only up to max cores tested
# Adjust ideal line if some actual speedups exceed max_cores (unlikely for S=ts/tp with MPI overhead)

# Plot ideal line. Consider the range of your actual speedups.
# If speedups are low, an ideal line going very high might dwarf the actual data.
# Let's plot ideal up to max_cores.


plt.title('MPI Speedup vs. Sequential C ($S = t_s/t_p$)', fontsize=16)
plt.xlabel('Number of Cores ($p$)', fontsize=14)
plt.ylabel('Speedup ($S$)', fontsize=14)

plt.xscale('log', base=2) 
plt.xticks(cores, labels=[str(c) for c in cores]) 

# Adjust y-axis if needed, e.g., if speedups are very high or very low.
# plt.ylim(bottom=0) # Ensure y-axis starts at 0 or slightly below smallest speedup

plt.legend(title='Grid Size ($N \\times N$)', fontsize=10, title_fontsize=12)
plt.grid(True, which="both", ls="-", alpha=0.5) 
plt.tight_layout() 

plt.savefig('mpi_speedup_plot_vs_seq1.png', dpi=300)
print("Plot saved as mpi_speedup_plot_vs_seq1.png")
plt.show()