from sim.scans import run_ft, run
import h5py
import matplotlib.pyplot as plt
import numpy as np
ttype = 'A2B2'
resolution = 100
run(resolution = resolution, ttype=ttype)


# file_path = "data/A2B1_100_01-24_14-02_A2.h5"
# # Replace with the actual file path
# with h5py.File(file_path, 'r') as f:
#     # Step 2: Load the datasets
#     theta_values = f['theta_values'][:]
#     phi_values = f['phi_values'][:]
#     c_magnitudes = f['c_magnitudes'][:]
# B_values = np.linspace(0.0001, 5, resolution)
# # Step 3: Select a specific B index (e.g., the first slice along B axis)
# b_index = 50  # Adjust this as needed
# for i in range(10):
#     b_index = 9*i
#     c_slice = c_magnitudes[:, :, b_index]
#
#     # Step 4: Create a color plot
#     theta, phi = np.meshgrid(theta_values, phi_values, indexing='ij')
#     plt.figure(figsize=(8, 6))
#     plt.pcolormesh(phi, theta, c_slice, shading='auto', cmap='viridis')
#     plt.colorbar(label='|c| Magnitude')
#     plt.xlabel('Phi (rad)')
#     plt.ylabel('Theta (rad)')
#     plt.title(f'{ttype} Overlap for B={B_values[b_index]:.2f} T')
#     plt.show()