import h5py
import numpy as np
import matplotlib.pyplot as plt
from transitions import transitioncompute
# File path (update this to the actual file name generated)
# filename = "fixed_100_11-25_15-40.h5"
#
# with h5py.File(filename, 'r') as f:
#     B_values = f['B_values'][:]
#     phi_values = f['phi_values'][:]
#     c_magnitudes = f['c_magnitudes'][:]
#     theta_fixed = f['theta_fixed'][()]

def magnitude_plot():

    # Create a meshgrid for plotting
    B_grid, phi_grid = np.meshgrid(B_values, phi_values)

    # Plot the magnitude |c|
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(phi_grid, B_grid, c_magnitudes, shading='auto', cmap='viridis')
    plt.colorbar(label='|c| Magnitude')
    plt.title(r"$|c|$ Magnitude as a Function of $\phi$ and $B$" + f"\n(Fixed θ = {theta_fixed:.2f} rad)")
    plt.xlabel(r"$\phi$ (rad)")
    plt.ylabel(r"$B$ (T)")
    plt.tight_layout()

    # Show the plot
    plt.show()

def fixed_plot(phi_index):

    phi_fixed = phi_values[phi_index]

    # Extract |c| for the chosen phi index
    c_magnitudes_fixed_phi = c_magnitudes[:, phi_index]

    # Plot |c| vs B for the chosen phi
    plt.figure(figsize=(8, 5))
    plt.plot(B_values, c_magnitudes_fixed_phi, marker='o', label=f"φ = {phi_fixed:.2f} rad")
    plt.title(r"$|c|$ vs $B$ for a Fixed $\phi$" + f"\n(Fixed θ = {theta_fixed:.2f} rad)")
    plt.xlabel(r"$B$ (T)")
    plt.ylabel(r"$|c|$ Magnitude")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # Show the plot
    plt.show()

labels = ["A1", "A2", "B1", "B2"]
model = transitioncompute([0,0,1])

amplitudes, dot_product_matrix = model.get_field_amplitudes()

plt.figure(figsize=(8, 6))
plt.imshow(dot_product_matrix, cmap="viridis", interpolation="nearest")

# Add labels
plt.colorbar(label="Dot Product Magnitude")
plt.xticks(range(len(labels)), labels)
plt.yticks(range(len(labels)), labels)
plt.title("Dot Product Matrix Heatmap")
plt.xlabel("Vectors")
plt.ylabel("Vectors")

# Annotate matrix values
for i in range(len(labels)):
    for j in range(len(labels)):
        plt.text(j, i, f"{dot_product_matrix[i, j]:.2f}",
                 ha="center", va="center", color="white" if dot_product_matrix[i, j] < dot_product_matrix.max() / 2 else "black")

# Show plot
plt.tight_layout()
plt.show()

plt.bar(labels, amplitudes)
plt.show()











