from src.transitions import transitioncompute
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils.analytic import approx_B
from utils.iqp_colors import uni

# Resolution
res = 100  # Reduce for faster computation

# Define parameter ranges
b_vals = np.linspace(0.001, 3, res)
theta_vals = np.linspace(0.0001, np.pi - 0.0001, res)
strain_vals = np.linspace(0.0001, 1e-3, res)

# Initialize 3D matrix to store minima values
cmat = np.zeros((res, res, res))

# Compute overlap values
for i in tqdm(range(res)):
    for j in range(res):
        for k in range(res):
            p = 0
            b = b_vals[i]
            t = theta_vals[j]

            Bx = b * np.sin(t) * np.cos(p)
            By = b * np.sin(t) * np.sin(p)
            Bz = b * np.cos(t)
            B = [Bx, By, Bz]

            exg = strain_vals[k]
            exyg = 0
            model = transitioncompute(B, strain=[exg, exyg])
            v1 = model.get_A2() / np.linalg.norm(model.get_A2())
            Ax2, Ay2 = model.convert_lab_frame(*v1)

            v2 = model.get_B1() / np.linalg.norm(model.get_B1())
            Ax, Ay = model.convert_lab_frame(*v2)

            c = np.abs(np.vdot([Ax2, Ay2], [Ax, Ay]))
            cmat[i, j, k] = c

# Extract minima surface
min_indices = np.argmin(cmat, axis=2)
B_surface, Theta_surface = np.meshgrid(b_vals, theta_vals)
Strain_surface = strain_vals[min_indices]

# Plot 3D surface
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(B_surface, Theta_surface, Strain_surface, cmap=uni, edgecolor='none')

ax.set_xlabel("B (T)")
ax.set_ylabel("Theta (rad)")
ax.set_zlabel("Strain")
ax.set_title("Minima Surface of \langle A2 | B1 \rangle")

plt.show()
