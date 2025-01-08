from src.transitions import transitioncompute
import numpy as np
import h5py
from tqdm.auto import tqdm
import datetime

resolution = 200
B_values = np.linspace(0, 5, resolution)
theta_values = np.linspace(0, np.pi, resolution)
phi_values = np.linspace(0, 2 * np.pi, resolution)

B_grid, theta_grid, phi_grid = np.meshgrid(B_values, theta_values, phi_values)
c_magnitudes = np.zeros_like(B_grid)

# Preallocate arrays for energy spectra and eigenvectors
energy_ground = np.zeros((len(theta_values), len(phi_values), len(B_values), 4))
energy_excited = np.zeros((len(theta_values), len(phi_values), len(B_values), 4))
eigenvectors_ground = np.zeros((len(theta_values), len(phi_values), len(B_values), 4, 4), dtype=complex)
eigenvectors_excited = np.zeros((len(theta_values), len(phi_values), len(B_values), 4, 4), dtype=complex)

# Calculate |c| for each combination of B, theta and phi
for i in tqdm(range(len(theta_values))):
    for j in range(len(phi_values)):
        for k in range(len(B_values)):
            # Convert spherical to Cartesian coordinates for B field
            Bx = B_grid[i, j, k] * np.sin(theta_grid[i, j, k]) * np.cos(phi_grid[i, j, k])
            By = B_grid[i, j, k] * np.sin(theta_grid[i, j, k]) * np.sin(phi_grid[i, j, k])
            Bz = B_grid[i, j, k] * np.cos(theta_grid[i, j, k])
            B = [Bx, By, Bz]
            model = transitioncompute(B)

            # Store energy levels and eigenvectors
            energy_ground[i, j, k, :], energy_excited[i, j, k, :] = model.return_levels()
            eigenvectors_ground[i, j, k, :, :], eigenvectors_excited[i, j, k, :, :] = model.return_vectors()

            # Calculate |c|

            # c = model.get_c_magnitudes()
            c = model.get_overlapA2()
            c_magnitudes[i, j, k] = c

timestamp = datetime.datetime.now().strftime("%m-%d_%H-%M_A2")
# Save data
with h5py.File(fr'data_{resolution}_{timestamp}.h5', 'w') as f:
    # Save parameter arrays
    f.create_dataset('B_values', data=B_values)
    f.create_dataset('theta_values', data=theta_values)
    f.create_dataset('phi_values', data=phi_values)

    # Save c magnitudes
    f.create_dataset('c_magnitudes', data=c_magnitudes)

    # Save energy spectra
    grp_ground = f.create_group('ground_state')
    grp_ground.create_dataset('energy_levels', data=energy_ground)
    grp_ground.create_dataset('eigenvectors', data=eigenvectors_ground)

    grp_excited = f.create_group('excited_state')
    grp_excited.create_dataset('energy_levels', data=energy_excited)
    grp_excited.create_dataset('eigenvectors', data=eigenvectors_excited)