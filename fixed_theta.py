from transitions import transitioncompute
import numpy as np
import h5py
from tqdm.auto import tqdm
import datetime

# Fixed theta value
theta_fixed = np.pi / 2

# Resolutions for B and phi
resolution = 100
B_values = np.linspace(0, 5, resolution)
phi_values = np.linspace(0, 2 * np.pi, resolution)

# Preallocate arrays
c_magnitudes = np.zeros((len(phi_values), len(B_values)))
energy_ground = np.zeros((len(phi_values), len(B_values), 4))
energy_excited = np.zeros((len(phi_values), len(B_values), 4))
eigenvectors_ground = np.zeros((len(phi_values), len(B_values), 4, 4), dtype=complex)
eigenvectors_excited = np.zeros((len(phi_values), len(B_values), 4, 4), dtype=complex)

# Calculate |c| for each combination of phi and B
for j in tqdm(range(len(phi_values))):
    for k in range(len(B_values)):
        # Convert spherical to Cartesian coordinates for B field
        Bx = B_values[k] * np.sin(theta_fixed) * np.cos(phi_values[j])
        By = B_values[k] * np.sin(theta_fixed) * np.sin(phi_values[j])
        Bz = B_values[k] * np.cos(theta_fixed)
        B = [Bx, By, Bz]

        # Compute the model
        model = transitioncompute(B)

        # Store energy levels and eigenvectors
        energy_ground[j, k, :], energy_excited[j, k, :] = model.return_levels()
        eigenvectors_ground[j, k, :, :], eigenvectors_excited[j, k, :, :] = model.return_vectors()

        # Calculate |c|
        c = model.get_c_magnitudes()
        c_magnitudes[j, k] = c

# Save results with timestamp
timestamp = datetime.datetime.now().strftime("%m-%d_%H-%M")
with h5py.File(fr'fixed_{resolution}_{timestamp}.h5', 'w') as f:
    # Save parameter arrays
    f.create_dataset('B_values', data=B_values)
    f.create_dataset('phi_values', data=phi_values)
    f.create_dataset('theta_values', data=theta_fixed)

    # Save c magnitudes
    f.create_dataset('c_magnitudes', data=c_magnitudes)

    # Save energy spectra
    grp_ground = f.create_group('ground_state')
    grp_ground.create_dataset('energy_levels', data=energy_ground)
    grp_ground.create_dataset('eigenvectors', data=eigenvectors_ground)

    grp_excited = f.create_group('excited_state')
    grp_excited.create_dataset('energy_levels', data=energy_excited)
    grp_excited.create_dataset('eigenvectors', data=eigenvectors_excited)



