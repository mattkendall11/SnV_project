from src.transitions import transitioncompute
import numpy as np
import h5py
from tqdm.auto import tqdm
import datetime


def run(resolution = 200, ttype = 'A1A2'):
    if ttype not in ['B1B2', 'A1A2', 'A1B2', 'A1B1', 'A2B1', 'A2B2']:
        raise ValueError(f"Invalid ttype: '{ttype}'. Must be one of ['B1B2', 'A1A2', 'A1B2', 'A1B1', 'A2B1', 'A2B2'].")

    B_values = np.linspace(0.0001, 5, resolution)
    theta_values = np.linspace(0.0001, np.pi, resolution)
    phi_values = np.linspace(0.0001, 2 * np.pi, resolution)

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
                if ttype == 'A1A2':
                    c = model.A1A2_overlap()
                if ttype == 'A1B2':
                    c = model.A1B2_overlap()
                if ttype == 'A1B1':
                    c = model.A1B1_overlap()
                if ttype == 'A2B1':
                    c = model.A2B1_overlap()
                if ttype == 'A2B2':
                    c = model.A2B2_overlap()
                if ttype == 'B1B2':
                    c = model.B1B2_overlap()

                c_magnitudes[i, j, k] = c

    timestamp = datetime.datetime.now().strftime("%m-%d_%H-%M_A2")
    # Save data
    with h5py.File(fr'data/{ttype}_{resolution}_{timestamp}.h5', 'w') as f:
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


def run_ft(resolution = 200):
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
            c = model.A1B2_overlap()
            c_magnitudes[j, k] = c

    # Save results with timestamp
    timestamp = datetime.datetime.now().strftime("%m-%d_%H-%M")
    with h5py.File(fr'data/fixed_{resolution}_{timestamp}.h5', 'w') as f:
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


def run_lf(resolution = 200, ttype = 'A1A2'):
    if ttype not in ['B1B2', 'A1A2', 'A1B2', 'A1B1', 'A2B1', 'A2B2']:
        raise ValueError(f"Invalid ttype: '{ttype}'. Must be one of ['B1B2', 'A1A2', 'A1B2', 'A1B1', 'A2B1', 'A2B2'].")

    B_values = np.linspace(0.0001, 5, resolution)
    theta_values = np.linspace(0.0001, np.pi, resolution)
    phi_values = np.linspace(0.0001, 2 * np.pi, resolution)

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

                v1 = model.get_A2() / np.linalg.norm(model.get_A2())
                Ax2, Ay2 = model.convert_lab_frame(*v1)

                v2 = model.get_B1() / np.linalg.norm(model.get_B1())

                Ax, Ay = model.convert_lab_frame(*v2)

                c = np.abs(np.vdot([Ax2, Ay2], [Ax, Ay]))

                c_magnitudes[i, j, k] = c

    timestamp = datetime.datetime.now().strftime("%m-%d_%H")
    # Save data
    with h5py.File(fr'data/{ttype}_{resolution}_{timestamp}_lf.h5', 'w') as f:
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

