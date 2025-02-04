from src.transitions import transitioncompute
import numpy as np
import h5py
from tqdm.auto import tqdm
import datetime


def run_strain(resolution = 200, ttype = 'A1A2'):
    if ttype not in ['B1B2', 'A1A2', 'A1B2', 'A1B1', 'A2B1', 'A2B2']:
        raise ValueError(f"Invalid ttype: '{ttype}'. Must be one of ['B1B2', 'A1A2', 'A1B2', 'A1B1', 'A2B1', 'A2B2'].")


    ag = np.linspace(0, -238e9, resolution)
    ae = np.linspace(0, -76e9, resolution)
    bg = np.linspace(0, 238e9, resolution)
    be = 76e9

    ag_grid, ae_grid, bg_grid = np.meshgrid(ag, ae, bg)
    c_magnitudes = np.zeros_like(ag_grid)

    # Preallocate arrays for energy spectra and eigenvectors
    energy_ground = np.zeros((len(ag_grid), len(ae_grid), len(bg_grid), 4))
    energy_excited = np.zeros((len(ag_grid), len(ae_grid), len(bg), 4))
    eigenvectors_ground = np.zeros((len(ag_grid), len(ae_grid), len(bg_grid), 4, 4), dtype=complex)
    eigenvectors_excited = np.zeros((len(ag_grid), len(ae_grid), len(bg_grid), 4, 4), dtype=complex)

    # Calculate |c| for each combination of B, theta and phi
    for i in tqdm(range(len(ag))):
        for j in range(len(ae)):
            for k in range(len(bg)):
                theta = 1.82
                phi = np.pi/2
                b = 1
                # Convert spherical to Cartesian coordinates for B field
                Bx = b * np.sin(theta) * np.cos(phi)
                By = b * np.sin(theta) * np.sin(phi)
                Bz = b * np.cos(theta)
                B = [Bx, By, Bz]
                a1,a2,b1 = ag_grid[i,j,k], ae_grid[i,j,k], bg_grid[i,j,k]

                model = transitioncompute(B, strain=[a1, b1, 0, a2, be, 0])

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

    timestamp = datetime.datetime.now().strftime("%m-%d_%H")
    # Save data
    with h5py.File(fr'data/{ttype}_{resolution}_strain.h5', 'w') as f:
        # Save parameter arrays
        f.create_dataset('ag', data=ag)
        f.create_dataset('ae', data=ae)
        f.create_dataset('bg', data=bg)

        # Save c magnitudes
        f.create_dataset('c_magnitudes', data=c_magnitudes)

        # Save energy spectra
        grp_ground = f.create_group('ground_state')
        grp_ground.create_dataset('energy_levels', data=energy_ground)
        grp_ground.create_dataset('eigenvectors', data=eigenvectors_ground)

        grp_excited = f.create_group('excited_state')
        grp_excited.create_dataset('energy_levels', data=energy_excited)
        grp_excited.create_dataset('eigenvectors', data=eigenvectors_excited)


def run_phi_ae():
    ae = np.linspace(0, -76e9, resolution)

