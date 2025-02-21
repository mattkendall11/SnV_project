from src.transitions import transitioncompute
import numpy as np
import h5py
from tqdm.auto import tqdm
import datetime


def run_strain(resolution = 200, ttype = 'A1A2'):
    if ttype not in ['B1B2', 'A1A2', 'A1B2', 'A1B1', 'A2B1', 'A2B2']:
        raise ValueError(f"Invalid ttype: '{ttype}'. Must be one of ['B1B2', 'A1A2', 'A1B2', 'A1B1', 'A2B1', 'A2B2'].")


    exg = np.linspace(0, 10**-3, resolution)
    exyg = np.linspace(0, 10**-3, resolution)
    exe = np.linspace(0, 10**-3, resolution)
    exye = 10**-4

    ag_grid, ae_grid, bg_grid = np.meshgrid(exg, exe, exyg)
    c_magnitudes = np.zeros_like(ag_grid)

    # Preallocate arrays for energy spectra and eigenvectors
    energy_ground = np.zeros((len(ag_grid), len(ae_grid), len(bg_grid), 4))
    energy_excited = np.zeros((len(ag_grid), len(ae_grid), len(bg_grid), 4))
    eigenvectors_ground = np.zeros((len(ag_grid), len(ae_grid), len(bg_grid), 4, 4), dtype=complex)
    eigenvectors_excited = np.zeros((len(ag_grid), len(ae_grid), len(bg_grid), 4, 4), dtype=complex)

    # Calculate |c| for each combination of B, theta and phi
    for i in tqdm(range(len(exg))):
        for j in range(len(exyg)):
            for k in range(len(exe)):
                theta = 1.82
                phi = np.pi/2
                b = 1
                # Convert spherical to Cartesian coordinates for B field
                Bx = b * np.sin(theta) * np.cos(phi)
                By = b * np.sin(theta) * np.sin(phi)
                Bz = b * np.cos(theta)
                B = [Bx, By, Bz]
                a1,a2,b1 = ag_grid[i,j,k], ae_grid[i,j,k], bg_grid[i,j,k]

                model = transitioncompute(B, strain=[a1, a2, b1, exye])

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
        f.create_dataset('exg', data=exg)
        f.create_dataset('exyg', data=exyg)
        f.create_dataset('exe', data=exe)

        # Save c magnitudes
        f.create_dataset('c_magnitudes', data=c_magnitudes)

        # Save energy spectra
        grp_ground = f.create_group('ground_state')
        grp_ground.create_dataset('energy_levels', data=energy_ground)
        grp_ground.create_dataset('eigenvectors', data=eigenvectors_ground)

        grp_excited = f.create_group('excited_state')
        grp_excited.create_dataset('energy_levels', data=energy_excited)
        grp_excited.create_dataset('eigenvectors', data=eigenvectors_excited)




