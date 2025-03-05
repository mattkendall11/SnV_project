from src.transitions import transitioncompute
import numpy as np

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
    timestamp = datetime.datetime.now().strftime("%m-%d")
    np.save(fr'data/full_scan_{resolution}_{timestamp}', c_magnitudes)


def run_ft(resolution = 200):
    # Fixed theta value
    theta_fixed = np.pi / 2

    # Resolutions for B and phi
    resolution = 100
    B_values = np.linspace(0, 5, resolution)
    phi_values = np.linspace(0, 2 * np.pi, resolution)
    c_magnitudes = np.zeros_like(B_values)

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


            # Calculate |c|
            c = model.A1B2_overlap()
            c_magnitudes[j, k] = c

    # Save results with timestamp
    timestamp = datetime.datetime.now().strftime("%m-%d")
    np.save(fr'data/ft_scan_{resolution}_{timestamp}', c_magnitudes)


def run_lf(resolution = 200, ttype = 'A1A2'):
    if ttype not in ['B1B2', 'A1A2', 'A1B2', 'A1B1', 'A2B1', 'A2B2']:
        raise ValueError(f"Invalid ttype: '{ttype}'. Must be one of ['B1B2', 'A1A2', 'A1B2', 'A1B1', 'A2B1', 'A2B2'].")

    B_values = np.linspace(0.0001, 5, resolution)
    theta_values = np.linspace(0.0001, np.pi, resolution)
    phi_values = np.linspace(0.0001, 2 * np.pi, resolution)

    B_grid, theta_grid, phi_grid = np.meshgrid(B_values, theta_values, phi_values)
    c_magnitudes = np.zeros_like(B_grid)

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

                v1 = model.get_A2() / np.linalg.norm(model.get_A2())
                Ax2, Ay2 = model.convert_lab_frame(*v1)

                v2 = model.get_B1() / np.linalg.norm(model.get_B1())
                Ax, Ay = model.convert_lab_frame(*v2)

                c = np.abs(np.vdot([Ax2, Ay2], [Ax, Ay]))

                c_magnitudes[i, j, k] = c

    timestamp = datetime.datetime.now().strftime("%m-%d")
    np.save(fr'data/lf_scan_{resolution}_{timestamp}', c_magnitudes)

def run_ft_lf(resolution = 200, ttype = 'A1A2'):
    if ttype not in ['B1B2', 'A1A2', 'A1B2', 'A1B1', 'A2B1', 'A2B2']:
        raise ValueError(f"Invalid ttype: '{ttype}'. Must be one of ['B1B2', 'A1A2', 'A1B2', 'A1B1', 'A2B1', 'A2B2'].")

    B_values = np.linspace(0.0001, 5, resolution)

    theta = np.pi/2
    phi_values = np.linspace(0.0001, 2 * np.pi, resolution)

    B_grid, phi_grid = np.meshgrid(B_values, phi_values)

    c_magnitudes = np.zeros_like(B_grid)

    # Calculate |c| for each combination of B, theta and phi
    for i in tqdm(range(len(B_values))):
        for j in range(len(phi_values)):

            # Convert spherical to Cartesian coordinates for B field
            Bx = B_grid[i, j] * np.sin(theta) * np.cos(phi_grid[i, j])
            By = B_grid[i, j] * np.sin(theta) * np.sin(phi_grid[i, j])
            Bz = B_grid[i, j] * np.cos(theta)
            B = [Bx, By, Bz]
            model = transitioncompute(B)
            if ttype == 'A2B1':
                v1 = model.get_A2() / np.linalg.norm(model.get_A2())
                v2 = model.get_B1() / np.linalg.norm(model.get_B1())
            if ttype == 'A1A2':
                v1 = model.get_A1() / np.linalg.norm(model.get_A1())
                v2 = model.get_A2() / np.linalg.norm(model.get_A2())
            if ttype == 'A1B2':
                v1 = model.get_A1() / np.linalg.norm(model.get_A1())
                v2 = model.get_B1() / np.linalg.norm(model.get_B2())
            if ttype == 'A1B1':
                v1 = model.get_A1() / np.linalg.norm(model.get_A1())
                v2 = model.get_B1() / np.linalg.norm(model.get_B1())
            if ttype == 'A2B2':
                v1 = model.get_A2() / np.linalg.norm(model.get_A2())
                v2 = model.get_B2() / np.linalg.norm(model.get_B2())

            Ax2, Ay2 = model.convert_lab_frame(*v1)

            Ax, Ay = model.convert_lab_frame(*v2)

            c = np.abs(np.vdot([Ax2, Ay2], [Ax, Ay]))

            c_magnitudes[i, j] = c

    timestamp = datetime.datetime.now().strftime("%m-%d")
    np.save(fr'data/ft_lf_{ttype}_{resolution}_{timestamp}', c_magnitudes)