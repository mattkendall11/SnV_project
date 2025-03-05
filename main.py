from sim.scans import run_ft, run, run_lf, run_ft_lf
from sim.strain import run_strain
import h5py
import matplotlib.pyplot as plt
import numpy as np
from utils.iqp_colors import div, uni
from utils.analytic import fit_ns
ttype = 'A1A2'
resolution = 300
colour = uni

#run_lf(resolution=resolution, ttype=ttype)
run_ft_lf(resolution=resolution, ttype=ttype)
file_path = fr"data/ft_lf_{ttype}_{resolution}_03-03.npy"


def analyse_angular(file_path):
    """
    Loads the precomputed c_magnitudes data and plots |c| for different B-field values.

    Parameters:
    - resolution: int, grid resolution used during data generation
    - ttype: str, transition type used during data generation
    - timestamp: str, date stamp used in the saved file
    """

    c_magnitudes = np.load(file_path)

    # Recompute parameter grids
    B_values = np.linspace(0.0001, 5, resolution)
    theta_values = np.linspace(0.0001, np.pi, resolution)
    phi_values = np.linspace(0.0001, 2 * np.pi, resolution)
    theta_grid, phi_grid = np.meshgrid(theta_values, phi_values, indexing='ij')

    # Plot slices at different B-field indices
    for i in range(20):
        b_index = 9 * i  # Adjust as needed
        if b_index >= resolution:
            break  # Ensure we don't exceed array bounds

        c_slice = c_magnitudes[:, :, b_index]

        # Create the plot
        plt.figure(figsize=(8, 6))
        plt.pcolormesh(theta_grid, phi_grid, c_slice, shading='auto', cmap=colour)
        plt.colorbar(label='|c| Magnitude')
        plt.xlabel('Theta (rad)')
        plt.ylabel('Phi (rad)')
        plt.title(f'{ttype} Overlap for B = {B_values[b_index]:.2f} T, min = {np.min(c_slice):.5f}')
        plt.show()

def analyse_phi_B(file_path):
    """
    Loads the precomputed c_magnitudes data and plots |c| for different B-field values.

    Parameters:
    - resolution: int, grid resolution used during data generation
    - ttype: str, transition type used during data generation
    - timestamp: str, date stamp used in the saved file
    """

    c_magnitudes = np.load(file_path)

    # Recompute parameter grids
    B_values = np.linspace(0.0001, 5, resolution)
    theta_values = np.linspace(0.0001, np.pi, resolution)
    phi_values = np.linspace(0.0001, 2 * np.pi, resolution)
    theta_grid, b_grid = np.meshgrid(theta_values, B_values, indexing='ij')

    for i in range(20):
        t = 9 * i  # Adjust as needed
        if t >= resolution:
            break  # Ensure we don't exceed array bounds

        c_slice = c_magnitudes[t, :, :]

        # Create the plot
        plt.figure(figsize=(8, 6))
        plt.pcolormesh(theta_grid, b_grid, c_slice, shading='auto', cmap=colour)
        plt.colorbar(label='|c| Magnitude')
        plt.xlabel('Theta (rad)')
        plt.ylabel('B (T)')
        plt.title(fr'{ttype} Overlap for $\theta$ = {theta_values[t]:.2f} , min = {np.min(c_slice):.5f}')
        plt.show()

def analyse_theta_B(file_path):
    """
    Loads the precomputed c_magnitudes data and plots |c| for different theta values.

    Parameters:
    - resolution: int, grid resolution used during data generation
    - ttype: str, transition type used during data generation
    - timestamp: str, date stamp used in the saved file
    """

    c_magnitudes = np.load(file_path)

    # Recompute parameter grids
    B_values = np.linspace(0.0001, 5, resolution)
    theta_values = np.linspace(0.0001, np.pi, resolution)
    phi_values = np.linspace(0.0001, 2 * np.pi, resolution)
    b_grid, phi_grid = np.meshgrid(B_values, phi_values, indexing='ij')

    # Plot slices at different B-field indices
    for i in range(20):
        p = 9 * i  # Adjust as needed
        if p >= resolution:
            break  # Ensure we don't exceed array bounds

        c_slice = c_magnitudes[:, p, :]

        # Create the plot
        plt.figure(figsize=(8, 6))
        plt.pcolormesh(b_grid, phi_grid, c_slice, shading='auto', cmap=colour)
        plt.colorbar(label='|c| Magnitude')
        plt.xlabel('B (T)')
        plt.ylabel('Phi (rad)')
        plt.title(fr'{ttype} Overlap for $\varphi$ = {phi_values[p]:.2f}, min = {np.min(c_slice):.5f}')
        plt.show()

def analyse_ft_phi_B(file_path, plot_analytic = False):
    """
    Loads the precomputed c_magnitudes data and plots |c| for different B-field values.

    Parameters:
    - resolution: int, grid resolution used during data generation
    - ttype: str, transition type used during data generation
    - timestamp: str, date stamp used in the saved file
    """

    c_slice = np.load(file_path)

    # Recompute parameter grids
    B_values = np.linspace(0.0001, 5, resolution)
    phi_values = np.linspace(0.0001, 2 * np.pi, resolution)


    b_grid, phi_grid = np.meshgrid(phi_values,B_values)
    plt.figure(figsize=(8, 6))
    plt.pcolormesh( b_grid,phi_grid, c_slice, shading='auto', cmap=colour)
    plt.colorbar(label='|c| Magnitude')
    plt.xlabel('Phi (rad)')
    plt.ylabel('B (T)')
    plt.title(fr'{ttype} Overlap for $\theta = \pi/2$  , min = {np.min(c_slice):.5f}')
    if plot_analytic:
        p_vals = np.arccos(np.sqrt(fit_ns(B_values)))
        plt.plot(   p_vals, B_values)
    plt.show()

analyse_ft_phi_B(file_path)