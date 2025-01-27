import h5py
import numpy as np
import matplotlib.pyplot as plt
from src.transitions import transitioncompute

# Load the data from the HDF5 file
file_path = 'fixed_100_12-02_15-01.h5'
vary = False
 # Replace with your file name
with h5py.File(file_path, 'r') as f:
    # Load parameter arrays
    B_values = f['B_values'][:]
    if vary:
        theta_values = f['theta_values'][:]
    phi_values = f['phi_values'][:]
    # Load c_magnitudes
    c_magnitudes = f['c_magnitudes'][:]


def constant_azimuth(ind):
    # Select a specific phi index, e.g., phi index 10
    phi_index = ind
    c_magnitudes_slice = c_magnitudes[:, :, phi_index]
    B_grid, theta_grid = np.meshgrid(B_values, theta_values)

    # Create a heatmap plot of c_magnitudes for this slice
    plt.figure(figsize=(10, 6))
    plt.contourf(theta_grid, B_grid, c_magnitudes_slice, cmap='viridis')
    plt.colorbar(label='c Magnitudes')
    plt.xlabel('Theta (rad)')
    plt.ylabel('Field Strength (B)')
    plt.title(f'Density Plot of c Magnitudes vs Theta and Field Strength (phi = {phi_values[phi_index]:.2f}), azymuthial angle constant')
    plt.savefig('constant_azimuthial.svg')
    plt.show()

def constant_polar(ind):
    # Select a specific theta index
    theta_index = ind
    c_magnitudes_slice = c_magnitudes[theta_index, :, :]  # Fix theta
    B_grid, phi_grid = np.meshgrid(B_values, phi_values)

    # Create a heatmap plot of c_magnitudes for this slice
    plt.figure(figsize=(10, 6))
    plt.contourf(phi_grid, B_grid, c_magnitudes_slice.T, cmap='viridis')  # Transpose for alignment
    plt.colorbar(label='c Magnitudes')
    plt.xlabel('Phi (rad)')
    plt.ylabel('Field Strength (B)')
    plt.title(f'Density Plot of c Magnitudes vs Phi and Field Strength (theta = {theta_values[theta_index]:.2f}), polar angle constant')
    plt.show()

def constant_field(ind):
    # Select a specific B index
    B_index = ind
    c_magnitudes_slice = c_magnitudes[:, :, B_index]  # Fix B
    theta_grid, phi_grid = np.meshgrid(theta_values, phi_values)

    # Create a heatmap plot of c_magnitudes for this slice
    plt.figure(figsize=(10, 6))
    plt.contourf(phi_grid, theta_grid, c_magnitudes_slice.T, cmap='viridis')  # Transpose for alignment
    plt.colorbar(label='c Magnitudes')
    plt.xlabel('Phi (rad)')
    plt.ylabel('Theta (rad)')
    plt.title(f'Density Plot of c Magnitudes vs Phi and Theta (B = {B_values[B_index]:.2f}), field strength constant')
    plt.show()

def constant_field_strength(ind):
    """
    Plot c magnitudes as a function of theta for a fixed field strength (B).
    
    Parameters:
    ind (int): Index of the desired field strength (B).
    """
    # Select the specific B index
    B_index = ind
    c_magnitudes_slice = c_magnitudes[:, :, B_index]  # Fix field strength B


    # Plot c magnitudes vs. theta for the given B value
    plt.figure(figsize=(10, 6))
    plt.plot(theta_values, c_magnitudes_slice[:,5], marker='o', linestyle='-', color='b')
    plt.xlabel('Theta (rad)')
    plt.ylabel('c Magnitudes (Average over Phi)')
    plt.title(f'c Magnitudes vs Theta (B = {B_values[B_index]:.2f})')
    plt.grid(True)
    plt.show()

def plot_constant_field_strength_grid(indices):
    """
    Create a 4x4 grid of c magnitudes vs theta plots for different field strengths (B).
    
    Parameters:
    indices (list of int): List of B indices to plot.
    """
    # Ensure we have 16 indices for a 4x4 grid
    if len(indices) != 16:
        raise ValueError("Please provide exactly 16 indices for a 4x4 grid.")

    # Create the figure and axes for a 4x4 grid
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    fig.suptitle('c Magnitudes vs Theta at Different Field Strengths (B)', fontsize=18)

    # Loop over indices and plot each on a subplot
    for i, ax in enumerate(axes.flatten()):
        B_index = indices[i]
        c_magnitudes_slice = c_magnitudes[:, :, B_index]
        

        # Plot on the current axis
        ax.plot(theta_values, c_magnitudes_slice[:,5], marker='o', linestyle='-', color='b')
        ax.set_title(f'B = {B_values[B_index]:.2f}')
        ax.set_xlabel('Theta (rad)')
        ax.set_ylabel('c Magnitudes (Avg over Phi)')
        ax.grid(True)

    # Adjust layout for better spacing
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_constant_theta_grid(indices):
    """
    Create a 4x4 grid of c magnitudes vs phi plots for different theta values.
    
    Parameters:
    indices (list of int): List of theta indices to plot.
    """
    # Ensure we have 16 indices for a 4x4 grid
    if len(indices) != 16:
        raise ValueError("Please provide exactly 16 indices for a 4x4 grid.")

    # Create the figure and axes for a 4x4 grid
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    fig.suptitle('c Magnitudes vs Phi at Different Theta Values', fontsize=18)

    # Loop over indices and plot each on a subplot
    for i, ax in enumerate(axes.flatten()):
        theta_index = indices[i]
        c_magnitudes_slice = c_magnitudes[theta_index, :, :]  # Fix theta

        # Plot on the current axis
        ax.plot(phi_values, c_magnitudes_slice[:,10], marker='o', linestyle='-', color='b')
        ax.set_title(f'Theta = {theta_values[theta_index]:.2f}')
        ax.set_xlabel('Phi (rad)')
        ax.set_ylabel('c Magnitudes (Avg over B)')
        ax.grid(True)

    # Adjust layout for better spacing
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
def spectra_plot():
    B_values = np.linspace(0,2,100)
    ground_values = []
    excited_values = []
    for b in B_values:
        mod = transitioncompute([0.5,0.5,b])
        eigenvalues_ground,  eigenvalues_excited = mod.return_levels()
        ground_values.append(eigenvalues_ground)
        excited_values.append(eigenvalues_excited)
     # Convert lists to numpy arrays for easier plotting
    ground_values = np.array(ground_values)
    excited_values = np.array(excited_values)

    # Create the plot with two subplots stacked on top of each other
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # Plot ground state eigenvalues
    ax[1].plot(B_values, ground_values[:, 0], label='1', color='blue')
    ax[1].plot(B_values, ground_values[:, 1], label='2', color='green')
    ax[1].plot(B_values, ground_values[:, 2], label='3', color='red')
    ax[1].plot(B_values, ground_values[:, 3], label='4', color='purple')
    ax[1].set_ylabel('Ground State Eigenvalues')
    ax[1].legend()


    # Plot excited state eigenvalues
    ax[0].plot(B_values, excited_values[:, 0], label='A', color='blue')
    ax[0].plot(B_values, excited_values[:, 1], label='B', color='green')
    ax[0].plot(B_values, excited_values[:, 2], label='C', color='red')
    ax[0].plot(B_values, excited_values[:, 3], label='D', color='purple')
    ax[1].set_xlabel('B Field (Tesla)')
    ax[0].set_ylabel('Excited State Eigenvalues')
    ax[0].legend()


    plt.tight_layout()
    plt.savefig('spectra.svg')
    plt.show()