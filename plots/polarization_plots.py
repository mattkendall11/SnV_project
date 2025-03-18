import numpy as np
import matplotlib.pyplot as plt
from utils.iqp_colors import light, dark

def plot_3d_vectors(vectors, colors=None, labels=None):
    """
    Plot multiple 3D vectors from the origin.

    Parameters:
    vectors (list): List of vectors, each vector is [x, y, z]
    colors (list): List of colors for each vector (optional)
    labels (list): List of labels for each vector (optional)
    """
    # Create figure and 3D axes
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Default colors if none provided
    if colors is None:
        colors = ['blue', 'green', 'red', 'purple', 'orange']

    # Default labels if none provided
    if labels is None:
        labels = [f'Vector {i + 1}' for i in range(len(vectors))]

    # Plot each vector
    for vector, color, label in zip(vectors, colors[:len(vectors)], labels[:len(vectors)]):
        ax.quiver(0, 0, 0,  # Start point
                  vector[0], vector[1], vector[2],  # Direction and length
                  color=color,
                  arrow_length_ratio=0.1,
                  label=label)

    # Get the maximum absolute value across all vectors for setting axis limits
    all_coords = np.array(vectors).flatten()
    max_val = max(abs(all_coords)) * 1.2

    # Set labels and limits
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-max_val, max_val])
    ax.set_ylim([-max_val, max_val])
    ax.set_zlim([-max_val, max_val])

    # Add legend
    ax.legend()

    # Add a title
    plt.title('3D Vectors from Origin')

    # Add the origin
    ax.scatter(0, 0, 0, color='black', marker='o', label='Origin')

    # Show grid
    ax.grid(True)

    return fig, ax


def plot_polar(phi_vals, mags):

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(phi_vals, mags, 'ro')  # 'ro' for red dots

    # Customize labels
    ax.set_theta_zero_location('E')  # Set zero degrees to the right (east)
    ax.set_theta_direction(-1)  # Counterclockwise

    plt.show()


def plot_2polar(phi_vals, mags, phi_vals2, mags2, labels):

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(phi_vals, mags, color = dark[0], label =labels[0])  # 'ro' for red dots
    ax.plot(phi_vals2, mags2, color = dark[1], label=labels[1])
    # Customize labels
    ax.set_theta_zero_location('E')  # Set zero degrees to the right (east)
    ax.set_theta_direction(-1)  # Counterclockwise
    plt.legend()
    plt.show()

def stokes_s3_and_ellipticity(Ex, Ey):
    """
    Compute the Stokes S3 parameter and the degree of ellipticity.

    Parameters:
    Ex (complex): x-component of the electric field.
    Ey (complex): y-component of the electric field.

    Returns:
    S3 (float): Stokes parameter S3.
    ellipticity (float): Degree of ellipticity (radians).
    """
    S3 = 2 * np.imag(Ex * np.conj(Ey)) / (np.abs(Ex) ** 2 + np.abs(Ey) ** 2)
    ellipticity = 0.5 * np.arcsin(S3)  # Ellipticity angle in radians
    return S3, ellipticity