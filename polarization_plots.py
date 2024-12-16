import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from matplotlib.animation import FuncAnimation


# Define a function to plot the Poincaré sphere and mark a given vector
def plot_poincare_sphere(vector, ax):
    # Ensure the input vector is normalized (optional for visualization)
    vector = vector / np.linalg.norm(vector)

    # Create a meshgrid for the sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the sphere
    ax.plot_surface(x, y, z, color='lightblue', alpha=0.3, edgecolor='k', linewidth=0.1)

    # Add axes lines
    ax.quiver(0, 0, 0, 1, 0, 0, color='r', label='S1 (Horizontal/Vertical)')  # S1-axis
    ax.quiver(0, 0, 0, 0, 1, 0, color='g', label='S2 (Diagonal/Anti-diagonal)')  # S2-axis
    ax.quiver(0, 0, 0, 0, 0, 1, color='b', label='S3 (Right/Left Circular)')  # S3-axis

    # Plot the input vector
    ax.quiver(0, 0, 0, vector[0], vector[1], vector[2], color='purple', linewidth=2, label='Input Vector')

    # Set labels
    ax.set_xlabel('S1')
    ax.set_ylabel('S2')
    ax.set_zlabel('S3')

    # Adjust the plot limits and view angle
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])


# Function to generate the model vector based on phi and theta
def generate_vector(phi, theta, mag=2):
    Bx = mag * np.sin(phi) * np.cos(theta)
    By = mag * np.sin(phi) * np.sin(theta)
    Bz = mag * np.cos(phi)
    return [Bx, By, Bz]


# Create a figure and 3D axis for the plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Set initial values for theta and phi
theta = np.pi / 4
phi = 0

# Create a function to update the plot for animation
def update(frame):
    # Clear the previous plot
    ax.clear()

    # Update phi value
    phi = frame * np.pi / 50  # Vary phi from 0 to 2*pi (adjust frame count for speed)

    # Generate the vector based on the current phi angle
    vector = generate_vector(phi, theta)

    # Plot the Poincaré sphere with the updated vector
    plot_poincare_sphere(vector, ax)

    # Add title and labels again after clearing
    ax.set_title('Poincaré Sphere')
    ax.set_xlabel('S1')
    ax.set_ylabel('S2')
    ax.set_zlabel('S3')

# Create the animation
ani = FuncAnimation(fig, update, frames=np.arange(0, 100), interval=100)
ani.save('poincare_sphere_animation.gif', writer='imagemagick', fps=30)
# Show the animation
plt.show()
