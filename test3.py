import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from transitions import transitioncompute

# Initialize the model
model = transitioncompute(B=[0.1, 0.1, 0])

# Set up the figure and axis for the plot
fig, ax = plt.subplots(figsize=(8, 6))

# Define the time range (from 0.1 to 1e-6)
time_values = np.logspace(np.log10(0.1), np.log10(1e-6), num=100)


# Create a function to update the heatmap for each time step
def update_heatmap(frame):
    time = time_values[frame]

    # Calculate the Hamiltonian matrix at the given time
    matrix = model.td_hamiltonian(10e15, time)

    # Clear previous heatmap and plot a new one
    ax.clear()

    # Plot the updated heatmap
    sns.heatmap(matrix, annot=True, cmap='viridis', cbar=False, ax=ax, cbar_kws={'label': 'Magnitude'})

    # Add title and labels
    ax.set_title(f'SnV Hamiltonian = {time:.2e}')



# Create the animation using FuncAnimation
ani = FuncAnimation(fig, update_heatmap, frames=len(time_values), repeat=False, interval=100)

# Display the animation
plt.tight_layout()
plt.show()

