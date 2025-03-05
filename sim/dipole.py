from src.transitions import transitioncompute
import numpy as np
import matplotlib.pyplot as plt


Bx = np.linspace(0,3,100)
By  = np.linspace(0,3,100)
cx,cy,cz = np.zeros((100,100)), np.zeros((100,100)), np.zeros((100,100))
for i in range(len(Bx)):
    for j  in range(len(By)):
        model =transitioncompute([Bx[i], By[j], 0.01])
        vector = model.get_B1()
        norm = np.linalg.norm(vector)

        # Normalize the vector
        normalized_vector = np.abs(vector / norm)
        cx[i,j], cy[i,j], cz[i,j] = normalized_vector


X, Y = np.meshgrid(Bx, By)

# Plot the heatmap
plt.figure(figsize=(8, 6))
heatmap = plt.pcolormesh(X, Y, cz, shading='auto', cmap='viridis')

# Add a colorbar
cbar = plt.colorbar(heatmap)
cbar.set_label('cz intensity', rotation=270, labelpad=15)

# Label the axes
plt.xlabel('bx')
plt.ylabel('by')
plt.title('Magnitude Plot')

# Show the plot
plt.show()