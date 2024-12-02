import numpy as np
import matplotlib.pyplot as plt
# Constants
hbar = 1.054571817e-34  # Reduced Planck's constant
muB = 9.2740100783e-24  # Bohr magneton
GHz = 1e9
lg = 815 * GHz  # Spin-orbit coupling (ground state)
lu = 2355 * GHz  # Spin-orbit coupling (excited state)
xg = 65 * GHz  # Jahn-Teller coupling (ground state)
yg = 0
xu = 855 * GHz  # Jahn-Teller coupling (excited state)
yu = 0
fg = 0.15
fu = 0.15
phi = np.pi / 3
B_array = np.linspace(0,2,1000)
phi_array = np.linspace(0,2*np.pi,1000)
E_ground = 0
E_excited = 484 * 1e12  # Energy of the excited state (Hz)

# Calculate field components

# Define a function to construct the Hamiltonian
def construct_hamiltonian(l, x, y, f, E, Bx, By, Bz=0):
    # Spin-orbit Hamiltonian
    HSpinOrbit = np.array([
        [0, 0, -1j * l / 2, 0],
        [0, 0, 0, 1j * l / 2],
        [1j * l / 2, 0, 0, 0],
        [0, -1j * l / 2, 0, 0]
    ])
    
    # Jahn-Teller Hamiltonian
    HJahnTeller = np.array([
        [x, 0, y, 0],
        [0, x, 0, y],
        [y, 0, -x, 0],
        [0, y, 0, -x]
    ])
    
    # Spin-Zeeman Hamiltonian
    HSpinZeeman = np.array([
        [Bz, Bx - 1j * By, 0, 0],
        [Bx + 1j * By, -Bz, 0, 0],
        [0, 0, Bz, Bx - 1j * By],
        [0, 0, Bx + 1j * By, -Bz]
    ])
    
    # Total Hamiltonian
    return E * np.eye(4) + HSpinOrbit + HJahnTeller + HSpinZeeman

def calculate_overlap(B, phi):
    Bx = 2 * hbar * muB * B * np.cos(phi)
    By = 2 * hbar * muB * B * np.sin(phi)
    # Ground and excited state Hamiltonians
    Ham_excited = construct_hamiltonian(l=lu, x=xu, y=yu, f=fu, E=E_excited, Bx=Bx, By=By)
    Ham_ground = construct_hamiltonian(l=lg, x=xg, y=yg, f=fg, E=E_ground, Bx=Bx, By=By)

    # Diagonalize Hamiltonians
    eigvals_excited, eigvecs_excited = np.linalg.eigh(Ham_excited)
    eigvals_ground, eigvecs_ground = np.linalg.eigh(Ham_ground)

    # Normalize eigenvectors
    ve = eigvecs_excited[:, 0]  # First eigenvector of the excited state
    vg1 = eigvecs_ground[:, 0]  # First eigenvector of the ground state
    vg2 = eigvecs_ground[:, 1]  # Second eigenvector of the ground state

    # Define Pauli matrices and P matrices
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])
    identity2 = np.eye(2)

    Px = np.kron(sigma_x, identity2)
    Py = np.kron(sigma_y, identity2)
    Pz = np.kron(2 * sigma_z, identity2)

    # Calculate field amplitudes
    Ax1 = np.vdot(vg1, Px @ ve)
    Ay1 = np.vdot(vg1, Py @ ve)
    Az1 = np.vdot(vg1, Pz @ ve)

    Ax2 = np.vdot(vg2, Px @ ve)
    Ay2 = np.vdot(vg2, Py @ ve)
    Az2 = np.vdot(vg2, Pz @ ve)

    vector1 = np.array([Ax1, Ay1, Az1])
    vector2 = np.array([Ax2, Ay2, Az2])
    # Compute overlap
    overlap = np.vdot(vector1, vector2)
    return np.abs(overlap)
overlap_values = []
for B in B_array:
    x = calculate_overlap(B, np.pi)
    overlap_values.append(x)
    print(x)
plt.plot(B_array, overlap_values, '+')
plt.xlabel('magnetic field strength (T)')
plt.ylabel('dipole transition overlap')
plt.title(fr'overlap at $\theta = \pi/2,$, with $\phi = \pi/4$')
plt.show()

overlap_values = []
for phi in phi_array:
    x = calculate_overlap(1, phi)
    overlap_values.append(x)
    print(x)
plt.plot(phi_array, overlap_values, '+')
plt.xlabel('phi')
plt.ylabel('dipole transition overlap')
plt.title(fr'overlap at $\theta = \pi/2,$, with $T = 1$')
plt.show()
overlap_values = np.zeros((len(B_array), len(phi_array)))

# Calculate overlap for each combination of B and phi
for i, B in enumerate(B_array):
    for j, phi in enumerate(phi_array):
        overlap_values[i, j] = calculate_overlap(B, phi)

# Create the density plot
plt.figure(figsize=(8, 6))
plt.pcolormesh(phi_array, B_array, overlap_values, shading='auto', cmap='viridis')
plt.colorbar(label='Dipole transition overlap')
plt.xlabel(r'$\phi$ (radians)')
plt.ylabel('Magnetic field strength $B$ (T)')
plt.title(r'Density plot of overlap as a function of $B$ and $\phi$')
plt.show()

