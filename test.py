import numpy as np
from src.transitions import transitioncompute
import matplotlib.pyplot as plt
from fractions import Fraction
from scipy.optimize import minimize
from tqdm.auto import tqdm
from utils.lalg import rotation_matrix_between_vectors
from plots.polarization_plots import plot_polar, plot_2polar

def check_min():
    x = np.linspace(1, np.pi-0.05, 10000)
    y = []
    for theta in x:

        b = 5
        phi = np.pi/4
        Bx = b*np.sin(theta) * np.cos(phi)
        By = b*np.sin(theta) * np.sin(phi)
        Bz = b*np.cos(theta)
        B = [Bx,By,Bz]

        model = transitioncompute(B)



        v1 = model.get_A2() / np.linalg.norm(model.get_A2())

        v2 = model.get_A2() / np.linalg.norm(model.get_A2())
        Ax2, Ay2 = model.convert_lab_frame(*v1)

        Ax, Ay = model.convert_lab_frame(*v2)
        c = np.abs(np.vdot([Ax2, Ay2], [Ax, Ay]))

        y.append(np.abs(c))
    factor = x[np.argmin(y)]
    print(factor, min(y))
    plt.plot(x,y)
    plt.show()


def scan_B_strength():
    b_vals = np.linspace(0.01,5,100)
    factors = []
    for b in tqdm(b_vals):
        x = np.linspace(1, 1.5, 10000)
        y = []
        for theta in x:

            phi = np.pi
            Bx = b*np.sin(theta) * np.cos(phi)
            By = b*np.sin(theta) * np.sin(phi)
            Bz = b*np.cos(theta)
            B = [Bx,By,Bz]

            model = transitioncompute(B)



            v1 = model.get_A2() / np.linalg.norm(model.get_A2())

            v2 = model.get_B1() / np.linalg.norm(model.get_B1())

            c = np.vdot(v1, v2)
            y.append(np.abs(c))
        factor = x[np.argmin(y)]
        factors.append(factor)
    plt.plot(b_vals, factors)
    plt.xlabel('B field strength')
    plt.ylabel('location of minima (theta)')
    plt.show()


def compute_polarization(field_vector):
    """
    Compute the polarization state of a photon given a complex field vector.

    Parameters:
        field_vector (np.ndarray): Complex field vector of size 3 [A_x, A_y, A_z].
                                   Each component can be a complex number.

    Returns:
        dict: Contains Ex, Ey, k-vector, polarization basis components, normalized electric field,
              and Stokes parameters.
    """
    # Ensure the input is a numpy array
    field_vector = np.array(field_vector, dtype=complex)


    k_vector = np.cross(field_vector.real, field_vector.imag)

    k_norm = np.linalg.norm(k_vector)
    k_vector /= k_norm
    # Define two orthogonal basis vectors perpendicular to k_vector
    basis_1 = np.cross(k_vector, [1, 0, 0]) if abs(k_vector[0]) < 1 else np.cross(k_vector, [0, 1, 0])
    basis_1 /= np.linalg.norm(basis_1)
    basis_2 = np.cross(k_vector, basis_1)

    # Project the field vector onto the basis
    A1 = np.dot(field_vector, basis_1)  # Projection on basis_1 (Ex)
    A2 = np.dot(field_vector, basis_2)  # Projection on basis_2 (Ey)

    # Circular polarization components
    A_L = (A1 + 1j * A2) / np.sqrt(2)
    A_R = (A1 - 1j * A2) / np.sqrt(2)

    # Normalize the electric field
    field_magnitude = np.linalg.norm(field_vector)
    E_normalized = field_vector / field_magnitude

    # Compute Stokes parameters
    S0 = abs(A1) ** 2 + abs(A2) ** 2
    S1 = abs(A1) ** 2 - abs(A2) ** 2
    S2 = 2 * np.real(A1 * np.conj(A2))
    S3 = 2 * np.imag(A1 * np.conj(A2))
    stokes_parameters = {"S0": S0, "S1": S1, "S2": S2, "S3": S3}

    return {
        "Ex": A1,  # Electric field component in the first basis
        "Ey": A2,  # Electric field component in the second basis
        "k_vector": k_vector,  # Normalized propagation direction
        "A_L": A_L,
        "A_R": A_R,
        "Normalized Field": E_normalized,
        "Stokes Parameters": stokes_parameters,
        'basis 1': basis_1,
        'basis 2': basis_2
    }

def plot_ellipticity(Ex, Ey, num_points=500):
    """
    Plot the ellipticity (polarization ellipse) given Ex and Ey components.

    Parameters:
        Ex (complex): Electric field component along the x-direction.
        Ey (complex): Electric field component along the y-direction.
        num_points (int): Number of points to sample for the ellipse.

    Returns:
        None: Displays the polarization ellipse plot.
    """
    # Time array for one oscillation cycle
    t = np.linspace(0, 2 * np.pi, num_points)

    # Electric field vector components as functions of time
    E_x = np.real(Ex) * np.cos(t) - np.imag(Ex) * np.sin(t)
    E_y = np.real(Ey) * np.cos(t) - np.imag(Ey) * np.sin(t)

    # Plot the ellipse
    plt.figure(figsize=(6, 6))
    plt.plot(E_x, E_y, label='Polarization Ellipse', color='blue')
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')

    # Set axis labels and equal aspect ratio
    plt.xlabel(r"$E_x$", fontsize=12)
    plt.ylabel(r"$E_y$", fontsize=12)
    plt.title("Polarization Ellipse", fontsize=14)
    plt.axis('equal')
    plt.grid(alpha=0.5)
    plt.legend()
    plt.show()

def find_min_index(matrix):

    min_value = float('inf')
    min_index = [-1, -1]

    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            if val < min_value:
                min_value = val
                min_index = [i, j]

    return min_index[0], min_index[1]
def make_matrix():
    res = 400
    hm = np.zeros((res,res))

    b_vals = np.linspace(0.001,3,res)
    theta_vals = np.linspace(0.0001,np.pi-0.001,res)
    for i in tqdm(range(res)):
        for j in range(res):
            b = b_vals[i]
            theta = theta_vals[j]
            phi = np.pi
            Bx = b * np.sin(theta) * np.cos(phi)
            By = b * np.sin(theta) * np.sin(phi)
            Bz = b * np.cos(theta)
            B = [Bx, By, Bz]

            model = transitioncompute(B, strain=[0,0,0,0])

            v1 = model.get_A2()/ np.linalg.norm(model.get_A2())
            Ax2, Ay2 = model.convert_lab_frame(*v1)


            v2 = model.get_B1()/np.linalg.norm(model.get_B1())

            Ax, Ay = model.convert_lab_frame(*v2)

            hm[i,j] = np.abs(np.vdot([Ax2, Ay2], [Ax, Ay]))
    print('finding min')
    k,l = find_min_index(hm)
    print(hm[k,l])
    bp, tp = b_vals[k], theta_vals[l]
    plt.figure(figsize=(8, 8))
    plt.imshow(hm, cmap='viridis', aspect='auto', extent=[theta_vals.min(), theta_vals.max(), b_vals.min(), b_vals.max(),], origin='lower')
    plt.colorbar(label="Value")
    plt.title("overlap in lab frame")
    plt.ylabel('B field strength')
    plt.xlabel('theta')
    plt.show()


b= 0.5
theta= 1.570796
phi= 0
exg= 0.0007656
exyg= 0

Bx = b * np.sin(theta) * np.cos(phi)
By = b * np.sin(theta) * np.sin(phi)
Bz = b * np.cos(theta)
B = [Bx, By, Bz]


model = transitioncompute(B, strain=[exg, exyg])

v1 = model.get_A1()/ np.linalg.norm(model.get_A1())
Ax2, Ay2 = model.convert_lab_frame(*v1)


v2 = model.get_A2()/np.linalg.norm(model.get_A2())
Ax, Ay = model.convert_lab_frame(*v2)

print(np.abs(np.vdot(v1, v2)))
print(np.abs(np.vdot([Ax2, Ay2], [Ax, Ay])))

phi_vals, mags = model.scan_polarisation(v1)
phi_vals2, mags2 = model.scan_polarisation(v2)
labels = ['A1', 'A2']
plot_2polar(phi_vals, mags, phi_vals2, mags2, labels)


plot_ellipticity(Ax, Ay)
plot_ellipticity(Ax2, Ay2)


print(Ax2,Ay2)