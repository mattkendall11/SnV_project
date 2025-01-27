import numpy as np
from src.transitions import transitioncompute
import matplotlib.pyplot as plt
from fractions import Fraction
from scipy.optimize import minimize

def check_min():
    x = np.linspace(1.3169328487746, 1.3169328487747, 100000)
    y = []
    for theta in x:

        b = 1.82
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
    decimal = factor/np.pi
    fraction = Fraction(decimal).limit_denominator()
    print(fraction)
    print(factor, np.min(y))
    plt.plot(x,y)
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
    print(k_vector)
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
        "A1": A1,
        "A2": A2,
        "A_L": A_L,
        "A_R": A_R,
        "Normalized Field": E_normalized,
        "Stokes Parameters": stokes_parameters
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

check_min()
b = 1.82
theta = 1.3169328487746776
#theta = np.pi/2
phi = np.pi
Bx = b * np.sin(theta) * np.cos(phi)
By = b * np.sin(theta) * np.sin(phi)
Bz = b * np.cos(theta)
B = [Bx, By, Bz]

model = transitioncompute(B)
ee, eg = model.return_levels()

v1 = model.get_A2() / np.linalg.norm(model.get_A2())

v2 = model.get_B1() / np.linalg.norm(model.get_B1())

v3 = model.get_A1() / np.linalg.norm(model.get_A1())

v4 = model.get_B2() / np.linalg.norm(model.get_B2())

c = np.vdot(v1, v2)
# print(fr'A2 :', v1)
# print('B1', v2)
# print(fr'A2 B1 overlap {np.abs(c):.16f}')

# result = compute_polarization(np.conj(v1))
result = compute_polarization(v1)
result2 = compute_polarization(v2)
result3 = compute_polarization(v3)
result4 = compute_polarization(v4)

Ex, Ey = result['Ex'], result['Ey']
Ex2, Ey2 = result2['Ex'], result2['Ey']
Ex3, Ey3 = result3['Ex'], result3['Ey']
Ex4, Ey4 = result4['Ex'], result4['Ey']

print(fr'A1: ', fr'Ex :{Ex3:.3f} , Ey: {Ey3:.3f}')
print(fr'A2: ', fr'Ex :{Ex:.3f} , Ey: {Ey:.3f}')
print(fr'B1: ', fr'Ex :{Ex2:.3f} , Ey: {Ey2:.3f}')
print(fr'B2: ', fr'Ex :{Ex4:.3f} , Ey: {Ey4:.3f}')

print(result['A_L'], result['A_R'])
print(result2['A_L'], result2['A_R'])
plot_ellipticity(Ex2, Ey2)
plot_ellipticity(Ex, Ey)