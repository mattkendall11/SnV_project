import numpy as np
import matplotlib.pyplot as plt

def rotation_matrix_between_vectors(v1, v2):
    """
    Calculate the rotation matrix that rotates v2 to align with v1.

    Parameters:
    v1 (numpy.ndarray): Target vector (3D)
    v2 (numpy.ndarray): Vector to be rotated (3D)

    Returns:
    numpy.ndarray: 3x3 rotation matrix
    """
    # Normalize the input vectors
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)

    # Calculate the cross product and dot product
    cross = np.cross(v2_norm, v1_norm)
    dot = np.dot(v2_norm, v1_norm)

    # Handle special cases
    if np.allclose(dot, 1.0):
        # Vectors are already aligned
        return np.eye(3)

    if np.allclose(dot, -1.0):
        # Vectors are opposite - need to find a perpendicular rotation
        # We'll use the Rodrigues' rotation formula with a 180-degree rotation
        # around an arbitrary perpendicular axis
        if not np.allclose(v1_norm[0], 0) or not np.allclose(v1_norm[1], 0):
            perpendicular = np.array([-v1_norm[1], v1_norm[0], 0])
        else:
            perpendicular = np.array([0, -v1_norm[2], v1_norm[1]])

        perpendicular /= np.linalg.norm(perpendicular)
        cross_matrix = np.array([
            [0, -perpendicular[2], perpendicular[1]],
            [perpendicular[2], 0, -perpendicular[0]],
            [-perpendicular[1], perpendicular[0], 0]
        ])

        return np.eye(3) + 2 * cross_matrix @ cross_matrix

    # Standard Rodrigues' rotation formula
    cross_norm = np.linalg.norm(cross)
    cross_matrix = np.array([
        [0, -cross[2], cross[1]],
        [cross[2], 0, -cross[0]],
        [-cross[1], cross[0], 0]
    ])

    # Rotation matrix calculation
    return (np.eye(3) +
            cross_matrix +
            (cross_matrix @ cross_matrix) * (1 / (1 + dot)))

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



def find_linear_polarization_transform(v1, v2):
    """
    Find a transformation matrix that converts two orthogonal complex 3D vectors
    into linearly polarized orthogonal vectors.

    Parameters:
    v1, v2: numpy arrays of shape (3,) containing complex numbers

    Returns:
    transform_matrix: 3x3 complex numpy array
    transformed_v1, transformed_v2: The transformed vectors
    """
    # Verify orthogonality of input vectors
    if abs(np.vdot(v1, v2)) > 1e-10:
        raise ValueError("Input vectors must be orthogonal")

    # Normalize the input vectors
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    # Find the phase of each component
    phases1 = np.angle(v1)
    phases2 = np.angle(v2)

    # Create diagonal matrix to remove phases from v1
    D1 = np.diag(np.exp(-1j * phases1))

    # Apply transformation to both vectors
    real_v1 = D1 @ v1
    temp_v2 = D1 @ v2

    # Find rotation matrix to align real_v1 with x-axis
    x_axis = np.array([1, 0, 0])
    rotation_axis = np.cross(real_v1.real, x_axis)
    rotation_angle = np.arccos(np.dot(real_v1.real, x_axis))

    if np.linalg.norm(rotation_axis) < 1e-10:
        R = np.eye(3)
    else:
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                      [rotation_axis[2], 0, -rotation_axis[0]],
                      [-rotation_axis[1], rotation_axis[0], 0]])
        R = np.eye(3) + np.sin(rotation_angle) * K + (1 - np.cos(rotation_angle)) * K @ K

    # Combined transformation
    transform_matrix = R @ D1

    # Apply transformation to both vectors
    transformed_v1 = transform_matrix @ v1
    transformed_v2 = transform_matrix @ v2

    return transform_matrix, transformed_v1, transformed_v2


def find_min_index(matrix):

    min_value = float('inf')
    min_index = [-1, -1]

    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            if val < min_value:
                min_value = val
                min_index = [i, j]

    return min_index[0], min_index[1]


def scan_levels():
    theta = np.pi/4
    phi = np.pi/4
    exg = 0
    exyg = 0
    b_vals = np.linspace(0,3,100)
    levels = []

    for b in b_vals:
        Bx = b * np.sin(theta) * np.cos(phi)
        By = b * np.sin(theta) * np.sin(phi)
        Bz = b * np.cos(theta)
        B = [Bx, By, Bz]

        model = transitioncompute(B, strain=[exg, exyg])
        e = model.return_levels()
        levels.append(e)

    plt.plot(b_vals, levels, color = 'b')

    plt.legend()
    plt.show()
