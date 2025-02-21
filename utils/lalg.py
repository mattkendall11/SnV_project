import numpy as np


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


# # Example usage
# v1 = np.array([1, 0, 0])
# v2 = np.array([0, 1, 0])
# rotation_mat = rotation_matrix_between_vectors(v1, v2)
# print("Rotation Matrix:")
# print(rotation_mat)
#
# # Verify rotation
# rotated_v2 = rotation_mat @ v2
# print("\nRotated v2:")
# print(rotated_v2)
# print("\nOriginal v1:")
# print(v1)
# print("\nClose to v1:", np.allclose(rotated_v2, v1))



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