from sympy.matrices.expressions.blockmatrix import bounds
from src.transitions import transitioncompute
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def objective(B, theta, phi, sx, sxy, Ex, Ey, Exc, Eyc):
    """
    Compute the objective function for optimization.

    Parameters:
    - B: Magnetic field strength
    - theta, phi: Magnetic field orientation angles
    - sx, sxy: Strain parameters
    - Ex, Ey: Real components of electric field
    - Exc, Eyc: Imaginary components of electric field

    Returns:
    - Objective function value to minimize
    """
    Bx = B * np.sin(theta) * np.cos(phi)
    By = B * np.sin(theta) * np.sin(phi)
    Bz = B * np.cos(theta)
    B = [Bx, By, Bz]

    model = transitioncompute(B, [sx, sxy])

    v1 = model.get_A1() / np.linalg.norm(model.get_A1())
    Ax, Ay = model.convert_lab_frame(*v1)
    A1 = [Ax, Ay]

    v2 = model.get_A2() / np.linalg.norm(model.get_A2())
    Ax2, Ay2 = model.convert_lab_frame(*v2)
    A2 = [Ax2, Ay2]

    v3 = model.get_B1() / np.linalg.norm(model.get_B1())
    Bx, By = model.convert_lab_frame(*v3)
    B1 = [Bx, By]

    v4 = model.get_B2() / np.linalg.norm(model.get_B2())
    Bx2, By2 = model.convert_lab_frame(*v4)
    B2 = [Bx2, By2]

    # Create complex field vector properly
    Field = np.array([Ex + Exc * 1j, Ey + Eyc * 1j])
    c = np.dot(A1, Field) ** 2 - np.dot(A2, Field) ** 2 - np.dot(B1, Field) ** 2 + np.dot(B2, Field) ** 2

    return -np.abs(c)  # Negative because we're minimizing


def field_magnitude_constraint(params):
    """
    Constraint function ensuring total field magnitude |field+cfield|^2 = 1

    Returns:
    - Value that should be 0 when constraint is satisfied
    """
    _, _, _, _, _, Ex, Ey, Exc, Eyc = params
    # Create complex field vector properly
    Field = np.array([Ex + Exc * 1j, Ey + Eyc * 1j])
    magnitude_squared = np.abs(Field[0]) ** 2 + np.abs(Field[1]) ** 2
    return magnitude_squared - 1  # Should be 0 when magnitude is 1


# Define parameter bounds
bounds = [
    (0.001, 1),  # Magnetic field strength
    (0.001, np.pi - 0.01),  # Polar angle
    (0.001, 2 * np.pi - 0.001),  # Azimuthal angle
    (0, 1e-3),  # Strain parameter sx
    (0, 1e-3),  # Strain parameter sxy
    (-1, 1),  # Electric field Ex
    (-1, 1),  # Electric field Ey
    (-1, 1),  # Electric field Exc (imaginary)
    (-1, 1)  # Electric field Eyc (imaginary)
]

# Define constraint
constraint = {
    'type': 'eq',
    'fun': field_magnitude_constraint
}

# Initial guess
x0 = [0.5, np.pi / 2, np.pi / 2, 5e-4, 5e-4, 0.5, 0.5, 0.5, 0.5]

# Normalize the initial field values to satisfy the constraint
Ex, Ey, Exc, Eyc = x0[5:9]
Field = [Ex, Ey]
cField = [Exc * 1j, Eyc * 1j]
total_field = np.array(Field) + np.array(cField)
magnitude = np.sqrt(np.abs(total_field[0]) ** 2 + np.abs(total_field[1]) ** 2)
x0[5:9] = [Ex / magnitude, Ey / magnitude, Exc / magnitude, Eyc / magnitude]

# Run optimization
result = minimize(
    lambda params: objective(*params),
    x0,
    bounds=bounds,
    constraints=[constraint],
    method='SLSQP',
    options={'ftol': 1e-6, 'disp': True, 'maxiter': 1000}
)

print("Optimization result:")
print(f"Success: {result.success}")
print(f"Message: {result.message}")
print(f"Final objective value: {result.fun}")

# Extract optimal values
B_opt, theta_opt, phi_opt, sx_opt, sxy_opt, Ex_opt, Ey_opt, Exc_opt, Eyc_opt = result.x

print("\nOptimal parameters:")
print(f"B = {B_opt:.6f}")
print(f"theta = {theta_opt:.6f}")
print(f"phi = {phi_opt:.6f}")
print(f"sx = {sx_opt:.6f}")
print(f"sxy = {sxy_opt:.6f}")
print(f"Ex = {Ex_opt:.6f}")
print(f"Ey = {Ey_opt:.6f}")
print(f"Exc = {Exc_opt:.6f}")
print(f"Eyc = {Eyc_opt:.6f}")

# Verify constraint
Field_opt = [Ex_opt, Ey_opt]
cField_opt = [Exc_opt * 1j, Eyc_opt * 1j]
total_field_opt = np.array(Field_opt) + np.array(cField_opt)
magnitude_opt = np.abs(total_field_opt[0]) ** 2 + np.abs(total_field_opt[1]) ** 2

print(f"\nTotal field magnitude squared: {magnitude_opt:.10f}")

# Calculate optimal magnetic field components
Bx_opt = B_opt * np.sin(theta_opt) * np.cos(phi_opt)
By_opt = B_opt * np.sin(theta_opt) * np.sin(phi_opt)
Bz_opt = B_opt * np.cos(theta_opt)

print("\nOptimal magnetic field components:")
print(f"Bx = {Bx_opt:.6f}")
print(f"By = {By_opt:.6f}")
print(f"Bz = {Bz_opt:.6f}")