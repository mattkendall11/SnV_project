import numpy as np
from src.transitions import transitioncompute
from typing import Tuple, List, Callable
import random
import warnings
from src.optimizers import optimize_function
from scipy.optimize import minimize, dual_annealing

def f_rad(b,theta,phi):

    Bx = b * np.sin(theta) * np.cos(phi)
    By = b * np.sin(theta) * np.sin(phi)
    Bz = b * np.cos(theta)
    B = [Bx, By, Bz]

    model = transitioncompute(B)

    v1 = model.get_A1()/np.linalg.norm(model.get_A1())
    Ax, Ay = model.convert_lab_frame(*v1)

    v2 = model.get_A2()/np.linalg.norm(model.get_A2())

    Ax2, Ay2 = model.convert_lab_frame(*v2)

    v3 = model.get_B1()/np.linalg.norm(model.get_B1())
    Bx, By = model.convert_lab_frame(*v3)

    v4 = model.get_B2()/np.linalg.norm(model.get_B2())

    Bx2, By2 = model.convert_lab_frame(*v4)
    return np.abs(np.vdot([Ax2, Ay2], [Bx2, By2])) + model.A1_branch() + model.B1_branch()

def f_rad_strain(b,theta,phi, Ex ,exy):

    Bx = b * np.sin(theta) * np.cos(phi)
    By = b * np.sin(theta) * np.sin(phi)
    Bz = b * np.cos(theta)
    B = [Bx, By, Bz]

    model = transitioncompute(B, [Ex, exy])

    v1 = model.get_A1()/np.linalg.norm(model.get_A1())
    Ax, Ay = model.convert_lab_frame(*v1)

    v2 = model.get_A2()/np.linalg.norm(model.get_A2())

    Ax2, Ay2 = model.convert_lab_frame(*v2)

    v3 = model.get_B1()/np.linalg.norm(model.get_B1())
    Bx, By = model.convert_lab_frame(*v3)

    v4 = model.get_B2()/np.linalg.norm(model.get_B2())

    Bx2, By2 = model.convert_lab_frame(*v4)
    return np.abs(np.vdot([Ax2, Ay2], [Bx2, By2])) + model.A1_branch()+ model.B1_branch()

def f_ellptical(b,theta, phi, ex, exy):
    Bx = b * np.sin(theta) * np.cos(phi)
    By = b * np.sin(theta) * np.sin(phi)
    Bz = b * np.cos(theta)
    B = [Bx, By, Bz]

    model = transitioncompute(B, strain=[ex, exy])

    v1 = model.get_A2() / np.linalg.norm(model.get_A2())
    Ax2, Ay2 = model.convert_lab_frame(*v1)
    c1 = 0.5 - Ax2
    c2 = 0.5j -Ay2
    return np.abs(c1)+np.abs(c2)


def optimize_da(f):
    bounds = [
        (0.001, 1),  # Magnetic field strength
        (0.001, np.pi - 0.1),  # Polar angle
        (0, 2 * np.pi),  # Azimuthal angle
        (0, 1e-3),  # Strain parameters
        (0, 1e-3)
    ]

    wrapped_f = lambda x: f(*x)
    result = dual_annealing(wrapped_f, bounds, maxiter=10000)
    return result

def optimize_da_ns(f):
    bounds = [(1e-9,3),(1e-9, np.pi-1e-9), (1e-9,2*np.pi-1e-9)]
    wrapped_f = lambda x: f(*x)
    result = dual_annealing(wrapped_f, bounds, maxiter=10000)
    return result



# result = optimize_da_ns(f_rad)
result = optimize_da(f_rad_strain)
print(result)

values = result['x']

print(f_rad_strain(*values))
