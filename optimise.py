import numpy as np
from src.transitions import transitioncompute
from typing import Tuple, List, Callable
import random
import warnings
from src.optimizers import optimize_function
from scipy.optimize import minimize, dual_annealing


def f(b,theta, phi, ex, exy):
    Bx = b * np.sin(theta) * np.cos(phi)
    By = b * np.sin(theta) * np.sin(phi)
    Bz = b * np.cos(theta)
    B = [Bx, By, Bz]

    model = transitioncompute(B, strain=[ex, exy])

    v1 = model.get_A1() / np.linalg.norm(model.get_A1())
    Ax2, Ay2 = model.convert_lab_frame(*v1)

    v2 = model.get_A2() / np.linalg.norm(model.get_A2())

    Ax, Ay = model.convert_lab_frame(*v2)

    return np.abs(np.vdot([Ax2, Ay2], [Ax, Ay]))

def f_nostrain(b,theta,phi):

    Bx = b * np.sin(theta) * np.cos(phi)
    By = b * np.sin(theta) * np.sin(phi)
    Bz = b * np.cos(theta)
    B = [Bx, By, Bz]

    model = transitioncompute(B)

    v1 = model.get_A1() / np.linalg.norm(model.get_A1())
    Ax2, Ay2 = model.convert_lab_frame(*v1)

    v2 = model.get_A2() / np.linalg.norm(model.get_A2())

    Ax, Ay = model.convert_lab_frame(*v2)

    return np.abs(np.vdot([Ax2, Ay2], [Ax, Ay]))

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
    bounds = [(1e-6,3),(1e-3, np.pi-0.01), (0.0001,2*np.pi-0.0001)]
    wrapped_f = lambda x: f_nostrain(*x)
    result = dual_annealing(wrapped_f, bounds, maxiter=10000)
    return result

result = optimize_da_ns(f)
print(result)
# if __name__ == "__main__":
#     optimize_function(f)