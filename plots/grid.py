import numpy as np
from sympy.codegen import Print

from src.transitions import transitioncompute

p = 0
b = 1e-3
t =np.pi/2
Bx = b * np.sin(t) * np.cos(p)
By = b * np.sin(t) * np.sin(p)
Bz = b * np.cos(t)
B = [Bx, By, Bz]

exg = 0
exyg = 0
model = transitioncompute(B, strain=[exg, exyg])
v1 = model.get_A1() / np.linalg.norm(model.get_A1())
Ax2, Ay2 = model.convert_lab_frame(*v1)
v2 = model.get_A2() / np.linalg.norm(model.get_A2())
Ax, Ay = model.convert_lab_frame(*v2)

c = np.abs(np.vdot([Ax2, Ay2], [Ax, Ay]))
# c = np.real(np.vdot([Ax2, Ay2], [Ax, Ay]))
print('A1A2')
print(c)
v1 = model.get_A1() / np.linalg.norm(model.get_A1())
Ax2, Ay2 = model.convert_lab_frame(*v1)
v2 = model.get_B1() / np.linalg.norm(model.get_B1())
Ax, Ay = model.convert_lab_frame(*v2)

c = np.abs(np.vdot([Ax2, Ay2], [Ax, Ay]))
# c = np.real(np.vdot([Ax2, Ay2], [Ax, Ay]))
print('A1B1')
print(c)
v1 = model.get_A1() / np.linalg.norm(model.get_A1())
Ax2, Ay2 = model.convert_lab_frame(*v1)
v2 = model.get_B2() / np.linalg.norm(model.get_B2())
Ax, Ay = model.convert_lab_frame(*v2)

c = np.abs(np.vdot([Ax2, Ay2], [Ax, Ay]))
# c = np.real(np.vdot([Ax2, Ay2], [Ax, Ay]))
print('A1B2')
print(c)
v1 = model.get_A2() / np.linalg.norm(model.get_A2())
Ax2, Ay2 = model.convert_lab_frame(*v1)
v2 = model.get_B1() / np.linalg.norm(model.get_B1())
Ax, Ay = model.convert_lab_frame(*v2)

c = np.abs(np.vdot([Ax2, Ay2], [Ax, Ay]))
# c = np.real(np.vdot([Ax2, Ay2], [Ax, Ay]))
print('A2B1')
print(c)
v1 = model.get_A2() / np.linalg.norm(model.get_A2())
Ax2, Ay2 = model.convert_lab_frame(*v1)
v2 = model.get_B2() / np.linalg.norm(model.get_B2())
Ax, Ay = model.convert_lab_frame(*v2)

c = np.abs(np.vdot([Ax2, Ay2], [Ax, Ay]))
# c = np.real(np.vdot([Ax2, Ay2], [Ax, Ay]))
print('A2B2')
print(c)
v1 = model.get_B1() / np.linalg.norm(model.get_B1())
Ax2, Ay2 = model.convert_lab_frame(*v1)
v2 = model.get_B2() / np.linalg.norm(model.get_B2())
Ax, Ay = model.convert_lab_frame(*v2)

c = np.abs(np.vdot([Ax2, Ay2], [Ax, Ay]))
# c = np.real(np.vdot([Ax2, Ay2], [Ax, Ay]))
print('b1B2')
print(c)