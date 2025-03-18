import numpy as np
import matplotlib.pyplot as plt
from src.transitions import transitioncompute


p = 5.902
b = 1e-9
t = np.pi
p_vals = np.linspace(0,np.pi,1000)
B_vals = np.linspace(0,3,1000)

Bx = b * np.sin(t) * np.cos(p)
By = b * np.sin(t) * np.sin(p)
Bz = b * np.cos(t)
B = [Bx, By, Bz]

model = transitioncompute(B)
a1rat = model.A1_rate()
a2rat = model.A2_rate()


