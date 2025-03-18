import numpy as np
import matplotlib.pyplot as plt
from src.transitions import transitioncompute


p = 5.902
b = 3
t = np.pi/2
p_vals = np.linspace(0,np.pi,1000)
B_vals = np.linspace(0,3,1000)
rates = []
for b in B_vals:
    Bx = b * np.sin(t) * np.cos(p)
    By = b * np.sin(t) * np.sin(p)
    Bz = b * np.cos(t)
    B = [Bx, By, Bz]

    model = transitioncompute(B)
    a1rat = model.A1_rate()
    a2rat = model.A2_rate()
    b1rat = model.B1_rate()
    b2rat = model.B2_rate()
    rates.append([a1rat,a2rat,b1rat,b2rat])

plt.plot(B_vals, rates)


