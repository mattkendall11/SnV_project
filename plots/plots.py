import matplotlib.pyplot as plt
from src.transitions import transitioncompute
import numpy as np
from polarization_plots import plot_2polar, plot_polar

p = 0
b = 1e-3
t = np.pi/2
Bx = b * np.sin(t) * np.cos(p)
By = b * np.sin(t) * np.sin(p)
Bz = b * np.cos(t)
B = [Bx, By, Bz]


model = transitioncompute(B)


A1 = model.get_A1()
A2 = model.get_A2()
B1 = model.get_B1()
B2 = model.get_B2()

A1ang, A1mag = model.scan_polarisation(A1)
A2ang, A2mag = model.scan_polarisation(A2)
B1ang, B1mag = model.scan_polarisation(B1)
B2ang, B2mag = model.scan_polarisation(B2)
plot_polar(B2ang, B2mag)
plot_2polar(A1ang, A1mag, A2ang, A2mag,labels=['A1', 'B2'])
