from src.transitions import transitioncompute
import numpy as np
from plots.polarization_plots import plot_polar, plot_2polar
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from utils.analytic import approx_B

res = 500
b_vals = np.linspace(0.001,3,res)
theta_vals = np.linspace(0, np.pi,res)
strain_vals = np.linspace(0.0001, 1e-3, res)
phi_vals = np.linspace

cmat = np.zeros((res,res))
for i in tqdm(range(res)):
    for j in range(res):

        p = np.pi/4
        b = b_vals[i]
        t = np.pi/2
        Bx = b * np.sin(t) * np.cos(p)
        By = b * np.sin(t) * np.sin(p)
        Bz = b * np.cos(t)
        B = [Bx, By, Bz]

        # exg = 0.001000
        # exyg = 0.000925
        exg = strain_vals[j]
        exyg = 0
        model = transitioncompute(B, strain=[exg, exyg])
        v1 = model.get_A1() / np.linalg.norm(model.get_A1())
        Ax2, Ay2 = model.convert_lab_frame(*v1)

        v2 = model.get_A2() / np.linalg.norm(model.get_A2())
        Ax, Ay = model.convert_lab_frame(*v2)

        c = np.abs(np.vdot([Ax2, Ay2], [Ax, Ay]))

        cmat[i,j] = c

b_vals_approx = []
for strain in strain_vals:
    q = approx_B(strain)
    b_vals_approx.append(q)

plt.figure(figsize=(8, 8))
plt.imshow(cmat, cmap='viridis', aspect='auto', extent=[strain_vals.min(), strain_vals.max(), b_vals.min(), b_vals.max(),], origin='lower')
plt.colorbar(label="Value")
plt.title(r"Overlap in lab frame, $\theta = \pi/2, \Phi = 0, \;\;\; e_{xy} = 0$ ")
plt.ylabel('B field (T)')
plt.xlabel(fr'$E_x$ strain')
plt.plot(strain_vals[230:], b_vals_approx[230:], color = 'r', label = 'theoretical approximate fit')
plt.legend()
plt.savefig('theory.svg')
plt.show()
