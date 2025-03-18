from src.transitions import transitioncompute
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from utils.analytic import approx_B
from utils.iqp_colors import uni, dark, light
'''

script to produce B vs strain plot with analytical fit
'''
res = 100
b_vals = np.linspace(0.001,3,res)
theta_vals = np.linspace(1e-9, np.pi-0.0001,res)
strain_vals = np.linspace(0, 1e-3, res)
phi_vals = np.linspace

cmat = np.zeros((res,res))
for i in tqdm(range(res)):
    for j in range(res):

        p = 5.902
        b = 1.052e-01
        t = theta_vals[i]
        Bx = b * np.sin(t) * np.cos(p)
        By = b * np.sin(t) * np.sin(p)
        Bz = b * np.cos(t)
        B = [Bx, By, Bz]

        exg = strain_vals[j]
        exyg = 0
        model = transitioncompute(B, strain=[exg, exyg])
        c =model.A1_rate()

        cmat[i,j] = c

b_vals_approx = []
for strain in strain_vals:
    q = approx_B(strain)
    b_vals_approx.append(q)

plt.figure(figsize=(7, 5))
plt.imshow(cmat, cmap=uni, aspect='auto', extent=[strain_vals.min(), strain_vals.max(), theta_vals.min(), theta_vals.max(),], origin='lower')
cbar = plt.colorbar(label=r"Decay rate")

plt.ylabel(fr'$\theta$', size = 'xx-large')
plt.xlabel(fr'Strain', size = 'x-large')

plt.savefig('strain_theta.svg')
plt.show()

