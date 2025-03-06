from src.transitions import transitioncompute
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from utils.analytic import approx_B
from utils.iqp_colors import uni, dark, light
'''

script to produce B vs strain plot with analytical fit
'''
res = 500
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
        v1 = model.get_A2() / np.linalg.norm(model.get_A2())
        Ax2, Ay2 = model.convert_lab_frame(*v1)

        v2 = model.get_B1() / np.linalg.norm(model.get_B1())
        Ax, Ay = model.convert_lab_frame(*v2)

        c = np.abs(np.vdot([Ax2, Ay2], [Ax, Ay]))

        cmat[i,j] = c

b_vals_approx = []
for strain in strain_vals:
    q = approx_B(strain)
    b_vals_approx.append(q)

plt.figure(figsize=(7, 5))
plt.imshow(cmat, cmap=uni, aspect='auto', extent=[strain_vals.min(), strain_vals.max(), theta_vals.min(), theta_vals.max(),], origin='lower')
cbar = plt.colorbar(label=r" $\langle A2|B1\rangle$")
cbar.ax.tick_params(labelsize='x-large')
custom_ticks = [1e-4, 0.5e-3, 1e-3]  # Adjust as needed
custom_labels = [r'$10^{-4}$', r'$5\times10^{-4}$', r'$10^{-3}$']

ax = plt.gca()  # Get current axis
ax.set_xticks(custom_ticks)  # Set specific tick positions
ax.set_xticklabels(custom_labels, fontsize='large')
# plt.title(r" $\langle A1|A2\rangle, \theta = \pi/2, \Phi = 0$ ", size = 'xx-large')
plt.ylabel(fr'$\theta$', size = 'xx-large')
plt.xlabel(fr'Strain', size = 'x-large')
# plt.plot(strain_vals[230:], b_vals_approx[230:], color = light[3], label = fr'Orthogonal condition', linestyle = 'dashed')
# plt.legend()

plt.savefig('strain_theta.svg')
plt.show()
