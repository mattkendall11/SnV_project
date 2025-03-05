from src.transitions import transitioncompute
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from utils.analytic import fit_ns, fita2b1
from utils.iqp_colors import uni, dark, light
'''

script to produce B vs strain plot with analytical fit
'''
res = 500
b_vals = np.linspace(0.001,3,res)
theta_vals = np.linspace(0, np.pi,res)
strain_vals = np.linspace(0.0001, 1e-3, res)
phi_vals = np.linspace(0.001, np.pi*2, res)

cmat1 = np.zeros((res,res))
cmat2 = np.zeros((res, res))
cmat3 = np.zeros((res,res))
cmat4 = np.zeros((res, res))
cmat5 = np.zeros((res,res))
cmat6 = np.zeros((res, res))
for i in tqdm(range(res)):
    for j in range(res):

        p = 0
        b = b_vals[j]
        t = theta_vals[i]
        Bx = b * np.sin(t) * np.cos(p)
        By = b * np.sin(t) * np.sin(p)
        Bz = b * np.cos(t)
        B = [Bx, By, Bz]

        # exg = 0.001000
        # exyg = 0.000925
        exg = 0
        exyg = 0
        model = transitioncompute(B, strain=[exg, exyg])
        v1 = model.get_A1() / np.linalg.norm(model.get_A1())
        Ax2, Ay2 = model.convert_lab_frame(*v1)
        v2 = model.get_A2() / np.linalg.norm(model.get_A2())
        Ax, Ay = model.convert_lab_frame(*v2)

        c = np.abs(np.vdot([Ax2, Ay2], [Ax, Ay]))
        # c = np.real(np.vdot([Ax2, Ay2], [Ax, Ay]))
        cmat1[i,j] = c

        v1 = model.get_A1() / np.linalg.norm(model.get_A1())
        Ax2, Ay2 = model.convert_lab_frame(*v1)
        v2 = model.get_B1() / np.linalg.norm(model.get_B1())
        Ax, Ay = model.convert_lab_frame(*v2)

        c = np.abs(np.vdot([Ax2, Ay2], [Ax, Ay]))
        # c = np.real(np.vdot([Ax2, Ay2], [Ax, Ay]))
        cmat2[i,j] = c

        v1 = model.get_A1() / np.linalg.norm(model.get_A1())
        Ax2, Ay2 = model.convert_lab_frame(*v1)
        v2 = model.get_B2() / np.linalg.norm(model.get_B2())
        Ax, Ay = model.convert_lab_frame(*v2)

        c = np.abs(np.vdot([Ax2, Ay2], [Ax, Ay]))
        # c = np.real(np.vdot([Ax2, Ay2], [Ax, Ay]))
        cmat3[i,j] = c

        v1 = model.get_A2() / np.linalg.norm(model.get_A2())
        Ax2, Ay2 = model.convert_lab_frame(*v1)
        v2 = model.get_B1() / np.linalg.norm(model.get_B1())
        Ax, Ay = model.convert_lab_frame(*v2)

        c = np.abs(np.vdot([Ax2, Ay2], [Ax, Ay]))
        # c = np.real(np.vdot([Ax2, Ay2], [Ax, Ay]))
        cmat4[i,j] = c


        v1 = model.get_A2() / np.linalg.norm(model.get_A2())
        Ax2, Ay2 = model.convert_lab_frame(*v1)
        v2 = model.get_B2() / np.linalg.norm(model.get_B2())
        Ax, Ay = model.convert_lab_frame(*v2)

        c = np.abs(np.vdot([Ax2, Ay2], [Ax, Ay]))
        # c = np.real(np.vdot([Ax2, Ay2], [Ax, Ay]))
        cmat5[i,j] = c


        v1 = model.get_B1() / np.linalg.norm(model.get_B1())
        Ax2, Ay2 = model.convert_lab_frame(*v1)
        v2 = model.get_B2() / np.linalg.norm(model.get_B2())
        Ax, Ay = model.convert_lab_frame(*v2)

        c = np.abs(np.vdot([Ax2, Ay2], [Ax, Ay]))
        # c = np.real(np.vdot([Ax2, Ay2], [Ax, Ay]))
        cmat6[i,j] = c



fig, axes = plt.subplots(2, 3, figsize=(15, 6))
cmap_list = [cmat1, cmat2, cmat3, cmat4, cmat5, cmat6]
titles = [
    r"$\langle A1|A2\rangle$",
    r"$\langle A1|B1\rangle$",
    r"$\langle A1|B2\rangle$",
    r"$\langle A2|B1\rangle$",
    r"$\langle A2|B2\rangle$",
    r"$\langle B1|B2\rangle$"

]

for ax, cmat, title in zip(axes.flat, cmap_list, titles):
    im = ax.imshow(
        cmat, cmap=uni, aspect='auto',
        extent=[b_vals.min(), b_vals.max(), theta_vals.min(), theta_vals.max()],
        origin='lower'
    )

    ax.set_title(title, size='x-large')
    ax.set_ylabel(fr'$\theta$', size='x-large')
    ax.set_xlabel(fr'B (T)', size='x-large')

# Add a single colorbar for all subplots
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax, label=r"Overlap")
cbar.ax.tick_params(labelsize='large')

plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig('B_theta_subplot.svg')
plt.show()