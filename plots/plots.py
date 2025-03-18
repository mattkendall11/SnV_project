import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from src.transitions import transitioncompute
import numpy as np
from polarization_plots import stokes_s3_and_ellipticity
from utils.iqp_colors import uni, light, dark


p = 5.902
b = 1.052e-01
t = 2.092
Ex = 0.0003721
Bx = b * np.sin(t) * np.cos(p)
By = b * np.sin(t) * np.sin(p)
Bz = b * np.cos(t)
B = [Bx, By, Bz]


model = transitioncompute(B, strain=[Ex,0])

strain_vals = np.linspace(0, 1e-3, 500)


elip_valsA2 = []
elip_valsB1 = []
stoke_valsA2 = []
stoke_valsB1 = []

for Exi in strain_vals:
    model = transitioncompute(B, strain=[Exi, 0])
    A2 = model.get_A2()
    B1 = model.get_B1()
    A2lfx1, A2lfy1 = model.convert_lab_frame(*A2)
    s3, elip = stokes_s3_and_ellipticity(A2lfx1, A2lfy1)

    B1lfx1, B1lfy1 = model.convert_lab_frame(*B1)
    s32, elip2 = stokes_s3_and_ellipticity(B1lfx1, B1lfy1)

    elip_valsA2.append(elip)
    elip_valsB1.append(elip2)
    stoke_valsA2.append(s3)
    stoke_valsB1.append(s32)

model = transitioncompute(B, strain=[Ex,0])

A1 = model.get_A1()/np.linalg.norm(model.get_A1())
A2 = model.get_A2()/np.linalg.norm(model.get_A2())
B1 = model.get_B1()/np.linalg.norm(model.get_B1())
B2 = model.get_B2()/np.linalg.norm(model.get_B2())

A1ang, A1mag = model.scan_polarisation(A1)
A2ang, A2mag = model.scan_polarisation(A2)
B1ang, B1mag = model.scan_polarisation(B1)
B2ang, B2mag = model.scan_polarisation(B2)
# plot_2polar(A2ang, A2mag, B1ang, B1mag,labels=['A2', 'B1'])
#
A2lfx, A2lfy = model.convert_lab_frame(*A2)
B1lfx, B1lfy = model.convert_lab_frame(*B1)


res = 500
cmat = np.load('data.npy')
theta_vals = np.linspace(1e-9, np.pi-0.0001,res)
strain_vals = np.linspace(0, 1e-3, res)
plt.plot(theta_vals, strain_vals, color = dark[0])

fig = plt.figure(figsize=(14,7))


gs = GridSpec(2, 3, height_ratios=[1.3, 1], width_ratios=[0.8, 0.8, 1])

ax2 = fig.add_subplot(gs[0, :])  # One large plot on top
ax1 = fig.add_subplot(gs[1, 0], projection = 'polar')  # Bottom-left
ax3 = fig.add_subplot(gs[1, 1])  # Bottom-middle
ax4 = fig.add_subplot(gs[1, 2])  # Bottom-right
pos = ax2.get_position()
ax2.set_position([pos.x0 +7, pos.y0+5, pos.width, pos.height])
# Hide axes of the polar plot
ax1.set_yticklabels([])  # Hide radial labels
ax1.set_yticks([])  # Remove radial ticks
ax1.spines['polar'].set_visible(False)
for spine in ax3.spines.values():
    spine.set_visible(False)
for spine in ax4.spines.values():
    spine.set_visible(False)
# Polar plot (top row, spanning all columns)
ax1.plot(A2ang, A2mag, color=dark[0], label='A2')
ax1.plot(B1ang, B1mag, color=dark[1], label='B1')

# Color plot (bottom-left)
im = ax2.imshow(cmat, cmap=uni, aspect='auto',
               extent=[strain_vals.min(), strain_vals.max(),
                       theta_vals.min(), theta_vals.max()],
               origin='lower')
cbar = fig.colorbar(im, ax=ax2, label=r"$\langle A2|B1\rangle$")
cbar.ax.tick_params(labelsize='large')
custom_ticks = [1e-4, 0.5e-3, 1e-3]
custom_labels = [r'$10^{-4}$', r'$5\times10^{-4}$', r'$10^{-3}$']
ax2.set_xticks(custom_ticks)
ax2.set_xticklabels(custom_labels)
ax2.set_ylabel(fr'$\theta$', size = 'x-large')
ax2.set_xlabel('Strain', size = 'x-large')

# Elliptical plot (bottom-middle)
t = np.linspace(0, 2 * np.pi, 360)
Ex, Ey = A2lfx, A2lfy
E_x = np.real(Ex) * np.cos(t) - np.imag(Ex) * np.sin(t)
E_y = np.real(Ey) * np.cos(t) - np.imag(Ey) * np.sin(t)
ax3.plot(E_x, E_y, label='A2', color=dark[0])
Ex, Ey = B1lfx, B1lfy
E_x = np.real(Ex) * np.cos(t) - np.imag(Ex) * np.sin(t)
E_y = np.real(Ey) * np.cos(t) - np.imag(Ey) * np.sin(t)
ax3.plot(E_x, E_y, color=dark[1], label='B1')
ax3.axhline(0, color='black', linewidth=0.5, linestyle='--')
ax3.axvline(0, color='black', linewidth=0.5, linestyle='--')
ax3.set_xlabel(r"$E_x$", size = 'x-large')
ax3.set_ylabel(r"$E_y$", size = 'x-large')
ax3.axis('equal')

# Strain plot (bottom-right)
ax4.plot(strain_vals, stoke_valsA2, color=dark[0])
ax4.plot(strain_vals, stoke_valsB1, color=dark[1])
ax4.set_ylabel(fr'$S_3$', size = 'x-large')
ax4.set_xlabel('Strain', size = 'x-large')
ax4.set_xticks(custom_ticks)
ax4.set_xticklabels(custom_labels)

# Subplot labels
fig.text(0.08, 0.9, 'a', fontsize=16, fontweight='bold')
fig.text(0.08, 0.4, 'b', fontsize=16, fontweight='bold')
fig.text(0.35, 0.4, 'c', fontsize=16, fontweight='bold')
fig.text(0.6, 0.4, 'd', fontsize=16, fontweight='bold')

plt.savefig('strain_big.svg')
plt.show()
