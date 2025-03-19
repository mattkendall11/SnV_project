import numpy as np
import matplotlib.pyplot as plt
from src.transitions import transitioncompute
from utils.iqp_colors import dark, uni
from tqdm.auto import tqdm


res = 100
p_vals = np.linspace(1e-9, 2*np.pi-1e-9,res)
b_vals = np.linspace(1e-9,3,res)
t_vals = np.linspace(1e-9, np.pi-1e-9, res)
# cost_values = np.zeros((res,res,res))
# for i in tqdm(range(100)):
#     for j in range(100):
#         for k in range(100):
#             b = b_vals[i]
#             t = t_vals[j]
#             p = p_vals[k]
#             Bx = b * np.sin(t) * np.cos(p)
#             By = b * np.sin(t) * np.sin(p)
#             Bz = b * np.cos(t)
#             B = [Bx, By, Bz]
#             model = transitioncompute(B)
#             v1 = model.get_A1() / np.linalg.norm(model.get_A1())
#             Ax, Ay = model.convert_lab_frame(*v1)
#
#             v2 = model.get_A2() / np.linalg.norm(model.get_A2())
#
#             Ax2, Ay2 = model.convert_lab_frame(*v2)
#
#             v3 = model.get_B1() / np.linalg.norm(model.get_B1())
#             Bx, By = model.convert_lab_frame(*v3)
#
#             v4 = model.get_B2() / np.linalg.norm(model.get_B2())
#
#             Bx2, By2 = model.convert_lab_frame(*v4)
#             cost = (1-np.abs(np.vdot([Ax, Ay], [Ax2, Ay2])))+ (1-np.abs(np.vdot([Bx, By], [Bx2, By2])))+ np.abs(np.vdot([Ax, Ay], [Bx2, By2]))
#             cost_values[i,j,k] = cost
#
# flat_index = np.argmin(cost_values)
#
# # Convert the flat index to 3D coordinates
# min_index = np.unravel_index(flat_index, cost_values.shape)
# print("Minimum value:", cost_values[min_index])
# print("Index of minimum value:", min_index)
t = np.pi/4
p = np.pi/4
rates = []
for b in b_vals:
    Bx = b * np.sin(t) * np.cos(p)
    By = b * np.sin(t) * np.sin(p)
    Bz = b * np.cos(t)
    B = [Bx, By, Bz]

    model = transitioncompute(B)
    a1rat = model.A1_branch()
    a2rat = model.A2_branch()
    b1rat = model.B1_branch()
    b2rat = model.B2_branch()
    rates.append([a1rat,a2rat, b1rat, b2rat])

rates = np.array(rates)
plt.plot(b_vals, rates[:,0], label = 'A1', color = dark[0])
plt.plot(b_vals, rates[:,1], label = 'A2', color = dark[1])
plt.plot(b_vals, rates[:,2], label = 'B1', color = dark[2])
plt.plot(b_vals, rates[:,3], label = 'B2', color = dark[3])
plt.legend()
plt.xlabel(fr'$\theta$', size = 'xx-large')
plt.ylabel(fr'$\Gamma$', size = 'xx-large')
plt.savefig('rates.svg')
plt.show()


# cmat = np.zeros((100,100))
# for i in range(100):
#     for j in range(100):
#         b = b_vals[i]
#         t = t_vals[j]
#         Bx = b * np.sin(t) * np.cos(p)
#         By = b * np.sin(t) * np.sin(p)
#         Bz = b * np.cos(t)
#         B = [Bx, By, Bz]
#
#         model = transitioncompute(B)
#
#         cmat[i,j] = model.B2_rate()
#
# plt.imshow(cmat, cmap=uni, aspect='auto', extent=[t_vals.min(), t_vals.max(), b_vals.min(), b_vals.max(),], origin='lower')
# cbar = plt.colorbar(label=r"Decay rate")
#
# plt.xlabel(fr'$\theta$', size = 'xx-large')
# plt.ylabel(fr'B', size = 'x-large')
# plt.title('B2', size = 'xx-large')
# plt.savefig('colorplot.svg')
# plt.show()