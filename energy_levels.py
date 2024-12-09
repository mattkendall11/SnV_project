import h5py
import numpy as np
import matplotlib.pyplot as plt
from transitions import transitioncompute
import matplotlib.ticker as ticker


from transitions import transitioncompute
import numpy as np
import h5py
from tqdm.auto import tqdm
import datetime

def plot_e_levels():
    B_vals = np.linspace(0,1,100)
    energy_ground = []
    energy_excited = []
    for b in B_vals:
        B = [0, 0, b]
        model = transitioncompute(B)


        # Store energy levels and eigenvectors
        eg, ee = model.return_levels()
        energy_ground.append(eg)

        energy_excited.append(ee)


    plt.plot( B_vals,energy_ground,  '+')
    plt.show()

    plt.plot(B_vals, energy_excited, 'o')
    plt.show()

    SnV = transitioncompute([0,0,9])
    frequencies = SnV.return_transitions()
    labels = ['A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3', 'B4', 'C1', 'C2', 'C3', 'C4', 'D1', 'D2', 'D3', 'D4']

    h = 6.62607015e-34
    c = 3*10**8
    plt.bar(labels, np.abs(frequencies))
    plt.ylabel('GHz')
    plt.show()


