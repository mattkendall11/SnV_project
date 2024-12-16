import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from dataclasses import dataclass
from model import QuantumHamiltonian

class transitioncompute(QuantumHamiltonian):
    """
    A class for analyzing quantum transitions and field amplitudes, extending the QuantumHamiltonian class.
    Includes functionality for calculating transition energies, field amplitudes, and polarization analysis.
    """

    def __init__(self, B):
        """Initialize the QuantumTransitionAnalyzer."""
        super().__init__()
        self.B = B
        self.Eg, self.Vg, self.Ee, self.Ve = QuantumHamiltonian.get_energy_spectra(self, B=self.B)


    def return_transitions(self):
        """
        Calculate all possible transitions between excited and ground states.

        Args:
            Ee (np.ndarray): Eigenvalues of excited hamiltonian
            Eg (np.ndarray): Eigenvalues of ground hamiltonian

        Returns:
            np.ndarray: Array of all possible transitions
        """
        transitions = []
        for e in self.Ee:
            for g in self.Eg:
                transitions.append(e - g)
        return np.array(transitions)

    def return_levels(self):
        '''
        :return: energy levels
        '''
        return self.Eg, self.Ee
    def return_vectors(self):
        '''

        :return: eigenvectors
        '''
        return self.Vg, self.Ve

    def return_field_amp(self, v1,v2) -> Tuple[complex, complex, complex]:
        """
        Calculate field amplitudes for given excited and ground state eigenvectors.
        Returns:
            Tuple[complex, complex, complex]: Field amplitudes (Ax, Ay, Az)
        """
        # Normalize vectors
        vg = v1 / np.linalg.norm(v1)
        ve = v2 / np.linalg.norm(v2)

        # Define polarization operators
        Px = np.kron(np.array([[0, 1], [1, 0]]), np.identity(2))
        Py = np.kron(np.array([[0, -1j], [1j, 0]]), np.identity(2))
        Pz = np.kron(2 * np.array([[1, 0], [0, 1]]), np.identity(2))

        # Calculate field amplitudes
        Ax = np.conj(vg).T @ Px @ ve
        Ay = np.conj(vg).T @ Py @ ve
        Az = np.conj(vg).T @ Pz @ ve

        return Ax, Ay, Az

    def convert_lab_frame(self,
                          theta: float = 54.7 * np.pi / 180,
                          phi: float = 0) -> Tuple[complex, complex]:
        """
        Convert field amplitudes to lab frame coordinates.

        Args:
            Ax (complex): X-component of field amplitude
            Ay (complex): Y-component of field amplitude
            Az (complex): Z-component of field amplitude
            theta (float): Angle theta in radians
            phi (float): Angle phi in radians

        Returns:
            Tuple[complex, complex]: Transformed coordinates (Ax_lab, Ay_lab)
        """
        Ax, Ay, Az = self.return_field_amp()
        M1 = np.array([[1, 0, 0],
                       [0, 1, 0]])

        M2 = np.array([[np.cos(theta), 0, np.sin(theta)],
                       [0, 1, 0],
                       [-np.sin(theta), 0, np.cos(theta)]])

        M3 = np.array([[np.cos(phi), -np.sin(phi), 0],
                       [np.sin(phi), np.cos(phi), 0],
                       [0, 0, 1]])

        A = np.array([Ax, Ay, Az])
        result = M1 @ M2 @ M3 @ A

        return result[0], result[1]

    def calculate_final_field(self,
                              Ax_l: complex,
                              Ay_l: complex,
                              phi: float) -> Tuple[complex, complex]:
        """
        Calculate final field after half-wave plate transformation.

        Args:
            Ax_l (complex): X-component in lab frame
            Ay_l (complex): Y-component in lab frame
            phi (float): Half-wave plate angle

        Returns:
            Tuple[complex, complex]: Final field components (Ax_final, Ay_final)
        """
        Ax_l, Ay_l =self.convert_lab_frame()
        M1 = np.array([[1, 0],
                       [0, 0]])
        M2 = np.array([[np.cos(phi), np.sin(phi)],
                       [np.sin(phi), -np.cos(phi)]])
        A_l = np.array([Ax_l, Ay_l])

        result = M1 @ M2 @ A_l
        return result[0], result[1]

    def scan_polarisation(self,
                          n_points: int = 360) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scan polarization by varying phi angle.

        Args:
            n_points (int): Number of points in the scan

        Returns:
            Tuple[np.ndarray, np.ndarray]: Phi values and corresponding magnitudes

        """
        Ax, Ay, Az = self.return_field_amp()
        phi_values = np.linspace(0, 2 * np.pi, n_points)
        magnitudes = []

        for phi in phi_values:
            Ax_l, Ay_l = self.convert_lab_frame(Ax, Ay, Az)
            Ax_f, Ay_f = self.calculate_final_field(Ax_l, Ay_l, phi)
            magnitude = np.abs(Ax_f) ** 2 + np.abs(Ay_f) ** 2
            magnitudes.append(magnitude)

        return phi_values, np.array(magnitudes)

    def get_magnitude_from_vector(self,
) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate magnitude from eigenvectors.

        Args:
            Ve (np.ndarray): Excited state eigenvector
            Vg (np.ndarray): Ground state eigenvector

        Returns:
            Tuple[np.ndarray, np.ndarray]: Phi values and magnitudes
        """
        Ax, Ay, Az = self.return_field_amp(self.Ve, self.Vg)
        return self.scan_polarisation(Ax, Ay, Az)

    def get_c_magnitudes(self):
        """
        Calculate transition strength between excited and ground states.

        Args:
            B (np.ndarray) magnetic field

        Returns:
            float: Magnitude of transition strength

        """

        Ve0 = self.Ve[:, 0]
        Vg0 = self.Vg[:, 0]
        Vg1 = self.Vg[:, 1]

        A1x, A1y, A1z = self.return_field_amp(Vg0, Ve0)
        A2x, A2y, A2z = self.return_field_amp(Vg1, Ve0)

        '''
        dagger one of the field amplitudes
        '''

        c = np.vdot([A1x, A1y, A1z], [A2x, A2y, A2z])

        return np.abs(c)

    def get_overlapA2(self):
        '''
        returns overlap of A2 transitions
        '''
        Ve1 = self.Ve[:, 1]
        Vg0 = self.Vg[:, 0]
        Vg1 = self.Vg[:, 1]

        A1x, A1y, A1z = self.return_field_amp(Vg0, Ve1)
        A2x, A2y, A2z = self.return_field_amp(Vg1, Ve1)

        '''
        dagger one of the field amplitudes
        '''

        c = np.vdot([A1x, A1y, A1z], [A2x, A2y, A2z])

        return np.abs(c)
    
    def get_field_amplitudes(self):
        '''

        '''
        Ve1, Ve2= self.Ve[:, 0], self.Ve[:, 1]
        Vg1, Vg2  = self.Vg[:, 0], self.Vg[:, 1]

        field_amps_ve1 = [self.return_field_amp(Vg, Ve1) for Vg in [Vg1, Vg2]]
        field_amps_ve2 = [self.return_field_amp(Vg, Ve2) for Vg in [Vg1, Vg2]]

        # Unpack results for Ve1
        A1x, A1y, A1z = field_amps_ve1[0]
        A2x, A2y, A2z = field_amps_ve1[1]


        # Unpack results for Ve2
        B1x, B1y, B1z = field_amps_ve2[0]
        B2x, B2y, B2z = field_amps_ve2[1]


        amps = [np.linalg.norm([A1x, A1y, A1z]), np.linalg.norm([A2x, A2y, A2z]),
                np.linalg.norm([B1x, B1y, B1z]), np.linalg.norm([B2x, B2y, B2z])]

        A1 = np.array([A1x, A1y, A1z])/amps[0]
        A2 = np.array([A2x, A2y, A2z])/amps[1]
        B1 = np.array([B1x, B1y, B1z])/amps[2]
        B2 = np.array([B2x, B2y, B2z])/amps[3]

        # Combine vectors into a single list for matrix calculation
        vectors = [A1, A2, B1, B2]
        labels = ["A1", "A2", "B1", "B2"]

        # Calculate the dot product matrix
        dot_product_matrix = np.zeros((len(vectors), len(vectors)))

        for i in range(len(vectors)):
            for j in range(len(vectors)):
                dot_product_matrix[i, j] = np.abs(np.vdot(vectors[i], vectors[j]))
        real_amps = [np.abs(x) for x in amps]
        return real_amps, dot_product_matrix

    def dipole_matrix(self):
        '''

        '''

        Px = np.kron(np.array([[0, 1], [1, 0]]), np.identity(2))
        Py = np.kron(np.array([[0, -1j], [1j, 0]]), np.identity(2))
        Pz = np.kron(2 * np.array([[1, 0], [0, 1]]), np.identity(2))

        Px_matrix = np.zeros((8,8))
        Py_matrix = np.zeros((8, 8))
        Pz_matrix = np.zeros((8, 8))

        eigenstates = np.concatenate((self.Vg, self.Ve), axis=0)
        norms = np.linalg.norm(eigenstates, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1
        # Divide each vector by its norm
        normalized_eigenstates = eigenstates / norms

        for i in range(4):
            for j in range(4,8):
                Px_matrix[i,j] = np.conj(normalized_eigenstates[i].T) @ Px @ normalized_eigenstates[j]
                Px_matrix[j,i] = np.conj(normalized_eigenstates[j].T) @ Px @ normalized_eigenstates[i]

                Py_matrix[i, j] = np.conj(normalized_eigenstates[i].T) @ Py @ normalized_eigenstates[j]
                Py_matrix[j, i] = np.conj(normalized_eigenstates[j].T) @ Py @ normalized_eigenstates[i]

                Pz_matrix[i, j] = np.conj(normalized_eigenstates[i].T) @ Pz @ normalized_eigenstates[j]
                Pz_matrix[j, i] = np.conj(normalized_eigenstates[j].T) @ Pz @ normalized_eigenstates[i]

        return [Px_matrix, Py_matrix, Pz_matrix]
    def energy_matrix(self):
        '''

        '''
        energies = np.concatenate((self.Eg, self.Ee), axis=0)
        H0 = np.zeros((8,8))

        for i in range(8):
            H0[i,i] = energies[i]

        return H0

    def E_field(self, w, t):
        """
        Compute the quantized electric field components for a single cavity mode.

        Parameters:
            omega (float): Angular frequency of the cavity mode (rad/s).
            V (float): Effective mode volume (m^3).

        Returns:
            numpy.ndarray: Array containing the electric field components [E_x, E_y, E_z].
        """
        # Constants
        hbar = 1.0545718e-34  # Reduced Planck's constant (J·s)
        epsilon_0 = 8.854187817e-12  # Permittivity of free space (F/m)

        V = 0.8
        # Prefactor for the electric field
        prefactor = 1j * np.sqrt(hbar * w / (2 * epsilon_0 * V))
        a = 5

        # Electric field components
        E_x = prefactor * (a *np.exp(1j*w*t) - a *np.exp(-1j*w*t))
        E_y = prefactor * (a *np.exp(1j*w*t) - a *np.exp(-1j*w*t))
        E_z = prefactor * (a *np.exp(1j*w*t) - a *np.exp(-1j*w*t))

        # Return as a vector
        return np.array([E_x, E_y, E_z])

    def td_hamiltonian(self,w,t):
        '''

        '''
        muE = [matrix * factor for matrix, factor in zip(self.dipole_matrix(), self.E_field(w,t))]
        muE = np.abs(muE[0]+muE[1]+muE[2])
        return self.energy_matrix() - muE

    def get_A2(self):

        Ve1 = self.Ve[:, 1]
        Vg1 = self.Vg[:, 1]

        A2x, A2y, A2z = self.return_field_amp(Vg1, Ve1)
        return [A2x, A2y, A2z]















