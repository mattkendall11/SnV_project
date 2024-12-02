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
        Ax = np.conj(vg) @ Px @ ve
        Ay = np.conj(vg) @ Py @ ve
        Az = np.conj(vg) @ Pz @ ve

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

    def plot_magnitude_polar(self,

                             title: str = "Magnitude of Final Vector vs φ",
                             save_path: Optional[str] = None) -> None:
        """
        Create polar plot of magnitude vs phi.

        Args:

            title (str): Plot title
            save_path (Optional[str]): Path to save the plot
        """
        phi_values, magnitudes = self.get_magnitude_from_vector()
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(111, projection='polar')
        ax.plot(phi_values, magnitudes)
        ax.set_title(title)
        ax.grid(True)

        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Polar plot saved to {save_path}")

        plt.show()

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
        Ve1, Ve2, Ve3, Ve4 = 







