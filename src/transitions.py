import numpy as np
from typing import Tuple
from src.model import Hamiltonian
from src.model import PhysicalConstants as Pc


class transitioncompute(Hamiltonian):
    """
    A class for analyzing quantum transitions and field amplitudes, extending the QuantumHamiltonian class.
    Includes functionality for calculating transition energies, field amplitudes, and polarization analysis.
    """

    def __init__(self, B, strain= [0,0,0,0,0,0]):
        """
        Initialize the model

        """
        super().__init__()
        self.B = B
        self.strain = strain
        self.E, self.V = Hamiltonian.get_energy_spectra(self, B=self.B, strain=self.strain)

        Px_orbital = np.array([[0,0,1,0],
                            [0,0,0,-1],
                            [1,0,0,0],
                            [0,-1,0,0]])

        self.Px = Pc.ec*Pc.afact*np.kron(Px_orbital, np.identity(2))

        Py_orbital = np.array([[0,0,0,-1],
                            [0,0,-1,0],
                            [0,-1,0,0],
                            [-1,0,0,0]])

        self.Py = Pc.ec*Pc.afact*np.kron(Py_orbital, np.identity(2))

        Pz_orbital = 2*np.array([[0,0,1,0],
                             [0,0,0,1],
                             [1,0,0,0],
                             [0,1,0,0]])

        self.Pz = Pc.ec*Pc.afact*np.kron(Pz_orbital, np.identity(2))



    def return_levels(self):
        '''
        return: energy levels
        '''
        return self.E

    def return_vectors(self):
        '''
        return: eigenvectors
        '''
        return self.V

    def return_field_amp(self, v1,v2) -> Tuple[complex, complex, complex]:
        """
        Calculate field amplitudes for given excited and ground state eigenvectors.
        Returns:
            Tuple[complex, complex, complex]: Field amplitudes (Ax, Ay, Az)
        """
        # Normalize vectors
        vg = v1 / np.linalg.norm(v1)
        ve = v2 / np.linalg.norm(v2)

        # Calculate field amplitudes
        Ax = np.conj(vg).T @ self.Px @ ve
        Ay = np.conj(vg).T @ self.Py @ ve
        Az = np.conj(vg).T @ self.Pz @ ve

        return Ax, Ay, Az

    def convert_lab_frame(self,Ax,Ay,Az,
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
        M1 = np.array([[1, 0, 0], [0, 1, 0]])

        M2 = np.array([[np.cos(theta), 0, np.sin(theta)],[0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])

        M3 = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])

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
        # Ax_l, Ay_l =self.convert_lab_frame()
        M1 = np.array([[1, 0], [0, 0]])
        M2 = np.array([[np.cos(phi), np.sin(phi)], [np.sin(phi), -np.cos(phi)]])
        A_l = np.array([Ax_l, Ay_l])

        result = M1 @ M2 @ A_l
        return result[0], result[1]

    def scan_polarisation(self,v,
                          n_points: int = 360) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scan polarization by varying phi angle.

        Args:
            n_points (int): Number of points in the scan

        Returns:
            Tuple[np.ndarray, np.ndarray]: Phi values and corresponding magnitudes

        """
        # Ax, Ay, Az = self.get_A1()
        Ax, Ay, Az = v[0], v[1], v[2]
        phi_values = np.linspace(0, 2 * np.pi, n_points)
        magnitudes = []

        for phi in phi_values:
            Ax_l, Ay_l = self.convert_lab_frame(Ax, Ay, Az)/np.linalg.norm(self.convert_lab_frame(Ax, Ay, Az))
            Ax_f, Ay_f = self.calculate_final_field(Ax_l, Ay_l, phi)
            magnitude = np.abs(Ax_f) ** 2 + np.abs(Ay_f) ** 2
            magnitudes.append(magnitude)

        return phi_values, np.array(magnitudes)

    def get_magnitude_from_vector(self,) -> Tuple[np.ndarray, np.ndarray]:
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

    def get_A1(self):
        '''
        returns A1 transition vector
        '''
        Ve0 = self.V[:, 4]
        Vg0 = self.V[:, 0]

        A1x, A1y, A1z = self.return_field_amp(Vg0, Ve0)


        return [A1x, A1y, A1z]

    def omega_A1(self):

        Ee = self.return_levels()[4]
        Eg = self.return_levels()[0]

        return Ee-Eg

    def get_A2(self):
        '''
        returns A2 transition vector
        '''
        Ve = self.V[:, 4]
        Vg = self.V[:, 1]

        A2x, A2y, A2z = self.return_field_amp(Vg, Ve)


        return [A2x, A2y, A2z]

    def omega_A2(self):

        Ee = self.return_levels()[4]
        Eg = self.return_levels()[1]

        return Ee-Eg


    def get_B1(self):
        '''
        returns B1 transition vector
        '''
        Ve = self.V[:, 5]
        Vg = self.V[:, 0]

        B1x, B1y, B1z = self.return_field_amp(Vg, Ve)


        return [B1x, B1y, B1z]

    def get_B2(self):
        '''
        returns B1 transition vector
        '''
        Ve = self.V[:, 5]
        Vg = self.V[:, 1]

        B2x, B2y, B2z = self.return_field_amp(Vg, Ve)


        return [B2x, B2y, B2z]

    def A1B2_overlap(self):

        v1 = self.get_A1()/np.linalg.norm(self.get_A1())

        v2 = self.get_B2()/np.linalg.norm(self.get_B2())

        c = np.vdot(v1,v2)

        return np.abs(c)

    def A1A2_overlap(self):

        v1 = self.get_A1()/np.linalg.norm(self.get_A1())

        v2 = self.get_A2()/np.linalg.norm(self.get_A2())
        Ax1, Ay1 = self.convert_lab_frame(*v1)
        Ax2, Ay2 = self.convert_lab_frame(*v2)
        c = np.vdot([Ax1, Ay1], [Ax2, Ay2])


        return np.abs(c)

    def A1B1_overlap(self):

        v1 = self.get_A1()/np.linalg.norm(self.get_A1())

        v2 = self.get_B1()/np.linalg.norm(self.get_B1())

        Ax1, Ay1 = self.convert_lab_frame(*v1)
        Ax2, Ay2 = self.convert_lab_frame(*v2)
        c = np.vdot([Ax1, Ay1], [Ax2, Ay2])

        return np.abs(c)

    def A2B2_overlap(self):

        v1 = self.get_A2()/np.linalg.norm(self.get_A2())

        v2 = self.get_B2()/np.linalg.norm(self.get_B2())

        Ax1, Ay1 = self.convert_lab_frame(*v1)
        Ax2, Ay2 = self.convert_lab_frame(*v2)
        c = np.vdot([Ax1, Ay1], [Ax2, Ay2])

        return np.abs(c)

    def A2B1_overlap(self):

        v1 = self.get_A2()/np.linalg.norm(self.get_A2())

        v2 = self.get_B1()/np.linalg.norm(self.get_B1())

        Ax1, Ay1 = self.convert_lab_frame(*v1)
        Ax2, Ay2 = self.convert_lab_frame(*v2)
        c = np.vdot([Ax1, Ay1], [Ax2, Ay2])

        return np.abs(c)

    def B1B2_overlap(self):

        v1 = self.get_B1()/np.linalg.norm(self.get_B1())

        v2 = self.get_B2()/np.linalg.norm(self.get_B2())

        Ax1, Ay1 = self.convert_lab_frame(*v1)
        Ax2, Ay2 = self.convert_lab_frame(*v2)
        c = np.vdot([Ax1, Ay1], [Ax2, Ay2])

        return np.abs(c)

    def A1_rate(self):

        dipole = self.get_A1()
        energy = self.omega_A1()
        numerator = ((energy)**3)*(np.linalg.norm(dipole)**2)*4*2.417*1/137
        denominator = 3*(Pc.c**2)*(Pc.ec**2)
        return numerator/denominator

    def A2_rate(self):

        dipole = self.get_A2()
        energy = self.omega_A2()
        numerator = ((energy)**3)*(np.linalg.norm(dipole)**2)*4*2.417*1/137
        denominator = 3*(Pc.c**2)*(Pc.ec**2)
        return numerator/denominator

    def A1_strength(self):
        dipole = np.abs(self.get_A1())
        return np.linalg.norm(dipole)

    def A2_strength(self):
        dipole = np.abs(self.get_A2())
        return np.linalg.norm(dipole)

    def B1_strength(self):
        dipole = np.abs(self.get_B1())
        return np.linalg.norm(dipole)

    def B2_strength(self):
        dipole = np.abs(self.get_B2())
        return np.linalg.norm(dipole)

















