import numpy as np
from scipy.linalg import eigh
from dataclasses import dataclass
from typing import Tuple, Optional, List


@dataclass
class PhysicalConstants:
    """Physical constants used in calculations."""
    muB: float = 9.2740100783e-24  # Bohr magneton in J/T
    hbar: float = 1.054571817e-34  # Reduced Planck's constant in J*s
    GHz: float = 1e9  # 1 GHz in Hz
    lg: float = 815e9  # Spin-orbit coupling in ground state (Hz)
    lu: float = 2355e9  # Spin-orbit coupling in excited state (Hz)
    x_g: float = 65e9 #Jahn teller coupling (Hz)
    y_g: float = 0  # Jahn-Teller coupling (ground state) (Hz)
    x_u: float = 855e9 # Jahn-Teller coupling (Hz)
    y_u: float = 0  # Jahn-Teller coupling (excited state) (Hz)
    fg: float = 0.15 # quenching factor
    fu: float = 0.15 # quenching factor
    ec:float = 1.6e-19 # elementary charge
    ep0:float = 8.84e-12 #permitivity
    c:float = 3e8 #speed of light
    me: float = 9.11e-31
    afact:float = np.sqrt(1.1366338470962457e-18)


class Hamiltonian:
    """
    A class for calculating and analyzing quantum Hamiltonians with spin-orbit coupling,
    Jahn-Teller effect, Zeeman splitting, and strain effects.

    Attributes:
        constants (PhysicalConstants): Physical constants used in calculations
        L (float): Orbital gyromagnetic ratio
        S (float): Spin gyromagnetic ratio
    """

    def __init__(self):
        """Initialize the QuantumHamiltonian with physical constants."""
        self.constants = PhysicalConstants()
        # self.L = self.constants.muB / self.constants.hbar  # Orbital gyromagnetic ratio
        self.L =self.constants.ec/(2*self.constants.me)
        self.S = 2 * self.L  # Spin gyromagnetic ratio


    @staticmethod
    def spin_orbit_hamiltonian(l: float) -> np.ndarray:
        """
        Calculate the spin-orbit Hamiltonian matrix.

        Args:
            l (float): Spin-orbit coupling constant

        Returns:
            np.ndarray: 4x4 spin-orbit Hamiltonian matrix
        """
        i = 1j
        return np.array([[0, 0, -l * i / 2, 0],
                         [0, 0, 0, l * i / 2],
                         [l * i / 2, 0, 0, 0],
                         [0, -i * l / 2, 0, 0]])
    @staticmethod
    def unpeterbed_hamiltonian():
        '''
        Returns identity of energy splitting
        '''
        return 484*10**12 *np.eye(4)

    @staticmethod
    def jahn_teller_hamiltonian(x: float, y: float) -> np.ndarray:
        """
        Calculate the Jahn-Teller Hamiltonian matrix.

        Args:
            x (float): Jahn-Teller coupling strength for x component
            y (float): Jahn-Teller coupling strength for y component

        Returns:
            np.ndarray: 4x4 Jahn-Teller Hamiltonian matrix
        """
        return np.array([[x, 0, y, 0],
                         [0, x, 0, y],
                         [y, 0, -x, 0],
                         [0, y, 0, -x]])

    def zeeman_hamiltonian(self, f: float, Bz: float) -> np.ndarray:
        """
        Calculate the Zeeman Hamiltonian matrix.

        Args:
            f (float): Quenching factor of orbital
            Bz (float): z component of magnetic field

        Returns:
            np.ndarray: 4x4 Zeeman Hamiltonian matrix
        """
        iBz_fL = 1j * f * self.L * Bz
        return np.array([[0, 0, iBz_fL, 0],
                         [0, 0, 0, iBz_fL],
                         [-iBz_fL, 0, 0, 0],
                         [0, -iBz_fL, 0, 0]])

    def spin_zeeman_hamiltonian(self, B: np.ndarray) -> np.ndarray:
        """
        Calculate the spin-Zeeman Hamiltonian matrix.

        Args:
            B (np.ndarray): Magnetic field vector [Bx, By, Bz]

        Returns:
            np.ndarray: 4x4 spin-Zeeman Hamiltonian matrix
        """
        Bx, By, Bz = B
        return self.S * np.array([[Bz, Bx - 1j * By, 0, 0],
                                  [Bx + 1j * By, -Bz, 0, 0],
                                  [0, 0, Bz, Bx - 1j * By],
                                  [0, 0, Bx + 1j * By, -Bz]])

    @staticmethod
    def strain_hamiltonian(ex: float, exy: float, d: float) -> np.ndarray:
        """
        Calculate the strain Hamiltonian matrix.

        Args:
            a (float): Alpha strain
            b (float): Beta strain
            d (float): Gamma strain

        Returns:
            np.ndarray: 4x4 strain Hamiltonian matrix
        """
        return np.array([[d*ex, 0, 2*d*exy, 0],
                         [0, d*ex, 0, 2*d*exy],
                         [2*d*exy, 0, -d*ex, 0],
                         [0, 2*d*exy, 0, -d*ex]])

    def get_energy_spectra(self,
                           B: np.ndarray,
                           strain: Tuple[float, float] = [0,0]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the energy eigenvalues and eigenvectors of the complete Hamiltonian.

        Args:
            l (float): Spin-orbit coupling constant
            x (float): Jahn-Teller x coupling strength
            y (float): Jahn-Teller y coupling strength
            f (float): Orbital quenching factor
            B (np.ndarray): Magnetic field vector [Bx, By, Bz]
            strain (np.ndarray): values of ground and excited state strain

        Returns:
            Eigenvalues and eigenvectors of the ground and excited Hamiltonian
        """
        # Default strain parameters
        ex, exy = strain[0], strain[1]

        # Construct full Hamiltonian
        Hg = (self.spin_orbit_hamiltonian(self.constants.lg) +
             self.jahn_teller_hamiltonian(self.constants.x_g, self.constants.y_g) +
             self.zeeman_hamiltonian(self.constants.fg, B[2]) +
             self.spin_zeeman_hamiltonian(B))

        He = (self.unpeterbed_hamiltonian()+self.spin_orbit_hamiltonian(self.constants.lu) +
             self.jahn_teller_hamiltonian(self.constants.x_u, self.constants.y_u) +
             self.zeeman_hamiltonian(self.constants.fu, B[2]) +
             self.spin_zeeman_hamiltonian(B))


        Hg += self.strain_hamiltonian(ex, exy, 0.787e15)
        He += self.strain_hamiltonian(ex, exy, 0.956e15)

        H = np.block([[He, np.zeros((4,4))],
                     [np.zeros((4,4)), Hg]])

        eigenvalues, eigenvectors = eigh(H)
        return eigenvalues, eigenvectors

