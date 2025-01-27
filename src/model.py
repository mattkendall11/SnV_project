import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Optional, List
import logging


@dataclass
class PhysicalConstants:
    """Physical constants used in calculations."""
    muB: float = 9.2740100783e-24  # Bohr magneton in J/T
    hbar: float = 1.054571817e-34  # Reduced Planck's constant in J*s
    GHz: float = 1e9  # 1 GHz in Hz
    lg: float = 815e9  # Spin-orbit coupling in ground state (Hz)
    lu: float = 2355e9  # Spin-orbit coupling in excited state (Hz)
    x_g: float = 65e9 #Jahn teller coupling
    y_g: float = 0  # Jahn-Teller coupling (ground state) (Hz)
    x_u: float = 855e9 # Jahn-Teller coupling
    y_u: float = 0  # Jahn-Teller coupling (excited state) (Hz)
    fg: float = 0.15 # quenching factor
    fu: float = 0.15 # quenching factor


class QuantumHamiltonian:
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
        self.L = self.constants.muB / self.constants.hbar  # Orbital gyromagnetic ratio
        self.S = 2 * self.constants.muB / self.constants.hbar  # Spin gyromagnetic ratio
        self._setup_logging()

    def _setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

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
    def strain_hamiltonian(a: float, b: float, d: float) -> np.ndarray:
        """
        Calculate the strain Hamiltonian matrix.

        Args:
            a (float): Alpha strain
            b (float): Beta strain
            d (float): Gamma strain

        Returns:
            np.ndarray: 4x4 strain Hamiltonian matrix
        """
        return np.array([[a - d, 0, b, 0],
                         [0, a - d, 0, b],
                         [b, 0, -a - d, 0],
                         [0, b, 0, -a - d]])

    def get_energy_spectra(self,
                           B: np.ndarray,
                           strain: bool = False,
                           strain_params: Optional[Tuple[float, float, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the energy eigenvalues and eigenvectors of the complete Hamiltonian.

        Args:
            l (float): Spin-orbit coupling constant
            x (float): Jahn-Teller x coupling strength
            y (float): Jahn-Teller y coupling strength
            f (float): Orbital quenching factor
            B (np.ndarray): Magnetic field vector [Bx, By, Bz]
            strain (bool, optional): Include strain effects. Defaults to False.
            strain_params (Tuple[float, float, float], optional):
                Strain parameters (a, b, d). Defaults to (-238e9, 238e9, 0).

        Returns:
            Eigenvalues and eigenvectors of the ground and excited Hamiltonian
        """
        # Default strain parameters
        if strain and strain_params is None:
            strain_params = (-238e9, 238e9, 0)

        # Construct full Hamiltonian
        Hg = (self.spin_orbit_hamiltonian(self.constants.lg) +
             self.jahn_teller_hamiltonian(self.constants.x_g, self.constants.y_g) +
             self.zeeman_hamiltonian(self.constants.fg, B[2]) +
             self.spin_zeeman_hamiltonian(B))

        He = (self.spin_orbit_hamiltonian(self.constants.lu) +
             self.jahn_teller_hamiltonian(self.constants.x_u, self.constants.y_u) +
             self.zeeman_hamiltonian(self.constants.fu, B[2]) +
             self.spin_zeeman_hamiltonian(B))

        if strain:
            Hg += self.strain_hamiltonian(*strain_params)
            He += self.strain_hamiltonian(*strain_params)


        # Find eigenvalues and eigenvectors
        eigenvalues_ground, eigenvectors_ground = eigh(Hg)
        eigenvalues_excited, eigenvectors_excited = eigh(He)
        return eigenvalues_ground, eigenvectors_ground, eigenvalues_excited, eigenvectors_excited




