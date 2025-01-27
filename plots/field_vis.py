import numpy as np
import matplotlib.pyplot as plt

# Functions to compute alpha, beta, k, and m
def compute_alpha(Gamma_x, Gamma_S, B, lambd):
    return np.sqrt((Gamma_x + Gamma_S * B)**2 + lambd**2)

def compute_beta(Gamma_x, Gamma_S, B, lambd):
    return np.sqrt((Gamma_x - Gamma_S * B)**2 + lambd**2)

def compute_k(alpha, Gamma_x, Gamma_S, B, lambd):
    return (alpha - Gamma_x - Gamma_S * B) / lambd

def compute_m(beta, Gamma_S, B, Gamma_x, lambd):
    return (beta + Gamma_S * B - Gamma_x) / lambd

def compute_k_prime(alpha, Gamma_x, Gamma_S, B, lambd):
    return (alpha + Gamma_x + Gamma_S * B) / lambd

def compute_m_prime(beta, Gamma_S, B, Gamma_x, lambd):
    return (beta - Gamma_S * B + Gamma_x) / lambd

# Define physical parameters
muB = 9.2740100783e-24  # Bohr magneton in J/T
hbar = 1.054571817e-34  # Reduced Planck constant
lg = 815e9  # Spin-orbit coupling in ground state (Hz)
lu = 2355e9  # Spin-orbit coupling in excited state (Hz)
xg = 65e9  # Ground Jahn-Teller coupling (Hz)
xu = 855e9  # Excited Jahn-Teller coupling (Hz)
S = 2 * muB / hbar  # Spin factor
B = 1  # Magnetic field (T)

# Compute alpha and beta for ground and excited states
alpha_g = compute_alpha(xg, S, B, lg)
alpha_e = compute_alpha(xu, S, B, lu)
beta_g = compute_beta(xg, S, B, lg)
beta_e = compute_beta(xu, S, B, lu)

# Compute k, k', m, m' for ground and excited states
kg, kdg, mg, mdg = (compute_k(alpha_g, xg, S, B, lg), compute_k_prime(alpha_g, xg, S, B, lg),
                    compute_m(beta_g, xg, S, B, lg), compute_m_prime(beta_g, xg, S, B, lg))
ke, kde, me, mde = (compute_k(alpha_e, xu, S, B, lu), compute_k_prime(alpha_e, xu, S, B, lu),
                    compute_m(beta_e, xu, S, B, lu), compute_m_prime(beta_e, xu, S, B, lu))



def plot_elipticity(Ex, Ey):
    # Compute Stokes parameters
    S0 = np.abs(Ex)**2 + np.abs(Ey)**2  # Total intensity
    S1 = np.abs(Ex)**2 - np.abs(Ey)**2  # Difference in intensities
    S2 = 2 * np.real(Ex * np.conj(Ey))
    S3 = -2 * np.imag(Ex * np.conj(Ey))  # Circular polarizationli

    # Ellipticity angle (chi) and orientation angle (psi)
    chi = 0.5 * np.arcsin(S3 / S0)  # Ellipticity angle
    psi = 0.5 * np.arctan(S2/ S1)  # Orientation angle


    # Generate the polarization ellipse
    theta = np.linspace(0, 2 * np.pi, 500)  # Parameter for the ellipse
    a = np.sqrt(S0)  # Semi-major axis
    b = a * np.sin(chi)  # Semi-minor axis
    x_ellipse = a * np.cos(theta) * np.cos(psi) - b * np.sin(theta) * np.sin(psi)
    y_ellipse = a * np.cos(theta) * np.sin(psi) + b * np.sin(theta) * np.cos(psi)

    # Plot the polarization ellipse
    plt.figure(figsize=(8, 8))
    plt.plot(x_ellipse, y_ellipse, label="Polarization Ellipse", color="blue")
    plt.xlabel("Ex (Electric Field X-component)")
    plt.ylabel("Ey (Electric Field Y-component)")
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
    plt.title(fr"Polarization Ellipse with Orientation (ψ):{psi:.2f} and Ellipticity (χ):{chi:.2f}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.axis('equal')
    plt.savefig('polarisation.svg')
    plt.show()

def plotA1B2():
    Ex = 2j * (mg - ke)  # x-component (complex)
    Ey = 2 * (ke + mg)   # y-component (complex)
    plot_elipticity(Ex, Ey)
    Ex = 2j * (kg - me)  # x-component (complex)
    Ey = 2 * (kg + me)  # y-component (complex)
    plot_elipticity(Ex, Ey)
    ex = 2*ke+2*mg
    ey = 2j*(-ke+mg)
    plot_elipticity(ex, ey)

plotA1B2()