import numpy as np
import matplotlib.pyplot as plt
from src.model import PhysicalConstants as pc

dg = 0.787e15
de = 0.956e15
chi = 54.7 * np.pi / 180
gs = 2*pc.muB/pc.hbar

def alpha_g(B, strain):
    t1 = ((pc.x_g+strain*dg)+ 2*pc.muB*B/pc.hbar )**2

    return np.sqrt(t1+pc.lg)

def alpha_e(B, strain):
    t2 = ((pc.x_u+strain*de)+ 2*pc.muB*B/pc.hbar )**2

    return np.sqrt(t2+pc.lu)

def beta_g(B, strain):
    t3 = ((pc.x_g+strain*dg)- 2*pc.muB*B/pc.hbar )**2

    return np.sqrt(t3+pc.lg)

def beta_e(B, strain):
    t4 = ((pc.x_u + strain * de)- 2 * pc.muB * B/pc.hbar )**2

    return np.sqrt(t4 + pc.lu)

def approx_B(strain):
    m = (pc.lu*dg - pc.lg*de)/((2*pc.muB/pc.hbar)*(pc.lu + pc.lg))
    c = (pc.lu*pc.x_g - pc.lg*pc.x_u)/((2*pc.muB/pc.hbar)*(pc.lu+pc.lg))
    field = m*strain + c
    return field

def Ag(B, strain):
    numer = alpha_g(B, strain*dg)- pc.x_g - gs*B
    return numer/pc.lg


def Ae(B, strain):
    numer = alpha_e(B, strain*de)- pc.x_u - gs*B
    return numer/pc.lu

def Bg(B, strain):
    numer = beta_g(B, strain*dg)- pc.x_g + gs*B
    return numer/pc.lg

def Be(B, strain):
    numer = beta_e(B, strain*de)- pc.x_u + gs*B
    return numer/pc.lu

def fit_ns(phi):
    s = (np.sin(2*phi))**2
    denom = gs * ( s + 4 * (np.sin(2*chi))**2) * (pc.lg**2 - pc.lu**2)
    num = 2 * pc.lu * pc.lg * pc.lg - pc.lg * pc.lu * pc.x_g - pc.lg * pc.lg * pc.x_u
    num2 = 4 * (np.sin(2*chi))**2 * (pc.lu*pc.lg*pc.x_g - pc.lg*pc.lg*pc.x_u)
    y =  s*num/denom+ num2/denom+ (pc.lu**2+pc.lg**2)/(gs*(s+np.sin(2*chi))**2)
    y[np.abs(y) > 3] = np.nan
    return  y

def fita2b1(phi):
    c = np.cos(phi)
    s = np.sin(chi)
    t1 = ((c+2*s)*pc.lg*pc.x_u+pc.lu*(-2*c*pc.lg+(c-2*s)*pc.x_g))
    print(t1)
    t2 = ((c-2*s)*pc.lg*pc.x_u + pc.lu*(-2*c*pc.lg + (c+2*s)*pc.x_g))
    print(t2)
    d = gs*gs*((c**2-4*s**2)*pc.lu**2 - (5+np.cos(2*phi)- 4*np.cos(2*chi))*pc.lu*pc.lg + ((c**2-4*s**2)*pc.lu**2))
    print(d)
    return np.sqrt(-t1+t2)/np.sqrt(-d)

# x = np.linspace(0.0001,2*np.pi-0.0001,100)
#
# y = fita2b1(x)
#
# plt.plot(x, y)
# plt.show()
bval = (pc.lg*pc.x_u - pc.lu*pc.x_g)/(gs*(pc.lu-pc.lg))
def bval2():
    numerator = 2*pc.lg*pc.x_u - pc.lu*pc.lg*pc.x_u*pc.x_u - 2*pc.lu*pc.x_g + pc.lg*pc.lu*pc.x_g*pc.x_g
    denominator = 2*gs*(pc.lu-pc.lg + pc.lg*pc.lu*pc.x_u - pc.lg*pc.lu*pc.x_g)
    return numerator/denominator


def compute_expression():
    gamma = gs
    lambda_g = pc.lg
    Gamma_xg = pc.x_g
    lambda_e = pc.lu
    Gamma_xe = pc.x_u

    numerator = np.sqrt(
        4 * gamma ** 2 * (
                (Gamma_xe / lambda_e) - 0.5 * Gamma_xe ** 2 - (Gamma_xg / lambda_g) - 0.5 * Gamma_xg ** 2
        ) + (
                gamma * (-Gamma_xe) - gamma * Gamma_xg + (gamma / lambda_e) + (gamma / lambda_g)
        ) ** 2
    ) + gamma * (-Gamma_xe) - gamma * Gamma_xg + (gamma / lambda_e) + (gamma / lambda_g)

    denominator = 2 * gamma ** 2

    return numerator / denominator


def compute_expression2():
    gamma = gs
    lambda_g = pc.lg
    Gamma_xg = pc.x_g
    lambda_e = pc.lu
    Gamma_xe = pc.x_u

    numerator = -np.sqrt(
        4 * gamma ** 2 * (
                (Gamma_xe / lambda_e) - 0.5 * Gamma_xe ** 2 - (Gamma_xg / lambda_g) - 0.5 * Gamma_xg ** 2
        ) + (
                gamma * (-Gamma_xe) - gamma * Gamma_xg + (gamma / lambda_e) + (gamma / lambda_g)
        ) ** 2
    ) + gamma * Gamma_xe + gamma * Gamma_xg - (gamma / lambda_e) - (gamma / lambda_g)

    denominator = 2 * gamma ** 2

    return numerator / denominator
