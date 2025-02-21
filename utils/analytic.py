import numpy as np

from src.model import PhysicalConstants as pc

dg = 0.787e15
de = 0.956e15

def alpha_g(B, strain):
    t1 = ((pc.x_g+strain*dg)+ 2*pc.muB*B/pc.hbar )**2
    print(t1, pc.lg**2)
    return np.sqrt(t1+pc.lg)

def alpha_e(B, strain):
    t2 = ((pc.x_u+strain*de)+ 2*pc.muB*B/pc.hbar )**2
    print(t2, pc.lu ** 2)
    return np.sqrt(t2+pc.lu)

def beta_g(B, strain):
    t3 = ((pc.x_g+strain*dg)- 2*pc.muB*B/pc.hbar )**2
    print(t3, pc.lg ** 2)
    return np.sqrt(t3+pc.lg)

def beta_e(B, strain):
    t4 = ((pc.x_u + strain * de)- 2 * pc.muB * B/pc.hbar )**2
    print(t4, pc.lu ** 2)
    return np.sqrt(t4 + pc.lu)

def approx_B(strain):
    m = (pc.lu*dg - pc.lg*de)/((2*pc.muB/pc.hbar)*(pc.lu + pc.lg))
    c = (pc.lu*pc.x_g - pc.lg*pc.x_u)/((2*pc.muB/pc.hbar)*(pc.lu+pc.lg))
    field = m*strain + c
    return field