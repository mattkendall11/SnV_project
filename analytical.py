import math

def calculate_alpha(Tx, gamma_s, B, lambda_):
    return math.sqrt((Tx + gamma_s * B)**2 + lambda_**2)

def calculate_beta(Tx, gamma_s, B, lambda_):
    return math.sqrt((Tx - gamma_s * B)**2 + lambda_**2)

def calculate_k(alpha, Tx, gamma_s, B, lambda_):
    return (alpha - Tx - gamma_s * B) / lambda_

def calculate_m(beta, gamma_s, B, Tx, lambda_):
    return (beta + gamma_s * B - Tx) / lambda_

def calculate_k_prime(alpha, Tx, gamma_s, B, lambda_):
    return (alpha + Tx + gamma_s * B) / lambda_

def calculate_m_prime(beta, gamma_s, B, Tx, lambda_):
    return (beta - gamma_s * B + Tx) / lambda_

Tx, gamma_s, B, lambda_ = 65e9, 2*9.2740100783e-24/1.054571817e-34,1,815e9

alphag = calculate_alpha(Tx, gamma_s, B, lambda_)
betag = calculate_beta(Tx, gamma_s, B, lambda_)

kg = calculate_k(alphag, Tx, gamma_s, B, lambda_)
mg = calculate_m(betag, gamma_s, B, Tx, lambda_)
k_primeg = calculate_k_prime(alphag, Tx, gamma_s, B, lambda_)
m_primeg = calculate_m_prime(betag, gamma_s, B, Tx, lambda_)


Tx, gamma_s, B, lambda_ = 855e9, 2*9.2740100783e-24/1.054571817e-34,1,2355e9

alphae = calculate_alpha(Tx, gamma_s, B, lambda_)
betae = calculate_beta(Tx, gamma_s, B, lambda_)

ke = calculate_k(alphag, Tx, gamma_s, B, lambda_)
me = calculate_m(betag, gamma_s, B, Tx, lambda_)
k_primee = calculate_k_prime(alphag, Tx, gamma_s, B, lambda_)
m_primee = calculate_m_prime(betag, gamma_s, B, Tx, lambda_)

