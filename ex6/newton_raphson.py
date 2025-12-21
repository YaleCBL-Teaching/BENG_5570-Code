#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 21 09:37:03 2025

@author: evanwillmarth
"""

import numpy as np
import math
import matplotlib.pyplot as plt


def compute_geometry(u, w, a, h, L):
    l1 = math.sqrt((a + u)**2 + (h - w)**2)
    l2 = math.sqrt((a - u)**2 + (h - w)**2)
    return l1, l2

def compute_strains(l1, l2, L):
    egl1 = (l1**2 - L**2) / (2 * L**2)
    egl2 = (l2**2 - L**2) / (2 * L**2)
    return egl1, egl2

def compute_N_forces(l1, l2, egl1, egl2, EA, L):
    N1 = (l1 / L) * EA * egl1
    N2 = (l2 / L) * EA * egl2
    return N1, N2

def compute_Kt(u, w, N1, N2, EA, L, a, h):
    K11 = N1 / l1 + (EA / L**3) * (a + u)**2 + N2 / l2 + (EA / L**3) * (a - u)**2
    K12 = - (EA / L**3) * (a + u) * (h - w) + (EA / L**3) * (a - u) * (h - w)
    K22 = N1 / l1 + (EA / L**3) * (h - w)**2 + N2 / l2 + (EA / L**3) * (h - w)**2
    return np.array([[K11, K12], [K12, K22]])

def compute_residual(u, w, N1, N2, P, lam, a, h):
    R1 = N1 * ((a + u) / l1) - N2 * ((a - u) / l2)
    R2 = -N1 * ((h - w) / l1) - N2 * ((h - w) / l2) - lam * P
    return np.array([R1, R2])


# Constant Parameters
a = 1.0
h = 3 * a
EA = 1.0
P = 1.0
psi = 1.0
tol = 1e-8
tol_cr = 1e-2
L = math.sqrt(a**2 + h**2)
s_hat = 0.001

D = np.array([0.0, 0.0])  # [u, w]
lam = 0.0

lambda_total = []
w_total = []
D_critical = []

F = np.array([0.0, 1.0])

for step in range(10000):

    if D[1] > 7.0:
        break

    lambda_total.append(lam)
    w_total.append(D[1])

    # Predictor Step
    uk, wk = D.copy()      #  SAVE previous D
    lam_k = lam            #  SAVE previous lambda

    u, w = uk, wk
    l1, l2 = compute_geometry(u, w, a, h, L)
    u, w = D
    l1, l2 = compute_geometry(u, w, a, h, L)
    egl1, egl2 = compute_strains(l1, l2, L)
    N1, N2 = compute_N_forces(l1, l2, egl1, egl2, EA, L)

    Kt = compute_Kt(u, w, N1, N2, EA, L, a, h)
    det_K_prev = np.linalg.det(Kt)

    delta_D_star = np.linalg.solve(Kt, F)
    delta_lambda = s_hat / math.sqrt(delta_D_star.dot(delta_D_star) + 2 * psi)

    kappa = float(F.T.dot(delta_D_star) / (delta_D_star.T.dot(delta_D_star)))
    if kappa > 0:
        lam += delta_lambda
        D += delta_lambda * delta_D_star
    else:
        lam -= delta_lambda
        D -= delta_lambda * delta_D_star

    # Newton–Raphson Correction
    while True:
        u, w = D
        l1, l2 = compute_geometry(u, w, a, h, L)
        egl1, egl2 = compute_strains(l1, l2, L)
        N1, N2 = compute_N_forces(l1, l2, egl1, egl2, EA, L)

        R = compute_residual(u, w, N1, N2, P, lam, a, h)
        if np.linalg.norm(R) < tol:
            break

        Kt = compute_Kt(u, w, N1, N2, EA, L, a, h)

        delta_D_R = -np.linalg.solve(Kt, R)
        delta_D_F = np.linalg.solve(Kt, F)

        s_val = math.sqrt((u - uk)**2 + (w - wk)**2 + psi**2 * (lam - lam_k)**2)
        f_vec = np.array([(u - uk) / s_val, (w - wk) / s_val])
        f_lambda = (psi**2) * (lam - lam_k) / s_val
        f_i = s_val - s_hat

        delta_lambda = -(f_i + f_vec.dot(delta_D_R)) / (f_lambda + f_vec.dot(delta_D_F))
        delta_D = delta_D_R + delta_lambda * delta_D_F

        D += delta_D
        lam += delta_lambda

    det_K_new = np.linalg.det(Kt)

    if np.sign(det_K_new) != np.sign(det_K_prev):
        if det_K_new < tol_cr:
            D_critical.append(D.copy())
        else:
            s_hat *= 0.5

# ====================
# Plot Result
# ====================

plt.plot(w_total, lambda_total)
plt.xlabel('w')
plt.ylabel(r'$\lambda$')
plt.show()





