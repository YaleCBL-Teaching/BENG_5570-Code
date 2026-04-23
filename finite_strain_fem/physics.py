"""Finite-strain solid-mechanics physics: per-element residual and tangent.

This module implements the spatial (updated-Lagrangian) form of the
internal force and consistent tangent for a hyperelastic material. The
formulas depend only on the spatial dimension and the per-Gauss-point
gradient arrays produced by the mesh -- the element type enters solely
through the sizes `N_NODE`, `N_GP`, and the weights `GW` (imported from
`hex8.py`).

Index conventions:
    a, b        element node index                   (0..N_NODE-1)
    i, j, k, l  spatial indices in current config    (0..N_DIM-1)
    I, J, K, L  material indices in reference config (0..N_DIM-1)
Uppercase = reference, lowercase = spatial. Tensor contractions use einsum.
"""
import numpy as np

from .hex8 import N_NODE, N_GP, GW

# spatial dimension of the solid-mechanics problem
N_DIM = 3

# displacement DOFs per element
N_DOF_ELEM = N_NODE * N_DIM


def finite_strain_element(u_elem, dNdX_e, detJ0_e, mat):
    """element tangent K_e, internal force f_int, and averaged Cauchy stress (spatial / updated-Lagrangian form)."""
    I = np.eye(N_DIM)
    K_e = np.zeros((N_NODE, N_DIM, N_NODE, N_DIM))
    f_int = np.zeros((N_NODE, N_DIM))
    sigma_avg = np.zeros((N_DIM, N_DIM))
    vol = 0.0
    u_nodes = u_elem.reshape(N_NODE, N_DIM)

    for gp in range(N_GP):
        # material shape function gradients (N_NODE, N_DIM)
        dNdX = dNdX_e[gp]

        # deformation gradient from current nodal displacements
        # F_iI = delta_iI + u_aI * dNdX_aI
        F = I + u_nodes.T @ dNdX

        # Jacobian
        J = np.linalg.det(F)
        if J <= 0:
            raise ValueError(f"Non-positive det F: {J}")

        # spatial shape function gradients
        # dNdx_ai = dNdX_aI * (F^-1)_Ii
        dNdx = dNdX @ np.linalg.inv(F)

        # current-config volume element with Gauss point weight
        dv = J * detJ0_e[gp] * GW[gp]

        # Green-Lagrange strain
        # E_IJ = 1/2 (F_kI F_kJ - delta_IJ)
        E = 0.5 * (F.T @ F - I)

        # constitutive response (material configuration): 2PK stress S, tangent dS/dE
        S, C_mat = mat.stress_tangent(E)

        # Cauchy stress: push-forward 2PK to spatial configuration
        # sigma_ij = (1/J) F_iI S_IJ F_jJ
        sigma = (F @ S @ F.T) / J

        # spatial elasticity
        # c_ijkl = (1/J) F_iI F_jJ F_kK F_lL C_IJKL
        c_mat = np.einsum("iI,jJ,kK,lL,IJKL->ijkl", F, F, F, F, C_mat) / J

        # accumulate internal nodal force
        # f_ai = sigma_ij * dNdx_aj
        f_int += np.einsum("ij,aj->ai", sigma, dNdx) * dv

        # material stiffness
        # K^mat_aibj = dNdx_ak * c_ikjl * dNdx_bl
        K_mat = np.einsum("ak,ikjl,bl->aibj", dNdx, c_mat, dNdx)

        # geometric stiffness
        # K^geo_aibj = (dNdx_ak sigma_kl dNdx_bl) * delta_ij
        G_ab = np.einsum("ak,kl,bl->ab", dNdx, sigma, dNdx)
        K_geo = np.einsum("ab,ij->aibj", G_ab, I)

        # accumulate stiffness
        K_e += (K_mat + K_geo) * dv
        sigma_avg += sigma * dv
        vol += dv

    sigma_avg /= vol
    return K_e.reshape(N_DOF_ELEM, N_DOF_ELEM), f_int.ravel(), sigma_avg
