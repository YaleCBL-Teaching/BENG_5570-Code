"""Hex8 element: shape functions, 2x2x2 Gauss quadrature, and the spatial
(updated-Lagrangian) residual and tangent for a St. Venant-Kirchhoff material.

Index conventions used in this file:
    a, b        element node index                   (0..N_NODE-1)
    i, j, k, l  spatial indices in current config    (0..N_DIM-1)
    I, J, K, L  material indices in reference config (0..N_DIM-1)
Uppercase = reference, lowercase = spatial. Tensor contractions use einsum.
"""

import numpy as np

# number of spatial dimensions
N_DIM = 3

# number of nodes per element
N_NODE = 8

# number of displacement DOFs per element
N_DOF_ELEM = N_NODE * N_DIM

# HEX8 TOPOLOGY
# corner signs of the 8 nodes in (xi, eta, zeta), VTK hex ordering
_SIGNS = np.array(
    [
        [-1, -1, -1],
        [+1, -1, -1],
        [+1, +1, -1],
        [-1, +1, -1],
        [-1, -1, +1],
        [+1, -1, +1],
        [+1, +1, +1],
        [-1, +1, +1],
    ],
    dtype=float,
)
# 12 edges of a hex8 (node-pair indices); used only for wireframe plotting
HEX_EDGES = np.array(
    [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
)

# QUADRATURE: 2x2x2 TENSOR-PRODUCT GAUSS ON [-1, 1]^3
_GP_1D = np.array([-1.0, 1.0]) / np.sqrt(3.0)

# (N_GP, N_DIM)
GP = np.array([(a, b, c) for a in _GP_1D for b in _GP_1D for c in _GP_1D])

# 2x2x2 = 8 Gauss points
N_GP = len(GP)

# unit weights for this rule
GW = np.ones(N_GP)


# shape-function natural-coord derivatives
def dN_dxi(xi, eta, zeta):
    """shape-function natural-coord derivatives dN_a / dxi_j at one point; returns (N_NODE, N_DIM)."""
    # N_a        = (1/8) prod_k (1 + s_ak * xi_k)
    # dN_a/dxi_j = (s_aj / 8) prod_{k != j} (1 + s_ak * xi_k)
    f = 1.0 + _SIGNS * np.array([xi, eta, zeta])  # (N_NODE, N_DIM)
    return (
        np.stack(
            [
                _SIGNS[:, 0] * f[:, 1] * f[:, 2],
                _SIGNS[:, 1] * f[:, 0] * f[:, 2],
                _SIGNS[:, 2] * f[:, 0] * f[:, 1],
            ],
            axis=1,
        )
        / 8.0
    )


# per-element residual and tangent
def hex8_element(u_elem, dNdX_e, detJ0_e, mat):
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
