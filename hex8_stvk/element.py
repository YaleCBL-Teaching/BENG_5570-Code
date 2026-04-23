"""Hex8 element: 2x2x2 Gauss rule, shape functions, and the spatial
(updated-Lagrangian) residual and tangent for a St. Venant-Kirchhoff material.

Index conventions used in this file:
    a, b        element node index         (0..7)
    i, j, k, l  spatial indices in current config   (0..2)
    I, J, K, L  material indices in reference config (0..2)
Tensor contractions are written in einsum form; uppercase = reference,
lowercase = spatial.
"""
import numpy as np


# ------------------------------------------------------------
# 2x2x2 GAUSS QUADRATURE AND SHAPE-FUNCTION DERIVATIVES
# ------------------------------------------------------------

# 8 Gauss points and weights (tensor product of 1-D 2-point rule on [-1, 1])
_GP_1D = np.array([-1.0, 1.0]) / np.sqrt(3.0)
GP = np.array([(a, b, c) for a in _GP_1D for b in _GP_1D for c in _GP_1D])  # (8, 3)
GW = np.ones(8)

# Corner signs of the 8 nodes in (xi, eta, zeta), VTK hex ordering
_S = np.array([[-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
               [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1]], dtype=float)

# 12 edges of a hex8 element (node-pair indices), used for wireframe plotting
HEX_EDGES = np.array([[0, 1], [1, 2], [2, 3], [3, 0],
                      [4, 5], [5, 6], [6, 7], [7, 4],
                      [0, 4], [1, 5], [2, 6], [3, 7]])


def dN_dxi(xi, eta, zeta):
    """Shape-function natural-derivatives dN_a / dxi_j at one point. Shape (8, 3)."""
    # N_a(xi, eta, zeta) = (1/8) prod_k (1 + s_ak * xi_k)
    # => dN_a/dxi_j = (s_aj / 8) * prod_{k != j} (1 + s_ak * xi_k)
    f = 1.0 + _S * np.array([xi, eta, zeta])                          # (8, 3)
    return np.stack([_S[:, 0] * f[:, 1] * f[:, 2],
                     _S[:, 1] * f[:, 0] * f[:, 2],
                     _S[:, 2] * f[:, 0] * f[:, 1]], axis=1) / 8.0


# dN/dxi at every Gauss point, precomputed once: shape (n_gp, 8_nodes, 3)
DN_DXI = np.stack([dN_dxi(*g) for g in GP])


# ------------------------------------------------------------
# ELEMENT ROUTINE
# ------------------------------------------------------------

def hex8_element(u_elem, dNdX_e, detJ0_e, mat):
    """
    Element residual and tangent in the spatial (updated-Lagrangian) form.

    u_elem  : (24,)             nodal displacements (flattened, node-major)
    dNdX_e  : (n_gp, 8, 3)      grad_X N_a at each Gauss point (reference config)
    detJ0_e : (n_gp,)           reference Jacobian determinants
    mat     : Material

    Returns Ke (24, 24), fint (24,), and the element-averaged Cauchy stress (3, 3).
    """
    I3 = np.eye(3)
    Ke = np.zeros((8, 3, 8, 3))
    fint = np.zeros((8, 3))
    sigma_avg = np.zeros((3, 3))
    vol = 0.0

    u_nodes = u_elem.reshape(8, 3)

    for g in range(len(GP)):
        dNdX = dNdX_e[g]                                               # (8, 3)

        # Deformation gradient:       F_iI = delta_iI + u_aI * dNdX_aI
        F = I3 + u_nodes.T @ dNdX
        J = np.linalg.det(F)
        if J <= 0:
            raise ValueError(f"Non-positive det F: {J}")

        # Spatial shape gradients:    dNdx_ai = dNdX_aI * F^-1_Ii
        dNdx = dNdX @ np.linalg.inv(F)

        # Current-config volume element
        dv = J * detJ0_e[g] * GW[g]

        # Green-Lagrange strain:      E_IJ = 1/2 (F_kI F_kJ - delta_IJ)
        Egl = 0.5 * (F.T @ F - I3)

        # Constitutive law: 2nd Piola-Kirchhoff stress S and tangent C = dS/dE
        S, Cmat = mat.stress_tangent(Egl)

        # Cauchy stress:              sigma_ij = (1/J) F_iI S_IJ F_jJ
        sigma = (F @ S @ F.T) / J

        sigma_avg += sigma * dv
        vol += dv

        # Spatial elasticity (push-forward of C):
        #                             c_ijkl = (1/J) F_iI F_jJ F_kK F_lL C_IJKL
        c = np.einsum('iI,jJ,kK,lL,IJKL->ijkl', F, F, F, F, Cmat) / J

        # Internal nodal force:       f_ai = sigma_ij * dNdx_aj
        fint += np.einsum('ij,aj->ai', sigma, dNdx) * dv

        # Material stiffness:         K^mat_aibj = dNdx_ak * c_ikjl * dNdx_bl
        Kmat = np.einsum('ak,ikjl,bl->aibj', dNdx, c, dNdx)

        # Geometric stiffness:        K^geo_aibj = (dNdx_ak sigma_kl dNdx_bl) * delta_ij
        G = np.einsum('ak,kl,bl->ab', dNdx, sigma, dNdx)
        Kgeo = np.einsum('ab,ij->aibj', G, I3)

        Ke += (Kmat + Kgeo) * dv

    sigma_avg /= vol
    return Ke.reshape(24, 24), fint.ravel(), sigma_avg
