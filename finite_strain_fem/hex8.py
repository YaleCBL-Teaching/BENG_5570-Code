"""Hex8 element definition: topology, shape functions, and 2x2x2 Gauss quadrature.

Everything in this file is specific to the hex8 element and would be swapped
out to use a different element type (tet4, hex20, ...). See `physics.py` for
the finite-strain routine, which consumes only the arrays produced here.
"""
import numpy as np

# number of nodes per element
N_NODE = 8

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
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ]
)

# QUADRATURE: 2x2x2 TENSOR-PRODUCT GAUSS ON [-1, 1]^3
_GP_1D = np.array([-1.0, 1.0]) / np.sqrt(3.0)
GP = np.array([(a, b, c) for a in _GP_1D for b in _GP_1D for c in _GP_1D])  # (N_GP, 3)
N_GP = len(GP)                                                              # 2x2x2 = 8
GW = np.ones(N_GP)                                                          # unit weights


# shape-function natural-coord derivatives
def dN_dxi(xi, eta, zeta):
    """shape-function natural-coord derivatives dN_a / dxi_j at one point; returns (N_NODE, 3)."""
    # N_a        = (1/8) prod_k (1 + s_ak * xi_k)
    # dN_a/dxi_j = (s_aj / 8) prod_{k != j} (1 + s_ak * xi_k)
    f = 1.0 + _SIGNS * np.array([xi, eta, zeta])
    return np.stack(
        [
            _SIGNS[:, 0] * f[:, 1] * f[:, 2],
            _SIGNS[:, 1] * f[:, 0] * f[:, 2],
            _SIGNS[:, 2] * f[:, 0] * f[:, 1],
        ],
        axis=1,
    ) / 8.0
