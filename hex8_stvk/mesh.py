"""Mesh dataclass and a structured brick-mesh generator."""
from dataclasses import dataclass, field
import numpy as np

from .element import DN_DXI


@dataclass
class Mesh:
    """
    Finite-element mesh plus reference-configuration quantities that are
    constant throughout the Newton iterations (dof_map, dNdX, detJ0).
    Caching these avoids recomputing them on every residual/tangent assembly.
    """
    nodes: np.ndarray            # (n_nodes, 3)
    elements: np.ndarray         # (n_elem, 8)   VTK hex node ordering
    dof_map: np.ndarray = field(init=False)   # (n_elem, 24) global DOF indices
    dNdX: np.ndarray = field(init=False)      # (n_elem, n_gp, 8, 3)  grad_X N_a
    detJ0: np.ndarray = field(init=False)     # (n_elem, n_gp)        det(J0)

    def __post_init__(self):
        n_elem = len(self.elements)

        # Scatter map from element-local DOF (24,) to global DOFs
        self.dof_map = (3 * self.elements[:, :, None] + np.arange(3)).reshape(n_elem, 24)

        # Reference Jacobian: J0_IJ = X_aI dN_a/dxi_J  (batched over elements and Gauss points)
        Xe = self.nodes[self.elements]                                 # (n_elem, 8, 3)
        J0 = np.einsum('eai,gaj->egij', Xe, DN_DXI)                    # (n_elem, n_gp, 3, 3)

        self.detJ0 = np.linalg.det(J0)
        if np.any(self.detJ0 <= 0):
            raise ValueError("Non-positive reference Jacobian in mesh.")

        # Reference shape gradients: dNdX_aI = dN_a/dxi_J * (J0^-1)_JI
        self.dNdX = np.einsum('gaj,egji->egai', DN_DXI, np.linalg.inv(J0))

    @property
    def ndof(self):
        return 3 * len(self.nodes)


def create_mesh(L, H, W, nx, ny, nz):
    """Structured brick mesh over [0,L] x [0,H] x [0,W] with nx*ny*nz hex8 elements."""
    nodes = np.array([[L*i/nx, H*j/ny, W*k/nz]
                      for k in range(nz + 1)
                      for j in range(ny + 1)
                      for i in range(nx + 1)], dtype=float)

    # Node id for the lattice point (i, j, k)
    def nid(i, j, k):
        return i + (nx + 1) * (j + (ny + 1) * k)

    # Each cell: bottom quad CCW, then top quad CCW (VTK hex ordering)
    elements = np.array(
        [[nid(i,   j,   k),   nid(i+1, j,   k),   nid(i+1, j+1, k),   nid(i,   j+1, k),
          nid(i,   j,   k+1), nid(i+1, j,   k+1), nid(i+1, j+1, k+1), nid(i,   j+1, k+1)]
         for k in range(nz) for j in range(ny) for i in range(nx)],
        dtype=int)

    return Mesh(nodes=nodes, elements=elements)
