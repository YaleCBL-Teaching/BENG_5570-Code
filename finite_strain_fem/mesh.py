"""Mesh dataclass and a structured brick-mesh generator.

Precomputes the reference-configuration quantities that stay constant
throughout the Newton iterations (dof_map, dNdX, detJ0), so the Newton
loop only does per-element kinematics / stress / tangent.
"""
from dataclasses import dataclass, field
import numpy as np

from .hex8 import dN_dxi, GP
from .physics import N_DIM, N_DOF_ELEM


# Shape-function natural-coord derivatives evaluated at every Gauss point,
# used only here to build each element's reference-config shape gradients.
_DN_DXI_GP = np.stack([dN_dxi(*g) for g in GP])       # (N_GP, N_NODE, N_DIM)


@dataclass
class Mesh:
    """finite-element mesh with cached reference-config quantities (dof_map, dNdX, detJ0)."""
    nodes: np.ndarray                          # (n_nodes, N_DIM)
    elements: np.ndarray                       # (n_elem, N_NODE)  VTK hex node ordering
    dof_map: np.ndarray = field(init=False)    # (n_elem, N_DOF_ELEM) global DOF indices
    dNdX: np.ndarray = field(init=False)       # (n_elem, N_GP, N_NODE, N_DIM)  grad_X N_a
    detJ0: np.ndarray = field(init=False)      # (n_elem, N_GP)  det(J0)

    def __post_init__(self):
        n_elem = len(self.elements)

        # Scatter map from element-local DOFs to global DOFs
        self.dof_map = (N_DIM * self.elements[:, :, None]
                        + np.arange(N_DIM)).reshape(n_elem, N_DOF_ELEM)

        # Reference Jacobian: J0_IJ = X_aI dN_a/dxi_J  (batched over elements and Gauss points)
        Xe = self.nodes[self.elements]                                 # (n_elem, N_NODE, N_DIM)
        J0 = np.einsum('eai,gaj->egij', Xe, _DN_DXI_GP)                # (n_elem, N_GP, N_DIM, N_DIM)

        self.detJ0 = np.linalg.det(J0)
        if np.any(self.detJ0 <= 0):
            raise ValueError("Non-positive reference Jacobian in mesh.")

        # Reference shape gradients: dNdX_aI = dN_a/dxi_J * (J0^-1)_JI
        self.dNdX = np.einsum('gaj,egji->egai', _DN_DXI_GP, np.linalg.inv(J0))

    @property
    def ndof(self):
        return N_DIM * len(self.nodes)


def apply_bc(mesh, clamped_nodes, loaded_nodes, load_dir, load_total):
    """clamp all DOFs of clamped_nodes and distribute load_total over loaded_nodes along load_dir; returns (fixed_dofs, f_ext)."""
    fixed = (N_DIM * clamped_nodes[:, None] + np.arange(N_DIM)).ravel()
    f_ext = np.zeros(mesh.ndof)
    f_ext[N_DIM * loaded_nodes + load_dir] = load_total / len(loaded_nodes)
    return fixed, f_ext


def create_mesh(L, H, W, nx, ny, nz):
    """structured brick mesh over [0,L] x [0,H] x [0,W] with nx*ny*nz hex8 elements."""
    nodes = np.array([[L*i/nx, H*j/ny, W*k/nz]
                      for k in range(nz + 1)
                      for j in range(ny + 1)
                      for i in range(nx + 1)], dtype=float)

    # Node id for the lattice point (i, j, k)
    def nid(i, j, k):
        return i + (nx + 1) * (j + (ny + 1) * k)

    # Each element: bottom quad CCW, then top quad CCW (VTK hex ordering)
    elements = np.array(
        [[nid(i,   j,   k),   nid(i+1, j,   k),   nid(i+1, j+1, k),   nid(i,   j+1, k),
          nid(i,   j,   k+1), nid(i+1, j,   k+1), nid(i+1, j+1, k+1), nid(i,   j+1, k+1)]
         for k in range(nz) for j in range(ny) for i in range(nx)],
        dtype=int)

    return Mesh(nodes=nodes, elements=elements)
