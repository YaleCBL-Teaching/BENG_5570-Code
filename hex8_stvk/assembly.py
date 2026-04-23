"""Global residual and sparse tangent assembly."""
import numpy as np
import scipy.sparse as sp

from .element import hex8_element


def assemble(u, mesh, mat, f_ext):
    """
    Assemble the global residual R = F_int - F_ext and sparse tangent K.

    Each element contributes a 24x24 block; we collect (row, col, val) triplets
    and build a COO matrix. Converting to CSR automatically sums entries with
    duplicate (row, col), which is exactly the FEM scatter operation.
    """
    n_elem = len(mesh.elements)
    ndof = mesh.ndof

    # Triplet buffers: 24*24 = 576 entries per element
    rows = np.empty(n_elem * 576, dtype=np.int32)
    cols = np.empty(n_elem * 576, dtype=np.int32)
    vals = np.empty(n_elem * 576)

    Rint = np.zeros(ndof)
    cell_sigma = np.zeros((n_elem, 3, 3))

    for e in range(n_elem):
        dofs = mesh.dof_map[e]
        Ke, fint, sig = hex8_element(u[dofs], mesh.dNdX[e], mesh.detJ0[e], mat)
        cell_sigma[e] = sig
        Rint[dofs] += fint

        s = slice(e * 576, (e + 1) * 576)
        rows[s] = np.repeat(dofs, 24)
        cols[s] = np.tile(dofs, 24)
        vals[s] = Ke.ravel()

    K = sp.coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()
    return Rint - f_ext, K, cell_sigma
