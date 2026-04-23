"""Global residual and sparse tangent assembly."""
import numpy as np
import scipy.sparse as sp

from .element import hex8_element, N_DIM, N_DOF_ELEM

_NBLOCK = N_DOF_ELEM ** 2          # triplet entries per element


def assemble(u, mesh, mat, f_ext):
    """assemble the global residual R = F_int - F_ext and sparse tangent K via COO triplets."""
    n_elem = len(mesh.elements)
    ndof = mesh.ndof

    rows = np.empty(n_elem * _NBLOCK, dtype=np.int32)
    cols = np.empty(n_elem * _NBLOCK, dtype=np.int32)
    vals = np.empty(n_elem * _NBLOCK)

    Rint = np.zeros(ndof)
    elem_sigma = np.zeros((n_elem, N_DIM, N_DIM))

    for e in range(n_elem):
        dofs = mesh.dof_map[e]
        Ke, fint, sig = hex8_element(u[dofs], mesh.dNdX[e], mesh.detJ0[e], mat)
        elem_sigma[e] = sig
        Rint[dofs] += fint

        s = slice(e * _NBLOCK, (e + 1) * _NBLOCK)
        rows[s] = np.repeat(dofs, N_DOF_ELEM)
        cols[s] = np.tile(dofs, N_DOF_ELEM)
        vals[s] = Ke.ravel()

    K = sp.coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()
    return Rint - f_ext, K, elem_sigma
