#!/usr/bin/env python3
"""
Spatial hex8 finite element, St. Venant-Kirchhoff material.
Author: Evan Wilmarth
"""

import os
import time
import argparse
from dataclasses import dataclass, field
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import meshio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection


# Index conventions used in comments below:
#   a, b     : element node index  (0..7)
#   i, j, k, l : spatial indices in current config  (0..2)
#   I, J, K, L : material indices in reference config (0..2)
#   delta_IJ : Kronecker delta
# Tensor contractions are written in einsum form; uppercase indices are reference,
# lowercase indices are spatial.


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
# MATERIAL: St. Venant-Kirchhoff
# ------------------------------------------------------------

@dataclass
class Material:
    """Isotropic St. Venant-Kirchhoff material. Stores Lamé parameters and C_IJKL."""
    E: float
    nu: float
    lam: float = field(init=False)
    mu: float = field(init=False)
    Cmat: np.ndarray = field(init=False)

    def __post_init__(self):
        # Lamé parameters
        self.lam = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.mu = self.E / (2 * (1 + self.nu))

        # Material elasticity tensor: C_IJKL = lam delta_IJ delta_KL
        #                                    + mu (delta_IK delta_JL + delta_IL delta_JK)
        I = np.eye(3)
        self.Cmat = (self.lam * np.einsum('ij,kl->ijkl', I, I)
                     + self.mu * (np.einsum('ik,jl->ijkl', I, I)
                                  + np.einsum('il,jk->ijkl', I, I)))


# ------------------------------------------------------------
# MESH: nodes, connectivity, and cached reference-config quantities
# ------------------------------------------------------------

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


# element routine
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

        # 2nd Piola-Kirchhoff stress: S_IJ = lam tr(E) delta_IJ + 2 mu E_IJ
        S = mat.lam * np.trace(Egl) * I3 + 2.0 * mat.mu * Egl

        # Cauchy stress:              sigma_ij = (1/J) F_iI S_IJ F_jJ
        sigma = (F @ S @ F.T) / J

        sigma_avg += sigma * dv
        vol += dv

        # Spatial elasticity (push-forward of C):
        #                             c_ijkl = (1/J) F_iI F_jJ F_kK F_lL C_IJKL
        c = np.einsum('iI,jJ,kK,lL,IJKL->ijkl', F, F, F, F, mat.Cmat) / J

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


# global assembly
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


# Newton-Raphson nonlinear solver
def nonlinear_solve(mesh, mat, fext, fixed,
                    steps=20, tol=1e-3, maxit=40, outdir="paraview_output"):
    """
    Load-stepped Newton-Raphson solve for u(F_ext). The external load is applied
    in `steps` equal increments; each increment is driven to |R_free| < tol via
    Newton, solving only on the free (non-Dirichlet) DOFs. A VTU file is written
    per converged step and collected in a PVD.
    """
    u = np.zeros(mesh.ndof)
    free = np.setdiff1d(np.arange(mesh.ndof), fixed)

    os.makedirs(outdir, exist_ok=True)
    vtu_files = []

    kw = len(str(steps))          # padding widths for aligned output
    iw = len(str(maxit))

    for k in range(steps):        # k: load step
        f_target = fext * (k + 1) / steps       # proportional load stepping
        print(f"k = {k+1:>{kw}d}/{steps}")
        converged = False

        for i in range(maxit):    # i: Newton iteration
            R, K, cell_sigma = assemble(u, mesh, mat, f_target)
            res = np.linalg.norm(R[free])

            tag = "  converged" if res < tol else ""
            print(f"  i = {i+1:>{iw}d}   |R| = {res:.3e}{tag}")

            if res < tol:
                converged = True
                break

            # Solve K_ff * du_free = -R_free  on the free sub-block
            du = np.zeros(mesh.ndof)
            du[free] = spla.spsolve(K[free][:, free], -R[free])
            u += du

        if not converged:
            raise RuntimeError(f"Newton failed at step {k+1}")

        vtu_name = f"step_{k:04d}.vtu"
        write_vtu(os.path.join(outdir, vtu_name), mesh, u.reshape(-1, 3), cell_sigma)
        vtu_files.append(vtu_name)

    write_pvd(os.path.join(outdir, "solution.pvd"), vtu_files)
    return u


# ParaView output
def write_vtu(path, mesh, disp, cell_sigma):
    """Write one converged load step to a .vtu file for ParaView."""
    # Von Mises from the deviatoric part of Cauchy:
    #   dev = sigma - (tr sigma / 3) I,     vm = sqrt(3/2 * dev:dev)
    tr = np.trace(cell_sigma, axis1=1, axis2=2) / 3.0
    dev = cell_sigma - tr[:, None, None] * np.eye(3)
    vm = np.sqrt(1.5 * np.einsum('eij,eij->e', dev, dev))

    meshio.Mesh(
        points=mesh.nodes,
        cells=[("hexahedron", mesh.elements)],
        point_data={
            "Displacement": disp,
            "DisplacementMagnitude": np.linalg.norm(disp, axis=1),
        },
        cell_data={
            "CauchyStress": [cell_sigma.reshape(-1, 9)],
            "vonMises": [vm],
        },
    ).write(path)


def write_pvd(path, vtu_files):
    """Write a PVD collection that groups the per-step .vtu files as a time series."""
    with open(path, "w") as f:
        f.write('<?xml version="1.0"?>\n'
                '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n'
                '  <Collection>\n')
        for i, vf in enumerate(vtu_files):
            f.write(f'    <DataSet timestep="{i}" group="" part="0" file="{vf}" />\n')
        f.write('  </Collection>\n</VTKFile>\n')


# structured mesh generator and wireframe plot
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


def plot_mesh(mesh, u, scale=1.0):
    """Wireframe plot of the deformed mesh. `scale` multiplies the displacement field."""
    xyz = mesh.nodes + scale * u.reshape(-1, 3)
    # For each element, pull the endpoints of all 12 edges: shape (n_elem*12, 2, 3)
    segments = xyz[mesh.elements[:, HEX_EDGES]].reshape(-1, 2, 3)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.add_collection3d(Line3DCollection(segments, colors='C0', linewidths=0.8))
    ax.auto_scale_xyz(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    ax.set_box_aspect([max(xyz[:, k].ptp(), 1e-9) for k in range(3)])
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    return fig, ax


# command-line interface
def parse_args():
    p = argparse.ArgumentParser(
        description="Cantilever beam driver for the hex8 St. Venant-Kirchhoff solver.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Geometry
    p.add_argument("-L", "--length", type=float, default=10.0, help="Length along x")
    p.add_argument("-H", "--height", type=float, default=1.0,  help="Height along y")
    p.add_argument("-W", "--width",  type=float, default=1.0,  help="Width along z")
    # Mesh resolution
    p.add_argument("--nx", type=int, default=8)
    p.add_argument("--ny", type=int, default=2)
    p.add_argument("--nz", type=int, default=2)
    # Material
    p.add_argument("-E", "--youngs-modulus", type=float, default=200e9, dest="E")
    p.add_argument("--nu", type=float, default=0.45, help="Poisson's ratio")
    # External load (total force on the tip face, split evenly over its nodes, in -y)
    p.add_argument("--load", type=float, default=-1e9,
                   help="Total tip load in the y direction (N); "
                        "for negative scientific notation use --load=-5e6")
    # Solver
    p.add_argument("--steps",  type=int,   default=2)
    p.add_argument("--tol",    type=float, default=1e-3)
    p.add_argument("--maxit",  type=int,   default=40)
    p.add_argument("--outdir", default="paraview_output")
    # Plot
    p.add_argument("--scale", type=float, default=5.0,
                   help="Displacement scale factor for the wireframe plot")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def print_header(args, mesh):
    bar = "=" * 64
    print(bar)
    print("  Hex8 St. Venant-Kirchhoff nonlinear FEM solver")
    print(bar)
    print(f"  Geometry    L = {args.length:g}   H = {args.height:g}   W = {args.width:g}")
    print(f"  Mesh        {args.nx} x {args.ny} x {args.nz} hex8   "
          f"({len(mesh.nodes)} nodes, {mesh.ndof} DOFs, {len(mesh.elements)} cells)")
    print(f"  Material    E = {args.E:.3e}   nu = {args.nu:g}")
    print(f"  Load        F = {args.load:+.3e} N   (tip face, y direction)")
    print(f"  Solver      steps = {args.steps}   tol = {args.tol:.1e}   "
          f"maxit = {args.maxit}")
    print(f"  Output      {args.outdir!r}")
    print(bar)


def print_footer(args, mesh, u, tip, elapsed):
    bar = "=" * 64
    disp = u.reshape(-1, 3)
    print(bar)
    print(f"  Tip deflection  u_y = {disp[tip, 1].mean():+.4e}")
    print(f"  Max |u|             = {np.linalg.norm(disp, axis=1).max():.4e}")
    print(f"  Wrote {args.steps} VTU files + PVD to {args.outdir!r}")
    print(f"  Elapsed         {elapsed:.2f} s")
    print(bar)


# beam problem
if __name__ == "__main__":
    args = parse_args()

    # Cantilever beam: length L along x, cross-section H x W in (y, z)
    mesh = create_mesh(args.length, args.height, args.width,
                       nx=args.nx, ny=args.ny, nz=args.nz)
    mat = Material(E=args.E, nu=args.nu)

    # Nodes on the two end faces
    base = np.where(np.isclose(mesh.nodes[:, 0], 0.0))[0]
    tip  = np.where(np.isclose(mesh.nodes[:, 0], args.length))[0]

    # Tip load: distributed equally over the x = L face nodes, in y direction
    Fext = np.zeros(mesh.ndof)
    Fext[3 * tip + 1] = args.load / len(tip)

    # Clamp the x = 0 face (all 3 DOFs of every node fixed)
    fixed = (3 * base[:, None] + np.arange(3)).ravel()

    print_header(args, mesh)
    t0 = time.perf_counter()
    u = nonlinear_solve(mesh, mat, Fext, fixed,
                        steps=args.steps, tol=args.tol, maxit=args.maxit,
                        outdir=args.outdir)
    print_footer(args, mesh, u, tip, time.perf_counter() - t0)

    if not args.no_plot:
        plot_mesh(mesh, u, scale=args.scale)
        plt.show()
