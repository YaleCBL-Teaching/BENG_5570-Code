#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 16:41:33 2026

@author: evanwillmarth
"""


import os
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# GAUSS POINTS
# ------------------------------------------------------------

gp = [-1/np.sqrt(3), 1/np.sqrt(3)]


# ------------------------------------------------------------
# SHAPE FUNCTIONS
# ------------------------------------------------------------

def shape_hex8(xi, eta, zeta):
    N = np.zeros(8)

    N[0] = (1-xi)*(1-eta)*(1-zeta)/8
    N[1] = (1+xi)*(1-eta)*(1-zeta)/8
    N[2] = (1+xi)*(1+eta)*(1-zeta)/8
    N[3] = (1-xi)*(1+eta)*(1-zeta)/8
    N[4] = (1-xi)*(1-eta)*(1+zeta)/8
    N[5] = (1+xi)*(1-eta)*(1+zeta)/8
    N[6] = (1+xi)*(1+eta)*(1+zeta)/8
    N[7] = (1-xi)*(1+eta)*(1+zeta)/8

    return N


def dN_dxi_hex8(xi, eta, zeta):
    dN = np.zeros((8, 3))

    dN[0] = [-(1-eta)*(1-zeta)/8, -(1-xi)*(1-zeta)/8, -(1-xi)*(1-eta)/8]
    dN[1] = [ (1-eta)*(1-zeta)/8, -(1+xi)*(1-zeta)/8, -(1+xi)*(1-eta)/8]
    dN[2] = [ (1+eta)*(1-zeta)/8,  (1+xi)*(1-zeta)/8, -(1+xi)*(1+eta)/8]
    dN[3] = [-(1+eta)*(1-zeta)/8,  (1-xi)*(1-zeta)/8, -(1-xi)*(1+eta)/8]
    dN[4] = [-(1-eta)*(1+zeta)/8, -(1-xi)*(1+zeta)/8,  (1-xi)*(1-eta)/8]
    dN[5] = [ (1-eta)*(1+zeta)/8, -(1+xi)*(1+zeta)/8,  (1+xi)*(1-eta)/8]
    dN[6] = [ (1+eta)*(1+zeta)/8,  (1+xi)*(1+zeta)/8,  (1+xi)*(1+eta)/8]
    dN[7] = [-(1+eta)*(1+zeta)/8,  (1-xi)*(1+zeta)/8,  (1-xi)*(1+eta)/8]

    return dN


# ------------------------------------------------------------
# MATERIAL ELASTICITY TENSOR
# ------------------------------------------------------------

def material_tensor(E, nu):
    lam = E*nu / ((1+nu)*(1-2*nu))
    mu  = E / (2*(1+nu))

    Cmat = np.zeros((3, 3, 3, 3))

    for I in range(3):
        for J in range(3):
            for K in range(3):
                for L in range(3):
                    Cmat[I, J, K, L] = (
                        lam*(I == J)*(K == L)
                        + mu*((I == K)*(J == L) + (I == L)*(J == K))
                    )

    return Cmat


# ------------------------------------------------------------
# ELEMENT ROUTINE
# ------------------------------------------------------------

def hex8_element(X, u_elem, E, nu):
    Cmat = material_tensor(E, nu)

    lam = E*nu / ((1+nu)*(1-2*nu))
    mu  = E / (2*(1+nu))

    Ke   = np.zeros((24, 24))
    fint = np.zeros(24)

    # for optional output: average Cauchy stress over the element
    sigma_avg = np.zeros((3, 3))
    vol = 0.0

    for xi in gp:
        for eta in gp:
            for zeta in gp:

                dNdxi = dN_dxi_hex8(xi, eta, zeta)

                # Reference Jacobian: J0_{IJ} = sum_a X_{aI} dN_a/dxi_J
                J0 = X.T @ dNdxi
                detJ0 = np.linalg.det(J0)

                if detJ0 <= 0:
                    raise ValueError(f"Non-positive reference Jacobian detected: detJ0 = {detJ0}")

                invJ0 = np.linalg.inv(J0)

                # Shape gradients w.r.t. reference coordinates
                # each row a is grad_X N_a in row-vector form
                dNdX = dNdxi @ invJ0

                # Deformation gradient F = I + grad_X u
                F = np.eye(3)
                for a in range(8):
                    ua = u_elem[3*a:3*a+3]
                    F += np.outer(ua, dNdX[a])

                J = np.linalg.det(F)
                if J <= 0:
                    raise ValueError(f"Non-positive deformation Jacobian detected: J = {J}")

                FinvT = np.linalg.inv(F).T

                # Spatial gradients grad_x N_a
                dNdx = np.zeros((8, 3))
                for a in range(8):
                    dNdx[a] = FinvT @ dNdX[a]

                weight = 1.0
                dv = J * detJ0 * weight

                # Green-Lagrange strain
                C = F.T @ F
                Egl = 0.5 * (C - np.eye(3))

                # St. Venant-Kirchhoff: S = lambda tr(E) I + 2 mu E
                S = lam*np.trace(Egl)*np.eye(3) + 2*mu*Egl

                # Cauchy stress
                sigma = (1.0/J) * F @ S @ F.T

                sigma_avg += sigma * dv
                vol += dv

                # Push-forward elasticity tensor
                c = np.zeros((3, 3, 3, 3))
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            for l in range(3):
                                val = 0.0
                                for I in range(3):
                                    for Jj in range(3):
                                        for K in range(3):
                                            for L in range(3):
                                                val += (
                                                    F[i, I] * F[j, Jj] * F[k, K] * F[l, L]
                                                    * Cmat[I, Jj, K, L]
                                                )
                                c[i, j, k, l] = val / J

                # INTERNAL FORCE
                for a in range(8):
                    for i in range(3):
                        val = 0.0
                        for j in range(3):
                            val += sigma[i, j] * dNdx[a, j]
                        fint[3*a + i] += val * dv

                # MATERIAL STIFFNESS
                for a in range(8):
                    for b in range(8):
                        for i in range(3):
                            for j in range(3):
                                val = 0.0
                                for k in range(3):
                                    for l in range(3):
                                        val += dNdx[a, k] * c[i, k, j, l] * dNdx[b, l]
                                Ke[3*a + i, 3*b + j] += val * dv

                # GEOMETRIC STIFFNESS
                for a in range(8):
                    for b in range(8):
                        g = 0.0
                        for k in range(3):
                            for l in range(3):
                                g += dNdx[a, k] * sigma[k, l] * dNdx[b, l]

                        for i in range(3):
                            Ke[3*a + i, 3*b + i] += g * dv

    sigma_avg /= vol
    return Ke, fint, sigma_avg


# ------------------------------------------------------------
# GLOBAL ASSEMBLY
# ------------------------------------------------------------

def assemble(nodes, elements, u, E, nu, f_ext):
    ndof = 3 * len(nodes)

    Rint = np.zeros(ndof)
    K = np.zeros((ndof, ndof))

    cell_sigma = np.zeros((len(elements), 3, 3))

    for e, elem in enumerate(elements):
        X = nodes[elem]

        dofs = []
        for n in elem:
            dofs += [3*n, 3*n+1, 3*n+2]

        u_elem = u[dofs]

        Ke, fint, sigma_avg = hex8_element(X, u_elem, E, nu)
        cell_sigma[e] = sigma_avg

        for a in range(24):
            Rint[dofs[a]] += fint[a]
            for b in range(24):
                K[dofs[a], dofs[b]] += Ke[a, b]

    R = Rint - f_ext
    return R, K, cell_sigma


# ------------------------------------------------------------
# VTU WRITER
# ------------------------------------------------------------

def write_vtu(filename, nodes, elements, displacement, cell_sigma=None):
    """
    Write an unstructured VTU file for ParaView.

    Mesh is written in the reference configuration.
    Displacement is written as point data.
    Optional element-average Cauchy stress is written as cell data.
    """
    npoints = len(nodes)
    ncells = len(elements)

    disp_mag = np.linalg.norm(displacement, axis=1)

    connectivity = []
    offsets = []
    cell_types = []

    off = 0
    for elem in elements:
        connectivity.extend(elem.tolist())
        off += len(elem)
        offsets.append(off)
        cell_types.append(12)  # VTK_HEXAHEDRON

    with open(filename, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        f.write('  <UnstructuredGrid>\n')
        f.write(f'    <Piece NumberOfPoints="{npoints}" NumberOfCells="{ncells}">\n')

        # Points
        f.write('      <Points>\n')
        f.write('        <DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
        for p in nodes:
            f.write(f'          {p[0]} {p[1]} {p[2]}\n')
        f.write('        </DataArray>\n')
        f.write('      </Points>\n')

        # Cells
        f.write('      <Cells>\n')

        f.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
        f.write('          ' + ' '.join(str(int(x)) for x in connectivity) + '\n')
        f.write('        </DataArray>\n')

        f.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
        f.write('          ' + ' '.join(str(int(x)) for x in offsets) + '\n')
        f.write('        </DataArray>\n')

        f.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
        f.write('          ' + ' '.join(str(int(x)) for x in cell_types) + '\n')
        f.write('        </DataArray>\n')

        f.write('      </Cells>\n')

        # Point data
        f.write('      <PointData Vectors="Displacement">\n')

        f.write('        <DataArray type="Float64" Name="Displacement" NumberOfComponents="3" format="ascii">\n')
        for u in displacement:
            f.write(f'          {u[0]} {u[1]} {u[2]}\n')
        f.write('        </DataArray>\n')

        f.write('        <DataArray type="Float64" Name="DisplacementMagnitude" NumberOfComponents="1" format="ascii">\n')
        for val in disp_mag:
            f.write(f'          {val}\n')
        f.write('        </DataArray>\n')

        f.write('      </PointData>\n')

        # Cell data
        f.write('      <CellData>\n')

        if cell_sigma is not None:
            # full tensor as 9 components
            f.write('        <DataArray type="Float64" Name="CauchyStress" NumberOfComponents="9" format="ascii">\n')
            for s in cell_sigma:
                f.write(
                    f'          {s[0,0]} {s[0,1]} {s[0,2]} '
                    f'{s[1,0]} {s[1,1]} {s[1,2]} '
                    f'{s[2,0]} {s[2,1]} {s[2,2]}\n'
                )
            f.write('        </DataArray>\n')

            # von Mises stress
            f.write('        <DataArray type="Float64" Name="vonMises" NumberOfComponents="1" format="ascii">\n')
            for s in cell_sigma:
                mean_stress = np.trace(s) / 3.0
                dev = s - mean_stress*np.eye(3)
                vm = np.sqrt(1.5 * np.sum(dev*dev))
                f.write(f'          {vm}\n')
            f.write('        </DataArray>\n')

        f.write('      </CellData>\n')

        f.write('    </Piece>\n')
        f.write('  </UnstructuredGrid>\n')
        f.write('</VTKFile>\n')


def write_pvd(filename, vtu_files):
    with open(filename, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        f.write('  <Collection>\n')
        for i, vf in enumerate(vtu_files):
            f.write(f'    <DataSet timestep="{i}" group="" part="0" file="{vf}" />\n')
        f.write('  </Collection>\n')
        f.write('</VTKFile>\n')


# ------------------------------------------------------------
# NEWTON SOLVER
# ------------------------------------------------------------

def nonlinear_solve(nodes, elements, E, nu, fext, fixed,
                    steps=20, tol=1e-3, maxit=40, outdir="paraview_output"):

    ndof = 3 * len(nodes)
    u = np.zeros(ndof)

    os.makedirs(outdir, exist_ok=True)
    vtu_files = []

    for step in range(steps):
        frac = (step + 1) / steps
        f_target = fext * frac

        converged = False

        for it in range(maxit):
            R, K, cell_sigma = assemble(nodes, elements, u, E, nu, f_target)

            # Apply Dirichlet BCs
            R[fixed] = 0.0

            Kmod = K.copy()
            for d in fixed:
                Kmod[d, :] = 0.0
                Kmod[:, d] = 0.0
                Kmod[d, d] = 1.0

            free = np.setdiff1d(np.arange(ndof), fixed)
            res = np.linalg.norm(R[free])

            print(f"step {step+1}/{steps}, iter {it+1}/{maxit}, |R_free| = {res:.6e}")

            if res < tol:
                converged = True
                break

            du = np.linalg.solve(Kmod, -R)
            u += du

        if not converged:
            raise RuntimeError(f"Newton failed to converge at load step {step+1}")

        # Re-assemble once more at converged state to output stress consistently
        R, K, cell_sigma = assemble(nodes, elements, u, E, nu, f_target)

        disp = u.reshape((-1, 3))
        vtu_name = f"step_{step:04d}.vtu"
        vtu_path = os.path.join(outdir, vtu_name)

        write_vtu(
            vtu_path,
            nodes,
            elements,
            disp,
            cell_sigma=cell_sigma
        )

        vtu_files.append(vtu_name)

    write_pvd(os.path.join(outdir, "solution.pvd"), vtu_files)
    return u


# ------------------------------------------------------------
# MESH GENERATOR
# ------------------------------------------------------------

def create_mesh(L, H, W, nx, ny, nz):
    nodes = []

    for k in range(nz+1):
        for j in range(ny+1):
            for i in range(nx+1):
                nodes.append([L*i/nx, H*j/ny, W*k/nz])

    nodes = np.array(nodes, dtype=float)

    def nid(i, j, k):
        return i + (nx+1)*(j + (ny+1)*k)

    elems = []

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                elems.append([
                    nid(i,   j,   k),
                    nid(i+1, j,   k),
                    nid(i+1, j+1, k),
                    nid(i,   j+1, k),
                    nid(i,   j,   k+1),
                    nid(i+1, j,   k+1),
                    nid(i+1, j+1, k+1),
                    nid(i,   j+1, k+1)
                ])

    return nodes, np.array(elems, dtype=int)


# ------------------------------------------------------------
# BEAM PROBLEM
# ------------------------------------------------------------

L = 10.0
H = 1.0
W = 1.0

nx = 8
ny = 2
nz = 2

nodes, elements = create_mesh(L, H, W, nx, ny, nz)

E = 200e9
nu = 0.3

ndof = 3 * len(nodes)
Fext = np.zeros(ndof)

# Apply load on x=L face
end_nodes = [i for i, p in enumerate(nodes) if np.isclose(p[0], L)]

# If you want TOTAL load = -1e7, divide by number of end nodes:
total_load = -1e7
nodal_load = total_load / len(end_nodes)

for i in end_nodes:
    Fext[3*i + 1] = nodal_load

# Clamp x=0 face
fixed = []
for i, node in enumerate(nodes):
    if abs(node[0]) < 1e-8:
        fixed += [3*i, 3*i+1, 3*i+2]
fixed = np.array(fixed, dtype=int)

u = nonlinear_solve(
    nodes, elements, E, nu, Fext, fixed,
    steps=20, tol=1e-3, maxit=40,
    outdir="paraview_output"
)


# ------------------------------------------------------------
# QUICK PLOT
# ------------------------------------------------------------

scale = 5.0

x = nodes[:, 0] + scale * u[0::3]
y = nodes[:, 1] + scale * u[1::3]
z = nodes[:, 2] + scale * u[2::3]

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(x, y, z)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()



