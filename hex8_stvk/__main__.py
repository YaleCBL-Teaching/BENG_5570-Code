"""Cantilever beam driver. Run with:  python -m hex8_stvk [options]"""
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

from .element import N_DIM
from .material import Material
from .mesh import create_mesh
from .solver import apply_bc, nonlinear_solve
from .plot import plot_mesh


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
    p.add_argument("--nx", type=int, default=8, help="Number of elements along x")
    p.add_argument("--ny", type=int, default=2, help="Number of elements along y")
    p.add_argument("--nz", type=int, default=2, help="Number of elements along z")
    # Material
    p.add_argument("-E", "--youngs-modulus", type=float, default=200e9, dest="E", help="Young's modulus")
    p.add_argument("--nu", type=float, default=0.45, help="Poisson's ratio")
    # External load (total force on the tip face, split evenly over its nodes, in -y)
    p.add_argument("--load", type=float, default=-1e9,
                   help="Total tip load in the y direction (N); "
                        "for negative scientific notation use --load=-5e6")
    # Solver
    p.add_argument("--steps",  type=int,   default=2, help="Number of load steps")
    p.add_argument("--tol-r",  type=float, default=1e-8, help="Relative residual tolerance")
    p.add_argument("--tol-u",  type=float, default=1e-8, help="Relative displacement increment tolerance")
    p.add_argument("--maxit",  type=int,   default=40, help="Maximum number of Newton iterations per load step")
    p.add_argument("--outdir", default="paraview_output", help="Directory to write VTU files to")
    # Plot
    p.add_argument("--scale", type=float, default=5.0,
                   help="Displacement scale factor for the wireframe plot")
    p.add_argument("--no-plot", action="store_true", help="Don't show the wireframe plot at the end")
    return p.parse_args()


def print_header(args, mesh):
    bar = "=" * 64
    print(bar)
    print("  Hex8 St. Venant-Kirchhoff nonlinear FEM solver")
    print(bar)
    print(f"  Geometry    L = {args.length:g}   H = {args.height:g}   W = {args.width:g}")
    print(f"  Mesh        {args.nx} x {args.ny} x {args.nz} hex8   "
          f"({len(mesh.nodes)} nodes, {mesh.ndof} DOFs, {len(mesh.elements)} elements)")
    print(f"  Material    E = {args.E:.3e}   nu = {args.nu:g}")
    print(f"  Load        F = {args.load:+.3e} N   (tip face, y direction)")
    print(f"  Solver      steps = {args.steps}   maxit = {args.maxit}")
    print(f"              tol(R/R_11) = {args.tol_r:.1e}   "
          f"tol(du/du_11) = {args.tol_u:.1e}")
    print(f"  Output      {args.outdir!r}")
    print(bar)


def print_footer(args, u, tip, elapsed):
    bar = "=" * 64
    disp = u.reshape(-1, N_DIM)
    print(bar)
    print(f"  Tip deflection  u_y = {disp[tip, 1].mean():+.4e}")
    print(f"  Wrote {args.steps} VTU files + PVD to {args.outdir!r}")
    print(f"  Elapsed         {elapsed:.2f} s")
    print(bar)


def main():
    args = parse_args()

    # Cantilever beam: length L along x, cross-section H x W in (y, z)
    mesh = create_mesh(args.length, args.height, args.width,
                       nx=args.nx, ny=args.ny, nz=args.nz)
    mat = Material(E=args.E, nu=args.nu)

    # Clamp the x = 0 face; distribute the tip load in -y over the x = L face.
    base = np.where(np.isclose(mesh.nodes[:, 0], 0.0))[0]
    tip  = np.where(np.isclose(mesh.nodes[:, 0], args.length))[0]
    fixed, Fext = apply_bc(mesh, clamped_nodes=base,
                           loaded_nodes=tip, load_dir=1, load_total=args.load)

    print_header(args, mesh)
    t0 = time.perf_counter()
    u = nonlinear_solve(mesh, mat, Fext, fixed,
                        steps=args.steps,
                        tol_r=args.tol_r, tol_u=args.tol_u,
                        maxit=args.maxit, outdir=args.outdir)
    print_footer(args, u, tip, time.perf_counter() - t0)

    if not args.no_plot:
        plot_mesh(mesh, u, scale=args.scale)
        plt.show()


if __name__ == "__main__":
    main()
