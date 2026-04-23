"""Load-stepped Newton-Raphson solver."""
import os
import numpy as np
import scipy.sparse.linalg as spla

from .assembly import assemble
from .io import write_vtu, write_pvd


def nonlinear_solve(mesh, mat, fext, fixed,
                    steps=20, tol_r=1e-8, tol_u=1e-8,
                    maxit=40, outdir="paraview_output"):
    """
    Load-stepped Newton-Raphson solve for u(F_ext). Load is applied in `steps`
    equal increments; each increment iterates until both relative measures are
    below their tolerances:

        |R_k,i| / |R_1,1|     residual vs. the very first residual
        |du_k,i| / |du_1,1|   increment vs. the very first increment

    Only free (non-Dirichlet) DOFs enter the norms. A VTU file is written per
    converged step and collected in a PVD.
    """
    u = np.zeros(mesh.ndof)
    free = np.setdiff1d(np.arange(mesh.ndof), fixed)

    os.makedirs(outdir, exist_ok=True)
    vtu_files = []

    kw = max(len(str(steps)), 1)      # padding widths for aligned columns
    iw = max(len(str(maxit)), 1)

    # Column header. Value fields are sized so the first digit sits under the
    # first '_' of the column label (e.g. "  1.0e+00  " inside "du_ki/du_11").
    header = f"  {'k':>{kw}}  {'i':>{iw}}   R_ki/R_11   du_ki/du_11"
    print(header)
    print("=" * len(header))

    res_11 = None                 # |R_1,1|  very-first residual
    du_11 = None                  # |du_1,1| very-first Newton increment

    for k in range(steps):        # k: load step
        f_target = fext * (k + 1) / steps       # proportional load stepping
        converged = False

        for i in range(maxit):    # i: Newton iteration
            R, K, cell_sigma = assemble(u, mesh, mat, f_target)
            res = np.linalg.norm(R[free])

            # Solve K_ff * du_free = -R_free on the free sub-block
            du_free = spla.spsolve(K[free][:, free], -R[free])
            du_norm = np.linalg.norm(du_free)

            # Fix reference scales on the very first Newton iteration. If
            # the reference is exactly zero (already at equilibrium), fall
            # back to 1.0 so the ratio is 0 rather than a divide-by-zero.
            if res_11 is None:
                res_11 = res if res > 0 else 1.0
            if du_11 is None:
                du_11 = du_norm if du_norm > 0 else 1.0

            r_rel = res / res_11
            u_rel = du_norm / du_11

            ok = (r_rel < tol_r) and (u_rel < tol_u)
            tag = "  converged" if ok else ""
            print(f"  {k+1:>{kw}d}  {i+1:>{iw}d}  "
                  f"  {r_rel:.1e}      {u_rel:.1e}{tag}")

            if ok:
                converged = True
                break

            u[free] += du_free

        if not converged:
            raise RuntimeError(f"Newton failed at step {k+1}")
        print("-" * len(header))

        vtu_name = f"step_{k:04d}.vtu"
        write_vtu(os.path.join(outdir, vtu_name), mesh, u.reshape(-1, 3), cell_sigma)
        vtu_files.append(vtu_name)

    write_pvd(os.path.join(outdir, "solution.pvd"), vtu_files)
    return u
