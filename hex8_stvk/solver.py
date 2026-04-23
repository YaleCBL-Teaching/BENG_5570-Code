"""Load-stepped Newton-Raphson solver."""
import os
import numpy as np
import scipy.sparse.linalg as spla

from .assembly import assemble
from .io import write_vtu, write_pvd


def apply_bc(mesh, clamped_nodes, loaded_nodes, load_dir, load_total):
    """Set up Dirichlet and Neumann boundary conditions.

    All three DOFs of `clamped_nodes` are fixed; `load_total` is spread evenly
    over `loaded_nodes` in direction `load_dir` (0=x, 1=y, 2=z).

    Returns (fixed_dofs, f_ext).
    """
    fixed = (3 * clamped_nodes[:, None] + np.arange(3)).ravel()
    f_ext = np.zeros(mesh.ndof)
    f_ext[3 * loaded_nodes + load_dir] = load_total / len(loaded_nodes)
    return fixed, f_ext


class Norms:
    """Relative-norm convergence check against the very first residual/increment.

    The first call fixes reference scales |R_1,1| and |du_1,1|; subsequent
    calls return the relative ratios and whether both are below tolerance.
    """
    def __init__(self, tol_r, tol_u):
        self.tol_r, self.tol_u = tol_r, tol_u
        self.res_11 = None
        self.du_11 = None

    def __call__(self, res, du):
        if self.res_11 is None:
            self.res_11 = res if res > 0 else 1.0
        if self.du_11 is None:
            self.du_11 = du if du > 0 else 1.0
        r_rel = res / self.res_11
        u_rel = du / self.du_11
        return r_rel, u_rel, (r_rel < self.tol_r and u_rel < self.tol_u)


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
    norms = Norms(tol_r, tol_u)

    os.makedirs(outdir, exist_ok=True)
    vtu_files = []

    kw = max(len(str(steps)), 1)
    iw = max(len(str(maxit)), 1)
    header = f"  {'k':>{kw}}  {'i':>{iw}}   R_ki/R_11   du_ki/du_11"
    print(header)
    print("=" * len(header))

    for k in range(steps):
        f_target = fext * (k + 1) / steps       # proportional load stepping
        converged = False

        for i in range(maxit):
            R, K, cell_sigma = assemble(u, mesh, mat, f_target)
            du_free = spla.spsolve(K[free][:, free], -R[free])

            r_rel, u_rel, ok = norms(np.linalg.norm(R[free]),
                                     np.linalg.norm(du_free))
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
