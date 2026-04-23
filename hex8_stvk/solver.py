"""Load-stepped Newton-Raphson solver."""
import os
import numpy as np
import scipy.sparse.linalg as spla

from .assembly import assemble
from .element import N_DIM
from .io import write_vtu, write_pvd


def apply_bc(mesh, clamped_nodes, loaded_nodes, load_dir, load_total):
    """clamp all DOFs of clamped_nodes and distribute load_total over loaded_nodes along load_dir; returns (fixed_dofs, f_ext)."""
    fixed = (N_DIM * clamped_nodes[:, None] + np.arange(N_DIM)).ravel()
    f_ext = np.zeros(mesh.ndof)
    f_ext[N_DIM * loaded_nodes + load_dir] = load_total / len(loaded_nodes)
    return fixed, f_ext


class Norms:
    """relative-norm convergence check; the first call fixes the references |R_1,1| and |du_1,1|."""
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
    """load-stepped Newton-Raphson solve for u(F_ext); writes a VTU per converged step and a PVD collection."""
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
        # proportional load stepping
        f_target = fext * (k + 1) / steps

        for i in range(maxit):
            R, K, elem_sigma = assemble(u, mesh, mat, f_target)
            du_free = spla.spsolve(K[free][:, free], -R[free])

            r_rel, u_rel, ok = norms(np.linalg.norm(R[free]),
                                     np.linalg.norm(du_free))
            tag = "  converged" if ok else ""
            print(f"  {k+1:>{kw}d}  {i+1:>{iw}d}  "
                  f"  {r_rel:.1e}      {u_rel:.1e}{tag}")

            if ok:
                break
            u[free] += du_free
        else:
            raise RuntimeError(f"Newton failed at step {k+1}")
        print("-" * len(header))

        vtu_name = f"step_{k:04d}.vtu"
        write_vtu(os.path.join(outdir, vtu_name), mesh, u.reshape(-1, N_DIM), elem_sigma)
        vtu_files.append(vtu_name)

    write_pvd(os.path.join(outdir, "solution.pvd"), vtu_files)
    return u
