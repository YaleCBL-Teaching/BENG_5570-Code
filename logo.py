import cv2
import numpy as np
import triangle  # pip install triangle
import meshio
from mpi4py import MPI
import dolfin as df
import pdb

df.parameters["linear_algebra_backend"] = "PETSc"

def find_polygons_with_holes(img_thresh, epsilon=2.0):
    """
    Given a thresholded image (0=background, 255=letters), find all outer
    and inner (hole) contours, approximate them, and build a polygon dictionary
    for python-triangle. Additionally:
      - Mirror letters across the x-axis (y -> -y).
      - Shift so the bottom of the mirrored letters is at y=0.
    
    Parameters
    ----------
    img_thresh : np.ndarray
        Binary (0 or 255) image where letters are the foreground.
    epsilon : float
        Approximation factor for simplifying contours. Larger = fewer vertices.
    
    Returns
    -------
    polygon : dict
        {
            "vertices": (N, 2) float array of all contour points,
            "segments": (M, 2) int array of boundary edges,
            "holes":    (K, 2) float array of seed points inside holes,
        }
    """
    # 1. Find contours with hierarchy (RETR_CCOMP or RETR_TREE)
    contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        # No contours found
        return {
            "vertices": np.array([]),
            "segments": np.array([]),
            "holes": np.array([])
        }
    hierarchy = hierarchy[0]  # shape: (num_contours, 4)

    # We'll accumulate all polygon data here
    all_vertices = []
    all_segments = []
    hole_points = []
    vertex_counter = 0

    # 2. Loop over contours and build the vertex/segment data
    for i, contour in enumerate(contours):
        # hierarchy[i] = [next, prev, child, parent]
        parent_idx = hierarchy[i][3]

        # Approximate the contour to ignore small details
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)
        approx = approx.squeeze(axis=1)  # from shape (N,1,2) to (N,2)

        # Skip degenerate polygons
        if len(approx) < 3:
            continue

        # 3. Mirror across the x-axis => y -> -y
        approx[:, 1] = -approx[:, 1]

        # Store these points and build boundary segments
        start_idx = vertex_counter
        n = len(approx)
        for (x, y) in approx:
            all_vertices.append([x, y])

        for j in range(n - 1):
            all_segments.append([start_idx + j, start_idx + j + 1])
        # close the loop
        all_segments.append([start_idx + n - 1, start_idx])

        vertex_counter += n

        # If there's a parent, this contour is a hole
        if parent_idx != -1:
            x_coords = approx[:, 0]
            y_coords = approx[:, 1]
            cx = np.mean(x_coords)
            cy = np.mean(y_coords)
            hole_points.append([cx, cy])

    # 4. Shift everything so the bottom is at y=0
    if len(all_vertices) == 0:
        return {
            "vertices": np.array([]),
            "segments": np.array([]),
            "holes": np.array([])
        }

    all_y = [v[1] for v in all_vertices]
    if len(hole_points) > 0:
        all_y += [h[1] for h in hole_points]
    min_y = min(all_y)

    # Shift vertices
    for v in all_vertices:
        v[1] -= min_y
    # Shift hole seeds
    for h in hole_points:
        h[1] -= min_y

    # 5. Build final polygon dictionary
    polygon = {
        "vertices": np.array(all_vertices, dtype=np.float64),
        "segments": np.array(all_segments, dtype=np.int32),
        "holes": (np.array(hole_points, dtype=np.float64)
                  if len(hole_points) > 0
                  else np.empty((0, 2), dtype=np.float64))
    }
    return polygon

def triangulate_with_holes(img_thresh, area_constraint=300.0, epsilon=2.0):
    """
    Build a polygon with holes from the thresholded image, then call python-triangle
    to produce a triangulation that excludes the hole regions.
    
    Parameters
    ----------
    img_thresh : np.ndarray
        Binary (0/255) image.
    area_constraint : float
        Maximum allowed area for each triangle (to get uniform sizes).
    epsilon : float
        Approximation factor for ignoring minor details in letter contours.
    
    Returns
    -------
    mesh_img : np.ndarray
        Grayscale image showing the final triangulation (holes carved out).
    tmesh : dict
        python-triangle result dictionary with 'vertices', 'triangles', etc.
    """
    h, w = img_thresh.shape

    # 1. Build polygon with hole info
    polygon = find_polygons_with_holes(img_thresh, epsilon=epsilon)

    # If no vertices => nothing to do
    if polygon["vertices"].size == 0:
        raise ValueError("No valid contours found in image.")

    # 2. Call python-triangle with constraints:
    #    p => piecewise linear complex
    #    D => refine Delaunay
    #    q30 => min angle 30 degrees (avoid skinny triangles)
    #    aXXX => limit max area
    opts = f"pDq30a{area_constraint}"
    tmesh = triangle.triangulate(polygon, opts)

    if "triangles" not in tmesh:
        raise RuntimeError("No 'triangles' produced by python-triangle. Possibly invalid geometry.")

    # 3. Draw the resulting mesh
    mesh_img = np.full((h, w), 255, dtype=np.uint8)
    tri_indices = tmesh["triangles"]
    vert_coords = tmesh["vertices"]  # shape (N,2)

    for simplex in tri_indices:
        pts = vert_coords[simplex]  # shape (3,2)
        pts = pts.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(mesh_img, [pts], True, color=0, thickness=1)

    return mesh_img, tmesh

def write_triangle_mesh_to_xdmf(tmesh, filename="letters.xdmf"):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    points = tmesh["vertices"]  # shape (N,2)
    cells = [("triangle", tmesh["triangles"])]

    # Only rank 0 writes
    if rank == 0:
        mesh = meshio.Mesh(points, cells)
        mesh.write(filename)
        print(f"[Rank 0] Wrote mesh to {filename}")
    
    # Optionally barrier to ensure rank 0 finishes writing
    comm.Barrier()

def fenics_finite_strain_neohookean(
    mesh_path,
    n_steps=100,
    max_accel=1.6,
    E=1.0e6,
    nu=0.4,
    rho=1.0,
):
    """
    Solve a 2D finite-strain Neo-Hookean problem in increments.

    Parameters
    ----------
    mesh_path : str
        Path to the 2D mesh file (XDMF or similar).
    n_steps : int
        Number of increments to apply the final acceleration.
    max_accel : float
        Final acceleration value (downward in y). We'll ramp from 0 to this.
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    rho : float
        Density.
    output_prefix : str
        Prefix for output files (displacement, principal stress, etc.).
    """

    # ----------------------------------------------------------------
    # 1. READ MESH (2D)
    # ----------------------------------------------------------------
    mesh = df.Mesh()
    with df.XDMFFile(mesh_path) as infile:
        infile.read(mesh)

    # ----------------------------------------------------------------
    # 2. FUNCTION SPACE (P2 for better accuracy)
    # ----------------------------------------------------------------
    degree = 2
    V = df.VectorFunctionSpace(mesh, "CG", degree)
    TFS = df.TensorFunctionSpace(mesh, "CG", degree)

    # ----------------------------------------------------------------
    # 3. BOUNDARY CONDITIONS (CLAMP BOTTOM)
    # ----------------------------------------------------------------
    eps = 20
    coords = mesh.coordinates()
    y_min = np.min(coords[:, 1])

    class BottomBoundary(df.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and abs(x[1] - y_min) < eps

    bottom_boundary = BottomBoundary()
    boundary_markers = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    bottom_boundary.mark(boundary_markers, 1)

    bc = df.DirichletBC(V, df.Constant((0.0, 0.0)), boundary_markers, 1)

    # ----------------------------------------------------------------
    # 4. MATERIAL (Neo-Hookean) SETUP
    # ----------------------------------------------------------------
    # Convert E, nu -> lam, mu
    mu_ = E/(2.0*(1.0 + nu))
    lam_ = (E*nu)/((1.0 + nu)*(1.0 - 2.0*nu))

    # Prepare the unknown displacement and test function
    u = df.Function(V, name="Displacement")  # current solution
    v = df.TestFunction(V)

    d = 2  # 2D
    I = df.Identity(d)
    F_ = I + df.grad(u)
    J = df.det(F_)
    C_ = F_.T*F_
    I1 = df.tr(C_)

    # Neo-Hookean strain energy density
    psi = (mu_/2.0)*(I1 - 2.0) - mu_*df.ln(J) + (lam_/2.0)*(df.ln(J))**2

    # We'll define the body force as a placeholder Constant, then ramp it
    accel = df.Constant(0.0)  # will set to partial fraction of max_accel
    # Body force:
    # f = df.as_vector([0.0, -rho*accel])  # negative y
    f = df.as_vector([-5*rho*accel, 0.0])  # negative y

    # Potential Pi = \int psi dV - \int f·u dV
    Pi = psi*df.dx - df.dot(f, u)*df.dx

    # Residual & Jacobian
    F_res = df.derivative(Pi, u, v)
    dF = df.derivative(F_res, u)

    # Create a NonlinearVariationalProblem
    problem = df.NonlinearVariationalProblem(F_res, u, bcs=[bc], J=dF)
    solver = df.NonlinearVariationalSolver(problem)
    
    prm = solver.parameters
    prm["nonlinear_solver"] = "newton"
    # prm["newton_solver"]["linear_solver"] = "gmres"              # or "gmres", "bicgstab", ...
    # prm["newton_solver"]["preconditioner"] = "hypre_amg"      # good parallel AMG
    prm["newton_solver"]["relative_tolerance"] = 1e-6
    prm["newton_solver"]["absolute_tolerance"] = 1e-8
    prm["newton_solver"]["maximum_iterations"] = 25

    # ----------------------------------------------------------------
    # 5. INCREMENTAL LOOP
    # ----------------------------------------------------------------

    # XDMF file (one file for all time steps)
    xdmf = df.XDMFFile(df.MPI.comm_world, "solution_timeseries.xdmf")
    # Optional parameters to control how data is stored
    xdmf.parameters["flush_output"] = True
    xdmf.parameters["functions_share_mesh"] = True
    xdmf.parameters["rewrite_function_mesh"] = False

    # Start from zero acceleration, ramp up to max_accel in n_steps
    for step in range(1, n_steps + 1):
        # fraction of load
        frac = step / float(n_steps)
        current_acc = frac * max_accel
        accel.assign(current_acc)
        print(f"[INFO] --- Load step {step}/{n_steps}, accel={current_acc:.2f} ---")

        # Solve
        solver.solve()  # solve for displacement at this load

        # ------------------------------------------------------
        # 5A. Compute stress (Cauchy or principal) at current step
        # ------------------------------------------------------
        F_ = I + df.grad(u)
        J = df.det(F_)
        # Example: full Cauchy stress for compressible neo-Hookean
        sigma_expr = (mu_ / J) * (F_ * F_.T - I) + lam_ * df.ln(J) * I
        # Create a Function in TFS to hold the stress
        sigma_func = df.project(sigma_expr, TFS)
        sigma_func.rename("Stress", "Cauchy Stress")  # Consistent name for stress

        # ------------------------------------------------------
        # 5B. Write displacement + stress to same XDMF with the
        #     current "time step" = step (or a real dt if you prefer).
        # ------------------------------------------------------
        xdmf.write(u, step)        # store displacement at time=step
        xdmf.write(sigma_func, step)  # store stress at time=step

    xdmf.close()
    print("[INFO] All time steps stored in 'incremental_solution.xdmf'.")


def main(input_path, output_prefix):
    """
    Example usage:
    - Reads an image with black background (gray=0) and letters in brighter intensity
    - Thresholds it
    - Calls triangulate_with_holes to produce uniform-size triangles excluding holes
    - Writes the final mesh to disk
    """
    img_gray = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise ValueError(f"Could not read the image at {input_path}.")
    h, w = img_gray.shape

    # If letters are bright and background is black (0), use a normal threshold:
    _, img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY)

    # For demonstration, you could also save the threshold for debugging
    # cv2.imwrite(f"{output_prefix}_threshold.png", img_thresh)

    # Triangulate with holes
    area_constraint = 500.0  # adjust to control triangle size
    epsilon = 1            # approximation factor to ignore small details
    mesh_img, tmesh = triangulate_with_holes(img_thresh, area_constraint, epsilon)

    # Save final
    out_path = f"{output_prefix}_final_mesh.png"
    cv2.imwrite(out_path, mesh_img)
    write_triangle_mesh_to_xdmf(tmesh, "letters.xdmf")

    print(f"[INFO] Saved triangulation (holes excluded) to {out_path}")

    fenics_finite_strain_neohookean("letters.xdmf")


if __name__ == "__main__":
    input_image_path = "yale_2560.png"
    output_prefix = "letters_output"
    main(input_image_path, output_prefix)
