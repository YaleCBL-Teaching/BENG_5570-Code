"""ParaView output: per-step VTU + PVD time-series collection."""

import numpy as np
import meshio


def write_vtu(path, mesh, disp, elem_sigma):
    """write one converged load step to a .vtu file for ParaView."""
    # Von Mises from the deviatoric part of Cauchy:
    #   dev = sigma - (tr sigma / 3) I,     vm = sqrt(3/2 * dev:dev)
    tr = np.trace(elem_sigma, axis1=1, axis2=2) / 3.0
    dev = elem_sigma - tr[:, None, None] * np.eye(3)
    vm = np.sqrt(1.5 * np.einsum("eij,eij->e", dev, dev))

    # Six unique components of the symmetric Cauchy stress (Voigt ordering)
    stress = {f"sigma_{ab}": [elem_sigma[:, i, j]]
              for ab, (i, j) in zip(("xx", "yy", "zz", "xy", "yz", "xz"),
                                    ((0, 0), (1, 1), (2, 2),
                                     (0, 1), (1, 2), (0, 2)))}

    meshio.Mesh(
        points=mesh.nodes,
        cells=[("hexahedron", mesh.elements)],
        point_data={"Displacement": disp},
        cell_data={**stress, "vonMises": [vm]},
    ).write(path)


def write_pvd(path, vtu_files):
    """write a PVD collection that groups the per-step .vtu files as a time series."""
    with open(path, "w") as f:
        f.write(
            '<?xml version="1.0"?>\n'
            '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n'
            "  <Collection>\n"
        )
        for i, vf in enumerate(vtu_files):
            f.write(f'    <DataSet timestep="{i}" group="" part="0" file="{vf}" />\n')
        f.write("  </Collection>\n</VTKFile>\n")
