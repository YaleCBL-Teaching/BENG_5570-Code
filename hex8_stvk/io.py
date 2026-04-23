"""ParaView output: per-step VTU + PVD time-series collection."""
import numpy as np
import meshio


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
