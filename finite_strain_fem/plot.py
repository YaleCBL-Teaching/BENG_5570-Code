"""Matplotlib wireframe plot of the deformed hex8 mesh."""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from .hex8 import HEX_EDGES
from .physics import N_DIM


def plot_mesh(mesh, u, scale=1.0):
    """wireframe plot of the deformed mesh; scale multiplies the displacement field."""
    xyz = mesh.nodes + scale * u.reshape(-1, N_DIM)
    # For each element, pull the endpoints of all 12 edges: shape (n_elem*12, 2, 3)
    segments = xyz[mesh.elements[:, HEX_EDGES]].reshape(-1, 2, 3)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.add_collection3d(Line3DCollection(segments, colors='C0', linewidths=0.8))
    ax.auto_scale_xyz(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    ax.set_box_aspect([max(xyz[:, k].ptp(), 1e-9) for k in range(3)])
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    return fig, ax
