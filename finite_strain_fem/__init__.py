"""Finite-strain nonlinear FEM (spatial / updated-Lagrangian) with hex8 elements."""
from .material import Material
from .mesh import Mesh, create_mesh, apply_bc
from .physics import finite_strain_element
from .assembly import assemble
from .solver import nonlinear_solve

__all__ = ["Material", "Mesh", "create_mesh", "apply_bc",
           "finite_strain_element", "assemble", "nonlinear_solve"]
