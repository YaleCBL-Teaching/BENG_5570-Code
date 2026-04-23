"""St. Venant-Kirchhoff isotropic material."""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Material:
    """Isotropic St. Venant-Kirchhoff material.

    `stress_tangent(E)` returns the 2nd Piola-Kirchhoff stress S and the
    material elasticity tensor C = dS/dE given a Green-Lagrange strain E.
    """
    E: float
    nu: float
    lam: float = field(init=False)
    mu: float = field(init=False)
    _Cmat: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        # Lamé parameters
        self.lam = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.mu = self.E / (2 * (1 + self.nu))

        # Material elasticity tensor: C_IJKL = lam delta_IJ delta_KL
        #                                    + mu (delta_IK delta_JL + delta_IL delta_JK)
        # Constant for SVK; cached once.
        I = np.eye(3)
        self._Cmat = (self.lam * np.einsum('ij,kl->ijkl', I, I)
                      + self.mu * (np.einsum('ik,jl->ijkl', I, I)
                                   + np.einsum('il,jk->ijkl', I, I)))

    def stress_tangent(self, E):
        """Return (S, C) given Green-Lagrange strain E (shape (3, 3))."""
        # 2nd Piola-Kirchhoff:  S_IJ = lam tr(E) delta_IJ + 2 mu E_IJ
        S = self.lam * np.trace(E) * np.eye(3) + 2.0 * self.mu * E
        return S, self._Cmat
