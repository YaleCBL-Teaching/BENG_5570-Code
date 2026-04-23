"""St. Venant-Kirchhoff isotropic material."""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Material:
    """Isotropic St. Venant-Kirchhoff material. Stores Lamé parameters and C_IJKL."""
    E: float
    nu: float
    lam: float = field(init=False)
    mu: float = field(init=False)
    Cmat: np.ndarray = field(init=False)

    def __post_init__(self):
        # Lamé parameters
        self.lam = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.mu = self.E / (2 * (1 + self.nu))

        # Material elasticity tensor: C_IJKL = lam delta_IJ delta_KL
        #                                    + mu (delta_IK delta_JL + delta_IL delta_JK)
        I = np.eye(3)
        self.Cmat = (self.lam * np.einsum('ij,kl->ijkl', I, I)
                     + self.mu * (np.einsum('ik,jl->ijkl', I, I)
                                  + np.einsum('il,jk->ijkl', I, I)))
