import numpy as np

from .._base import SOMatrixBase


class SO3Matrix(SOMatrixBase):
    dim = 3
    dof = 3

    @classmethod
    def wedge(cls, phi):
        phi = np.atleast_2d(phi)
        Phi = np.zeros((phi.shape[0], cls.dim, cls.dim), dtype=phi.dtype)
        Phi[:, 0, 1] = -phi[:, 2]
        Phi[:, 1, 0] = phi[:, 2]
        Phi[:, 0, 2] = phi[:, 1]
        Phi[:, 2, 0] = -phi[:, 1]
        Phi[:, 1, 2] = -phi[:, 0]
        Phi[:, 2, 1] = phi[:, 0]
        return Phi.squeeze(0) if Phi.shape[0] == 1 else Phi

    @classmethod
    def vee(cls, Phi):
        Phi = np.atleast_3d(Phi)
        phi = np.zeros((Phi.shape[0], cls.dof), dtype=Phi.dtype)
        phi[:, 0] = Phi[:, 2, 1]
        phi[:, 1] = Phi[:, 0, 2]
        phi[:, 2] = Phi[:, 1, 0]
        return phi.squeeze(0) if phi.shape[0] == 1 else phi

