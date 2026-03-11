import numpy as np

from .so3 import SO3Matrix
from .._base import SEMatrixBase


class SE3Matrix(SEMatrixBase):
    dim = 4
    dof = 6
    RotationType = SO3Matrix

    def as_matrix(self):
        R = self.rot.as_matrix()
        t = np.reshape(self.trans, (self.dim - 1, 1))
        bottom = np.append(np.zeros(self.dim - 1), 1.0)
        return np.vstack([np.hstack([R, t]), bottom])

