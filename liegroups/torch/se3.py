import torch

from .so3 import SO3Matrix
from .._base import SEMatrixBase


class SE3Matrix(SEMatrixBase):
    dim = 4
    dof = 6
    RotationType = SO3Matrix

    def as_matrix(self):
        R = self.rot.as_matrix()
        if R.dim() < 3:
            R = R.unsqueeze(0)
        t = self.trans
        if t.dim() < 2:
            t = t.unsqueeze(0)
        t = t.unsqueeze(2)
        bottom = t.new_zeros(t.shape[0], 1, self.dim)
        bottom[:, 0, -1] = 1.0
        T = torch.cat([torch.cat([R, t], dim=2), bottom], dim=1)
        return T.squeeze(0) if T.shape[0] == 1 else T

