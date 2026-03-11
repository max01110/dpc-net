import torch

from . import utils
from .._base import SOMatrixBase


class SO3Matrix(SOMatrixBase):
    dim = 3
    dof = 3

    @classmethod
    def wedge(cls, phi):
        if phi.dim() < 2:
            phi = phi.unsqueeze(0)
        Phi = phi.new_zeros(phi.shape[0], cls.dim, cls.dim)
        Phi[:, 0, 1] = -phi[:, 2]
        Phi[:, 1, 0] = phi[:, 2]
        Phi[:, 0, 2] = phi[:, 1]
        Phi[:, 2, 0] = -phi[:, 1]
        Phi[:, 1, 2] = -phi[:, 0]
        Phi[:, 2, 1] = phi[:, 0]
        return Phi.squeeze(0) if Phi.shape[0] == 1 else Phi

    @classmethod
    def vee(cls, Phi):
        if Phi.dim() < 3:
            Phi = Phi.unsqueeze(0)
        phi = Phi.new_zeros(Phi.shape[0], cls.dof)
        phi[:, 0] = Phi[:, 2, 1]
        phi[:, 1] = Phi[:, 0, 2]
        phi[:, 2] = Phi[:, 1, 0]
        return phi.squeeze(0) if phi.shape[0] == 1 else phi

    @classmethod
    def exp(cls, phi):
        if phi.dim() < 2:
            phi = phi.unsqueeze(0)
        angle = phi.norm(p=2, dim=1)
        mat = phi.new_empty(phi.shape[0], cls.dim, cls.dim)
        small = utils.isclose(angle, 0.0)
        small_inds = small.nonzero(as_tuple=False).view(-1)
        large_inds = (~small).nonzero(as_tuple=False).view(-1)

        if small_inds.numel() > 0:
            eye = torch.eye(cls.dim, dtype=phi.dtype, device=phi.device)
            mat[small_inds] = eye + cls.wedge(phi[small_inds])

        if large_inds.numel() > 0:
            angle_l = angle[large_inds]
            axis = phi[large_inds] / angle_l.unsqueeze(1)
            s = angle_l.sin().unsqueeze(1).unsqueeze(2)
            c = angle_l.cos().unsqueeze(1).unsqueeze(2)
            eye = torch.eye(cls.dim, dtype=phi.dtype, device=phi.device).unsqueeze(0)
            eye = eye.expand(large_inds.numel(), cls.dim, cls.dim)
            outer = utils.outer(axis, axis)
            mat[large_inds] = c * eye + (1 - c) * outer + s * cls.wedge(axis)

        return cls(mat.squeeze(0) if mat.shape[0] == 1 else mat)

    def as_matrix(self):
        return self.mat

