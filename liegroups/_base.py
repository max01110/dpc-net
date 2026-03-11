from abc import ABCMeta, abstractmethod


class LieGroupBase(metaclass=ABCMeta):
    """Common abstract base class defining basic interface for Lie groups."""

    @property
    @classmethod
    @abstractmethod
    def dim(cls):
        pass

    @property
    @classmethod
    @abstractmethod
    def dof(cls):
        pass


class MatrixLieGroupBase(LieGroupBase):
    @abstractmethod
    def as_matrix(self):
        pass


class SOMatrixBase(MatrixLieGroupBase):
    def __init__(self, mat):
        self.mat = mat


class SEMatrixBase(MatrixLieGroupBase):
    def __init__(self, rot, trans):
        self.rot = rot
        self.trans = trans


class VectorLieGroupBase(LieGroupBase):
    def __init__(self, data):
        self.data = data

