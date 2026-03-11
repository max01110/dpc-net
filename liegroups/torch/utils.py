import torch


def isclose(mat1, mat2, tol=1e-6):
    return (mat1 - mat2).abs_().lt(tol)


def allclose(mat1, mat2, tol=1e-6):
    return isclose(mat1, mat2, tol).all()


def trace(mat):
    if mat.dim() < 3:
        mat = mat.unsqueeze(dim=0)
    eye = torch.eye(mat.shape[1], dtype=mat.dtype, device=mat.device)
    tr = (eye * mat).sum(dim=1).sum(dim=1)
    return tr.view(mat.shape[0])

