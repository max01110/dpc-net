import torch
import torch.nn as nn
from lie_algebra import *


def _assert_no_grad(variable):
    assert not variable.requires_grad, (
        "nn criterions don't compute the gradient w.r.t. targets - please "
        "mark these variables as not requiring gradients"
    )


class SO3GeodesicLossFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, target_C_inv, precision):
        ctx.save_for_backward(input, target_C_inv, precision)
        num_samples = input.size(0)
        phi_star = so3_log(target_C_inv)
        f_phi = so3_log(so3_exp(input).bmm(target_C_inv))

        loss = 0.5 * (f_phi.mm(precision).mm(f_phi.t())).trace() \
             - 0.5 * (phi_star.mm(precision).mm(phi_star.t())).trace()
        loss *= (1.0 / num_samples)
        return input.new([loss])

    @staticmethod
    def backward(ctx, grad_output):
        input, target_C_inv, precision = ctx.saved_tensors
        batch_size = input.size(0)

        f_phi = so3_log(so3_exp(input).bmm(target_C_inv))
        so3_log_jacobs = so3_inv_left_jacobian(f_phi).bmm(so3_left_jacobian(input))

        f_phi = f_phi.view(-1, 1, 3)
        grad_losses = f_phi.bmm(precision.expand_as(so3_log_jacobs)).bmm(so3_log_jacobs)
        grad_loss = grad_losses.view(batch_size, 3)

        grad_loss *= (1.0 / batch_size)

        out = grad_output.expand_as(grad_loss) * grad_loss
        return out, None, None


class SO3GeodesicLoss(nn.Module):
    def __init__(self):
        super(SO3GeodesicLoss, self).__init__()

    def forward(self, input, target_C_inv, precision):
        _assert_no_grad(target_C_inv)
        _assert_no_grad(precision)
        return SO3GeodesicLossFn.apply(input, target_C_inv, precision)


class SE3GeodesicLossFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, target_T_inv, precision):
        ctx.save_for_backward(input, target_T_inv, precision)
        num_samples = input.size(0)

        xi_star = se3_log(target_T_inv)
        g_xi = se3_log(se3_exp(input).bmm(target_T_inv))

        loss_corr = (0.5 / num_samples) * (g_xi.mm(precision).mm(g_xi.t())).trace()
        loss_base = (0.5 / num_samples) * (xi_star.mm(precision).mm(xi_star.t())).trace()

        loss = loss_corr - loss_base
        return input.new([loss])

    @staticmethod
    def backward(ctx, grad_output):
        input, target_T_inv, precision = ctx.saved_tensors
        batch_size = input.size(0)

        logs = se3_log(se3_exp(input).bmm(target_T_inv))
        se3_log_jacobs = se3_inv_left_jacobian(logs).bmm(se3_left_jacobian(input))

        logs = logs.view(-1, 1, 6)
        grad_losses = logs.bmm(precision.expand_as(se3_log_jacobs)).bmm(se3_log_jacobs)
        grad_loss = grad_losses.view(batch_size, 6)

        grad_loss *= (1.0 / batch_size)

        out = grad_output.expand_as(grad_loss) * grad_loss
        return out, None, None


class SE3GeodesicLoss(nn.Module):
    def __init__(self):
        super(SE3GeodesicLoss, self).__init__()

    def forward(self, input, target_T_inv, precision):
        _assert_no_grad(target_T_inv)
        _assert_no_grad(precision)
        return SE3GeodesicLossFn.apply(input, target_T_inv, precision)


def compute_loss_rot(image_quad, target, model, loss_fn, precision, config, mode='train'):
    use_cuda = config.get('use_cuda', torch.cuda.is_available()) and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    target_C_inv = target.transpose(1, 2).contiguous().to(device)
    precision_dev = precision.to(device)
    img_1 = image_quad[0].to(device)
    img_2 = image_quad[2].to(device)

    output = model(img_1, img_2)
    loss = loss_fn(output, target_C_inv, precision_dev)

    return loss, output


def compute_loss_yaw(image_quad, target, model, loss_fn, precision, config, mode='train'):
    use_cuda = config.get('use_cuda', torch.cuda.is_available()) and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    target_yaw = target.to(device)
    img_1 = image_quad[0].to(device)
    img_2 = image_quad[2].to(device)

    output = model(img_1, img_2)
    loss = loss_fn(output, target_yaw)

    return loss, output


def compute_loss(image_quad, target, model, loss_fn, precision, config, mode='train', debug=False):
    use_cuda = config.get('use_cuda', torch.cuda.is_available()) and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    target_T_inv = se3_inv(target).to(device)
    precision_dev = precision.to(device)
    stereo_img_1 = torch.cat((image_quad[0], image_quad[1]), 1).to(device)
    stereo_img_2 = torch.cat((image_quad[2], image_quad[3]), 1).to(device)

    output = model(stereo_img_1, stereo_img_2)
    loss = loss_fn(output, target_T_inv, precision_dev)

    if debug:
        print('loss: {}'.format(loss.item()))

    return loss, output
