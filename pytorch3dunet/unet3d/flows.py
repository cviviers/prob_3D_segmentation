"""
Collection of flow strategies
"""

from __future__ import print_function
import numpy as np
import torch
import sys
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from pytorch3dunet.unet3d.layers import MaskedConv2d, MaskedLinear


# ACTIVATION_DERIVATIVES = {
#     F.elu: lambda x: torch.ones_like(x) * (x >= 0) + torch.exp(x) * (x < 0),
#     torch.tanh: lambda x: 1 - torch.tanh(x) ** 2
# }

# class Planar(nn.Module):
#     def __init__(self, D=6, activation=torch.tanh):
#         super().__init__()
#         self.D = D
#         self.w = nn.Parameter(torch.empty(D))
#         self.b = nn.Parameter(torch.empty(1))
#         self.u = nn.Parameter(torch.empty(D))
#         self.activation = activation
#         self.activation_derivative = ACTIVATION_DERIVATIVES[activation]

#         nn.init.normal_(self.w)
#         nn.init.normal_(self.u)
#         nn.init.normal_(self.b)

#     def forward(self, z: torch.Tensor):
#         lin = (z @ self.w + self.b).unsqueeze(1)  # shape: (B, 1)
#         f = z + self.u * self.activation(lin)  # shape: (B, D)
#         phi = self.activation_derivative(lin) * self.w  # shape: (B, D)
#         log_det = torch.log(torch.abs(1 + phi @ self.u) + 1e-4) # shape: (B,)
        

#         return log_det, f


class Planar(nn.Module):
    """
    PyTorch implementation of planar flows as presented in "Variational Inference with Normalizing Flows"
    by Danilo Jimenez Rezende, Shakir Mohamed. Model assumes amortized flow parameters.
    """

    def __init__(self):

        super(Planar, self).__init__()

        self.h = nn.Tanh()
        self.softplus = nn.Softplus()

    def der_h(self, x):
        """ Derivative of tanh """

        return 1 - self.h(x) ** 2
    
        
    def forward(self, zk, u, w, b):
        """
        Forward pass. Assumes amortized u, w and b. Conditions on diagonals of u and w for invertibility
        will be be satisfied inside this function. Computes the following transformation:
        z' = z + u h( w^T z + b)
        or actually
        z'^T = z^T + h(z^T w + b)u^T
        Assumes the following input shapes:
        shape u = (batch_size, z_size, 1)
        shape w = (batch_size, 1, z_size)
        shape b = (batch_size, 1, 1)
        shape z = (batch_size, z_size).
        """

        zk = zk.unsqueeze(-1)
        bs = u.shape[0]
        total = zk.shape[0]
        latent_dim = zk.shape[1]
        sample_size = total // bs

        if total != bs:
            u = u.unsqueeze(1).repeat(1, sample_size, 1, 1)
            w = w.unsqueeze(1).repeat(1, sample_size, 1, 1)
            b = b.unsqueeze(1).repeat(1, sample_size, 1, 1)
            u = u.reshape(bs*sample_size, latent_dim, 1)
            w = w.reshape(bs*sample_size, 1, latent_dim)
            b = b.reshape(bs*sample_size, 1, 1)

        # reparameterize u such that the flow becomes invertible (see appendix paper)
        uw = torch.bmm(w, u)
        m_uw = -1. + self.softplus(uw)
        w_norm_sq = torch.sum(w ** 2, dim=2, keepdim=True)
        u_hat = u + ((m_uw - uw) * w.transpose(2, 1) / w_norm_sq)

        # compute flow with u_hat
        wzb = torch.bmm(w, zk) + b
        # magnitude_u = torch.sum(torch.abs(u_hat))/u_hat.shape[0]
        # magnitude_wzb = torch.sum(torch.abs(wzb))/wzb.shape[0]
        z = zk + u_hat * self.h(wzb)
        z = z.squeeze(2)
            
        # compute logdetJ
        psi = w * self.der_h(wzb)
        # jacobian = 1 + torch.bmm(torch.bmm(u_hat, self.der_h(torch.bmm(w, zk) + b)), w) 
        log_det_jacobian = torch.log(torch.abs(1 + torch.bmm(psi, u_hat)))
        log_det_jacobian = log_det_jacobian.squeeze(2).squeeze(1)

        return log_det_jacobian, z
        
class Radial(nn.Module):
    """
    PyTorch implementation of planar flows as presented in "Variational Inference with Normalizing Flows"
    by Danilo Jimenez Rezende, Shakir Mohamed. Model assumes amortized flow parameters.
    """

    def __init__(self, shape):

        super(Radial, self).__init__()
        
        self.d_cpu = torch.prod(torch.tensor(shape))
        self.register_buffer('d', self.d_cpu)
        lim = 1.0 / np.prod(shape)


    def forward(self, zk, z0, alpha, beta): #TODO: fix
        """
        Forward pass. Assumes amortized u, w and b. Conditions on diagonals of u and w for invertibility
        will be be satisfied inside this function. Computes the following transformation:
        z' = z + u h( w^T z + b)
        or actually
        z'^T = z^T + h(z^T w + b)u^T
        Assumes the following input shapes:
        shape u = (batch_size, z_size, 1)
        shape w = (batch_size, 1, z_size)
        shape b = (batch_size, 1, 1)
        shape z = (batch_size, z_size).
        """
        
        beta = torch.log(1 + torch.exp(beta)) - torch.abs(alpha)
        zk = zk.unsqueeze(-1)
        bs = alpha.shape[0]
        total = zk.shape[0]
        latent_dim = zk.shape[1]
        sample_size = total // bs
        if total != bs:
            alpha = alpha.unsqueeze(1).repeat(1, sample_size, 1, 1)
            beta = beta.unsqueeze(1).repeat(1, sample_size, 1, 1)
            z0 = z0.unsqueeze(1).repeat(1, sample_size, 1, 1)
            alpha = alpha.reshape(bs*sample_size, 1, 1)
            beta = beta.reshape(bs*sample_size, 1, 1)
            z0 = z0.reshape(bs*sample_size, latent_dim, 1)
        
        dz = zk - z0
        
        r = torch.norm(dz, dim=list(range(1, z0.dim() - 1))).unsqueeze(-1)
        h_arr = beta / (torch.abs(alpha) + r)
        h_arr_ = - beta * r / (torch.abs(alpha) + r) ** 2
        zk = zk + h_arr * dz
        log_det_jacobian = (self.d - 1) * torch.log(1 + h_arr) + torch.log(1 + h_arr + h_arr_)
        #log_det always positive?
        return log_det_jacobian, zk.squeeze(-1)


class Sylvester(nn.Module):
    """
    Sylvester normalizing flow.
    """

    def __init__(self, num_ortho_vecs):

        super(Sylvester, self).__init__()

        self.num_ortho_vecs = num_ortho_vecs

        self.h = nn.Tanh()

        triu_mask = torch.triu(torch.ones(num_ortho_vecs, num_ortho_vecs), diagonal=1).unsqueeze(0)
        diag_idx = torch.arange(0, num_ortho_vecs).long()

        self.register_buffer('triu_mask', Variable(triu_mask))
        self.triu_mask.requires_grad = False
        self.register_buffer('diag_idx', diag_idx)

    def der_h(self, x):
        return self.der_tanh(x)

    def der_tanh(self, x):
        return 1 - self.h(x) ** 2

    def _forward(self, zk, r1, r2, q_ortho, b, sum_ldj=True):
        """
        All flow parameters are amortized. Conditions on diagonals of R1 and R2 for invertibility need to be satisfied
        outside of this function. Computes the following transformation:
        z' = z + QR1 h( R2Q^T z + b)
        or actually
        z'^T = z^T + h(z^T Q R2^T + b^T)R1^T Q^T
        :param zk: shape: (batch_size, z_size)
        :param r1: shape: (batch_size, num_ortho_vecs, num_ortho_vecs)
        :param r2: shape: (batch_size, num_ortho_vecs, num_ortho_vecs)
        :param q_ortho: shape (batch_size, z_size , num_ortho_vecs)
        :param b: shape: (batch_size, 1, self.z_size)
        :return: z, log_det_j
        """

        # Amortized flow parameters
        zk = zk.unsqueeze(1)

        # Save diagonals for log_det_j
        diag_r1 = r1[:, self.diag_idx, self.diag_idx]
        diag_r2 = r2[:, self.diag_idx, self.diag_idx]

        r1_hat = r1
        r2_hat = r2

        qr2 = torch.bmm(q_ortho, r2_hat.transpose(2, 1))
        qr1 = torch.bmm(q_ortho, r1_hat)

        r2qzb = torch.bmm(zk, qr2) + b
        z = torch.bmm(self.h(r2qzb), qr1.transpose(2, 1)) + zk
        z = z.squeeze(1)

        # Compute log|det J|
        # Output log_det_j in shape (batch_size) instead of (batch_size,1)
        diag_j = diag_r1 * diag_r2
        diag_j = self.der_h(r2qzb).squeeze(1) * diag_j
        diag_j += 1.
        log_diag_j = diag_j.abs().log()

        if sum_ldj:
            log_det_j = log_diag_j.sum(-1)
        else:
            log_det_j = log_diag_j

        return z, log_det_j

    def forward(self, zk, r1, r2, q_ortho, b, sum_ldj=True):

        return self._forward(zk, r1, r2, q_ortho, b, sum_ldj)


class IAF(nn.Module):
    """
    PyTorch implementation of inverse autoregressive flows as presented in
    "Improving Variational Inference with Inverse Autoregressive Flow" by Diederik P. Kingma, Tim Salimans,
    Rafal Jozefowicz, Xi Chen, Ilya Sutskever, Max Welling.
    Inverse Autoregressive Flow with either MADE MLPs or Pixel CNNs. Contains several flows. Each transformation
     takes as an input the previous stochastic z, and a context h. The structure of each flow is then as follows:
     z <- autoregressive_layer(z) + h, allow for diagonal connections
     z <- autoregressive_layer(z), allow for diagonal connections
     :
     z <- autoregressive_layer(z), do not allow for diagonal connections.
     Note that the size of h needs to be the same as h_size, which is the width of the MADE layers.
     """

    def __init__(self, z_size, num_flows=2, num_hidden=0, h_size=50, forget_bias=1., conv2d=False):
        super(IAF, self).__init__()
        self.z_size = z_size
        self.num_flows = num_flows
        self.num_hidden = num_hidden
        self.h_size = h_size
        self.conv2d = conv2d
        if not conv2d:
            ar_layer = MaskedLinear
        else:
            ar_layer = MaskedConv2d
        self.activation = torch.nn.ELU
        # self.activation = torch.nn.ReLU

        self.forget_bias = forget_bias
        self.flows = []
        self.param_list = []

        # For reordering z after each flow
        flip_idx = torch.arange(self.z_size - 1, -1, -1).long()
        self.register_buffer('flip_idx', flip_idx)

        for k in range(num_flows):
            arch_z = [ar_layer(z_size, h_size), self.activation()]
            self.param_list += list(arch_z[0].parameters())
            z_feats = torch.nn.Sequential(*arch_z)
            arch_zh = []
            for j in range(num_hidden):
                arch_zh += [ar_layer(h_size, h_size), self.activation()]
                self.param_list += list(arch_zh[-2].parameters())
            zh_feats = torch.nn.Sequential(*arch_zh)
            linear_mean = ar_layer(h_size, z_size, diagonal_zeros=True)
            linear_std = ar_layer(h_size, z_size, diagonal_zeros=True)
            self.param_list += list(linear_mean.parameters())
            self.param_list += list(linear_std.parameters())

            if torch.cuda.is_available():
                z_feats = z_feats.cuda()
                zh_feats = zh_feats.cuda()
                linear_mean = linear_mean.cuda()
                linear_std = linear_std.cuda()
            self.flows.append((z_feats, zh_feats, linear_mean, linear_std))

        self.param_list = torch.nn.ParameterList(self.param_list)

    def forward(self, z, h_context):

        logdets = 0.
        for i, flow in enumerate(self.flows):
            if (i + 1) % 2 == 0 and not self.conv2d:
                # reverse ordering to help mixing
                z = z[:, self.flip_idx]

            h = flow[0](z)
            h = h + h_context
            h = flow[1](h)
            mean = flow[2](h)
            gate = F.sigmoid(flow[3](h) + self.forget_bias)
            z = gate * z + (1 - gate) * mean
            logdets += torch.sum(gate.log().view(gate.size(0), -1), 1)
        return z, logdets