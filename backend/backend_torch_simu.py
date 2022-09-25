'''
This script is defining the multiplication behaviors.
The input matrix is first translated into binary codes,
and the weight changes for each vector.

v2: exp weights
simu: 1st frame of exp weights
'''
import torch
import os
import sys
sys.path.append('.')
from utility import utils
# import matplotlib.pyplot as plt


class WeightMultiplication_torch():
    def __init__(self, dim_input,
                 dim_hidden, wi_scaling, wh_scaling, avg_node_per_graph, n_bit,
                 noise, gen_weights=True, device='cpu'):

        self.float_type = torch.double
        self.device = device
        # Dimension
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.avg_node_per_graph = avg_node_per_graph

        # Hardware data statistics (only nonzero parts)
        cond_mean = 14.2473
        cond_std = 2.4624


        if gen_weights:
            self.wi_raw = (torch.randn((self.dim_hidden, self.dim_input + 1)) * cond_std) + cond_mean
            self.wh_raw = (torch.randn((self.dim_hidden, self.dim_hidden)) * cond_std) + cond_mean

            self.wi_raw = utils.zero_mask(self.wi_raw, sparsity=0.).to(self.device)
            self.wh_raw = utils.zero_mask(self.wh_raw, sparsity=0.).to(self.device)

        # Scale Wi
        self.wi_scaling = wi_scaling
        self.wi_ori = self.wi_raw * self.wi_scaling

        # Scale Wh
        self.wh_scaling = wh_scaling
        self.wh_ori = self.wh_raw * self.wh_scaling
        # Bit and base
        self.n_bit = None
        self.base = None
        self.update_nbits(n_bit)

        # VMM dynamic scaling origin shift
        self.wi_sum_1d = None
        self.wh_sum_1d = None

        # noise
        self.noise = noise

        # Update dim
        self.update_dim(dim_hidden)

    def update_nbits(self, n_bit):
        # Bit and base
        self.n_bit = n_bit
        self.base = 2 ** torch.arange(self.n_bit).to(self.device)

    def update_weights(self, wi_scaling, wh_scaling, dim_hidden=None):
        self.wi_scaling, self.wh_scaling = wi_scaling, wh_scaling
        self.wi_ori = self.wi_raw * wi_scaling
        self.wh_ori = self.wh_raw * wh_scaling
        if dim_hidden:
            self.update_dim(dim_hidden)

    def update_dim(self, dim_hidden=None):
        # Update hidden dim
        if dim_hidden:
            self.dim_hidden = dim_hidden

        # Dim check
        assert self.wi_ori.shape[0] >= self.dim_hidden
        assert self.wi_ori.shape[1] >= self.dim_input + 1
        assert self.wh_ori.shape[0] >= self.dim_hidden
        assert self.wh_ori.shape[1] >= self.dim_hidden

        # Dimension enforcing
        self.wi = self.wi_ori[0:self.dim_hidden,
                              0:(self.dim_input + 1)]  # Bias +1
        self.wh = self.wh_ori[0:self.dim_hidden, 0:self.dim_hidden]

        # VMM dynamic scaling origin shift
        self.wi_sum_1d = torch.sum(self.wi, dim=1)
        self.wh_sum_1d = torch.sum(self.wh, dim=1)

    def w_multiply(self, in_mat, weight_type):
        # Weight multiplication
        # in_mat: input matrix, one sample per column
        # wi_wh_wphi: 'wi', 'wh'

        assert weight_type in ['wi', 'wh'], 'weight must be wi or wh'

        n_sample_per_batch = in_mat.shape[1]  # Samples per batch

        if weight_type == 'wi':
            w = self.wi
            w_sum_1d = self.wi_sum_1d
            dim_out = self.dim_hidden
        elif weight_type == 'wh':
            w = self.wh
            w_sum_1d = self.wh_sum_1d
            dim_out = self.dim_hidden
        else:
            raise Exception('Must be wi or wh')

        w_non_noise = w

        # Initialize output
        result = torch.zeros(dim_out, n_sample_per_batch).to(self.device)

        # Quantization
        in_mat_q, a, b = self.binarization(in_mat)
        in_mat_q = in_mat_q.to(dtype=self.float_type)

        # Multiplication
        for i_sample in range(n_sample_per_batch):
            for i_bit in range(self.n_bit):

                if self.noise:
                    w = w_non_noise + w_non_noise * torch.randn(w.shape).to(self.device) * self.noise

                result[:, i_sample] += torch.squeeze(
                    torch.mm(w, in_mat_q[:, i_sample, i_bit].reshape(-1, 1)) * self.base[i_bit])

        # Rescale back
        a = a.repeat(dim_out, 1)
        result = result * a + torch.outer(w_sum_1d, b)

        return result

    def binarization(self, input):
        # Binarize matrices
        # input: 2D matrix to be binarized

        # Column-wise min/max of input
        input_min = torch.min(input, dim=0)[0]
        input_max = torch.max(input, dim=0)[0]

        n_levels = 2 ** self.n_bit - 1

        # Scaling coefficient (see output)
        a = (input_max - input_min) / n_levels
        b = input_min

        input_min = input_min.repeat(input.shape[0], 1)
        input_max = input_max.repeat(input.shape[0], 1)

        # Cast to integers
        # note that, though the first line would result in nan value if max == min,
        # the round function will return 0 for nan values.
        input_int = (input - input_min) / (input_max - input_min) * n_levels
        input_int = torch.round(input_int).to(torch.int)
        # Binarize
        input_b = input_int.unsqueeze(-1).bitwise_and(self.base).ne(0).byte()
        return input_b, a, b
