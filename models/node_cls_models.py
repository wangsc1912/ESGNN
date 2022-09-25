import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('.')
from tqdm import tqdm
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.utils import add_self_loops, degree
import numpy as np
from utility import utils


class Reservoir(MessagePassing):
    def __init__(self,
                 num_node_feat: int,
                 hid_dim: int,
                 num_class: int,
                 n_iter: int,
                 leaky: float,
                 sparsity= 1.,
                 weight_dist='gaussian',
                 wi_scaling=1.,
                 wh_scaling=1.,
                 hid_init_dist='zero',
                 hid_init_scale=1.,
                 noise=0,
                 self_loop=False) -> None:
        super(Reservoir, self).__init__(aggr='add')

        self.leaky = leaky

        self.num_node_feat = num_node_feat
        self.hid_dim = hid_dim
        self.num_class = num_class
        self.self_loop = self_loop
        self.n_iter = n_iter

        self.noise = noise

        self.weight_dist = weight_dist

        self.sparsity = sparsity

        self.input_layer = nn.Linear(num_node_feat, hid_dim)
        self.hidden_layer = nn.Linear(hid_dim, hid_dim)
        self.input_layer.weight = torch.nn.Parameter(utils.weight_init(self.input_layer.weight, weight_init=weight_dist, sparsity=1), requires_grad=False)
        self.orign_in_weight = self.input_layer.weight
        self.hidden_layer.weight = torch.nn.Parameter(utils.weight_init(self.hidden_layer.weight, weight_init=weight_dist, sparsity=sparsity), requires_grad=False)
        self.orign_hid_weight = self.hidden_layer.weight

        for w in self.input_layer.parameters():
            w.requires_grad = False
        for w in self.hidden_layer.parameters():
            w.requires_grad = False
        self.act = torch.tanh

        self.hid_init_dist = hid_init_dist
        self.hid_init_scale = hid_init_scale

    def reset_parameters(self, noise=0):
        self.input_layer.weight = torch.nn.Parameter(utils.weight_init(self.input_layer.weight, weight_init=self.weight_dist, sparsity=1), requires_grad=False)
        self.orign_in_weight = self.input_layer.weight
        self.hidden_layer.weight = torch.nn.Parameter(utils.weight_init(self.hidden_layer.weight, weight_init=self.weight_dist, sparsity=self.sparsity), requires_grad=False)
        self.orign_hid_weight = self.hidden_layer.weight

    def forward(self, x, edge_index, noise=0.):
        self.noise = noise
        if self.noise:
            input_feat = []
            for i in range(x.shape[0]):
                x_single = x[i]
                self.input_layer.weight = nn.Parameter(utils.weight_add_noise(self.orign_in_weight.data, self.noise), requires_grad=False)
                input_feat_single = self.input_layer(x_single)
                input_feat.append(input_feat_single)
                input_feat_single = 0
            input_feat = torch.stack(input_feat, dim=0)
        else:
            input_feat = self.input_layer(x)
        # add self-loop
        if self.self_loop:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # normalization
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # iteration process
        if self.hid_init_dist == 'zero':
            state_old = torch.zeros_like(input_feat)
        elif self.hid_init_dist == 'gaussian':
            state_old = torch.randn_like(input_feat) * self.hid_init_scale
        elif self.hid_init_dist == 'uniform':
            state_old = torch.rand_like(input_feat) * self.hid_init_scale
        for i_iter in range(self.n_iter):
            if self.noise:
                state_old_out = []
                for i in range(state_old.shape[0]):
                    state_old_single = state_old[i]
                    self.hidden_layer.weight = nn.Parameter(utils.weight_add_noise(self.orign_hid_weight, self.noise), requires_grad=False)
                    state_old_single = self.hidden_layer(state_old_single)
                    state_old_out.append(state_old_single)
                state_old = torch.stack(state_old_out)
            else:
                state_old = self.hidden_layer(state_old)
            # extract neighbor info
            neighbor_info = self.propagate(edge_index, x=state_old, norm=norm)

            # aggregation
            post_activation = self.act(input_feat + neighbor_info)
            state = (1 - self.leaky) * state_old + self.leaky * post_activation
            state_old = state

        return state

    def message(self, x_j) -> torch.Tensor:
        return super().message(x_j)


class Readout(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Readout, self).__init__()
        self.lin = nn.Linear(input_dim, output_dim)

    def reset_parameters(self):
        pass

    def forward(self, x, edge_index=None):
        x = self.lin(x)
        x = F.log_softmax(x, dim=1)
        return x


class NodeClsESGNN(nn.Module):
    def __init__(self, dataset, hidden_dim, n_iter=10, sparsity=1, weight_dist='gaussian', hid_init_dist='zero',
                 hid_init_scale=1., noise=0):
        super(NodeClsESGNN, self).__init__()
        self.num_feat = dataset.num_features
        self.num_class = dataset.num_classes
        self.hidden_dim = hidden_dim
        self.n_iter = n_iter
        self.noise = noise
        self.reservoir = Reservoir(self.num_feat,
                                   self.hidden_dim,
                                   self.num_class,
                                   self.n_iter,
                                   0.2,
                                   sparsity=sparsity,
                                   weight_dist=weight_dist,
                                   hid_init_dist=hid_init_dist,
                                   hid_init_scale=hid_init_scale,
                                   noise=self.noise
                                   )

        # self.readout = Readout(hidden_dim, self.num_class)
        self.readout = CLS(hidden_dim, self.num_class)

        self.layers = [self.reservoir, self.readout]

    def update_params(self, **kwargs):
        for k, v in kwargs.items():
            if k == 'hid_dim':
                self.reservoir = Reservoir(self.num_feat, v, self.num_class, self.n_iter, 0.2)
                self.readout = CLS(v, self.num_class)
            if k == 'n_iter':
                self.reservoir = Reservoir(self.num_feat, self.hidden_dim, self.num_class, v, 0.2)
            if k == 'weight_dist':
                self.reservoir.weight_dist = v
                self.reservoir.reset_parameters()
            if k == 'noise':
                self.noise = v
                self.reservoir.noise = v
        pass

    def reset_parameters(self):
        self.reservoir.reset_parameters(noise=self.noise)
        self.readout.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index.to(data.x.device)
        with torch.no_grad():
            node_embedding = F.relu(self.reservoir(x, edge_index))

        x = self.readout(node_embedding, edge_index)
        return x, node_embedding

    @torch.no_grad()
    def inference(self, x_all, subgraph_loader):
        pbar = tqdm(total=len(subgraph_loader.dataset) * len(self.convs))
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        for i, layer in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(self.device)
                x = layer(x, batch.edge_index.to(self.device))
                # x = F.relu(self.reservoir(x, batch.edge_index.to(self.device)))
                if i < len(self.convs) - 1:
                    x = x.relu_()
                xs.append(x[:batch.batch_size].cpu())
                pbar.update(batch.batch_size)
            x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all


class CLS(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super(CLS, self).__init__()
        self.conv = GCNConv(d_in, d_out, cached=True)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, mask=None):
        x = self.conv(x, edge_index)
        x = F.log_softmax(x, dim=1)
        # x = F.softmax(x, dim=1)
        return x


class CRD(torch.nn.Module):
    def __init__(self, d_in, d_out, p):
        super(CRD, self).__init__()
        self.conv = GCNConv(d_in, d_out, cached=True)
        self.p = p

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, mask=None):
        x = F.relu(self.conv(x, edge_index))
        x = F.dropout(x, p=self.p, training=self.training)
        return x


class GCN(torch.nn.Module):
    def __init__(self, dataset, hidden_dim, dropout):
        super(GCN, self).__init__()
        self.crd = CRD(dataset.num_features, hidden_dim, dropout)
        self.cls = CLS(hidden_dim, dataset.num_classes)

    def reset_parameters(self):
        self.crd.reset_parameters()
        self.cls.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index.to(data.x.device)
        x = self.crd(x, edge_index, data.train_mask)
        x = self.cls(x, edge_index, data.train_mask)
        return x, None
