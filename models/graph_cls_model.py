import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import mm, cat
import numpy as np


class GraphClsESGNN(nn.Module):
    def __init__(self,
                 dim_hidden,
                 n_iter,
                 leaky,
                 lambdas,
                 bias,
                 backend,
                 gcn=False,
                 pooling='sum',
                 activation=torch.tanh):
        # Initialize ESN network
        # dim_hidden : ESN hidden dimension
        # n_iter : no. of iterations
        # leaky : ESN neuron leaky rate
        # lamdas : pseudo inverse denominator residual
        # bias : constant bias amplitude
        # backend : VMM provider

        super(GraphClsESGNN, self).__init__()

        # Net dimension
        self.dim_hidden = dim_hidden

        # Other parameters
        self.n_iter = n_iter
        self.leaky = leaky
        self.activation = activation
        self.lambdas = lambdas

        # GCN_flag
        self.gcn = gcn
        # Wi/Wh multiplication backend
        self.backend = backend
        self.device = self.backend.device

        # Bias amplitude
        self.bias = torch.tensor([bias], device=self.device)

        # Readout weight
        self.w_readout = []

        self.pooling = pooling
        self.last_activation = torch.relu


    def update_params(self, **kwargs):
        for key, value in kwargs.items():
            if key == 'num_iter':
                self.n_iter = value
            elif key == 'leaky':
                self.leaky = value
            elif key == 'bias':
                self.bias = value
            elif key == 'hidden_dim':
                self.dim_hidden = value
            elif key == 'pooling':
                self.pooling = value
            elif key == 'last_act':
                self.last_activation = value

    def batch_embed(self, list_Ui, list_A, targets, idx, rand_state_init=False, rand_init_ratio=1):
        # Batch graph embedding
        # list_Ui: list of Ui
        # list_A: list of A
        if len(targets.shape) > 1:
            targets = np.argmax(targets, axis=1)
        else:
            targets = np.where(targets == -1, 0, targets)

        # Embeddings
        embeddings = []

        # Embed all graphs
        for i, (Ui_single, A_single, target, idx) in enumerate(zip(list_Ui, list_A, targets, idx)):
            embedding = self.forward(Ui_single, A_single, target, idx, rand_state_init, rand_init_ratio)
            embeddings.append(embedding)

        embeddings = torch.squeeze(torch.stack(embeddings, dim=-1))
        embeddings_b = cat((embeddings, self.bias * torch.ones(1, embeddings.shape[1], device=self.device)), dim=0)

        return embeddings_b

    def forward(self, Ui, A, target=[], idx=[], rand_state_init=False, rand_init_ratio=1):
        # Graph embedding with ESN
        # Ui: Node input vectors (one vector per column)
        # A: Adjacency matrix

        # number of vertices in the graph
        N = A.shape[0]

        # Input Matrix
        Ui_b = torch.cat((Ui, self.bias * torch.ones(1, N).to(self.device)), dim=0)  # Include bias
        WiU = self.backend.w_multiply(Ui_b, weight_type='wi')



        if self.gcn:
            A = A + torch.eye(A.shape[0])
            # degree matrix
            D_array = torch.sum(A, dim=1)
            # normalized adj mat
            D_inter = (1 / torch.sqrt(D_array)) * torch.eye(A.shape[0])
            A = torch.mm(torch.mm(D_inter, A), D_inter)
            state_old = WiU
        else:
            state_old = torch.zeros(self.dim_hidden, N)  # Initial state : 0
            if rand_state_init == 'gaussian':
                state_old = torch.randn_like(state_old) * rand_init_ratio

            elif rand_state_init == 'uniform':
                state_old = torch.rand_like(state_old) * rand_init_ratio
        state_old = state_old.to(self.device)

        # Recurrent Matrix
        for i in range(self.n_iter):
            mul_output = self.backend.w_multiply(state_old, weight_type='wh')
            if self.gcn:
                pre_activation = torch.mm(mul_output, A)
            else:
                wxa = torch.mm(mul_output, A)
                pre_activation = WiU + wxa
            post_activation = self.activation(pre_activation)
            state = state_old * (1 - self.leaky) + post_activation * self.leaky
            state_old = state

        if self.n_iter == 0:
            state = WiU

        # pooling
        if self.pooling == 'sum':
            pooled_vec = torch.sum(state, dim=1, keepdim=True)
        elif self.pooling == 'mean':
            pooled_vec = torch.mean(state, dim=1, keepdim=True)
        elif self.pooling == 'max':
            pooled_vec = torch.max(state, dim=1, keepdim=True).values
        elif self.pooling == 'mean_max_cat':
            pooled_vec = torch.cat((torch.max(state, dim=1, keepdim=True).values, torch.mean(state, dim=1, keepdim=True)), dim=0)

        # activation and get final embedding
        if self.last_activation is not None:
            embedding = self.last_activation(pooled_vec)
        else:
            embedding = pooled_vec

        return embedding

    def train_only(self, embeddings_b, tr_targets):
        # Train readout use linear regression
        # embeddings_b: embeddings with bias
        # tr_targets: target readout outputs

        # Linear regression
        A = mm(tr_targets, torch.transpose(embeddings_b, dim0=0, dim1=1))
        B = mm(embeddings_b, torch.transpose(embeddings_b, dim0=0, dim1=1))
        B_inv = torch.inverse(B + self.lambdas * torch.eye(B.shape[0]))
        self.w_readout = mm(A, B_inv)
        return mm(self.w_readout, embeddings_b)

    def test_only(self, embeddings_b):
        # Readout forward pass
        # embeddings_b: embeddings with bias

        return mm(self.w_readout, embeddings_b)


class Readout(nn.Module):
    def __init__(self, dim_in, dim_out, lambdas, aggregation='sum', bias=True, device='cpu'):
        super(Readout, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.aggregation = aggregation
        self.bias = bias
        # self.w_readout = nn.Linear(dim_in, dim_out, bias=bias)
        self.w_readout = []
        self.lambdas = lambdas

    def batch_embed(self, list_Ui, list_A, targets, idx, list_plts=[]):
        # Batch graph embedding
        # list_Ui: list of Ui
        # list_A: list of A
        if len(targets.shape) > 1:
            targets = np.argmax(targets, axis=1)
        else:
            targets = np.where(targets == -1, 0, targets)

        # Embeddings
        embeddings = []

        # Embed all graphs
        for i, (Ui_single, A_single, target, idx) in enumerate(zip(list_Ui, list_A, targets, idx)):
            # embedding = self.forward(Ui_single, A_single, target, idx, plot_states=(i in list_plts))
            embedding = self.embed(Ui_single, A_single, target, idx, edge=False)
            embeddings.append(embedding)
        # for i, (Ui_single, A_single) in enumerate(zip(list_Ui, list_A)):
        #     embedding = self.forward(Ui_single, A_single, plot_states=(i in list_plts))
        #     embeddings.append(embedding)
        embeddings = torch.stack(embeddings, dim=-1)
        if embeddings.shape[0] != 1:
            embeddings = torch.squeeze(embeddings)
        embeddings_b = cat((embeddings, self.bias * torch.ones(1, embeddings.shape[1])), dim=0)
        # torch.save({
        #     'embebddings_b': embeddings_b.cpu()
        # }, 'exp_data_plots/embeddings.pt')
        # print('embedding saved.')
        return embeddings_b

    def embed(self, nodes, A_single, target, idx, edge=False):
        # Readout forward pass
        # embeddings_b: embeddings with bias
        if not edge:
            if self.aggregation == 'sum':
                x = torch.sum(nodes, dim=1)
            elif self.aggregation == 'mean':
                x = torch.mean(nodes, dim=1)
            elif self.aggregation == 'max':
                x = torch.mean(nodes, dim=1)
        else:
            nodes = mm(nodes, A_single)
            if self.aggregation == 'sum':
                x = torch.sum(nodes, dim=1)
            elif self.aggregation == 'mean':
                x = torch.mean(nodes, dim=1)
            elif self.aggregation == 'max':
                x = torch.mean(nodes, dim=1)

        # x = self.w_readout(x)
        #TODO: check the dim
        # x = F.softmax(x, dim=0)
        return x

    def forward(self, x):
        # Readout forward pass
        # embeddings_b: embeddings with bias

        return mm(self.w_readout, x)

    def train_only(self, embeddings_b, tr_targets):
        # Train readout use linear regression
        # embeddings_b: embeddings with bias
        # tr_targets: target readout outputs

        # Linear regression
        A = mm(tr_targets, torch.transpose(embeddings_b, dim0=0, dim1=1))
        B = mm(embeddings_b, torch.transpose(embeddings_b, dim0=0, dim1=1))
        B_inv = torch.inverse(B + self.lambdas * torch.eye(B.shape[0]))
        self.w_readout = mm(A, B_inv)
        return mm(self.w_readout, embeddings_b)
