import torch
import argparse
# from backend.backend_hardware import WeightMultiplication     # for random resistive array hardware calls via pynq
import backend.backend_torch_simu as backend_torch_simu
from models.graph_cls_model import GraphClsESGNN
from utility.graph_cls_data_proc import load_dataset
import numpy as np
import utility.utils as utils
from utility.utils import split_wrt_masks
import os
import itertools


# configurations
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0', choices=['cpu', 'cuda:0', 'cuda:1'],
                        help="choose cuda device")
    parser.add_argument('--dataset', type=str, default='mutag', choices=[
                        'collab', 'mutag'], help="choose dateset")
    parser.add_argument('--dim_hidden', type=int, default=50,
                        help='Hidden dimension of the kernel')
    parser.add_argument('--wi_scaling', type=float,
                        default=0.0016, help="Input_scaling")
    parser.add_argument('--wh_scaling', type=float, default=0.006,
                        help="Recurrent_scaling")
    parser.add_argument('--n_bit', type=int, default=4, help="Number of bits")
    parser.add_argument('--lambdas', type=float,
                        default=1e-3, help='lambda for regression')
    parser.add_argument('--leaky', type=float, default=0.2,
                        help="ESN leaky parameter")
    parser.add_argument('--n_iter', type=int, default=4, help="ESN iterations")
    parser.add_argument('--is_plt', type=str, default=False,
                        help="plot figures")
    parser.add_argument('--simulation', type=str, default=True,
                        help="Simulation or Experment")
    parser.add_argument('--noise', type=float, default=0.02,
                        help="Noise level (only for simulation)")
    parser.add_argument('--save_fmt', type=str,
                        default='pdf', help='figure save format')
    args = parser.parse_args()

    dataset = args.dataset
    if args.simulation:
        args.data_root_wh = f'./weights/{dataset}_selected_4b/wh'
        args.data_root_wi = f'./weights/{dataset}_selected_4b/wi'
    elif dataset == 'collab':
        args.data_root_wh = f'./weights/{dataset}_selected_4b/wh/pos'
        args.data_root_wi = f'./weights/{dataset}_selected_4b/wi/pos'
    elif dataset == 'mutag':
        args.data_root_wh = f'./weights/{dataset}_selected_4b/wh/neg'
        args.data_root_wi = f'./weights/{dataset}_selected_4b/wi/neg'

    return args


options = parse_args()
if options.dataset == 'collab':
    size_limit = 200
else:
    size_limit = 0
print(options)

# Torch setting
torch.set_default_tensor_type(torch.cuda.DoubleTensor)

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
device = torch.device(options.device if torch.cuda.is_available() else 'cpu')
torch.device(device)

# Datasets
dim_input, avg_node_per_graph, adj_mats, node_labels, target, tr_samples, test_samples, nested_trs, nested_vals, idx = \
    load_dataset(options.dataset, sparse=False, size_limit=(200 if options.dataset == 'collab' else 0), device=device)

if options.dataset == 'mutag':
    num_class = 2
elif options.dataset == 'collab':
    num_class = 3
num_sample = len(adj_mats)
dataset_info = (adj_mats, node_labels, target)
idx = list(range(num_sample))

if options.simulation:
    WeightMultiplication = backend_torch_simu.WeightMultiplication_torch
# else:
    # For in-memory computing using hardware-implemented random weight matrices (via pynq.dma to command the Xilinx FPGA)
    # WeightMultiplication = WeightMultiplication


def create_and_train_esn(options,
                         dataset_info,
                         state_init='gaussian',
                         init_ratio=0):
    # Load dataset
    adj_mats, node_labels, target = dataset_info

    # Wi_Wh multiplication backend
    backend_torch = WeightMultiplication(dim_input=dim_input,
                                         dim_hidden=options.dim_hidden,
                                         wi_scaling=options.wi_scaling,
                                         wh_scaling=options.wh_scaling,
                                         avg_node_per_graph=avg_node_per_graph,
                                         n_bit=options.n_bit,
                                         noise=options.noise,
                                         gen_weights=True,
                                         device=device)

    # Model
    model = GraphClsESGNN(dim_hidden=options.dim_hidden,
                        n_iter=options.n_iter,
                        leaky=options.leaky,
                        lambdas=options.lambdas,
                        bias=1,
                        backend=backend_torch)
    model.to(device)

    model.backend = backend_torch

    # embeddings_b: [hid_dim + 1, num_graph]
    embeddings_b = model.batch_embed(
        node_labels, adj_mats, target, idx, rand_state_init=state_init, rand_init_ratio=init_ratio)

    # Random division train/test sets
    train_accs = []
    test_accs = []

    for i in range(len(tr_samples)):

        # Train/Test partition for cross validation
        tr_embeddings_b, te_embeddings_b, tr_targets, te_targets = split_wrt_masks(
            tr_samples[i], test_samples[i], embeddings_b, target)

        # Train
        train_output = model.train_only(tr_embeddings_b, tr_targets)
        train_acc = torch.sum(
            tr_targets == train_output.sign()) / tr_targets.shape[-1]
        if options.dataset == 'collab':
            train_acc = torch.sum(torch.argmax(train_output, 0) == torch.argmax(
                tr_targets, 0)) / tr_targets.shape[-1]

        # Test
        te_output = model.test_only(te_embeddings_b)
        te_acc = torch.sum(torch.sign(te_output) ==
                           te_targets) / te_targets.shape[-1]

        # Adjust collab label to non-one-hot type
        if options.dataset == 'collab':
            te_acc = torch.sum(torch.argmax(te_output, dim=0) == torch.argmax(
                te_targets, dim=0)) / te_targets.shape[-1]
            te_targets = torch.argmax(te_targets, dim=0)
            te_output = torch.argmax(te_output, dim=0)
            tr_targets = torch.argmax(tr_targets, dim=0)
            train_output = torch.argmax(train_output, dim=0)

        train_accs.append(train_acc.cpu().numpy())
        test_accs.append(te_acc.cpu().numpy())

    return test_accs, train_accs


if __name__ == "__main__":
    # Build model and train
    test_accs, _ = create_and_train_esn(
        options, dataset_info, state_init='gaussian', init_ratio=0)
    test_acc_list = [acc.item() for acc in test_accs]
    print(f'fold-wis accuracy: {test_acc_list}')
    print(f'10-fold avearge test accuracy: {np.mean(test_accs) * 100:.2f}%')
