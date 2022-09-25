import torch
import numpy as np
import os
import time
from datetime import datetime


def split_wrt_masks(tr_masks: list, te_masks: list, embeddings: torch.tensor, target: np.ndarray):
    '''
    tr_masks: list or ndarray with dtype(int)
    te_masks: list or ndarray with dtype(int)

    '''
    if type(tr_masks) == np.ndarray:
        tr_masks = tr_masks.astype(np.int)
    if type(te_masks) == np.ndarray:
        te_masks = te_masks.astype(np.int)
    tr_embeddings = embeddings[:, tr_masks]
    te_embeddings = embeddings[:, te_masks]

    # tr_targets = torch.Tensor([target[i] for i in tr_masks])
    # te_targets = torch.Tensor([target[i] for i in te_masks])
    tr_targets = torch.tensor(target[tr_masks], dtype=torch.double)
    te_targets = torch.tensor(target[te_masks], dtype=torch.double)
    if target.ndim == 1:  # Binary
        tr_targets = torch.unsqueeze(tr_targets, dim=0)
        te_targets = torch.unsqueeze(te_targets, dim=0)
    else:
        tr_targets = tr_targets.T
        te_targets = te_targets.T

    return tr_embeddings, te_embeddings, tr_targets, te_targets


def onehot2digital(label, dim):
    # [label, n_graph]
    return np.argmax(label, axis=dim)


def tensor2array(tensor):
    if tensor.is_cuda:
        tensor = tensor.cpu().numpy()
    else:
        tensor = tensor.numpy()
    return tensor


def to_array(mat):
    if type(mat) == torch.Tensor:
        return tensor2array(mat)
    elif type(mat) == np.ndarray:
        return mat


def cross_validation_split(origin_idx, num_fold, shuffle=False):
    if shuffle:
        np.random.shuffle(origin_idx)
    num_data = len(origin_idx)
    num_per_fold = int(np.ceil(num_data / num_fold))
    idx_fold_list = [origin_idx[fold_idx * num_per_fold: (fold_idx + 1) * num_per_fold] for fold_idx in range(num_fold)]
    train_samples, test_samples = [], []
    for fold in range(num_fold):
        idx_fold_list_new = idx_fold_list.copy()
        idx_fold_list_new.pop(fold)
        train_data_fold = []
        for lst in idx_fold_list_new:
            train_data_fold.extend(lst)
        train_samples.append(train_data_fold)
        test_samples.append(idx_fold_list[fold])

    return train_samples, test_samples


def zero_mask_torch(tensor, zero_ratio=0.5, non_negative=False):
    mask = torch.rand_like(tensor) - zero_ratio 
    mask = torch.where(mask <= 0, torch.tensor(0., dtype=mask.dtype), 1)
    # mask = torch.where(mask > 0, torch.tensor(1., dtype=mask.dtype), mask)
    return mask * tensor


def zero_mask(tensor, sparsity=0.5, non_negative=False):
    mask = torch.randn_like(tensor)
    idx = int(len(mask.flatten()) * sparsity)
    mask_sorted, _ = mask.flatten().sort()
    mask_sparse = torch.where(mask >= mask_sorted[idx], 1., 0.)
    # mask_sparse = torch.where(mask_sparse <= mask_sorted[idx], 1., mask_sparse)
    return mask_sparse * tensor


def empty_dir(folder):
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))


def weight_init(tensor, weight_init='gaussian', a=0, b=1, sparsity=1):
    with torch.no_grad():
        if weight_init == 'gaussian':
            tensor = tensor.normal_(a, b)
        if weight_init == 'uniform':
            a, b = -1, 1
            tensor = tensor.uniform_(a, b)

        sparsity_mask = torch.where(torch.rand_like(tensor) < sparsity, True, False)
        tensor = tensor * sparsity_mask
        return tensor


def get_time():
    t = time.time()
    dt = datetime.fromtimestamp(t)
    dt = datetime.strftime(dt, '%m%d%H%M')
    return dt


def filename_with_time(name: str, save_path: str, format='') -> str:
    dt = get_time()
    filename = f'{name}_{dt}'
    if format:
        filename = filename + '.' + format
    filename = os.path.join(save_path, filename)
    return filename
