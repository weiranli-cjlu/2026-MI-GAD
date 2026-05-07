from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import auc, precision_recall_curve
from sklearn.neighbors import kneighbors_graph


def log(message: str, verbose: bool = True, *, flush: bool = True) -> None:
    """Print only when verbose=True."""
    if verbose:
        print(message, flush=flush)


def ensure_dir(path: str | os.PathLike) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_name: str, verbose: bool = True) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        log("CUDA is not available; falling back to CPU.", verbose)
        return torch.device("cpu")
    return torch.device(device_name)


def normalize_adj(adj: sp.spmatrix) -> sp.coo_matrix:
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def row_normalize(mx: sp.spmatrix) -> sp.spmatrix:
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    return sp.diags(r_inv).dot(mx)


def sparse_mx_to_torch_sparse_tensor(sparse_mx: sp.spmatrix) -> torch.Tensor:
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape).coalesce()


def preprocess_features(features: sp.spmatrix | np.ndarray) -> np.ndarray:
    if sp.issparse(features):
        features = row_normalize(features).todense()
    return np.asarray(features)


def load_knn_adj(features: np.ndarray, k: int = 10) -> torch.Tensor:
    adj = kneighbors_graph(features, k)
    adj = sp.coo_matrix(adj)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_mx_to_torch_sparse_tensor(adj)


def calculate_auprc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    return auc(recall, precision)


def negative_sampling(raw_adj: torch.Tensor) -> torch.Tensor:
    """Sample one negative edge for each positive edge in a sparse adjacency matrix."""
    if not raw_adj.is_sparse:
        raise ValueError("negative_sampling expects a sparse adjacency tensor.")

    device = raw_adj.device
    adj = raw_adj.coalesce()
    indices = adj.indices()
    num_nodes = raw_adj.size(0)
    num_positive = indices.size(1)

    existing_edges = (indices[0] * num_nodes + indices[1]).unique().sort()[0]
    negative_edges = torch.empty((2, num_positive), dtype=torch.long, device=device)

    neg_sample_idx = 0
    while neg_sample_idx < num_positive:
        batch_size = num_positive - neg_sample_idx
        sampled_i = torch.randint(0, num_nodes, (batch_size,), dtype=torch.long, device=device)
        sampled_j = torch.randint(0, num_nodes, (batch_size,), dtype=torch.long, device=device)

        mask = sampled_i < sampled_j
        sampled_i = sampled_i[mask]
        sampled_j = sampled_j[mask]
        if sampled_i.numel() == 0:
            continue

        sampled_encoded = sampled_i * num_nodes + sampled_j
        positions = torch.searchsorted(existing_edges, sampled_encoded)
        valid_positions = positions < existing_edges.size(0)
        not_exist = ~valid_positions
        not_exist[valid_positions] = existing_edges[positions[valid_positions]] != sampled_encoded[valid_positions]

        sampled_i = sampled_i[not_exist]
        sampled_j = sampled_j[not_exist]
        num_new_samples = sampled_i.size(0)
        if num_new_samples > 0:
            end = min(neg_sample_idx + num_new_samples, num_positive)
            take = end - neg_sample_idx
            negative_edges[:, neg_sample_idx:end] = torch.stack([sampled_i[:take], sampled_j[:take]], dim=0)
            neg_sample_idx = end

    neg_values = torch.ones(num_positive, dtype=raw_adj.dtype, device=device)
    return torch.sparse_coo_tensor(negative_edges, neg_values, raw_adj.size()).coalesce()
