from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from scipy.sparse import csc_matrix

from .utils import normalize_adj, sparse_mx_to_torch_sparse_tensor


@dataclass
class GraphData:
    adj_norm: object
    features: np.ndarray
    labels: np.ndarray
    adj_label: object
    test_id: np.ndarray | None


def load_mat(dataset: str, data_dir: str | Path) -> GraphData:
    """Load a .mat graph dataset and return tensors/arrays used by MI-GAD."""
    data_path = Path(data_dir) / f"{dataset}.mat"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    data_mat = sio.loadmat(data_path)
    adj = data_mat["Network"] if "Network" in data_mat else data_mat["A"]
    features = data_mat["Attributes"] if "Attributes" in data_mat else data_mat["X"]
    labels = data_mat["Label"] if "Label" in data_mat else data_mat["gnd"]
    test_id = data_mat["k"].flatten() if dataset == "Elliptic-all" else None

    labels = labels.flatten()
    if dataset in {"Books", "Enron", "twitter"}:
        adj = csc_matrix(adj)

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_norm = normalize_adj(adj)

    return GraphData(
        adj_norm=sparse_mx_to_torch_sparse_tensor(adj_norm),
        features=sp.lil_matrix(features).toarray(),
        labels=labels,
        adj_label=sparse_mx_to_torch_sparse_tensor(adj),
        test_id=test_id,
    )
