from __future__ import annotations

import torch
import torch.nn.functional as F


def local_affinity(features: torch.Tensor, adj_matrix: torch.Tensor, dataset: str) -> torch.Tensor:
    if adj_matrix.is_sparse:
        row_sum = torch.sparse.sum(adj_matrix, dim=1).to_dense()
        agg_features = torch.sparse.mm(adj_matrix, features if dataset in {"Elliptic-all", "dgraphfin"} else F.normalize(features, p=2, dim=1))
    else:
        row_sum = adj_matrix.sum(dim=1)
        agg_features = torch.matmul(adj_matrix, features if dataset in {"Elliptic-all", "dgraphfin"} else F.normalize(features, p=2, dim=1))

    row_sum = torch.clamp(row_sum, min=1e-30)
    normalized = features if dataset in {"Elliptic-all", "dgraphfin"} else F.normalize(features, p=2, dim=1)
    return (normalized * agg_features).sum(dim=1) / row_sum


def global_inconsistency_loss(z: torch.Tensor, test_id, dataset: str) -> tuple[torch.Tensor, torch.Tensor]:
    z = F.normalize(z, p=2, dim=-1)
    center = z[test_id].mean(dim=0) if dataset in {"Elliptic-all", "dgraphfin"} and test_id is not None else z.mean(dim=0)
    loss_per_sample = (z - center).pow(2).sum(dim=1)
    return loss_per_sample, loss_per_sample.mean()
