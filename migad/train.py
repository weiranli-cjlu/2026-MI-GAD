from __future__ import annotations

import csv
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from pandas import DataFrame
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler

from .config import Config
from .data import load_mat
from .losses import global_inconsistency_loss, local_affinity
from .models import GAD
from .utils import (
    calculate_auprc,
    ensure_dir,
    log,
    negative_sampling,
    resolve_device,
    set_seed,
)


def minmax_1d(values: np.ndarray) -> np.ndarray:
    return MinMaxScaler().fit_transform(values.reshape(-1, 1)).reshape(-1)


def build_score(
    sc1: torch.Tensor, global_loss: torch.Tensor, alpha: float, beta: float
) -> np.ndarray:
    score1 = minmax_1d((-sc1).detach().cpu().numpy())
    score2 = minmax_1d(global_loss.detach().cpu().numpy())
    return score1 * alpha + score2 * beta


def evaluate_auc(labels: np.ndarray, score: np.ndarray, dataset: str, test_id) -> float:
    if dataset in {"dgraphfin", "Elliptic-all"} and test_id is not None:
        return roc_auc_score(labels[test_id], score[test_id])
    return roc_auc_score(labels, score)


def train_one_run(
    config: Config, run_idx: int, data, device: torch.device
) -> dict[str, float]:
    set_seed(run_idx)
    log(f"\n# Run: {run_idx}", config.verbose)

    features = torch.as_tensor(data.features, dtype=torch.float32, device=device)
    if config.dataset in {"Reddit"}:
        features = (features - features.mean(0)) / (features.std(0) + 1e-30)

    adj_label = data.adj_label.to(device)
    num_nodes = features.shape[0]
    labels = data.labels

    model = GAD(
        feat_size=features.size(1),
        hidden_size=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    bce = nn.BCEWithLogitsLoss(reduction="none")

    best_loss = float("inf")
    best_epoch = 0
    wait = 0
    max_eval_auc = 0.0
    checkpoint = ensure_dir(config.output_dir) / f"best_model.pth"

    start_time = time.time()
    loop = range(config.epoch)
    if config.tqdm:
        loop = tqdm(loop, desc="Epoch", position=1, leave=False)
    for epoch in loop:
        model.train()
        optimizer.zero_grad()

        node_embed = model(features)
        sc_pos = local_affinity(node_embed, adj_label, config.dataset)
        neg_adj = negative_sampling(adj_label)
        sc_neg = local_affinity(node_embed, neg_adj, config.dataset)
        global_loss_sample, global_loss = global_inconsistency_loss(
            node_embed, data.test_id, config.dataset
        )

        pred = torch.cat([sc_pos, sc_neg], dim=0)
        target = torch.cat([torch.ones(num_nodes), torch.zeros(num_nodes)], dim=0).to(
            device
        )
        affinity_loss = bce(pred, target).mean()
        loss = affinity_loss * config.alpha + global_loss * config.beta

        score = build_score(sc_pos, global_loss_sample, config.alpha, config.beta)
        epoch_auc = evaluate_auc(labels, score, config.dataset, data.test_id)

        if loss.item() < best_loss and epoch > config.epoch // 2:
            best_loss = loss.item()
            best_epoch = epoch
            wait = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            wait += 1

        if wait == config.patience:
            log("Early stopping!", config.verbose)
            break

        loss.backward()
        optimizer.step()

        log(
            f"Epoch: {epoch:04d} train_loss={loss.item():.5f} auc={epoch_auc:.6f}",
            config.verbose,
        )

    log(f"Training time: {time.time() - start_time:.2f}s", config.verbose)
    log(f"Loading {best_epoch}th epoch", config.verbose)
    if checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=device))

    scores = np.zeros((config.tests, num_nodes))
    run_aucs: list[float] = []
    for test_idx in range(config.tests):
        model.eval()
        with torch.no_grad():
            node_embed = model(features)
            sc_pos = local_affinity(node_embed, adj_label, config.dataset)
            global_loss_sample, _ = global_inconsistency_loss(
                node_embed, data.test_id, config.dataset
            )
            score = build_score(sc_pos, global_loss_sample, config.alpha, config.beta)
            scores[test_idx] = score
            auc_score = evaluate_auc(labels, score, config.dataset, data.test_id)
            run_aucs.append(auc_score)
            max_eval_auc = max(max_eval_auc, auc_score)
            log(
                f"Test: {test_idx:04d} Auc: {auc_score:.6f} Best_Auc: {max_eval_auc:.6f}",
                config.verbose,
            )

    final_score = scores.mean(axis=0)
    auc_score = evaluate_auc(labels, final_score, config.dataset, data.test_id)
    auprc = calculate_auprc(labels, final_score)
    log(f"auprc: {auprc:.6f}", config.verbose)
    log(f"auc: {auc_score:.6f}", config.verbose)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "auc": auc_score,
        "auprc": auprc,
        "best_epoch": float(best_epoch),
        "best_loss": best_loss,
    }


def append_result_csv(
    result_csv: str | Path, config: Config, summary: dict[str, object]
) -> None:
    csv_path = Path(result_csv).expanduser()
    if csv_path.parent != Path("."):
        ensure_dir(csv_path.parent)

    row = {
        "datetime": datetime.now().isoformat(timespec="minutesminutes"),
        "dataset": config.dataset,
        "trials": int(config.runs),
        "auc": f'{float(summary["mean_auc"])*100:.2f} ± {float(summary["std_auc"])*100:.2f}({float(summary["max_auc"])*100:.2f})',
        "prc": f'{float(summary["mean_auprc"])*100:.2f} ± {float(summary["std_auprc"])*100:.2f}({float(summary["max_auprc"])*100:.2f})',
        "auroc_mean": float(summary["mean_auc"]),
        "auroc_std": float(summary["std_auc"]),
        "auroc_max": float(summary["max_auc"]),
        "auprc_mean": float(summary["mean_auprc"]),
        "auprc_std": float(summary["std_auprc"]),
        "auprc_max": float(summary["max_auprc"]),
    }

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    DataFrame([row]).to_csv(header=write_header, index=False, mode="a")


def run_experiment(config: Config) -> dict[str, object]:
    ensure_dir(config.output_dir)
    log(str(asdict(config)), config.verbose)

    data = load_mat(config.dataset, config.data_dir)
    device = resolve_device(config.device, config.verbose)

    all_auc: list[float] = []
    all_auprc: list[float] = []
    runs: list[dict[str, float]] = []

    loop = range(config.runs)
    if config.tqdm:
        loop = tqdm(loop, desc="Run", position=0, leave=True)

    for run_idx in loop:
        result = train_one_run(config, run_idx, data, device)
        runs.append(result)
        all_auc.append(result["auc"])
        all_auprc.append(result["auprc"])

    summary = {
        "runs": runs,
        "all_auc": all_auc,
        "all_auprc": all_auprc,
        "mean_auc": float(np.mean(all_auc)),
        "std_auc": float(np.std(all_auc)),
        "max_auc": float(np.max(all_auc)),
        "mean_auprc": float(np.mean(all_auprc)),
        "std_auprc": float(np.std(all_auprc)),
        "max_auprc": float(np.max(all_auprc)),
    }

    if not config.not_show_res:
        print(f"MI-GAD {config.runs} runs")
        print(
            f"FINAL TESTING AUC:{summary['mean_auc'] * 100:.4f}, std:{summary['std_auc'] * 100:.4f}, max:{summary['max_auc'] * 100:.4f}"
        )
        print(
            f"FINAL TESTING AUPRC:{summary['mean_auprc'] * 100:.4f}, std:{summary['std_auprc'] * 100:.4f}, max:{summary['max_auprc'] * 100:.4f}"
        )

    if config.result_csv:
        append_result_csv(config.result_csv, config, summary)
    return summary
