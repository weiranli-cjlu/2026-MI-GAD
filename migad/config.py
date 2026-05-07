from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    dataset: str = "Amazon"
    data_dir: str = str(Path.home() / "datasets" / "GAD" / "mat")
    hidden_dim: int = 64
    epoch: int = 100
    tests: int = 1
    lr: float = 2e-3
    dropout: float = 0.0
    alpha: float = 1.0
    beta: float = 0.3
    device: str = "cuda"
    patience: int = 500
    weight_decay: float = 0.0
    runs: int = 5
    output_dir: str = "outputs"
    verbose: bool = True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MI-GAD: Multiscale Inconsistency Learning for Graph Anomaly Detection"
    )
    parser.add_argument(
        "--dataset",
        default=Config.dataset,
        help="Dataset name, e.g. Facebook/Amazon/Flickr/ACM/BlogCatalog",
    )
    parser.add_argument(
        "--data_dir",
        default=Config.data_dir,
        help="Directory containing <dataset>.mat files",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=Config.hidden_dim,
        help="Hidden embedding dimension",
    )
    parser.add_argument(
        "--epoch", type=int, default=Config.epoch, help="Training epochs"
    )
    parser.add_argument(
        "--tests",
        type=int,
        default=Config.tests,
        help="Number of evaluation rounds per run",
    )
    parser.add_argument("--lr", type=float, default=Config.lr, help="Learning rate")
    parser.add_argument(
        "--dropout", type=float, default=Config.dropout, help="Dropout rate"
    )
    parser.add_argument(
        "--alpha", type=float, default=Config.alpha, help="Affinity loss / score weight"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=Config.beta,
        help="Global inconsistency loss / score weight",
    )
    parser.add_argument("--device", default=Config.device, type=str, help="cuda/cpu")
    parser.add_argument(
        "--patience", type=int, default=Config.patience, help="Early stopping patience"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=Config.weight_decay, help="Weight decay"
    )
    parser.add_argument(
        "--runs", type=int, default=Config.runs, help="Number of independent runs"
    )
    parser.add_argument(
        "--output_dir",
        default=Config.output_dir,
        help="Directory for checkpoints and logs",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def parse_args() -> Config:
    namespace = build_parser().parse_args()
    return Config(**vars(namespace))
