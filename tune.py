from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import optuna

from migad.config import Config
from migad.train import run_experiment
from migad.utils import ensure_dir

TUNE_PARAM_KEYS = [
    "hidden_dim",
    "lr",
    "dropout",
    "alpha",
    "beta",
    "weight_decay",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optuna hyper-parameter tuning for MI-GAD."
    )

    # Dataset / runtime options.
    parser.add_argument(
        "--dataset", default=Config.dataset, help="Dataset name, e.g. Amazon/Facebook"
    )
    parser.add_argument(
        "--data_dir",
        default=Config.data_dir,
        help="Directory containing <dataset>.mat files",
    )
    parser.add_argument("--device", default=Config.device, help="cuda/cpu")
    parser.add_argument(
        "--output_dir",
        default="tune_outputs",
        help="Directory for Optuna outputs and trial checkpoints",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--not_show_train_res", type=bool, default=True)

    # Training budget used inside each trial.
    parser.add_argument(
        "--n_trials", type=int, default=50, help="Number of Optuna trials"
    )
    parser.add_argument(
        "--timeout", type=int, default=None, help="Maximum tuning time in seconds"
    )
    parser.add_argument(
        "--epoch", type=int, default=Config.epoch, help="Training epochs per trial"
    )
    parser.add_argument(
        "--runs_per_trial",
        type=int,
        default=1,
        help="Independent runs per trial; use 1 for speed, 3-5 for stability",
    )
    parser.add_argument(
        "--tests", type=int, default=Config.tests, help="Evaluation rounds per run"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=Config.patience,
        help="Early stopping patience per trial",
    )

    # Fixed optional search constraints.
    parser.add_argument(
        "--metric",
        choices=["mean_auc", "mean_auprc"],
        default="mean_auc",
        help="Metric optimized by Optuna",
    )
    parser.add_argument(
        "--sampler_seed", type=int, default=42, help="Seed for Optuna sampler"
    )
    parser.add_argument("--study_name", default=None, help="Optuna study name")
    parser.add_argument(
        "--storage",
        default=None,
        help="Optional Optuna storage URL, e.g. sqlite:///tune.db",
    )
    parser.add_argument("--load_if_exists", action="store_true")

    # Optional final verification with a larger number of runs.
    parser.add_argument(
        "--final_runs",
        type=int,
        default=0,
        help="Re-run best params with this many runs after tuning; 0 disables",
    )

    return parser


def suggest_config(trial: optuna.Trial, base: Config) -> Config:
    """Create a Config object from Optuna suggestions.

    The ranges are intentionally broad but conservative for MI-GAD. For paper-table
    reproduction, first tune with runs_per_trial=1 for speed, then verify the best
    parameters with --final_runs 5 or by running run.py manually.
    """

    hidden_dim = trial.suggest_categorical("hidden_dim", [16, 32, 64, 128, 256])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.6)
    alpha = trial.suggest_float("alpha", 0.5, 2.0)
    beta = trial.suggest_float("beta", 0.05, 1.0)
    weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True)

    return replace(
        base,
        hidden_dim=hidden_dim,
        lr=lr,
        dropout=dropout,
        alpha=alpha,
        beta=beta,
        weight_decay=weight_decay,
        verbose=False,
    )


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    if args.study_name is None:
        args.study_name = f"{args.dataset}_tune"
    output_dir = ensure_dir(args.output_dir)

    base_config = Config(
        dataset=args.dataset,
        data_dir=args.data_dir,
        epoch=args.epoch,
        tests=args.tests,
        device=args.device,
        patience=args.patience,
        runs=args.runs_per_trial,
        output_dir=str(output_dir),
        verbose=False,
        tqdm=False,
        not_show_res=args.not_show_train_res
    )

    sampler = optuna.samplers.TPESampler(seed=args.sampler_seed, multivariate=True)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=max(5, args.n_trials // 10), n_warmup_steps=0
    )
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=args.storage is not None,
    )

    def objective(trial: optuna.Trial) -> float:
        config = suggest_config(trial, base_config)
        summary = run_experiment(config)

        value = float(summary[args.metric])
        trial.set_user_attr("mean_auc", float(summary["mean_auc"]))
        trial.set_user_attr("std_auc", float(summary["std_auc"]))
        trial.set_user_attr("mean_auprc", float(summary["mean_auprc"]))
        trial.set_user_attr("std_auprc", float(summary["std_auprc"]))
        trial.set_user_attr("config", asdict(config))

        if args.verbose:
            print(
                f"Trial {trial.number:04d}: {args.metric}={value:.6f}, "
                f"auc={summary['mean_auc']:.6f}, auprc={summary['mean_auprc']:.6f}, "
                f"params={trial.params}",
                flush=True,
            )
        return value

    study.optimize(
        objective, n_trials=args.n_trials, timeout=args.timeout, gc_after_trial=True, show_progress_bar=True
    )

    best_config = replace(
        base_config,
        **{key: study.best_trial.params[key] for key in TUNE_PARAM_KEYS},
        verbose=False,
    )
    best_payload = {
        "best_value": study.best_value,
        "optimized_metric": args.metric,
        "best_params": study.best_trial.params,
        "best_config": asdict(best_config),
        "best_trial_number": study.best_trial.number,
    }

    if args.final_runs and args.final_runs > 0:
        final_config = replace(
            best_config,
            runs=args.final_runs,
            verbose=args.verbose,
        )
        final_summary = run_experiment(final_config)
        best_payload["final_summary"] = final_summary
        best_payload["final_config"] = asdict(final_config)

    save_json(output_dir / f"{args.study_name}_best_params.json", best_payload)
    study.trials_dataframe().to_csv(output_dir / f"{args.study_name}_trials.csv", index=False)

    print("\n================ OPTUNA RESULT ================")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best {args.metric}: {study.best_value:.6f}")
    print("Best params:")
    for key, value in study.best_trial.params.items():
        print(f"  --{key} {value}")
    print(f"Saved: {output_dir / f'{args.study_name}_best_params.json'}")
    print(f"Saved: {output_dir / f'{args.study_name}_trials.csv'}")
    print("================================================")


if __name__ == "__main__":
    main()
