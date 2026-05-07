# MI-GAD Refactored

Refactored code for **MI-GAD: Beyond Local Patterns: Multiscale Inconsistency Learning for Graph Anomaly Detection**.

This version keeps the original training logic but reorganizes the project and adds a `--verbose` argument to control logs.

## Structure

```text
MI-GAD_refactored/
├── run.py                  # CLI entry point
├── README.md
└── migad/
    ├── __init__.py
    ├── config.py           # argparse + Config dataclass
    ├── data.py             # .mat dataset loading
    ├── losses.py           # local affinity and global inconsistency losses
    ├── models.py           # GAD model and GraphConvolution
    ├── train.py            # train/evaluate experiment loop
    └── utils.py            # seed, sparse conversion, metrics, logging, negative sampling
```

## Usage

```bash
uv venv -p 3.12
uv pip install torch==2.11.0 numpy scipy matplotlib scikit-learn tqdm --torch-backend=cu128

# Facebook
python run.py --dataset Facebook --lr 2e-3 --epoch 100 --beta 0.2

# Amazon
python run.py --dataset Amazon --lr 2e-3 --epoch 100 --beta 0.3

# Silent mode: no per-epoch or summary logs
python run.py --dataset Amazon --lr 2e-3 --epoch 100 --beta 0.3 --verbose false
```

By default, data is read from:

```text
~/datasets/GAD/mat/<dataset>.mat
```

Use `--data_dir` to override it:

```bash
python run.py --dataset Amazon --data_dir /path/to/mat/files
```

Checkpoints are saved to `outputs/best_model_run<id>.pth` by default. Use `--output_dir` to change the path.

## Notes

- `--verbose` preserves the original behavior of printing arguments, run IDs, epoch losses, test AUC
, and final summary will always show.
- Module-level argument parsing has been removed, so importing `migad.data`, `migad.models`, or `migad.utils` no longer triggers CLI parsing.
