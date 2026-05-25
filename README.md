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
uv pip install torch==2.11.0 numpy scipy matplotlib scikit-learn tqdm optuna pandas --torch-backend=cu128

# Facebook
python run.py --dataset Facebook --lr 2e-3 --epoch 100 --beta 0.2

# Amazon
python run.py --dataset Amazon --lr 2e-3 --epoch 100 --beta 0.3

# Silent mode: no per-epoch or summary logs
python run.py --dataset Amazon --lr 2e-3 --epoch 100 --beta 0.3 --verbose
```

```bash
python run.py --dataset cs --hidden_dim 256 --lr 0.004138 --dropout 0.18277 --alpha 0.6465 --beta 0.70002 --weight_decay 1.58768 --runs 50 --tqdm --result-csv outputs/result.csv
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

## 实验

```bash
python run.py --dataset Facebook \
    --hidden_dim 16 --epoch 100 \
    --lr 0.0069782812651260325 --dropout 0.0530955012311517 \
    --alpha 0.7939742936287177 --beta 0.09296592446501117 \
    --patience 500 --weight_decay 4.233032996527588e-07 \
    --runs 10 --result-csv outputs/result.csv --tqdm

python run.py --dataset cora \
    --hidden_dim 256 --epoch 100 \
    --lr 0.0009718319944817398 --dropout 0.3136396976291964 \
    --alpha 1.1413115275378245 --beta 0.07414817040689042 \
    --patience 500 --weight_decay 3.4630370261191806e-08 \
    --runs 10 --result-csv outputs/result.csv --tqdm

python run.py --dataset citeseer \
    --hidden_dim 256 --epoch 100 \
    --lr 0.0009718319944817398 --dropout 0.3136396976291964 \
    --alpha 1.1413115275378245 --beta 0.07414817040689042 \
    --patience 500 --weight_decay 3.4630370261191806e-08 \
    --runs 10 --result-csv outputs/result.csv --tqdm

python run.py --dataset tolokers \
    --hidden_dim 256 --epoch 100 \
    --lr 0.007320837856369199 --dropout 0.01734469509859761 \
    --alpha 1.6368285307920063 --beta 0.6290747898599562 \
    --patience 500 --weight_decay 0.0003165516687397805 \
    --runs 10 --result-csv outputs/result.csv --tqdm

python run.py --dataset Amazon \
    --hidden_dim 16 --epoch 100 \
    --lr 0.0019379671536779212 --dropout 0.02372717750664053 \
    --alpha 7588127966413724 --beta 0.19428321679592314 \
    --patience 500 --weight_decay 8.689551819762203e-06 \
    --runs 10 --result-csv outputs/result.csv --tqdm

python run.py --dataset ACM \
    --hidden_dim 256 --epoch 100 \
    --lr 0.0009718319944817398 --dropout 0.3136396976291964 \
    --alpha 1.1413115275378245 --beta 0.07414817040689042 \
    --patience 500 --weight_decay 3.4630370261191806e-08 \
    --runs 10 --result-csv outputs/result.csv --tqdm

python run.py --dataset Flickr \
    --hidden_dim 64 --epoch 100 \
    --lr 0.007428087840236459 --dropout 0.19201648936337873 \
    --alpha 0.5599634490694338 --beta 0.3076017528761459 \
    --patience 500 --weight_decay 0.0007220941578931524 \
    --runs 10 --result-csv outputs/result.csv --tqdm

python run.py --dataset BlogCatalog \
    --hidden_dim 64 --epoch 100 \
    --lr 0.001217284708112243 --dropout 0.08455453498485759 \
    --alpha 1.7032954711310595 --beta 0.12082311149578229 \
    --patience 500 --weight_decay 0.0008598737339212274 \
    --runs 10 --result-csv outputs/result.csv --tqdm
    
python run.py --dataset YelpChi \
    --hidden_dim 256 --epoch 100 \
    --lr 0.004711938313793034 --dropout 0.23447635020248575 \
    --alpha 0.9983751062029745 --beta 0.5182720098717788 \
    --patience 500 --weight_decay 0.0008396987422844343 \
    --runs 10 --result-csv outputs/result.csv --tqdm
```