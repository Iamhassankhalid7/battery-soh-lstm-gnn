# battery-soh-lstm-gnn
# Hybrid LSTM-GNN for Battery State of Health (SOH) Prediction

MSc Artificial Intelligence with Business Strategy — Master's Dissertation
Aston University, 2025/26

## Overview

Battery State of Health (SOH) determines how much usable capacity a battery has left compared to when it was new, and is critical for safety and lifecycle planning in EVs and energy storage systems. This project benchmarks several deep learning architectures for forecasting SOH from historical charge/discharge cycle data, culminating in a **hybrid LSTM + Graph Neural Network (GNN)** model that combines temporal degradation patterns with structural relationships between battery units.

## Models implemented

| Model | File | Architecture | Notes |
|---|---|---|---|
| LSTM (baseline) | `lstm.py` | 2-layer LSTM, hidden=128 | Sequential/temporal degradation only |
| GraphSAGE | `sagegraph_v2.py` | 3-layer GraphSAGE, hidden=192 | Structural relationships between battery units |
| Temporal ResGCN | `TGNN_v2.py` | Residual GCN, hidden=192 | Temporal-graph baseline |
| Temporal ResGCN + GRU | `TGNN_GRU.py` | Residual GCN + GRU head | Adds a recurrent head to the graph baseline |
| **Hybrid LSTM-GNN (final)** | `lstmgnn_v2.py` | LSTM (128) + GNN (192) combined | Main dissertation model |

All models:
- Use a 12-step sliding window over cycle history
- Are trained on a **battery-wise** 70/10/20 train/val/test split (no leakage between a battery's own cycles across splits)
- Use early stopping (patience 10–12 epochs), Adam optimizer, weight decay for regularization
- Are evaluated with MAE, RMSE, and R²

## Results
| Model | MAE | RMSE |
|---|---|---|---|
| LSTM |0.0092 | 0.0241 |
| GraphSAGE | 0.0088 | 0.0278 |
| Temporal ResGCN (+GRU) | 0.0094 | 0.0287 |
| **Hybrid LSTM-GNN** | 0.0088 | 0.0277 |

Each `results_*/` folder also includes:
- Training/validation loss curves
- Predicted vs. true SOH plots
- Per-battery MAE/RMSE breakdown (top offenders)
- Worst-case battery trajectory plot


## Repository structure

```
├── lstm.py                  # Baseline LSTM model
├── sagegraph_v2.py          # GraphSAGE model
├── TGNN_v2.py                # Temporal ResGCN baseline
├── TGNN_GRU.py                # Temporal ResGCN + GRU variant
├── lstmgnn_v2.py             # Final hybrid LSTM-GNN model
├── metadata.csv              # Battery cycle dataset
├── models_lstm/              # Trained LSTM weights
├── models_sage/              # Trained GraphSAGE weights
├── models_hybrid/             # Trained hybrid model weights
├── models_temporal_v3/        # Trained temporal ResGCN weights
├── results_lstm/              # LSTM evaluation plots
├── results_sage/              # GraphSAGE evaluation plots
├── results_hybrid/             # Hybrid model evaluation plots
└── results_temporal_v3/        # Temporal ResGCN evaluation plots
```

## How to run

```bash
pip install -r requirements.txt
python lstm.py          # train/evaluate baseline LSTM
python sagegraph_v2.py  # train/evaluate GraphSAGE
python lstmgnn_v2.py    # train/evaluate the final hybrid model
```

Each script trains its model, saves weights to the corresponding `models_*/` folder, and writes evaluation plots to `results_*/`.

## Requirements

```
numpy
pandas
torch
scikit-learn
matplotlib
```

## Dataset

`metadata.csv` contains battery charge/discharge cycle records used to compute SOH (capacity relative to nominal capacity, 2.0 Ah). Data preprocessing includes per-battery imputation and forward-fill handling for missing cycle records.

## Author

Hassan Khalid — MSc Artificial Intelligence with Business Strategy, Aston University.
https://www.linkedin.com/in/hassanabdullahbinkhalid/ · Hassanabdullahbinkhalid@gmail.com
