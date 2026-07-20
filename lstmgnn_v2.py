# soh_hybrid_lstm_gnn_full13_ffill_UPGRADED.py
import os, re, random
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Metrics + plotting
from sklearn.metrics import mean_absolute_error, mean_squared_error
try:
    
    from sklearn.metrics import root_mean_squared_error
except Exception:
    root_mean_squared_error = None

import matplotlib.pyplot as plt


#Config
CSV_PATH   = "metadata.csv"
assert os.path.basename(CSV_PATH) == "metadata.csv"

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
SEED       = 42
NOMINAL_C  = 2.0
WINDOW     = 12

TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT = 0.7, 0.1, 0.2

BATCH_TR   = 64
BATCH_EVAL = 128
EPOCHS     = 60
PATIENCE   = 12
LR         = 5e-4
WD         = 3e-4
DROPOUT    = 0.25

H_LSTM     = 128
LAYERS_LSTM= 2
H_GNN      = 192
LAYERS_GNN = 3

MODEL_DIR  = "models_hybrid"
PLOTS_DIR  = "results_hybrid"


def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

set_seed(SEED)


#Utils
class StandardScalerNP:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.mean_ = np.nanmean(X, axis=0)
        self.scale_ = np.nanstd(X, axis=0, ddof=0)
        self.scale_[~np.isfinite(self.scale_)] = 1.0
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        Z = (X - self.mean_) / self.scale_
        Z[~np.isfinite(Z)] = 0.0
        return Z


def parse_matlab_datevec(s):
    s = str(s).strip().strip('[]').replace(',', ' ')
    p = [t for t in s.split() if t]
    if len(p) < 6:
        return pd.NaT
    y, m, d, hh, mm, ss = [float(x) for x in p[:6]]
    si = int(ss)
    micro = int(round((ss - si) * 1_000_000))
    try:
        return datetime(int(y), int(m), int(d), int(hh), int(mm), si, micro)
    except Exception:
        return pd.NaT


def _find_col(df, candidates):
    lower = {c.lower().strip(): c for c in df.columns}
    tight = {re.sub(r'[\s_\-]+', '', c.lower()): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        k = cand.lower().strip()
        if k in lower:
            return lower[k]
        k2 = re.sub(r'[\s_\-]+', '', cand.lower())
        if k2 in tight:
            return tight[k2]
    return None


def _ensure_col(df, target, candidates, required=False):
    src = _find_col(df, candidates + [target])
    if src is None:
        if required:
            raise KeyError(f"Missing column '{target}' (tried {candidates})")
        return None
    if src != target:
        df.rename(columns={src: target}, inplace=True)
    return target


def split_by_battery(df, tr=0.7, va=0.1, te=0.2, seed=42):
    rng = np.random.default_rng(seed)
    bids = df["battery_id"].unique().tolist()
    rng.shuffle(bids)
    n = len(bids)
    ntr = int(n * tr)
    nva = int(n * va)
    tr_b = bids[:ntr]
    va_b = bids[ntr:ntr + nva]
    te_b = bids[ntr + nva:]
    tr_idx = df.index[df["battery_id"].isin(tr_b)].to_numpy()
    va_idx = df.index[df["battery_id"].isin(va_b)].to_numpy()
    te_idx = df.index[df["battery_id"].isin(te_b)].to_numpy()
    return tr_idx, va_idx, te_idx


#Build per-cycle
def build_per_cycle(csv_path, nominal_c=2.0):
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]

    _ensure_col(df, "battery_id", ["battery_id", "Battery_ID", "cell", "cell_id", "id", "unit"], True)
    _ensure_col(df, "start_time", ["start_time", "start", "time", "timestamp", "datetime", "date"], False)
    _ensure_col(df, "type", ["type", "operation", "op", "mode"], False)
    _ensure_col(df, "ambient_temperature", ["ambient_temperature", "temperature", "temp", "Temperature_measured", "T", "Temp"], False)
    _ensure_col(df, "Re", ["Re", "r_e", "electrolyte_resistance", "R_e", "ohmic_resistance"], False)
    _ensure_col(df, "Rct", ["Rct", "r_ct", "charge_transfer_resistance", "R_ct"], False)
    _ensure_col(df, "Capacity", ["Capacity", "capacity", "Qd", "Q_discharge", "discharge_capacity", "Cap"], False)

    print(f"[DBG] raw rows: {len(df)}")

    # time
    if "start_time" in df.columns:
        dt1 = df["start_time"].apply(parse_matlab_datevec)
        if dt1.isna().all():
            dt2 = pd.to_datetime(df["start_time"], errors="coerce", utc=False, infer_datetime_format=True)
            if dt2.isna().all():
                df["_orig_idx"] = np.arange(len(df))
                df["start_dt"] = df.groupby("battery_id")["_orig_idx"].rank(method="first")
            else:
                df["start_dt"] = dt2
        else:
            df["start_dt"] = dt1
    else:
        df["_orig_idx"] = np.arange(len(df))
        df["start_dt"] = df.groupby("battery_id")["_orig_idx"].rank(method="first")

    # numeric
    for c in ["ambient_temperature", "Re", "Rct", "Capacity"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # sort & per-battery ffill/bfill for sensors (not for SOH)
    df = df.sort_values(["battery_id", "start_dt"]).reset_index(drop=True)
    before = {c: int(df[c].isna().sum()) for c in ["ambient_temperature", "Re", "Rct"] if c in df.columns}
    for c in ["Re", "Rct", "ambient_temperature"]:
        if c in df.columns:
            df[c] = df.groupby("battery_id")[c].ffill().bfill()
    after = {c: int(df[c].isna().sum()) for c in ["ambient_temperature", "Re", "Rct"] if c in df.columns}
    print(f"[DBG] NaNs before fill: {before} | after: {after}")

    # index & SOH
    df["cycle_idx"] = df.groupby("battery_id").cumcount()
    if "Capacity" in df.columns:
        df["SOH_raw"] = (df["Capacity"] / nominal_c).astype(float)
    else:
        df["SOH_raw"] = np.nan
    df["SOH_ff"] = df.groupby("battery_id")["SOH_raw"].ffill()

    # engineer
    eps = 1e-6
    if "Re" in df.columns:
        df["Re0"] = df.groupby("battery_id")["Re"].transform("first")
        df["dRe"] = df["Re"] - df["Re0"]
        df["logRe"] = np.log(df["Re"].clip(lower=eps))
        df["Re_rm3"] = df.groupby("battery_id")["Re"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    if "Rct" in df.columns:
        df["Rct0"] = df.groupby("battery_id")["Rct"].transform("first")
        df["dRct"] = df["Rct"] - df["Rct0"]
        df["logRct"] = np.log(df["Rct"].clip(lower=eps))
        df["Rct_rm3"] = df.groupby("battery_id")["Rct"].transform(lambda s: s.rolling(3, min_periods=1).mean())

    if "ambient_temperature" in df.columns:
        df["T0"] = df.groupby("battery_id")["ambient_temperature"].transform("first")
        df["Temp_prev"] = df.groupby("battery_id")["ambient_temperature"].shift(1)
        df["dTemp"] = df["ambient_temperature"] - df["T0"]
        df["dTemp_prev"] = df["ambient_temperature"] - df["Temp_prev"]

    df["SOH_prev"] = df.groupby("battery_id")["SOH_ff"].shift(1)
    df["max_cycle"] = df.groupby("battery_id")["cycle_idx"].transform("max")
    df["cycle_frac"] = df["cycle_idx"] / df["max_cycle"].replace(0, 1)

    FEATURES = [
        "ambient_temperature", "dTemp", "dTemp_prev", "SOH_prev",
        "Re", "Rct", "dRe", "dRct", "logRe", "logRct", "Re_rm3", "Rct_rm3",
        "cycle_frac",
    ]

    print(f"[INFO] per-cycle rows (all types): {len(df)} | batteries: {df['battery_id'].nunique()} | candidate features: {len(FEATURES)}")
    return df, FEATURES


#Impute + scale
def impute_and_scale(df, FEATURES, train_idx):
    # medians from TRAIN for features
    med = df.loc[train_idx, FEATURES].median(axis=0, numeric_only=True)

    for c in FEATURES:
        # ensure numeric float
        df.loc[:, c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
        if df[c].isna().any():
            v = med.get(c, np.nan)
            if not np.isfinite(v):
                v = float(np.nanmean(df.loc[train_idx, c].values))
            if not np.isfinite(v):
                v = 0.0
            df.loc[:, c] = df[c].fillna(v).astype(np.float32)

    scaler = StandardScalerNP().fit(df.loc[train_idx, FEATURES].values.astype(np.float32))
    block = df.loc[:, FEATURES].values.astype(np.float32)
    scaled = scaler.transform(block).astype(np.float32)
    scaled[~np.isfinite(scaled)] = 0.0


    scaled_df = pd.DataFrame(scaled, columns=FEATURES, index=df.index, dtype=np.float32)
    df[FEATURES] = scaled_df
    return df


#Graph
def build_temporal_adj(Nfeat, T):
    n = Nfeat * T
    A = np.zeros((n, n), dtype=np.float32)

    def idx(f, t):
        return f + Nfeat * t

    for t in range(T - 1):
        for f in range(Nfeat):
            i, j = idx(f, t), idx(f, t + 1)
            A[i, j] = 1.0
            A[j, i] = 1.0

    for t in range(T):
        base = t * Nfeat
        for f1 in range(Nfeat):
            for f2 in range(Nfeat):
                if f1 == f2:
                    continue
                A[base + f1, base + f2] = 1.0

    for i in range(n):
        A[i, i] = 1.0

    return torch.from_numpy(A)


#Dataset
class HybridWindowDataset(Dataset):
   
    def __init__(self, df, keep_idx, features, T):
        self.df = df
        self.keep = set(keep_idx.tolist() if isinstance(keep_idx, np.ndarray) else keep_idx)
        self.features = features
        self.T = T
        self.samples = []

        use_type = "type" in df.columns
        type_str = df["type"].astype(str).str.lower() if use_type else None

        for bid, g in df.groupby("battery_id"):
            g = g.sort_values("cycle_idx")
            if len(g) < T + 1:
                continue

            if use_type and type_str.loc[g.index].str.contains("discha", na=False).any():
                tgt_mask = type_str.loc[g.index].str.contains("discha", na=False).values
            else:
                tgt_mask = np.isfinite(g["SOH_ff"].values)

            for i in range(T, len(g)):
                if not tgt_mask[i]:
                    continue

                rows = g.iloc[i - T:i]
                yrow = g.iloc[i]

                if (not set(rows.index).issubset(self.keep)) or (yrow.name not in self.keep):
                    continue

                soh_prev = float(g.loc[rows.index[-1], "SOH_ff"])
                soh_true = float(g.loc[yrow.name, "SOH_ff"])
                if not (np.isfinite(soh_prev) and np.isfinite(soh_true)):
                    continue

                Xwin = self.df.loc[rows.index, self.features].to_numpy()
                if not np.isfinite(Xwin).all():
                    continue

                self.samples.append((rows.index.tolist(), yrow.name, str(bid)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, k):
        rows, yidx, bid = self.samples[k]
        X = self.df.loc[rows, self.features].to_numpy(dtype=np.float32)         # [T,F]
        X_seq = torch.from_numpy(X)                                             # [T,F]
        nodes = torch.from_numpy(X.T.reshape(-1, 1).astype(np.float32))         # [T*F,1]
        A = build_temporal_adj(X.shape[1], self.T)                              # [n,n]
        soh_t = float(self.df.loc[yidx, "SOH_ff"])
        soh_p = float(self.df.loc[rows[-1], "SOH_ff"])
        y = soh_t - soh_p                                                       # ΔSOH
        return X_seq, nodes, A, float(y), float(soh_p), float(soh_t), bid


def collate(batch):
    Xs, Ns, As, ys, sp, st, bids = zip(*batch)
    X_seq = torch.stack(Xs, 0)
    nodes = torch.stack(Ns, 0)
    A = torch.stack(As, 0).float()
    y = torch.tensor(ys, dtype=torch.float32)
    sp = torch.tensor(sp, dtype=torch.float32)
    st = torch.tensor(st, dtype=torch.float32)
    return X_seq, nodes, A, y, sp, st, list(bids)


#Model
class ResidualGraphBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.25):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()
        self.res_proj = (nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity())

    def forward(self, X, A):
        deg = A.sum(-1, keepdim=True) + 1e-6
        Ahat = A / deg
        H = torch.bmm(Ahat, X)
        H = self.lin(H)
        H = self.norm(H)
        H = self.act(H)
        H = self.drop(H)
        return H + self.res_proj(X)


class GNNBranch(nn.Module):
    def __init__(self, node_in=1, hidden=192, layers=3, dropout=0.25):
        super().__init__()
        blocks = []
        d_in = node_in
        for _ in range(layers):
            blocks.append(ResidualGraphBlock(d_in, hidden, dropout))
            d_in = hidden
        self.blocks = nn.ModuleList(blocks)
        self.attn = nn.Linear(hidden, 1)

    def forward(self, X_nodes, A):
        H = X_nodes
        for blk in self.blocks:
            H = blk(H, A)
        scores = self.attn(H).squeeze(-1)
        alpha = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (alpha * H).sum(dim=1)


class LSTMBranch(nn.Module):
    def __init__(self, in_dim, hidden=128, layers=2, dropout=0.25):
        super().__init__()
        self.lstm = nn.LSTM(
            in_dim, hidden,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, X_seq):
        out, _ = self.lstm(X_seq)
        return self.norm(out[:, -1, :])


class HybridLSTM_GNN(nn.Module):
    def __init__(self, in_dim, h_lstm=128, layers_lstm=2, h_gnn=192, layers_gnn=3, dropout=0.25):
        super().__init__()
        self.lstm_branch = LSTMBranch(in_dim, h_lstm, layers_lstm, dropout)
        self.gnn_branch = GNNBranch(1, h_gnn, layers_gnn, dropout)
        fused = h_lstm + h_gnn
        self.head = nn.Sequential(
            nn.Linear(fused, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, X_seq, X_nodes, A):
        h_l = self.lstm_branch(X_seq)
        h_g = self.gnn_branch(X_nodes, A)
        return self.head(torch.cat([h_l, h_g], dim=-1)).squeeze(-1)


#Metrics helpers
def _mae_rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mae = mean_absolute_error(y_true, y_pred)

    if root_mean_squared_error is not None:
        rmse = root_mean_squared_error(y_true, y_pred)
    else:
        # fallback compatible with older sklearn
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return float(mae), float(rmse)


def run_epoch(loader, model, lossf, optimizer=None, train=False):
    model.train(train)

    losses = []
    y_true_d, y_pred_d = [], []
    y_true_soh, y_pred_soh = [], []

    for X_seq, nodes, A, y, sp, st, _ in loader:
        X_seq = X_seq.to(DEVICE).float()
        nodes = nodes.to(DEVICE).float()
        A = A.to(DEVICE).float()
        y = y.to(DEVICE).float()
        sp = sp.to(DEVICE).float()
        st = st.to(DEVICE).float()

        if train and optimizer:
            optimizer.zero_grad()

        pred_d = model(X_seq, nodes, A)         
        loss = lossf(pred_d, y)

        if train and optimizer:
            loss.backward()
            optimizer.step()

        losses.append(loss.detach().cpu().item())

        pred_d_cpu = pred_d.detach().cpu().numpy()
        y_cpu = y.detach().cpu().numpy()

        pred_soh_cpu = (sp + pred_d).detach().cpu().numpy()
        true_soh_cpu = st.detach().cpu().numpy()

        y_true_d.append(y_cpu)
        y_pred_d.append(pred_d_cpu)
        y_true_soh.append(true_soh_cpu)
        y_pred_soh.append(pred_soh_cpu)

    if len(y_true_d) == 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

    y_true_d = np.concatenate(y_true_d)
    y_pred_d = np.concatenate(y_pred_d)
    y_true_soh = np.concatenate(y_true_soh)
    y_pred_soh = np.concatenate(y_pred_soh)

    d_mae, d_rmse = _mae_rmse(y_true_d, y_pred_d)
    s_mae, s_rmse = _mae_rmse(y_true_soh, y_pred_soh)
    loss_mean = float(np.mean(losses)) if losses else np.nan

    return d_mae, d_rmse, s_mae, s_rmse, loss_mean


def collect_predictions(loader, model):
    model.eval()
    bids_all = []
    true_d_all, pred_d_all = [], []
    true_soh_all, pred_soh_all = [], []

    with torch.no_grad():
        for X_seq, nodes, A, y, sp, st, bids in loader:
            X_seq = X_seq.to(DEVICE).float()
            nodes = nodes.to(DEVICE).float()
            A = A.to(DEVICE).float()
            y = y.to(DEVICE).float()
            sp = sp.to(DEVICE).float()
            st = st.to(DEVICE).float()

            pred_d = model(X_seq, nodes, A)

            pred_d_cpu = pred_d.detach().cpu().numpy()
            y_cpu = y.detach().cpu().numpy()

            pred_soh_cpu = (sp + pred_d).detach().cpu().numpy()
            true_soh_cpu = st.detach().cpu().numpy()

            bids_all.extend(list(bids))
            true_d_all.append(y_cpu)
            pred_d_all.append(pred_d_cpu)
            true_soh_all.append(true_soh_cpu)
            pred_soh_all.append(pred_soh_cpu)

    true_d_all = np.concatenate(true_d_all) if true_d_all else np.array([])
    pred_d_all = np.concatenate(pred_d_all) if pred_d_all else np.array([])
    true_soh_all = np.concatenate(true_soh_all) if true_soh_all else np.array([])
    pred_soh_all = np.concatenate(pred_soh_all) if pred_soh_all else np.array([])

    return bids_all, true_d_all, pred_d_all, true_soh_all, pred_soh_all


def per_battery_metrics(bids, true_soh, pred_soh):
    out = {}
    for bid, t, p in zip(bids, true_soh, pred_soh):
        out.setdefault(bid, {"t": [], "p": []})
        out[bid]["t"].append(float(t))
        out[bid]["p"].append(float(p))

    per = {}
    for bid, d in out.items():
        mae, rmse = _mae_rmse(d["t"], d["p"])
        per[bid] = {"mae": mae, "rmse": rmse, "n": len(d["t"])}
    return per


#Plotting 
def make_plots(tag, history, test_metrics, true_soh, pred_soh, per_batt, bids_test):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # 1) Train/Val curves (SOH)
    epochs = [h["epoch"] for h in history]
    tr_loss = [h["tr_loss"] for h in history]
    va_loss = [h["va_loss"] for h in history]
    tr_mae = [h["tr_soh_mae"] for h in history]
    va_mae = [h["va_soh_mae"] for h in history]
    tr_rmse = [h["tr_soh_rmse"] for h in history]
    va_rmse = [h["va_soh_rmse"] for h in history]

    plt.figure()
    plt.plot(epochs, tr_mae, label="Train SOH MAE")
    plt.plot(epochs, va_mae, label="Val SOH MAE")
    plt.plot(epochs, tr_rmse, label="Train SOH RMSE")
    plt.plot(epochs, va_rmse, label="Val SOH RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.title(f"{tag} - Training Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{tag}_train_curves.png"), dpi=160)
    plt.close()

    # 2) Loss curves
    plt.figure()
    plt.plot(epochs, tr_loss, label="Train Loss")
    plt.plot(epochs, va_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("SmoothL1 Loss")
    plt.title(f"{tag} - Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{tag}_loss_curves.png"), dpi=160)
    plt.close()

    # 3) Test metrics bar plot
    labels = ["ΔSOH MAE", "ΔSOH RMSE", "SOH MAE", "SOH RMSE"]
    vals = [test_metrics["delta_mae"], test_metrics["delta_rmse"], test_metrics["soh_mae"], test_metrics["soh_rmse"]]
    plt.figure()
    plt.bar(labels, vals)
    plt.ylabel("Error")
    plt.title(f"{tag} - Test Metrics")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{tag}_test_metrics.png"), dpi=160)
    plt.close()

    # 4) True vs Pred scatter (SOH)
    if len(true_soh) > 0:
        plt.figure()
        plt.scatter(true_soh, pred_soh, s=12)
        mn = float(np.nanmin([true_soh.min(), pred_soh.min()]))
        mx = float(np.nanmax([true_soh.max(), pred_soh.max()]))
        plt.plot([mn, mx], [mn, mx])
        plt.xlabel("True SOH")
        plt.ylabel("Predicted SOH")
        plt.title(f"{tag} - True vs Pred SOH")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{tag}_true_vs_pred_soh.png"), dpi=160)
        plt.close()

        # 5) SOH over sample index (trend)
        idx = np.arange(len(true_soh))
        plt.figure()
        plt.plot(idx, true_soh, label="True SOH")
        plt.plot(idx, pred_soh, label="Pred SOH")
        plt.xlabel("Test sample index")
        plt.ylabel("SOH")
        plt.title(f"{tag} - SOH over Test Samples")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{tag}_soh_over_samples.png"), dpi=160)
        plt.close()

    # 6) Per-battery MAE top-25
    items_mae = sorted(per_batt.items(), key=lambda kv: kv[1]["mae"], reverse=True)
    top_mae = items_mae[:min(25, len(items_mae))]
    if top_mae:
        bnames = [k for k, _ in top_mae]
        maes = [v["mae"] for _, v in top_mae]
        plt.figure(figsize=(10, 5))
        plt.bar(bnames, maes)
        plt.xticks(rotation=60, ha="right")
        plt.ylabel("SOH MAE")
        plt.title(f"{tag} - Worst Batteries by SOH MAE (Top {len(top_mae)})")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{tag}_per_battery_mae_top.png"), dpi=160)
        plt.close()

    # 7) Per-battery RMSE top-25
    items_rmse = sorted(per_batt.items(), key=lambda kv: kv[1]["rmse"], reverse=True)
    top_rmse = items_rmse[:min(25, len(items_rmse))]
    if top_rmse:
        bnames = [k for k, _ in top_rmse]
        rmses = [v["rmse"] for _, v in top_rmse]
        plt.figure(figsize=(10, 5))
        plt.bar(bnames, rmses)
        plt.xticks(rotation=60, ha="right")
        plt.ylabel("SOH RMSE")
        plt.title(f"{tag} - Worst Batteries by SOH RMSE (Top {len(top_rmse)})")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{tag}_per_battery_rmse_top.png"), dpi=160)
        plt.close()

    # 8) Worst battery plot
    if len(items_mae) > 0 and len(true_soh) == len(bids_test):
        worst_bid = items_mae[0][0]
        mask = np.array([b == worst_bid for b in bids_test], dtype=bool)
        if mask.any():
            t_b = true_soh[mask]
            p_b = pred_soh[mask]
            x = np.arange(len(t_b))
            plt.figure()
            plt.plot(x, t_b, label="True SOH")
            plt.plot(x, p_b, label="Pred SOH")
            plt.xlabel("Sample index (within battery)")
            plt.ylabel("SOH")
            plt.title(f"{tag} - Worst Battery ({worst_bid}) SOH: True vs Pred")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, f"{tag}_worst_battery_{worst_bid}_soh.png"), dpi=160)
            plt.close()

    print(f"[plots saved] {PLOTS_DIR}/ (png files)")


#Main
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df, FEATURES = build_per_cycle(CSV_PATH, nominal_c=NOMINAL_C)

    tr_idx, va_idx, te_idx = split_by_battery(df, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, SEED)

    pc = impute_and_scale(df.copy(), FEATURES, tr_idx)

    train_ds = HybridWindowDataset(pc, tr_idx, FEATURES, WINDOW)
    val_ds   = HybridWindowDataset(pc, va_idx, FEATURES, WINDOW)
    test_ds  = HybridWindowDataset(pc, te_idx, FEATURES, WINDOW)

    if min(len(train_ds), len(val_ds), len(test_ds)) == 0:
        print(f"[WARN] Not enough samples after split. Got train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}.")
        return

    train_ld = DataLoader(train_ds, batch_size=BATCH_TR, shuffle=True,  collate_fn=collate)
    val_ld   = DataLoader(val_ds,   batch_size=BATCH_EVAL, shuffle=False, collate_fn=collate)
    test_ld  = DataLoader(test_ds,  batch_size=BATCH_EVAL, shuffle=False, collate_fn=collate)

    model = HybridLSTM_GNN(len(FEATURES), H_LSTM, LAYERS_LSTM, H_GNN, LAYERS_GNN, DROPOUT).to(DEVICE)
    lossf = nn.SmoothL1Loss(beta=0.01)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    tag = f"hybrid_w{WINDOW}_hl{H_LSTM}_hg{H_GNN}_do{DROPOUT}_full13_ff"

    best_val = float("inf")
    best_state = None
    pat = 0
    history = []

    for ep in range(1, EPOCHS + 1):
        tr_d_mae, tr_d_rmse, tr_s_mae, tr_s_rmse, tr_loss = run_epoch(train_ld, model, lossf, opt, True)
        va_d_mae, va_d_rmse, va_s_mae, va_s_rmse, va_loss = run_epoch(val_ld, model, lossf, None, False)

        history.append({
            "epoch": ep,
            "tr_loss": tr_loss,
            "va_loss": va_loss,
            "tr_soh_mae": tr_s_mae,
            "va_soh_mae": va_s_mae,
            "tr_soh_rmse": tr_s_rmse,
            "va_soh_rmse": va_s_rmse,
        })

        if ep % 5 == 0:
            print(
                f"[hybrid] epoch {ep:02d} | "
                f"train Δ(MAE/RMSE) {tr_d_mae:.4f}/{tr_d_rmse:.4f} | "
                f"val   Δ(MAE/RMSE) {va_d_mae:.4f}/{va_d_rmse:.4f} | "
                f"val SOH (MAE/RMSE) {va_s_mae:.4f}/{va_s_rmse:.4f}"
            )

        # Early stop on VAL SOH MAE
        if va_s_mae < best_val - 1e-4:
            best_val = va_s_mae
            best_state = model.state_dict()
            pat = 0
        else:
            pat += 1

        if pat >= PATIENCE:
            print(f"[hybrid] Early stop at {ep} (best val SOH MAE {best_val:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test metrics + predictions for plots
    te_d_mae, te_d_rmse, te_s_mae, te_s_rmse, te_loss = run_epoch(test_ld, model, lossf, None, False)
    print(
        f"[hybrid] TEST | ΔSOH (MAE/RMSE) = {te_d_mae:.4f}/{te_d_rmse:.4f} | "
        f"SOH (MAE/RMSE) = {te_s_mae:.4f}/{te_s_rmse:.4f}"
    )

    bids, true_d, pred_d, true_soh, pred_soh = collect_predictions(test_ld, model)
    per_batt = per_battery_metrics(bids, true_soh, pred_soh)

    # Save model checkpoint
    ckpt = os.path.join(MODEL_DIR, f"{tag}.pt")
    torch.save(best_state if best_state is not None else model.state_dict(), ckpt)
    print(f"[saved] model checkpoint: {ckpt}")

    test_metrics = {
        "delta_mae": te_d_mae,
        "delta_rmse": te_d_rmse,
        "soh_mae": te_s_mae,
        "soh_rmse": te_s_rmse,
    }

    make_plots(tag, history, test_metrics, true_soh, pred_soh, per_batt, bids)


if __name__ == "__main__":
    main()
