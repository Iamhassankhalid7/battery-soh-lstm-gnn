# soh_temporal_baseline_v3_w12_
import os, random, re
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt

# Config
CSV_PATH       = "metadata.csv"
assert os.path.basename(CSV_PATH) == "metadata.csv", "Only metadata.csv is allowed."

NOMINAL_C      = 2.0
EOL_C          = 1.4
PREDICT_DELTA  = True
SPLIT_MODE     = "battery"  


WINDOW    = 12
DROPOUT   = 0.25

# Keep hidden configurable here 
HIDDEN    = 192
NOISE_STD = 0.00

# Train params
BATCH_TR       = 64
BATCH_EVAL     = 128
EPOCHS         = 70
PATIENCE       = 12
LR             = 5e-4
WD             = 3e-4
SEED           = 42
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_DIR      = "models_temporal_v3"
RESULTS_DIR    = "results_temporal_v3"

def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
set_seed(SEED)

#Utils
def parse_matlab_datevec(s):
    s = str(s).strip().strip("[]").replace(",", " ")
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
    tight = {re.sub(r"[\s_\-]+", "", c.lower()): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        k = cand.lower().strip()
        if k in lower:
            return lower[k]
        k2 = re.sub(r"[\s_\-]+", "", cand.lower())
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

#Splits
def temporal_split_index(df, train=0.6, val=0.1, test=0.3):
    tr, va, te = [], [], []
    for _, g in df.groupby("battery_id"):
        g = g.sort_values("cycle_idx")
        n = len(g)
        if n == 0:
            continue
        ntr = int(n * train)
        nv  = int(n * val)
        tr += g.index[:ntr].tolist()
        va += g.index[ntr:ntr + nv].tolist()
        te += g.index[ntr + nv:].tolist()
    return np.array(tr), np.array(va), np.array(te)

def battery_split_index(df, train=0.7, val=0.1, test=0.2, seed=42):
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(df["battery_id"].unique()))
    rng.shuffle(uniq)
    n = len(uniq)
    ntr = int(n * train)
    nva = int(n * val)
    tr_ids = set(uniq[:ntr])
    va_ids = set(uniq[ntr:ntr + nva])
    te_ids = set(uniq[ntr + nva:])

    idx = np.arange(len(df))
    tr = idx[df["battery_id"].isin(tr_ids)]
    va = idx[df["battery_id"].isin(va_ids)]
    te = idx[df["battery_id"].isin(te_ids)]
    return tr, va, te

def add_rul(group, eol_capacity):
    g = group.sort_values("cycle_idx").copy()
    cap = g["Capacity"].values
    eol = int(np.where(cap <= eol_capacity)[0][0]) if np.any(cap <= eol_capacity) else len(g) - 1
    idx = np.arange(len(g))
    g["RUL"] = np.maximum(eol - idx, 0).astype(float)
    bid = g["battery_id"].iloc[0] if "battery_id" in g.columns else group.name
    g["battery_id"] = bid
    return g

#Build per-cycle 
def build_per_cycle(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]

    _ensure_col(df, "battery_id", ["battery_id", "Battery_ID", "cell", "cell_id", "id", "unit"], True)
    _ensure_col(df, "start_time", ["start_time", "start", "time", "timestamp", "datetime", "date"], True)
    _ensure_col(df, "type", ["type", "operation", "op", "mode"], False)
    _ensure_col(df, "ambient_temperature", ["ambient_temperature", "temperature", "temp", "Temperature_measured"], False)
    _ensure_col(df, "Re", ["Re", "r_e", "electrolyte_resistance"], False)
    _ensure_col(df, "Rct", ["Rct", "r_ct", "charge_transfer_resistance"], False)
    _ensure_col(df, "Capacity", ["Capacity", "capacity", "Qd", "Q_discharge", "discharge_capacity"], True)

    df["start_dt"] = df["start_time"].apply(parse_matlab_datevec)
    df = df.sort_values(["battery_id", "start_dt"]).reset_index(drop=True)

    for c in ["ambient_temperature", "Re", "Rct", "Capacity"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

     
    before = {c: int(df[c].isna().sum()) for c in ["ambient_temperature","Re","Rct"] if c in df.columns}
    for c in ["Re", "Rct", "ambient_temperature"]:
        if c in df.columns:
            df[c] = df.groupby("battery_id")[c].ffill().bfill()
            if df[c].isna().any():
                df[c] = df[c].fillna(df[c].median())
    after = {c: int(df[c].isna().sum()) for c in ["ambient_temperature","Re","Rct"] if c in df.columns}
    print(f"[DBG] NaNs before fill: {before} | after: {after}")

    # Case-insensitive discharge 
    if "type" in df.columns:
        tl = df["type"].astype(str).str.lower()
        dmask = tl.str.contains("discha", na=False)
        df = df[dmask] if dmask.any() else df

    df = df.sort_values(["battery_id", "start_dt"]).reset_index(drop=True)
    df["cycle_idx"] = df.groupby("battery_id").cumcount()

    df["SOH"] = df["Capacity"].astype(float) / NOMINAL_C

    eps = 1e-6
    if "Re" in df.columns:
        df["Re0"]    = df.groupby("battery_id")["Re"].transform("first")
        df["dRe"]    = df["Re"] - df["Re0"]
        df["logRe"]  = np.log(df["Re"].clip(lower=eps))
        df["Re_rm3"] = df.groupby("battery_id")["Re"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    if "Rct" in df.columns:
        df["Rct0"]    = df.groupby("battery_id")["Rct"].transform("first")
        df["dRct"]    = df["Rct"] - df["Rct0"]
        df["logRct"]  = np.log(df["Rct"].clip(lower=eps))
        df["Rct_rm3"] = df.groupby("battery_id")["Rct"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    if "ambient_temperature" in df.columns:
        df["T0"]         = df.groupby("battery_id")["ambient_temperature"].transform("first")
        df["Temp_prev"]  = df.groupby("battery_id")["ambient_temperature"].shift(1)
        df["dTemp"]      = df["ambient_temperature"] - df["T0"]
        df["dTemp_prev"] = df["ambient_temperature"] - df["Temp_prev"]

    df["SOH_prev"]   = df.groupby("battery_id")["SOH"].shift(1)
    df["max_cycle"]  = df.groupby("battery_id")["cycle_idx"].transform("max")
    df["cycle_frac"] = df["cycle_idx"] / df["max_cycle"].replace(0, 1)

    FEATURES = [c for c in [
        "ambient_temperature", "dTemp", "dTemp_prev", "SOH_prev",
        "Re", "Rct", "dRe", "dRct", "logRe", "logRct", "Re_rm3", "Rct_rm3",
        "cycle_frac",
    ] if c in df.columns]

    need = ["battery_id", "cycle_idx", "start_dt", "SOH"] + FEATURES
    df = df.dropna(subset=need).reset_index(drop=True)


    try:
        per_cycle = df.groupby("battery_id", group_keys=False).apply(
            lambda g: add_rul(g, EOL_C), include_groups=False
        )
    except TypeError:
        per_cycle = df.groupby("battery_id", group_keys=False).apply(lambda g: add_rul(g, EOL_C))

    per_cycle = per_cycle.reset_index(drop=True)
    print(f"[INFO] per-cycle rows: {len(per_cycle)} | batteries: {per_cycle['battery_id'].nunique()} | features: {len(FEATURES)}")
    return per_cycle, FEATURES

#Graph building 
def build_spatiotemporal_graph(N, T, start_times, dt_mu=None, dt_sigma=None, use_edge_feat=True, add_spatial=True):
    n = N * T
    A = np.zeros((n, n), dtype=np.float32)
    de = 3 if use_edge_feat else 0
    EF = np.zeros((n, n, de), dtype=np.float32) if use_edge_feat else None

    def idx(f, t):
        return f + N * t

    # temporal edges
    for t in range(T - 1):
        dt_hours = (start_times[t + 1] - start_times[t]).total_seconds() / 3600.0
        dt_z = ((dt_hours - (dt_mu or 0.0)) / (dt_sigma or 1.0)) if (use_edge_feat and dt_sigma) else 0.0
        for f in range(N):
            i, j = idx(f, t), idx(f, t + 1)
            A[i, j] = A[j, i] = 1.0
            if use_edge_feat:
                EF[i, j, 0] = EF[j, i, 0] = dt_z
                EF[i, j, 1] = EF[j, i, 1] = 1.0  # temporal flag
                EF[i, j, 2] = EF[j, i, 2] = 0.0  # spatial flag

    # spatial edges
    if add_spatial:
        for t in range(T):
            base = N * t
            for f in range(N):
                for g in range(f + 1, N):
                    i, j = base + f, base + g
                    A[i, j] = A[j, i] = 1.0
                    if use_edge_feat:
                        EF[i, j, 0] = EF[j, i, 0] = 0.0
                        EF[i, j, 1] = EF[j, i, 1] = 0.0
                        EF[i, j, 2] = EF[j, i, 2] = 1.0

    # self loops
    for i in range(n):
        A[i, i] = 1.0
        if use_edge_feat:
            EF[i, i, :] = 0.0

    return torch.from_numpy(A), (torch.from_numpy(EF) if use_edge_feat else None)

#Dataset 
class WindowGraphDataset(Dataset):
    def __init__(self, df, keep_idx, features, T, predict_delta, use_edge_feat=True, add_spatial=True):
        self.df = df
        self.keep = set(keep_idx.tolist() if isinstance(keep_idx, np.ndarray) else keep_idx)
        self.features = features
        self.T = T
        self.predict_delta = predict_delta
        self.use_edge_feat = use_edge_feat
        self.add_spatial = add_spatial
        self.samples = []
        self._collect_samples()

        # dt stats for normalization 
        dts = []
        for rows, _, _ in self.samples:
            st = df.loc[rows, "start_dt"].tolist()
            for t in range(len(st) - 1):
                dts.append((st[t + 1] - st[t]).total_seconds() / 3600.0)
        self.dt_mu = float(np.mean(dts)) if len(dts) else 0.0
        self.dt_sigma = float(np.std(dts)) if len(dts) and np.std(dts) > 0 else 1.0

    def _collect_samples(self):
        for bid, g in self.df.groupby("battery_id"):
            g = g.sort_values("cycle_idx")
            if len(g) < self.T + 1:
                continue
            for i in range(self.T, len(g)):
                rows = g.iloc[i - self.T:i]
                yrow = g.iloc[i]
                if set(rows.index).issubset(self.keep) and (yrow.name in self.keep):
                    self.samples.append((rows.index.tolist(), yrow.name, str(bid)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, k):
        rows, yidx, bid = self.samples[k]
        X = self.df.loc[rows, self.features].to_numpy(dtype=np.float32)  # [T,F]
        nodes = torch.from_numpy(X.T.reshape(-1, 1).astype(np.float32))  # [F*T,1]
        start_times = self.df.loc[rows, "start_dt"].tolist()
        N = X.shape[1]
        T = self.T
        A, EF = build_spatiotemporal_graph(
            N, T, start_times,
            dt_mu=self.dt_mu, dt_sigma=self.dt_sigma,
            use_edge_feat=self.use_edge_feat, add_spatial=self.add_spatial
        )
        soh_true = float(self.df.loc[yidx, "SOH"])
        soh_prev = float(self.df.loc[rows[-1], "SOH"])
        y = (soh_true - soh_prev) if self.predict_delta else soh_true
        return nodes, A, EF, float(y), float(soh_prev), float(soh_true), bid

def collate(batch):
    Xs, As, EFs, ys, sp, st, bids = zip(*batch)
    X = torch.stack(Xs, 0)
    A = torch.stack(As, 0)
    EF = torch.stack(EFs, 0) if EFs[0] is not None else None
    y  = torch.tensor(ys, dtype=torch.float32)
    sp = torch.tensor(sp, dtype=torch.float32)
    st = torch.tensor(st, dtype=torch.float32)
    return X, A, EF, y, sp, st, list(bids)

#Models
class ResGraphLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.lin  = nn.Linear(in_dim, out_dim)
        self.skip = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim, bias=False)
        self.ln   = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)
        self.act  = nn.GELU()

    def forward(self, X, A):
        deg = (A.sum(-1, keepdim=True) + 1e-6)
        Ahat = A / deg
        H = torch.bmm(Ahat, X)
        H = self.lin(H)
        H = H + self.skip(X)
        H = self.ln(H)
        H = self.act(H)
        return self.drop(H)

class GNN(nn.Module):
    def __init__(self, node_in=1, hidden=192, layers=3, dropout=0.25):
        super().__init__()
        self.layers = nn.ModuleList([
            ResGraphLayer(node_in if i == 0 else hidden, hidden, dropout)
            for i in range(layers)
        ])
        self.pool_gate = nn.Linear(hidden, 1)  # attention pooling
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, X, A, EF=None):
        H = X
        for g in self.layers:
            H = g(H, A)
        w = torch.softmax(self.pool_gate(H).squeeze(-1), dim=-1)  # [B,n]
        H = torch.bmm(w.unsqueeze(1), H).squeeze(1)               # [B,hidden]
        return self.head(H).squeeze(-1)

#Train / Eval (MAE + RMSE)
def _mae_rmse(y_true: np.ndarray, y_pred: np.ndarray):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return float(mae), float(rmse)

def run_epoch(loader, model, lossf, optimizer=None, train=False, noise_std=0.0):
    model.train(train)

    losses = []
    soh_true_all = []
    soh_pred_all = []
    delta_true_all = []
    delta_pred_all = []

    for X, A, EF, y_delta, soh_prev, soh_true, _ in loader:
        X = X.to(DEVICE).float()
        A = A.to(DEVICE).float()
        y_delta = y_delta.to(DEVICE).float()

        if train and noise_std > 0:
            X = X + torch.randn_like(X) * noise_std

        if train and optimizer:
            optimizer.zero_grad()

        pred_delta = model(X, A, EF)
        loss = lossf(pred_delta, y_delta)

        if train and optimizer:
            loss.backward()
            optimizer.step()

        losses.append(loss.item())

        # Collect metrics on CPU
        pred_delta_cpu = pred_delta.detach().cpu().numpy()
        y_delta_cpu    = y_delta.detach().cpu().numpy()
        soh_prev_cpu   = soh_prev.detach().cpu().numpy()
        soh_true_cpu   = soh_true.detach().cpu().numpy()

        soh_pred_cpu = soh_prev_cpu + pred_delta_cpu

        delta_true_all.append(y_delta_cpu)
        delta_pred_all.append(pred_delta_cpu)
        soh_true_all.append(soh_true_cpu)
        soh_pred_all.append(soh_pred_cpu)

    loss_mean = float(np.mean(losses)) if losses else float("nan")

    delta_true = np.concatenate(delta_true_all) if delta_true_all else np.array([])
    delta_pred = np.concatenate(delta_pred_all) if delta_pred_all else np.array([])
    soh_true_v = np.concatenate(soh_true_all)   if soh_true_all   else np.array([])
    soh_pred_v = np.concatenate(soh_pred_all)   if soh_pred_all   else np.array([])

    d_mae, d_rmse = _mae_rmse(delta_true, delta_pred) if len(delta_true) else (float("nan"), float("nan"))
    s_mae, s_rmse = _mae_rmse(soh_true_v, soh_pred_v) if len(soh_true_v) else (float("nan"), float("nan"))

    return loss_mean, d_mae, d_rmse, s_mae, s_rmse

@torch.no_grad()
def evaluate_test(loader, model):
    model.eval()
    rows = []
    for X, A, EF, y_delta, soh_prev, soh_true, bids in loader:
        X = X.to(DEVICE).float()
        A = A.to(DEVICE).float()
        pred_delta = model(X, A, EF).detach().cpu().numpy()

        soh_prev_np = soh_prev.numpy()
        soh_true_np = soh_true.numpy()
        soh_pred_np = soh_prev_np + pred_delta

        for i in range(len(bids)):
            rows.append((bids[i], float(soh_true_np[i]), float(soh_pred_np[i])))

    df = pd.DataFrame(rows, columns=["battery_id", "true", "pred"])
    return df

def per_battery_stats(df_test: pd.DataFrame) -> pd.DataFrame:
    d = df_test.copy()
    d["ae"] = np.abs(d["pred"] - d["true"])
    d["se"] = (d["pred"] - d["true"]) ** 2
    g = d.groupby("battery_id").agg(
        mae=("ae", "mean"),
        mse=("se", "mean"),
        n=("ae", "size"),
    )
    g["rmse"] = np.sqrt(g["mse"])
    g = g.drop(columns=["mse"]).sort_values("mae")
    return g

#Plotting
def _ensure_dir(p): os.makedirs(p, exist_ok=True)

def plot_loss_curves(tag, train_losses, val_losses):
    plt.figure()
    plt.plot(train_losses, label="train loss")
    plt.plot(val_losses, label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("SmoothL1 loss")
    plt.title(f"{tag} - Loss Curves")
    plt.legend()
    _ensure_dir(RESULTS_DIR)
    out = os.path.join(RESULTS_DIR, f"{tag}_loss_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot saved] {out}")

def plot_training_curves(tag, tr_mae, va_mae, tr_rmse, va_rmse):
    plt.figure()
    plt.plot(tr_mae,  label="train SOH MAE")
    plt.plot(va_mae,  label="val SOH MAE")
    plt.plot(tr_rmse, label="train SOH RMSE")
    plt.plot(va_rmse, label="val SOH RMSE")
    plt.xlabel("epoch")
    plt.ylabel("error")
    plt.title(f"{tag} - Training Curves")
    plt.legend()
    _ensure_dir(RESULTS_DIR)
    out = os.path.join(RESULTS_DIR, f"{tag}_train_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot saved] {out}")

def plot_true_vs_pred(tag, df_test):
    plt.figure()
    plt.scatter(df_test["true"].values, df_test["pred"].values, s=15)
    mn = min(df_test["true"].min(), df_test["pred"].min())
    mx = max(df_test["true"].max(), df_test["pred"].max())
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel("True SOH")
    plt.ylabel("Predicted SOH")
    plt.title(f"{tag} - True vs Pred SOH")
    _ensure_dir(RESULTS_DIR)
    out = os.path.join(RESULTS_DIR, f"{tag}_true_vs_pred_soh.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot saved] {out}")

def plot_worst_battery(tag, df_test, worst_bid):
    g = df_test[df_test["battery_id"] == worst_bid].reset_index(drop=True)
    plt.figure()
    plt.plot(g["true"].values, label="True SOH")
    plt.plot(g["pred"].values, label="Pred SOH")
    plt.xlabel("test window idx")
    plt.ylabel("SOH")
    plt.title(f"Worst battery: {worst_bid}")
    plt.legend()
    _ensure_dir(RESULTS_DIR)
    out = os.path.join(RESULTS_DIR, f"{tag}_worst_battery_soh.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot saved] {out}")

#Main run
def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(CSV_PATH)

    per_cycle, FEATURES = build_per_cycle(CSV_PATH)

    tag = f"temp_resgcn_w{WINDOW}_h{HIDDEN}_do{DROPOUT}_nz{NOISE_STD}_sp1_batsplit"

    # split
    if SPLIT_MODE == "battery":
        train_idx, val_idx, test_idx = battery_split_index(per_cycle, seed=SEED)
    else:
        train_idx, val_idx, test_idx = temporal_split_index(per_cycle)

    # train-only scaling (sklearn)
    scaler = StandardScaler()
    Xtr = per_cycle.loc[train_idx, FEATURES].to_numpy(np.float32)
    scaler.fit(Xtr)

    pc = per_cycle.copy()
    Xall = pc.loc[:, FEATURES].to_numpy(np.float32)
    Xall_s = scaler.transform(Xall).astype(np.float32)


    for j, c in enumerate(FEATURES):
        pc[c] = Xall_s[:, j]

    # datasets/loaders
    train_ds = WindowGraphDataset(pc, train_idx, FEATURES, WINDOW, PREDICT_DELTA, use_edge_feat=True, add_spatial=True)
    val_ds   = WindowGraphDataset(pc, val_idx,   FEATURES, WINDOW, PREDICT_DELTA, use_edge_feat=True, add_spatial=True)
    test_ds  = WindowGraphDataset(pc, test_idx,  FEATURES, WINDOW, PREDICT_DELTA, use_edge_feat=True, add_spatial=True)

    if min(len(train_ds), len(val_ds), len(test_ds)) == 0:
        print(f"[{tag}] Not enough samples.")
        return

    print(f"[INFO] windows: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} | batteries={pc['battery_id'].nunique()}")

    train_ld = DataLoader(train_ds, batch_size=BATCH_TR,   shuffle=True,  collate_fn=collate)
    val_ld   = DataLoader(val_ds,   batch_size=BATCH_EVAL, shuffle=False, collate_fn=collate)
    test_ld  = DataLoader(test_ds,  batch_size=BATCH_EVAL, shuffle=False, collate_fn=collate)

    # model
    model = GNN(node_in=1, hidden=HIDDEN, layers=3, dropout=DROPOUT).to(DEVICE)
    lossf = nn.SmoothL1Loss(beta=0.01)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    # history
    tr_loss_hist, va_loss_hist = [], []
    tr_soh_mae_hist, va_soh_mae_hist = [], []
    tr_soh_rmse_hist, va_soh_rmse_hist = [], []

    best_val = float("inf")
    best_state = None
    pat = 0

    for ep in range(1, EPOCHS + 1):
        tr_loss, tr_d_mae, tr_d_rmse, tr_s_mae, tr_s_rmse = run_epoch(train_ld, model, lossf, opt, True, NOISE_STD)
        va_loss, va_d_mae, va_d_rmse, va_s_mae, va_s_rmse = run_epoch(val_ld,   model, lossf, None, False, 0.0)

        tr_loss_hist.append(tr_loss); va_loss_hist.append(va_loss)
        tr_soh_mae_hist.append(tr_s_mae); va_soh_mae_hist.append(va_s_mae)
        tr_soh_rmse_hist.append(tr_s_rmse); va_soh_rmse_hist.append(va_s_rmse)

        if ep % 5 == 0:
            print(
                f"[{tag}] epoch {ep:02d} | "
                f"train Δ(MAE/RMSE) {tr_d_mae:.4f}/{tr_d_rmse:.4f} | "
                f"val   Δ(MAE/RMSE) {va_d_mae:.4f}/{va_d_rmse:.4f} | "
                f"val SOH (MAE/RMSE) {va_s_mae:.4f}/{va_s_rmse:.4f}"
            )

        # early stopping on val SOH MAE
        if va_s_mae < best_val - 1e-4:
            best_val = va_s_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1

        if pat >= PATIENCE:
            print(f"[{tag}] Early stop at {ep} (best val SOH MAE {best_val:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # test metrics (SOH)
    df_test = evaluate_test(test_ld, model)
    test_mae = mean_absolute_error(df_test["true"].values, df_test["pred"].values)
    test_rmse = np.sqrt(mean_squared_error(df_test["true"].values, df_test["pred"].values))

    print(f"[{tag}] TEST | SOH (MAE/RMSE) = {test_mae:.4f}/{test_rmse:.4f}")

    # per-battery stats
    batt = per_battery_stats(df_test)
    print("\nPer-battery SOH MAE (best 5):")
    print(batt.head(5).to_string())
    print("\nPer-battery SOH MAE (worst 5):")
    print(batt.tail(5).to_string())

    worst_bid = batt.index[-1]
    print(f"[INFO] worst battery by MAE: {worst_bid}")

    # plots
    plot_loss_curves(tag, tr_loss_hist, va_loss_hist)
    plot_training_curves(tag, tr_soh_mae_hist, va_soh_mae_hist, tr_soh_rmse_hist, va_soh_rmse_hist)
    plot_true_vs_pred(tag, df_test)
    plot_worst_battery(tag, df_test, worst_bid)


    os.makedirs(MODEL_DIR, exist_ok=True)
    ckpt = os.path.join(MODEL_DIR, f"{tag}.pt")
    torch.save(model.state_dict(), ckpt)
    print(f"[saved] {ckpt}")

if __name__ == "__main__":
    main()
