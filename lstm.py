# soh_lstm_windows_UPGRADED.py
import os, random, re
from datetime import datetime
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Metrics + plotting
from sklearn.metrics import mean_absolute_error, mean_squared_error
try:
    from sklearn.metrics import root_mean_squared_error
except Exception:
    root_mean_squared_error = None

import matplotlib.pyplot as plt


#Config
CSV_PATH      = "metadata.csv"
assert os.path.basename(CSV_PATH) == "metadata.csv", "Only metadata.csv is allowed."

NOMINAL_C     = 2.0
SEED          = 42

WINDOW_T      = 12
STEP          = 1
BATCH_TR      = 64
BATCH_EVAL    = 128

EPOCHS        = 70
PATIENCE      = 10
LR            = 5e-4
WD            = 3e-4
DROPOUT       = 0.3
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

HIDDEN        = 128
LAYERS        = 2
BIDIR         = False

MODEL_DIR     = "models_lstm"
PLOTS_DIR     = "results_lstm"   # NEW: plots saved here


def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
set_seed(SEED)


#Utils
class StandardScalerNP:
    def __init__(self): self.mean_=None; self.scale_=None
    def fit(self,X):
        X=np.asarray(X,dtype=np.float32)
        self.mean_=np.nanmean(X,axis=0)
        self.scale_=np.nanstd (X,axis=0,ddof=0)
        self.scale_[~np.isfinite(self.scale_)] = 1.0
        self.scale_[self.scale_==0] = 1.0
        return self
    def transform(self,X):
        X=np.asarray(X,dtype=np.float32)
        Z=(X-self.mean_)/self.scale_
        Z[~np.isfinite(Z)] = 0.0
        return Z

def parse_matlab_datevec(s):
    s=str(s).strip().strip('[]').replace(',',' ')
    p=[t for t in s.split() if t]
    if len(p)<6: return pd.NaT
    y,m,d,hh,mm,ss=[float(x) for x in p[:6]]
    si=int(ss); micro=int(round((ss-si)*1_000_000))
    try: return datetime(int(y),int(m),int(d),int(hh),int(mm),si,micro)
    except: return pd.NaT

def _find_col(df, candidates):
    lower={c.lower().strip():c for c in df.columns}
    tight={re.sub(r'[\s_\-]+','',c.lower()):c for c in df.columns}
    for cand in candidates:
        if cand in df.columns: return cand
        k=cand.lower().strip()
        if k in lower: return lower[k]
        k2=re.sub(r'[\s_\-]+','',cand.lower())
        if k2 in tight: return tight[k2]
    return None

def _ensure_col(df, target, candidates, required=False):
    src=_find_col(df, candidates+[target])
    if src is None:
        if required: raise KeyError(f"Missing column '{target}' (tried {candidates})")
        return None
    if src != target: df.rename(columns={src:target}, inplace=True)
    return target


# Build per-cycle
def build_per_cycle(csv_path):
    df=pd.read_csv(csv_path)
    df.columns=[str(c).strip() for c in df.columns]

    _ensure_col(df,"battery_id",["battery_id","Battery_ID","cell","cell_id","id","unit"],True)
    _ensure_col(df,"start_time",["start_time","start","time","timestamp","datetime","date"],True)
    _ensure_col(df,"type",["type","operation","op","mode"],False)
    _ensure_col(df,"ambient_temperature",["ambient_temperature","temperature","temp","Temperature_measured"],False)
    _ensure_col(df,"Re",["Re","r_e","electrolyte_resistance"],False)
    _ensure_col(df,"Rct",["Rct","r_ct","charge_transfer_resistance"],False)
    _ensure_col(df,"Capacity",["Capacity","capacity","Qd","Q_discharge","discharge_capacity"],True)

    # parse + sort
    df["start_dt"]=df["start_time"].apply(parse_matlab_datevec)
    df=df.sort_values(["battery_id","start_dt"]).reset_index(drop=True)

    # numeric coercion (before fills)
    for c in ["ambient_temperature","Re","Rct","Capacity"]:
        if c in df.columns: df[c]=pd.to_numeric(df[c],errors="coerce")

    # fill BEFORE filtering
    before = {c: int(df[c].isna().sum()) for c in ["ambient_temperature","Re","Rct"] if c in df.columns}
    for c in ["Re","Rct","ambient_temperature"]:
        if c in df.columns:
            df[c]=df.groupby("battery_id")[c].ffill().bfill()
            if df[c].isna().any(): df[c]=df[c].fillna(df[c].median())
    after = {c: int(df[c].isna().sum()) for c in ["ambient_temperature","Re","Rct"] if c in df.columns}
    print(f"[DBG] NaNs before fill: {before} | after: {after}")

    # keep discharge rows if present
    if "type" in df.columns:
        tl=df["type"].astype(str).str.lower()
        dmask=tl.str.contains("discha", na=False)
        df = df[dmask] if dmask.any() else df

    # cycle index after filtering
    df=df.sort_values(["battery_id","start_dt"]).reset_index(drop=True)
    df["cycle_idx"]=df.groupby("battery_id").cumcount()

    # SOH
    df["SOH"]=(df["Capacity"]/NOMINAL_C).astype(float)

    # engineered features 
    eps=1e-6
    if "Re" in df.columns:
        df["Re0"]   = df.groupby("battery_id")["Re"].transform("first")
        df["dRe"]   = df["Re"]-df["Re0"]
        df["logRe"] = np.log(df["Re"].clip(lower=eps))
        df["Re_rm3"]= df.groupby("battery_id")["Re"].transform(lambda s: s.rolling(3,min_periods=1).mean())

    if "Rct" in df.columns:
        df["Rct0"]    = df.groupby("battery_id")["Rct"].transform("first")
        df["dRct"]    = df["Rct"]-df["Rct0"]
        df["logRct"]  = np.log(df["Rct"].clip(lower=eps))
        df["Rct_rm3"] = df.groupby("battery_id")["Rct"].transform(lambda s: s.rolling(3,min_periods=1).mean())

    if "ambient_temperature" in df.columns:
        df["T0"]         = df.groupby("battery_id")["ambient_temperature"].transform("first")
        df["Temp_prev"]  = df.groupby("battery_id")["ambient_temperature"].shift(1)
        df["dTemp"]      = df["ambient_temperature"]-df["T0"]
        df["dTemp_prev"] = df["ambient_temperature"]-df["Temp_prev"]

    df["SOH_prev"]  = df.groupby("battery_id")["SOH"].shift(1)
    df["max_cycle"] = df.groupby("battery_id")["cycle_idx"].transform("max")
    df["cycle_frac"]= df["cycle_idx"]/df["max_cycle"].replace(0,1)

    FEATURES = [c for c in [
        "ambient_temperature","dTemp","dTemp_prev","SOH_prev",
        "Re","Rct","dRe","dRct","logRe","logRct","Re_rm3","Rct_rm3",
        "cycle_frac",
    ] if c in df.columns]

    need=["battery_id","cycle_idx","start_dt","SOH"]
    df=df.dropna(subset=[c for c in need if c in df.columns]).reset_index(drop=True)

    print(f"[INFO] per-cycle rows (after discharge filter if any): {len(df)} | batteries: {df['battery_id'].nunique()} | features: {len(FEATURES)}")
    return df, FEATURES


#Windows with IN-WINDOW imputation
def build_all_windows(df, FEATURES, T=12, step=1):
    feat_medians = df[FEATURES].median().to_dict()

    X_list, y_list, sp_list, st_list, bid_list = [], [], [], [], []
    for bid, g in df.groupby("battery_id"):
        g=g.sort_values("cycle_idx")
        if len(g) < T+1: continue
        for i in range(T, len(g), step):
            rows = g.iloc[i-T:i].copy()
            yrow = g.iloc[i]
            block = rows[FEATURES].copy()

            
            for f in FEATURES:
                if block[f].isna().any():
                    block.loc[block[f].isna(), f] = feat_medians.get(f, 0.0)

            soh_prev = rows["SOH"].iloc[-1]
            soh_true = yrow["SOH"]
            if not np.isfinite(soh_prev) or not np.isfinite(soh_true):
                continue

            y = float(soh_true - soh_prev)
            X_list.append(block.to_numpy(dtype=np.float32))
            y_list.append(np.float32(y))
            sp_list.append(np.float32(soh_prev))
            st_list.append(np.float32(soh_true))
            bid_list.append(str(bid))

    if not X_list: return None
    X = np.stack(X_list, axis=0)          # [N, T, F]
    y = np.array(y_list, dtype=np.float32)
    sp= np.array(sp_list, dtype=np.float32)
    st= np.array(st_list, dtype=np.float32)
    bids = np.array(bid_list, dtype=object)
    return X, y, sp, st, bids


#Battery-level split
def split_by_battery_ids(bids, train=0.7, val=0.1, test=0.2, seed=42):
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(bids)))
    rng.shuffle(uniq)
    n = len(uniq); ntr=int(n*train); nva=int(n*val)
    tr_ids = set(uniq[:ntr]); va_ids = set(uniq[ntr:ntr+nva]); te_ids = set(uniq[ntr+nva:])
    idx = np.arange(len(bids))
    tr = idx[np.isin(bids, list(tr_ids))]
    va = idx[np.isin(bids, list(va_ids))]
    te = idx[np.isin(bids, list(te_ids))]
    return tr, va, te


#Dataset & Model
class WinDataset(Dataset):
    def __init__(self, X, y, sp, st, bids):
        self.X=torch.from_numpy(X)
        self.y=torch.from_numpy(y)
        self.sp=torch.from_numpy(sp)
        self.st=torch.from_numpy(st)
        self.bids=np.array(bids, dtype=object)

    def __len__(self): return self.X.shape[0]

    def __getitem__(self,i):
        return self.X[i], self.y[i], self.sp[i], self.st[i], str(self.bids[i])


class LSTMRegressor(nn.Module):
    def __init__(self, in_dim, hidden=128, layers=2, dropout=0.3, bidir=False):
        super().__init__()
        self.lstm=nn.LSTM(
            input_size=in_dim, hidden_size=hidden, num_layers=layers,
            dropout=dropout if layers>1 else 0.0,
            batch_first=True, bidirectional=bidir
        )
        out_dim=hidden*(2 if bidir else 1)
        self.head=nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim//2, 1)
        )
    def forward(self,X):
        out,_=self.lstm(X)
        hlast=out[:,-1,:]
        return self.head(hlast).squeeze(-1)


#Metrics helpers
def _mae_rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mae = mean_absolute_error(y_true, y_pred)
    if root_mean_squared_error is not None:
        rmse = root_mean_squared_error(y_true, y_pred)
    else:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return float(mae), float(rmse)


#Train / Eval 
def run_epoch(loader, model, lossf, optimizer=None, train=False):
    model.train(train)

    losses = []
    y_true_d, y_pred_d = [], []
    y_true_soh, y_pred_soh = [], []
    bids_all = []

    for X,y,sp,st,bids in loader:
        X=X.to(DEVICE).float()
        y=y.to(DEVICE).float()
        sp=sp.to(DEVICE).float()
        st=st.to(DEVICE).float()

        if train and optimizer:
            optimizer.zero_grad()

        pred_d = model(X)
        loss = lossf(pred_d, y)

        if train and optimizer:
            loss.backward()
            optimizer.step()

        losses.append(loss.detach().cpu().item())

        pred_d_cpu = pred_d.detach().cpu().numpy()
        y_cpu      = y.detach().cpu().numpy()

        pred_soh_cpu = (sp + pred_d).detach().cpu().numpy()
        true_soh_cpu = st.detach().cpu().numpy()

        y_true_d.append(y_cpu); y_pred_d.append(pred_d_cpu)
        y_true_soh.append(true_soh_cpu); y_pred_soh.append(pred_soh_cpu)
        bids_all.extend(list(bids))

    if len(y_true_d) == 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, [])

    y_true_d   = np.concatenate(y_true_d)
    y_pred_d   = np.concatenate(y_pred_d)
    y_true_soh = np.concatenate(y_true_soh)
    y_pred_soh = np.concatenate(y_pred_soh)

    d_mae, d_rmse = _mae_rmse(y_true_d, y_pred_d)
    s_mae, s_rmse = _mae_rmse(y_true_soh, y_pred_soh)
    loss_mean = float(np.mean(losses)) if losses else np.nan

    return d_mae, d_rmse, s_mae, s_rmse, loss_mean, bids_all


def collect_predictions(loader, model):
    model.eval()
    bids_all = []
    true_soh_all, pred_soh_all = [], []

    with torch.no_grad():
        for X,y,sp,st,bids in loader:
            X=X.to(DEVICE).float()
            sp=sp.to(DEVICE).float()
            st=st.to(DEVICE).float()

            pred_d = model(X)
            pred_soh = (sp + pred_d).detach().cpu().numpy()
            true_soh = st.detach().cpu().numpy()

            bids_all.extend(list(bids))
            true_soh_all.append(true_soh)
            pred_soh_all.append(pred_soh)

    true_soh_all = np.concatenate(true_soh_all) if true_soh_all else np.array([])
    pred_soh_all = np.concatenate(pred_soh_all) if pred_soh_all else np.array([])

    return bids_all, true_soh_all, pred_soh_all


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

    epochs = [h["epoch"] for h in history]
    tr_loss = [h["tr_loss"] for h in history]
    va_loss = [h["va_loss"] for h in history]
    tr_mae  = [h["tr_soh_mae"] for h in history]
    va_mae  = [h["va_soh_mae"] for h in history]
    tr_rmse = [h["tr_soh_rmse"] for h in history]
    va_rmse = [h["va_soh_rmse"] for h in history]

    # Train curves
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

    # Loss curves
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

    # Test summary
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

    # True vs Pred scatter
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

        # SOH over samples
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

    # Per-battery MAE top-25
    items_mae = sorted(per_batt.items(), key=lambda kv: kv[1]["mae"], reverse=True)
    top_mae = items_mae[:min(25, len(items_mae))]
    if top_mae:
        bnames = [k for k,_ in top_mae]
        maes   = [v["mae"] for _,v in top_mae]
        plt.figure(figsize=(10,5))
        plt.bar(bnames, maes)
        plt.xticks(rotation=60, ha="right")
        plt.ylabel("SOH MAE")
        plt.title(f"{tag} - Worst Batteries by SOH MAE (Top {len(top_mae)})")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{tag}_per_battery_mae_top.png"), dpi=160)
        plt.close()

    # Per-battery RMSE top-25
    items_rmse = sorted(per_batt.items(), key=lambda kv: kv[1]["rmse"], reverse=True)
    top_rmse = items_rmse[:min(25, len(items_rmse))]
    if top_rmse:
        bnames = [k for k,_ in top_rmse]
        rmses  = [v["rmse"] for _,v in top_rmse]
        plt.figure(figsize=(10,5))
        plt.bar(bnames, rmses)
        plt.xticks(rotation=60, ha="right")
        plt.ylabel("SOH RMSE")
        plt.title(f"{tag} - Worst Batteries by SOH RMSE (Top {len(top_rmse)})")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{tag}_per_battery_rmse_top.png"), dpi=160)
        plt.close()

    # Worst battery trace 
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

    # 1) per-cycle table
    df, FEATURES = build_per_cycle(CSV_PATH)

    # 2) windows
    built=build_all_windows(df, FEATURES, T=WINDOW_T, step=STEP)
    if built is None:
        print("No windows built at all. Try smaller WINDOW_T (e.g., 6 or 8) or check data.")
        return
    X,y,sp,st,bids=built
    print(f"[INFO] Built windows: {X.shape[0]} | per-window shape {X.shape[1:]} | batteries: {len(np.unique(bids))}")

    # 3) NaN safety
    mask = np.isfinite(X).all(axis=(1,2)) & np.isfinite(y) & np.isfinite(sp) & np.isfinite(st)
    X,y,sp,st,bids = X[mask], y[mask], sp[mask], st[mask], bids[mask]
    if X.shape[0]==0:
        print("All windows invalid after NaN filtering. Check data.")
        return

    # 4) split by battery IDs
    tr_idx, va_idx, te_idx = split_by_battery_ids(bids, train=0.7, val=0.1, test=0.2, seed=SEED)

    # 5) scale: fit on train windows only
    scaler = StandardScalerNP()
    X_tr_flat = X[tr_idx].reshape(-1, X.shape[-1])
    scaler.fit(X_tr_flat)

    def apply_scale(Xarr):
        Xflat = Xarr.reshape(-1, Xarr.shape[-1])
        Xflat = scaler.transform(Xflat)
        return Xflat.reshape(Xarr.shape)

    X_tr = apply_scale(X[tr_idx]); y_tr=y[tr_idx]; sp_tr=sp[tr_idx]; st_tr=st[tr_idx]; bids_tr=bids[tr_idx]
    X_va = apply_scale(X[va_idx]); y_va=y[va_idx]; sp_va=sp[va_idx]; st_va=st[va_idx]; bids_va=bids[va_idx]
    X_te = apply_scale(X[te_idx]); y_te=y[te_idx]; sp_te=sp[te_idx]; st_te=st[te_idx]; bids_te=bids[te_idx]

    tr_ds=WinDataset(X_tr, y_tr, sp_tr, st_tr, bids_tr)
    va_ds=WinDataset(X_va, y_va, sp_va, st_va, bids_va)
    te_ds=WinDataset(X_te, y_te, sp_te, st_te, bids_te)

    tr_ld=DataLoader(tr_ds,batch_size=BATCH_TR,shuffle=True)
    va_ld=DataLoader(va_ds,batch_size=BATCH_EVAL,shuffle=False)
    te_ld=DataLoader(te_ds,batch_size=BATCH_EVAL,shuffle=False)

    # 6) model
    model=LSTMRegressor(in_dim=X.shape[-1], hidden=HIDDEN, layers=LAYERS, dropout=DROPOUT, bidir=BIDIR).to(DEVICE)
    lossf=nn.SmoothL1Loss(beta=0.01)
    opt=torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    tag=f"lstm_win{WINDOW_T}_h{HIDDEN}_L{LAYERS}_bi{int(BIDIR)}_do{DROPOUT}_byBAT_impute"

    best_val=float("inf"); best_state=None; pat=0
    history=[]

    for ep in range(1, EPOCHS+1):
        tr_d_mae, tr_d_rmse, tr_s_mae, tr_s_rmse, tr_loss, _ = run_epoch(tr_ld, model, lossf, opt, True)
        va_d_mae, va_d_rmse, va_s_mae, va_s_rmse, va_loss, _ = run_epoch(va_ld, model, lossf, None, False)

        history.append({
            "epoch": ep,
            "tr_loss": tr_loss,
            "va_loss": va_loss,
            "tr_soh_mae": tr_s_mae,
            "va_soh_mae": va_s_mae,
            "tr_soh_rmse": tr_s_rmse,
            "va_soh_rmse": va_s_rmse,
        })

        if ep%5==0:
            print(f"[lstm] epoch {ep:02d} | "
                  f"train Δ(MAE/RMSE) {tr_d_mae:.4f}/{tr_d_rmse:.4f} | "
                  f"val   Δ(MAE/RMSE) {va_d_mae:.4f}/{va_d_rmse:.4f} | "
                  f"val SOH (MAE/RMSE) {va_s_mae:.4f}/{va_s_rmse:.4f}")

        if va_s_mae < best_val - 1e-4:
            best_val = va_s_mae
            best_state = model.state_dict()
            pat = 0
        else:
            pat += 1

        if pat >= PATIENCE:
            print(f"[lstm] Early stop at {ep} (best val SOH MAE {best_val:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    te_d_mae, te_d_rmse, te_s_mae, te_s_rmse, te_loss, _ = run_epoch(te_ld, model, lossf, None, False)
    print(f"[lstm] TEST | ΔSOH (MAE/RMSE) = {te_d_mae:.4f}/{te_d_rmse:.4f} | "
          f"SOH (MAE/RMSE) = {te_s_mae:.4f}/{te_s_rmse:.4f}")

    # collect predictions for plots
    bids_all, true_soh, pred_soh = collect_predictions(te_ld, model)
    per_batt = per_battery_metrics(bids_all, true_soh, pred_soh)

    # save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"{tag}.pt"))
    print(f"[saved] {MODEL_DIR}/{tag}.pt")

    test_metrics = {"delta_mae": te_d_mae, "delta_rmse": te_d_rmse, "soh_mae": te_s_mae, "soh_rmse": te_s_rmse}
    make_plots(tag, history, test_metrics, true_soh, pred_soh, per_batt, bids_all)


if __name__ == "__main__":
    main()
