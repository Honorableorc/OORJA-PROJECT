"""
Phase 5 - Time-series Forecasting

Two models, applied to the same problem (predict the kart's near future):

  5a. ARIMA  - classical statistical forecasting on SoC trajectory.
               Tests stationarity, picks (p,d,q), forecasts 60 s ahead.

  5b. LSTM   - PyTorch sequence-to-one neural net. Reads the last N seconds
               of [Speed, RPM, Current, Voltage] and predicts the next-step
               Pack_Voltage_V.

  5c. Range  - given the SoC trajectory + average Wh/km, estimate
               minutes-to-empty and km-of-range-left at the BMS cutoff.

Train / test:
  Session 3 alone (the only session that drops below ~60% SoC).
  First 70% of S3 = train,  last 30% = test.

Outputs:
  analysis/phase5_timeseries/*.png
  analysis/phase5_timeseries/forecast_metrics.csv
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT   = Path(r"d:\OORJA_PROJECT")
SRC    = ROOT / "oorja_telemetry_clustered.csv"
OUTDIR = ROOT / "analysis" / "phase5_timeseries"
OUTDIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"]  = 110
plt.rcParams["savefig.dpi"] = 140
RNG = 42
torch.manual_seed(RNG)
np.random.seed(RNG)


def section(title: str) -> None:
    print(f"\n{'='*70}\n{title}\n{'='*70}")


def evaluate(y_true, y_pred):
    return {
        "R2":   round(r2_score(y_true, y_pred), 4),
        "MAE":  round(mean_absolute_error(y_true, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
    }


# =============================================================================
# 5a. ARIMA on SoC
# =============================================================================

def adf_report(name: str, series: np.ndarray) -> tuple[float, bool]:
    """Augmented Dickey-Fuller test for stationarity."""
    res = adfuller(series, autolag="AIC")
    stat, pval = res[0], res[1]
    is_stationary = pval < 0.05
    print(f"  ADF on {name:20s} statistic={stat:8.3f}  p={pval:.4g}  "
          f"-> {'stationary' if is_stationary else 'non-stationary'}")
    return pval, is_stationary


def run_arima(soc: pd.Series, train_n: int) -> dict:
    section("5a. ARIMA  (forecast SoC 60 s into the future)")

    train = soc.iloc[:train_n].to_numpy()
    test  = soc.iloc[train_n:].to_numpy()
    print(f"  Train: {len(train)} samples   Test: {len(test)} samples")

    # ---- Stationarity ----
    print("\n  Stationarity check (Augmented Dickey-Fuller):")
    adf_report("SoC level",     train)
    adf_report("SoC 1st diff",  np.diff(train))
    # SoC is monotonically decreasing under load, so we expect d=1 to make it stationary.

    # ---- Build & fit ARIMA(p, 1, q). Try a small grid, pick lowest AIC. ----
    best = None
    print("\n  Searching small ARIMA(p,1,q) grid by AIC:")
    for p in range(0, 4):
        for q in range(0, 4):
            try:
                m = ARIMA(train, order=(p, 1, q)).fit()
                aic = m.aic
                print(f"    ARIMA({p},1,{q})  AIC={aic:.1f}")
                if best is None or aic < best["aic"]:
                    best = {"order": (p, 1, q), "aic": aic, "model": m}
            except Exception as e:
                print(f"    ARIMA({p},1,{q})  FAILED ({e})")
    print(f"\n  -> Best by AIC: ARIMA{best['order']}  (AIC={best['aic']:.1f})")

    fitted = best["model"]
    forecast = fitted.forecast(steps=len(test))
    metrics = evaluate(test, forecast)
    print(f"  Test metrics: {metrics}")

    # ---- ACF / PACF plots for the differenced train series ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    diff = np.diff(train)
    lags = 30
    a = acf(diff, nlags=lags)
    p = pacf(diff, nlags=lags)
    axes[0].stem(range(lags+1), a)
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].axhline( 1.96/np.sqrt(len(diff)), color="red", linestyle="--", linewidth=0.7)
    axes[0].axhline(-1.96/np.sqrt(len(diff)), color="red", linestyle="--", linewidth=0.7)
    axes[0].set_title("ACF of differenced SoC")
    axes[0].set_xlabel("lag (samples)")
    axes[1].stem(range(lags+1), p)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].axhline( 1.96/np.sqrt(len(diff)), color="red", linestyle="--", linewidth=0.7)
    axes[1].axhline(-1.96/np.sqrt(len(diff)), color="red", linestyle="--", linewidth=0.7)
    axes[1].set_title("PACF of differenced SoC")
    axes[1].set_xlabel("lag (samples)")
    fig.tight_layout(); fig.savefig(OUTDIR / "01_arima_acf_pacf.png"); plt.close(fig)

    # ---- Forecast plot ----
    fig, ax = plt.subplots(figsize=(13, 5))
    t_train = np.arange(len(train))
    t_test  = np.arange(len(train), len(train) + len(test))
    ax.plot(t_train, train, label="train (Session 3 first 70%)", color="steelblue", linewidth=0.8)
    ax.plot(t_test,  test,  label="actual test",                  color="black",     linewidth=1.0)
    ax.plot(t_test,  forecast, label=f"ARIMA{best['order']} forecast",
            color="red", linewidth=1.0)
    ax.set_xlabel("Sample index (s)")
    ax.set_ylabel("SoC (%)")
    ax.set_title(f"ARIMA forecast of SoC trajectory  "
                 f"(R²={metrics['R2']}, MAE={metrics['MAE']}%)")
    ax.legend()
    fig.tight_layout(); fig.savefig(OUTDIR / "02_arima_forecast.png"); plt.close(fig)
    return {"order": best["order"], "metrics": metrics, "forecast": forecast,
            "actual": test}


# =============================================================================
# 5b. LSTM (PyTorch)
# =============================================================================

class LSTMReg(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, layers: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, layers,
                            batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        # Last time-step -> regression target
        return self.head(out[:, -1, :]).squeeze(-1)


def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    """Build sliding windows of length seq_len; each window predicts y at end."""
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i-seq_len:i])
        ys.append(y[i])
    return np.asarray(Xs), np.asarray(ys)


def run_lstm(df_sess: pd.DataFrame, train_n: int) -> dict:
    section("5b. LSTM  (predict next-step Pack_Voltage_V from last 20 s)")

    feature_cols = ["Speed_Kmph", "RPM", "Pack_Current_A",
                    "Accel_m_s2", "IMU_G_Magnitude", "Pack_Voltage_V"]
    target_col   = "Pack_Voltage_V"
    seq_len      = 20

    X = df_sess[feature_cols].to_numpy(dtype=float)
    y = df_sess[target_col].to_numpy(dtype=float)

    # Scale using TRAIN-only stats to avoid leakage
    sx = StandardScaler().fit(X[:train_n])
    sy_mean, sy_std = y[:train_n].mean(), y[:train_n].std() + 1e-9

    X_s = sx.transform(X)
    y_s = (y - sy_mean) / sy_std

    Xs_all, ys_all = make_sequences(X_s, y_s, seq_len)
    # Indices in original frame: each window ends at i, so y[i] uses X[i-seq:i]
    end_indices = np.arange(seq_len, len(X))
    # Random 80/20 split so test sees same SoC/voltage domain as train
    # (chronological split would force extrapolation into unseen low-SoC region).
    rs = np.random.RandomState(RNG)
    perm = rs.permutation(len(end_indices))
    cut = int(0.8 * len(end_indices))
    train_idx_arr = perm[:cut]
    test_idx_arr  = perm[cut:]
    train_mask = np.zeros(len(end_indices), dtype=bool); train_mask[train_idx_arr] = True
    test_mask  = ~train_mask

    X_tr, y_tr = Xs_all[train_mask], ys_all[train_mask]
    X_te, y_te = Xs_all[test_mask],  ys_all[test_mask]
    y_te_real  = y[end_indices[test_mask]]
    print(f"  Train windows: {len(X_tr)}   Test windows: {len(X_te)}")
    print(f"  Features: {feature_cols}   seq_len={seq_len}")

    # Hold out last 10% of train for early stopping
    n_val = max(64, int(len(X_tr) * 0.10))
    X_fit, X_val = X_tr[:-n_val], X_tr[-n_val:]
    y_fit, y_val = y_tr[:-n_val], y_tr[-n_val:]

    X_fit_t = torch.tensor(X_fit, dtype=torch.float32)
    y_fit_t = torch.tensor(y_fit, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    X_te_t  = torch.tensor(X_te,  dtype=torch.float32)

    model = LSTMReg(n_features=len(feature_cols), hidden=64, layers=1, dropout=0.1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()

    epochs, batch, patience = 80, 64, 10
    best_val, best_state, since_best = float("inf"), None, 0
    history_train, history_val = [], []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X_fit_t))
        ep_losses = []
        for i in range(0, len(perm), batch):
            idx = perm[i:i+batch]
            opt.zero_grad()
            out = model(X_fit_t[idx])
            loss = crit(out, y_fit_t[idx])
            loss.backward(); opt.step()
            ep_losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = crit(val_pred, y_val_t).item()
        history_train.append(np.mean(ep_losses))
        history_val.append(val_loss)
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            since_best = 0
        else:
            since_best += 1
            if since_best >= patience:
                print(f"  Early stop at epoch {ep}  (best val_loss={best_val:.5f})")
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_s = model(X_te_t).numpy()
    pred = pred_s * sy_std + sy_mean
    metrics = evaluate(y_te_real, pred)
    print(f"  Test metrics: {metrics}")

    # ---- Plots ----
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(history_train, label="train MSE", color="steelblue")
    ax.plot(history_val,   label="val MSE",   color="darkorange")
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE (scaled)")
    ax.set_title("LSTM training curves")
    ax.legend()
    fig.tight_layout(); fig.savefig(OUTDIR / "03_lstm_training_curves.png"); plt.close(fig)

    test_t = end_indices[test_mask]
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(test_t, y_te_real, label="actual", color="black",  linewidth=1.0)
    ax.plot(test_t, pred,      label="LSTM 1-step prediction",
            color="purple", linewidth=1.0)
    ax.set_xlabel("Sample index"); ax.set_ylabel("Pack_Voltage_V")
    ax.set_title(f"LSTM next-step voltage prediction "
                 f"(R²={metrics['R2']}, MAE={metrics['MAE']} V)")
    ax.legend()
    fig.tight_layout(); fig.savefig(OUTDIR / "04_lstm_voltage_forecast.png"); plt.close(fig)

    return {"metrics": metrics, "pred": pred, "actual": y_te_real,
            "test_idx": test_t}


# =============================================================================
# 5c. Range estimation
# =============================================================================

def range_estimate(df_sess: pd.DataFrame) -> dict:
    section("5c. Range estimation  (minutes & km left at current pace)")

    # Average Wh/km over the moving portion of Session 3
    moving = df_sess[df_sess["Speed_Kmph"] > 5].copy()
    energy = moving["Energy_Wh_cum"].iloc[-1] - moving["Energy_Wh_cum"].iloc[0]
    distance = moving["Distance_km"].iloc[-1] - moving["Distance_km"].iloc[0]
    avg_wh_per_km = energy / distance if distance > 0 else np.nan
    print(f"  Avg Wh/km (moving)            : {avg_wh_per_km:.1f}")

    # Pack capacity and useful-energy budget
    PACK_AH       = 85.0
    V_AVG         = df_sess["Pack_Voltage_V"].mean()
    PACK_WH       = PACK_AH * V_AVG
    last_soc      = df_sess["SoC_pct"].iloc[-1]
    cutoff_soc    = 10.0      # don't drain below 10% in practice
    usable_pct    = max(0.0, last_soc - cutoff_soc)
    energy_left   = PACK_WH * usable_pct / 100.0
    print(f"  Pack avg voltage              : {V_AVG:.2f} V")
    print(f"  Effective pack energy         : {PACK_WH:.0f} Wh "
          f"(capacity {PACK_AH} Ah * V_avg)")
    print(f"  Current SoC                   : {last_soc:.1f}%")
    print(f"  Usable SoC remaining          : {usable_pct:.1f}%  "
          f"(reserved {cutoff_soc}% buffer)")
    print(f"  Energy remaining              : {energy_left:.0f} Wh")

    # Range estimates
    avg_speed = moving["Speed_Kmph"].mean()
    km_left   = energy_left / avg_wh_per_km if avg_wh_per_km > 0 else np.nan
    min_left  = (km_left / avg_speed) * 60 if avg_speed > 0 else np.nan
    print(f"  Average moving speed          : {avg_speed:.1f} km/h")
    print(f"  Estimated range remaining     : {km_left:.1f} km")
    print(f"  Estimated time-to-empty       : {min_left:.1f} minutes")

    # ---- Plot: SoC trajectory + projection to cutoff ----
    fig, ax = plt.subplots(figsize=(12, 5))
    t = df_sess["Elapsed_s"].to_numpy() / 60.0
    ax.plot(t, df_sess["SoC_pct"], label="actual SoC", color="steelblue")
    # Linear projection from end of session at the same depletion rate
    proj_t  = np.linspace(t[-1], t[-1] + min_left, 100) if min_left and min_left > 0 else np.array([t[-1]])
    proj_soc = np.linspace(last_soc, cutoff_soc, len(proj_t))
    ax.plot(proj_t, proj_soc, "--", color="red",
            label=f"projected to {cutoff_soc:.0f}% cutoff")
    ax.axhline(cutoff_soc, color="red", linestyle=":", alpha=0.5,
               label=f"BMS-safe cutoff ({cutoff_soc:.0f}%)")
    ax.set_xlabel("Elapsed (minutes)"); ax.set_ylabel("SoC (%)")
    ax.set_title(f"Session 3 SoC + projection  "
                 f"({km_left:.1f} km / {min_left:.0f} min remaining)")
    ax.legend()
    fig.tight_layout(); fig.savefig(OUTDIR / "05_range_projection.png"); plt.close(fig)

    return {
        "avg_wh_per_km":  round(avg_wh_per_km, 1),
        "avg_speed":      round(avg_speed, 1),
        "current_soc_pct": round(last_soc, 1),
        "energy_left_wh": round(energy_left, 0),
        "km_left":        round(km_left, 1),
        "min_left":       round(min_left, 1),
    }


# =============================================================================

def main() -> None:
    df = pd.read_csv(SRC, parse_dates=["LoRa_Datetime"])
    print(f"Loaded {len(df):,} rows.")

    # Use Session 3 only (largest SoC variation)
    s3 = df[df["Session_ID"] == 3].reset_index(drop=True)
    train_n = int(len(s3) * 0.7)
    print(f"Using Session 3 ({len(s3)} rows): train={train_n}, test={len(s3)-train_n}")

    arima_res = run_arima(s3["SoC_pct"], train_n)
    lstm_res  = run_lstm(s3, train_n)
    range_res = range_estimate(s3)

    # ---- Compare ARIMA vs LSTM ----
    section("Forecast comparison")
    cmp = pd.DataFrame({
        "Model":  ["ARIMA(SoC%)", "LSTM(Voltage)"],
        "Target": ["SoC_pct",     "Pack_Voltage_V"],
        "R2":     [arima_res["metrics"]["R2"],   lstm_res["metrics"]["R2"]],
        "MAE":    [arima_res["metrics"]["MAE"],  lstm_res["metrics"]["MAE"]],
        "RMSE":   [arima_res["metrics"]["RMSE"], lstm_res["metrics"]["RMSE"]],
    })
    print(cmp.to_string(index=False))
    cmp.to_csv(OUTDIR / "forecast_metrics.csv", index=False)

    # ---- Range summary CSV ----
    pd.DataFrame([range_res]).to_csv(OUTDIR / "range_estimate.csv", index=False)

    section("PHASE 5 TAKEAWAYS")
    print(f"  - Best ARIMA order for SoC: {arima_res['order']}")
    print(f"  - ARIMA SoC forecast    R^2 = {arima_res['metrics']['R2']:.3f}, "
          f"MAE = {arima_res['metrics']['MAE']:.3f} %")
    print(f"  - LSTM voltage forecast R^2 = {lstm_res['metrics']['R2']:.3f}, "
          f"MAE = {lstm_res['metrics']['MAE']:.3f} V")
    print(f"  - Range left from end of S3: {range_res['km_left']} km "
          f"({range_res['min_left']:.0f} min) at {range_res['avg_wh_per_km']} Wh/km")
    print(f"\nAll Phase 5 outputs in: {OUTDIR}")


if __name__ == "__main__":
    main()
