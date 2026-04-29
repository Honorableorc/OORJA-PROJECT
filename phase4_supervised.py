"""
Phase 4 - Supervised Machine Learning (Regression)

Predicts 4 different targets, each with 6 models, with proper
time-based train/test split (Sessions 1+2 -> train, Session 3 -> test).

Targets:
  4a. Pack_Current_A     <-  driving inputs (speed, RPM, accel, IMU)
  4b. Pack_Voltage_V     <-  current + SoC  (validates battery model)
  4c. Pack_Power_kW      <-  driving inputs (gear ratio + dynamics)
  4d. Wh_per_km (rolling) <- 30-second window stats (efficiency)

Models:
  1. Linear Regression  -- baseline
  2. Ridge Regression   -- regularized linear
  3. Random Forest      -- non-linear, gives feature importance
  4. Gradient Boosting  -- sklearn (XGBoost not installed)
  5. SVR (RBF kernel)   -- support vector regression
  6. PyTorch MLP        -- 2-layer neural network

Outputs:
  analysis/phase4_supervised/{current,voltage,power,efficiency}/*.png
  analysis/phase4_supervised/{...}/metrics.csv
  analysis/phase4_supervised/master_summary.csv
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT     = Path(r"d:\OORJA_PROJECT")
SRC      = ROOT / "oorja_telemetry_clustered.csv"
OUTDIR   = ROOT / "analysis" / "phase4_supervised"
OUTDIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"]  = 110
plt.rcParams["savefig.dpi"] = 140
RNG = 42
torch.manual_seed(RNG)
np.random.seed(RNG)


# =============================================================================
# PyTorch MLP
# =============================================================================

class MLP(nn.Module):
    def __init__(self, n_in: int, hidden=(64, 32), dropout: float = 0.1):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X_tr, y_tr, X_te, y_te, epochs=200, batch=128, lr=1e-3,
              patience=20, verbose=False):
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_te_t = torch.tensor(X_te, dtype=torch.float32)

    # Hold out last 10% of train as validation for early stopping
    n_val = max(64, int(len(X_tr) * 0.10))
    X_val_t = X_tr_t[-n_val:]
    y_val_t = y_tr_t[-n_val:]
    X_fit_t = X_tr_t[:-n_val]
    y_fit_t = y_tr_t[:-n_val]

    model = MLP(n_in=X_tr.shape[1])
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    crit  = nn.MSELoss()

    best_val, best_state, since_best = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X_fit_t))
        for i in range(0, len(perm), batch):
            idx = perm[i:i+batch]
            opt.zero_grad()
            pred = model(X_fit_t[idx])
            loss = crit(pred, y_fit_t[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = crit(val_pred, y_val_t).item()
        if val_loss < best_val - 1e-5:
            best_val, best_state, since_best = val_loss, {
                k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            since_best += 1
            if since_best >= patience:
                break
        if verbose and ep % 20 == 0:
            print(f"    ep {ep:3d}  val_loss={val_loss:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(X_te_t).numpy()


# =============================================================================
# Generic per-target pipeline
# =============================================================================

def evaluate(y_true, y_pred):
    return {
        "R2":   round(r2_score(y_true, y_pred), 4),
        "MAE":  round(mean_absolute_error(y_true, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
    }


def section(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


def run_target(df: pd.DataFrame, target: str, predictors: list[str],
               outdir: Path, target_pretty: str, target_unit: str,
               split: str = "session") -> dict:
    """
    Train 6 models on one target. Returns dict of model_name -> metrics.
    split = "session" -> train on S1+S2, test on S3 (default; realistic).
    split = "random"  -> 80/20 random split (use when train/test must cover
                         the same domain, e.g. voltage which depends on SoC).
    """
    outdir.mkdir(parents=True, exist_ok=True)
    section(f"Target: {target_pretty}  ({target} {target_unit})  [split={split}]")
    print(f"  Predictors: {predictors}")

    work = df.dropna(subset=predictors + [target]).copy().reset_index(drop=True)
    if split == "random":
        rs = np.random.RandomState(RNG)
        idx = rs.permutation(len(work))
        cut = int(0.8 * len(work))
        train_idx, test_idx = idx[:cut], idx[cut:]
        train_mask = work.index.isin(train_idx)
        test_mask  = work.index.isin(test_idx)
    else:
        train_mask = work["Session_ID"].isin([1, 2])
        test_mask  = work["Session_ID"] == 3

    X_tr_raw = work.loc[train_mask, predictors].to_numpy(dtype=float)
    y_tr     = work.loc[train_mask, target].to_numpy(dtype=float)
    X_te_raw = work.loc[test_mask,  predictors].to_numpy(dtype=float)
    y_te     = work.loc[test_mask,  target].to_numpy(dtype=float)
    print(f"  Train: {len(y_tr):,}   Test: {len(y_te):,}")

    # Scale for non-tree models
    scaler = StandardScaler().fit(X_tr_raw)
    X_tr_s = scaler.transform(X_tr_raw)
    X_te_s = scaler.transform(X_te_raw)

    # ---- Train models ----
    results = {}
    preds   = {}
    importances = {}

    # 1. Linear
    m = LinearRegression().fit(X_tr_s, y_tr)
    p = m.predict(X_te_s)
    results["LinearRegression"] = evaluate(y_te, p); preds["LinearRegression"] = p

    # 2. Ridge
    m = Ridge(alpha=1.0, random_state=RNG).fit(X_tr_s, y_tr)
    p = m.predict(X_te_s)
    results["Ridge"] = evaluate(y_te, p); preds["Ridge"] = p

    # 3. Random Forest (no scaling needed, but use same X for consistency)
    rf = RandomForestRegressor(n_estimators=200, max_depth=None,
                               n_jobs=-1, random_state=RNG).fit(X_tr_raw, y_tr)
    p = rf.predict(X_te_raw)
    results["RandomForest"] = evaluate(y_te, p); preds["RandomForest"] = p
    importances["RandomForest"] = rf.feature_importances_

    # 4. Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                   learning_rate=0.05, random_state=RNG).fit(X_tr_raw, y_tr)
    p = gb.predict(X_te_raw)
    results["GradientBoosting"] = evaluate(y_te, p); preds["GradientBoosting"] = p
    importances["GradientBoosting"] = gb.feature_importances_

    # 5. SVR (slow on big data; subsample to 3000 if needed)
    if len(y_tr) > 3000:
        idx = np.random.RandomState(RNG).choice(len(y_tr), 3000, replace=False)
        svr = SVR(kernel="rbf", C=10.0, gamma="scale").fit(X_tr_s[idx], y_tr[idx])
    else:
        svr = SVR(kernel="rbf", C=10.0, gamma="scale").fit(X_tr_s, y_tr)
    p = svr.predict(X_te_s)
    results["SVR"] = evaluate(y_te, p); preds["SVR"] = p

    # 6. PyTorch MLP
    p = train_mlp(X_tr_s, y_tr, X_te_s, y_te, epochs=200, patience=25)
    results["PyTorchMLP"] = evaluate(y_te, p); preds["PyTorchMLP"] = p

    # ---- Print metrics ----
    metrics_df = pd.DataFrame(results).T.sort_values("R2", ascending=False)
    print("\n  Metrics (sorted by R^2):")
    print(metrics_df.to_string())
    metrics_df.to_csv(outdir / "metrics.csv")

    best_model = metrics_df.index[0]
    print(f"\n  Best model: {best_model}  (R^2 = {metrics_df.loc[best_model, 'R2']:.4f})")

    # ---- Plots ----
    # 1. Bar chart of R^2 / MAE
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(x=metrics_df.index, y=metrics_df["R2"], ax=axes[0],
                hue=metrics_df.index, palette="Set2", legend=False)
    axes[0].set_ylim(min(0, metrics_df["R2"].min() - 0.05), 1.0)
    axes[0].set_ylabel("R²"); axes[0].set_title(f"R² (higher = better) — {target_pretty}")
    axes[0].tick_params(axis="x", rotation=20)
    for i, v in enumerate(metrics_df["R2"]):
        axes[0].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    sns.barplot(x=metrics_df.index, y=metrics_df["MAE"], ax=axes[1],
                hue=metrics_df.index, palette="Set2", legend=False)
    axes[1].set_ylabel(f"MAE ({target_unit})")
    axes[1].set_title(f"MAE (lower = better) — {target_pretty}")
    axes[1].tick_params(axis="x", rotation=20)
    for i, v in enumerate(metrics_df["MAE"]):
        axes[1].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout(); fig.savefig(outdir / "01_model_comparison.png"); plt.close(fig)

    # 2. Best model predicted vs actual
    p_best = preds[best_model]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_te, p_best, s=8, alpha=0.4, color="steelblue")
    lim = [min(y_te.min(), p_best.min()), max(y_te.max(), p_best.max())]
    ax.plot(lim, lim, "r--", label="y = x (perfect)")
    ax.set_xlabel(f"Actual {target_pretty} ({target_unit})")
    ax.set_ylabel(f"Predicted {target_pretty} ({target_unit})")
    ax.set_title(f"{best_model}: Predicted vs Actual\n"
                 f"R²={results[best_model]['R2']:.3f}  "
                 f"MAE={results[best_model]['MAE']:.3f}  "
                 f"RMSE={results[best_model]['RMSE']:.3f}")
    ax.legend()
    fig.tight_layout(); fig.savefig(outdir / "02_best_pred_vs_actual.png"); plt.close(fig)

    # 3. Residuals over test-set time (sorted by Elapsed_s for plotting)
    test_sub = work.loc[test_mask, ["Elapsed_s"]].copy()
    test_sub["y"]    = y_te
    test_sub["pred"] = p_best
    test_sub = test_sub.sort_values("Elapsed_s")
    test_t    = test_sub["Elapsed_s"].to_numpy() / 60.0
    y_te_t    = test_sub["y"].to_numpy()
    p_best_t  = test_sub["pred"].to_numpy()
    residuals = y_te_t - p_best_t
    fig, axes = plt.subplots(2, 1, figsize=(13, 7))
    axes[0].plot(test_t, y_te_t, label="actual", linewidth=0.7, color="black")
    axes[0].plot(test_t, p_best_t, label=f"predicted ({best_model})",
                 linewidth=0.7, color="orangered", alpha=0.8)
    axes[0].set_ylabel(f"{target_pretty} ({target_unit})")
    axes[0].legend(); axes[0].set_title(f"Test set (Session 3) — actual vs predicted")
    axes[1].plot(test_t, residuals, color="darkred", linewidth=0.6)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_ylabel("Residual (actual - pred)"); axes[1].set_xlabel("Elapsed (min)")
    axes[1].set_title(f"Residuals (mean={residuals.mean():.3f}, std={residuals.std():.3f})")
    fig.tight_layout(); fig.savefig(outdir / "03_residuals_time.png"); plt.close(fig)

    # 4. Feature importance (RF + GB)
    fig, axes = plt.subplots(1, 2, figsize=(15, max(4, len(predictors)*0.32)))
    for ax, model_name in zip(axes, ["RandomForest", "GradientBoosting"]):
        imp = importances[model_name]
        order = np.argsort(imp)[::-1]
        feats_sorted = [predictors[i] for i in order]
        sns.barplot(x=imp[order], y=feats_sorted, ax=ax,
                    hue=feats_sorted, palette="viridis", legend=False)
        ax.set_title(f"{model_name} feature importance")
        ax.set_xlabel("Importance")
    fig.tight_layout(); fig.savefig(outdir / "04_feature_importance.png"); plt.close(fig)

    print(f"  -> plots & metrics saved to {outdir.relative_to(ROOT)}/")
    return {"metrics": metrics_df, "best_model": best_model,
            "best_pred": preds[best_model], "y_te": y_te}


# =============================================================================
# Build a windowed-efficiency target (Wh / km over rolling 30-s window)
# =============================================================================

def add_window_efficiency(df: pd.DataFrame, window_s: int = 30) -> pd.DataFrame:
    df = df.copy()
    # Energy in window (Wh) / distance in window (km)
    energy_win = df["Energy_Wh_cum"] - df["Energy_Wh_cum"].shift(window_s)
    dist_win   = df["Distance_km"]   - df["Distance_km"].shift(window_s)
    eff = energy_win / dist_win.replace(0, np.nan)
    df[f"WhPerKm_w{window_s}"] = eff

    # Window features (predictors)
    df[f"Speed_w{window_s}_mean"] = df["Speed_Kmph"].rolling(window_s, min_periods=1).mean()
    df[f"Speed_w{window_s}_std"]  = df["Speed_Kmph"].rolling(window_s, min_periods=1).std().fillna(0)
    df[f"Accel_w{window_s}_mean"] = df["Accel_m_s2"].rolling(window_s, min_periods=1).mean()
    df[f"Accel_w{window_s}_std"]  = df["Accel_m_s2"].rolling(window_s, min_periods=1).std().fillna(0)
    df[f"RPM_w{window_s}_mean"]   = df["RPM"].rolling(window_s, min_periods=1).mean()
    df[f"Vibr_w{window_s}_mean"]  = df["IMU_Vibration"].rolling(window_s, min_periods=1).mean()
    df[f"Gmag_w{window_s}_mean"]  = df["IMU_G_Magnitude"].rolling(window_s, min_periods=1).mean()
    return df


# =============================================================================

def main() -> None:
    df = pd.read_csv(SRC, parse_dates=["LoRa_Datetime"])
    print(f"Loaded {len(df):,} rows.")
    print(f"Sessions: train = 1 + 2 ({(df['Session_ID']<3).sum():,} rows), "
          f"test = 3 ({(df['Session_ID']==3).sum():,} rows)")

    master = []   # collect best-row per target

    # ---- 4a. Pack_Current_A ----
    target = "Pack_Current_A"
    predictors = ["Speed_Kmph", "RPM", "Accel_m_s2",
                  "MPU_Gx_LP", "MPU_Gy_LP", "MPU_Gz_LP",
                  "IMU_G_Magnitude", "IMU_Vibration", "Gear_Ratio"]
    res = run_target(df, target, predictors, OUTDIR / "current",
                     "Pack Current", "A")
    best_row = res["metrics"].iloc[0].to_dict(); best_row["Target"] = target
    best_row["Best_Model"] = res["best_model"]
    master.append(best_row)

    # ---- 4b. Pack_Voltage_V (validates our battery model) ----
    # Voltage depends on SoC. Sessions 1+2 cover SoC 63-85% but Session 3 only
    # covers 37-63% -> session split forces extrapolation outside training range
    # and every model degenerates. Use random 80/20 split instead so the model
    # is evaluated on its ability to LEARN the V = OCV(SoC) - I*R relationship,
    # not on out-of-distribution generalization.
    target = "Pack_Voltage_V"
    predictors = ["Pack_Current_A", "SoC_pct", "Speed_Kmph", "Pack_Power_kW"]
    res = run_target(df, target, predictors, OUTDIR / "voltage",
                     "Pack Voltage", "V", split="random")
    best_row = res["metrics"].iloc[0].to_dict(); best_row["Target"] = target
    best_row["Best_Model"] = res["best_model"]
    master.append(best_row)

    # ---- 4c. Pack_Power_kW ----
    target = "Pack_Power_kW"
    predictors = ["Speed_Kmph", "RPM", "Accel_m_s2",
                  "MPU_Gx_LP", "MPU_Gy_LP", "MPU_Gz_LP",
                  "IMU_G_Magnitude", "IMU_Vibration", "Gear_Ratio"]
    res = run_target(df, target, predictors, OUTDIR / "power",
                     "Pack Power", "kW")
    best_row = res["metrics"].iloc[0].to_dict(); best_row["Target"] = target
    best_row["Best_Model"] = res["best_model"]
    master.append(best_row)

    # ---- 4d. Wh per km (windowed) ----
    df_w = add_window_efficiency(df, window_s=30)
    target = "WhPerKm_w30"
    predictors = ["Speed_w30_mean", "Speed_w30_std",
                  "Accel_w30_mean", "Accel_w30_std",
                  "RPM_w30_mean",   "Vibr_w30_mean", "Gmag_w30_mean"]
    # Drop the first 30 rows (NaN window) and any zero-distance windows
    df_eff = df_w.dropna(subset=[target] + predictors)
    df_eff = df_eff[(df_eff[target] > 0) & (df_eff[target] < 500)]   # filter junk
    print(f"\nEfficiency target rows after filter: {len(df_eff):,}")
    res = run_target(df_eff, target, predictors, OUTDIR / "efficiency",
                     "Energy Efficiency", "Wh/km")
    best_row = res["metrics"].iloc[0].to_dict(); best_row["Target"] = target
    best_row["Best_Model"] = res["best_model"]
    master.append(best_row)

    # ---- Master summary ----
    section("MASTER SUMMARY (best model per target)")
    master_df = pd.DataFrame(master)[["Target", "Best_Model", "R2", "MAE", "RMSE"]]
    print(master_df.to_string(index=False))
    master_df.to_csv(OUTDIR / "master_summary.csv", index=False)
    print(f"\n  -> {OUTDIR / 'master_summary.csv'}")

    print("\nAll Phase 4 outputs in:", OUTDIR)


if __name__ == "__main__":
    main()
