"""
Phase 3 - Unsupervised Machine Learning

Lets the data speak. Four models:
  3a. K-Means      -> find natural driving regimes (elbow + silhouette to pick k)
  3b. DBSCAN       -> density-based clustering (also flags lonely points)
  3c. PCA          -> squeeze ~13 features into 2 numbers for visualization
  3d. IsolationForest -> hunt anomalies (specific to "weird" moments)

Inputs:
  oorja_telemetry_processed.csv (unscaled, engineered features)

Outputs:
  analysis/phase3_unsupervised/*.png
  analysis/phase3_unsupervised/cluster_profiles.csv
  analysis/phase3_unsupervised/anomalies.csv
  oorja_telemetry_clustered.csv  (original data + cluster + anomaly columns)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

ROOT   = Path(r"d:\OORJA_PROJECT")
SRC    = ROOT / "oorja_telemetry_processed.csv"
DST    = ROOT / "oorja_telemetry_clustered.csv"
OUTDIR = ROOT / "analysis" / "phase3_unsupervised"
OUTDIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"]  = 110
plt.rcParams["savefig.dpi"] = 140

# Features used by every unsupervised model.
# We deliberately skip MONOTONIC channels (Distance, Energy, SoC, Charge, Elapsed)
# because they would just rank time, not driving behaviour.
FEATURES = [
    "Speed_Kmph", "RPM", "Accel_m_s2",
    "MPU_Gx_LP", "MPU_Gy_LP", "MPU_Gz_LP",
    "IMU_G_Magnitude", "IMU_Vibration",
    "Pack_Current_A", "Pack_Voltage_V", "Pack_Power_kW",
    "Gear_Ratio",
]


def save(fig, name: str) -> None:
    p = OUTDIR / name
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    print(f"  -> {p.name}")


def section(title: str) -> None:
    print(f"\n{'='*64}\n{title}\n{'='*64}")


# =============================================================================
# 3a. K-MEANS
# =============================================================================

def run_kmeans(X_std: np.ndarray, df: pd.DataFrame) -> tuple[np.ndarray, int, dict]:
    section("3a. K-MEANS  (find natural driving regimes)")

    # ---- Elbow + silhouette to pick k ----
    k_range = range(2, 9)
    inertias, sils = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=0)
        labels = km.fit_predict(X_std)
        inertias.append(km.inertia_)
        # Silhouette on a sample (full set is fine here, ~3.6k points)
        sils.append(silhouette_score(X_std, labels, sample_size=2000, random_state=0))
        print(f"  k={k}  inertia={km.inertia_:>10.0f}  silhouette={sils[-1]:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(list(k_range), inertias, "o-", color="steelblue")
    axes[0].set_title("K-Means elbow plot")
    axes[0].set_xlabel("k (number of clusters)")
    axes[0].set_ylabel("Inertia (lower = tighter clusters)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(list(k_range), sils, "o-", color="darkorange")
    axes[1].set_title("Silhouette score vs k")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette (higher = better separated)")
    axes[1].grid(True, alpha=0.3)
    save(fig, "01_kmeans_k_selection.png")

    best_k = list(k_range)[int(np.argmax(sils))]
    print(f"\n  -> Best k by silhouette: k = {best_k}")

    # ---- Fit final K-Means with chosen k ----
    km = KMeans(n_clusters=best_k, n_init=20, random_state=0)
    labels = km.fit_predict(X_std)

    # ---- Cluster profiles (mean of each feature per cluster, in ORIGINAL units) ----
    profile = df[FEATURES].copy()
    profile["Cluster"] = labels
    means = profile.groupby("Cluster").mean().round(2)
    sizes = pd.Series(labels).value_counts().sort_index()
    means["Size"]    = sizes.values
    means["Share_%"] = (sizes.values / len(labels) * 100).round(1)
    print("\nCluster profiles (mean values per cluster, original units):")
    print(means.to_string())
    means.to_csv(OUTDIR / "cluster_profiles.csv")

    # ---- Heatmap of cluster centers (z-scaled view) ----
    centers_df = pd.DataFrame(km.cluster_centers_, columns=FEATURES)
    fig, ax = plt.subplots(figsize=(12, max(4, best_k*0.6)))
    sns.heatmap(centers_df, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                ax=ax, cbar_kws={"label": "z-score"})
    ax.set_title(f"K-Means cluster centers (k={best_k}) - z-scored features")
    ax.set_ylabel("Cluster #")
    save(fig, "02_kmeans_centers_heatmap.png")

    return labels, best_k, {"inertias": inertias, "silhouettes": sils, "profile": means}


# =============================================================================
# 3b. DBSCAN
# =============================================================================

def run_dbscan(X_std: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    section("3b. DBSCAN  (density-based clustering + outliers)")

    # eps via k-distance plot
    k = 8
    nn = NearestNeighbors(n_neighbors=k).fit(X_std)
    dists, _ = nn.kneighbors(X_std)
    kd = np.sort(dists[:, -1])
    # Heuristic: pick eps near the knee of the curve (90th percentile)
    eps = float(np.quantile(kd, 0.90))

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(kd, color="steelblue")
    ax.axhline(eps, color="red", linestyle="--", label=f"chosen eps = {eps:.2f}")
    ax.set_title(f"DBSCAN k-distance plot (k={k}) - knee picks eps")
    ax.set_xlabel("Sample (sorted by distance to 8th neighbour)")
    ax.set_ylabel("Distance")
    ax.legend()
    save(fig, "03_dbscan_kdistance.png")
    print(f"  Chosen eps={eps:.3f}, min_samples={k}")

    db = DBSCAN(eps=eps, min_samples=k).fit(X_std)
    labels = db.labels_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = int((labels == -1).sum())
    print(f"  -> {n_clusters} clusters, {n_noise:,} noise points "
          f"({n_noise/len(labels)*100:.1f}%)")

    # Brief per-cluster headcount
    counts = pd.Series(labels).value_counts().sort_index()
    print("  Cluster sizes:")
    for c, n in counts.items():
        tag = "noise" if c == -1 else f"cluster {c}"
        print(f"    {tag:12s}  {n:5d}  ({n/len(labels)*100:5.1f}%)")
    return labels


# =============================================================================
# 3c. PCA
# =============================================================================

def run_pca(X_std: np.ndarray, df: pd.DataFrame, kmeans_labels: np.ndarray,
            best_k: int) -> np.ndarray:
    section("3c. PCA  (squeeze ~12 features into 2 axes for visualization)")
    pca = PCA(n_components=4, random_state=0)
    X_pca = pca.fit_transform(X_std)
    var = pca.explained_variance_ratio_
    cum = np.cumsum(var)
    print(f"  Explained variance: PC1={var[0]:.1%}  PC2={var[1]:.1%}  "
          f"PC3={var[2]:.1%}  PC4={var[3]:.1%}  (cum to 4: {cum[3]:.1%})")

    # Loadings (which feature drives each PC)
    loadings = pd.DataFrame(pca.components_[:2].T, index=FEATURES,
                            columns=["PC1", "PC2"]).round(2)
    print("\n  Loadings on PC1 / PC2:")
    print(loadings.to_string())

    # ---- Scatter: 4 panels, coloured by drive_state, session, k-means, speed ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1],
                    hue=df["Drive_State"], palette="tab10",
                    s=8, alpha=0.6, ax=axes[0, 0])
    axes[0, 0].set_title("PCA - coloured by Drive_State")
    axes[0, 0].set_xlabel(f"PC1 ({var[0]:.0%})")
    axes[0, 0].set_ylabel(f"PC2 ({var[1]:.0%})")

    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1],
                    hue=df["Session_ID"], palette="Set1",
                    s=8, alpha=0.6, ax=axes[0, 1])
    axes[0, 1].set_title("PCA - coloured by Session")
    axes[0, 1].set_xlabel(f"PC1 ({var[0]:.0%})")
    axes[0, 1].set_ylabel(f"PC2 ({var[1]:.0%})")

    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1],
                    hue=kmeans_labels, palette="viridis",
                    s=8, alpha=0.6, ax=axes[1, 0])
    axes[1, 0].set_title(f"PCA - coloured by K-Means cluster (k={best_k})")
    axes[1, 0].set_xlabel(f"PC1 ({var[0]:.0%})")
    axes[1, 0].set_ylabel(f"PC2 ({var[1]:.0%})")

    sc = axes[1, 1].scatter(X_pca[:, 0], X_pca[:, 1],
                            c=df["Speed_Kmph"], cmap="plasma", s=8, alpha=0.6)
    axes[1, 1].set_title("PCA - coloured by Speed (km/h)")
    axes[1, 1].set_xlabel(f"PC1 ({var[0]:.0%})")
    axes[1, 1].set_ylabel(f"PC2 ({var[1]:.0%})")
    plt.colorbar(sc, ax=axes[1, 1], label="Speed (km/h)")

    save(fig, "04_pca_scatter_grid.png")

    # Loadings biplot
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(X_pca[:, 0], X_pca[:, 1], s=4, alpha=0.15, color="lightgray")
    scale = 4.0
    for i, feat in enumerate(FEATURES):
        x, y = pca.components_[0, i] * scale, pca.components_[1, i] * scale
        ax.arrow(0, 0, x, y, color="darkred", alpha=0.7,
                 head_width=0.07, length_includes_head=True)
        ax.text(x*1.10, y*1.10, feat, color="darkred", fontsize=9, ha="center")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel(f"PC1 ({var[0]:.1%})")
    ax.set_ylabel(f"PC2 ({var[1]:.1%})")
    ax.set_title("PCA loadings biplot - which features point where")
    save(fig, "05_pca_biplot.png")

    # Variance bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(1, 5), var * 100, color="steelblue", label="Per PC")
    ax.plot(range(1, 5), cum * 100, "ro-", label="Cumulative")
    ax.set_xticks(range(1, 5))
    ax.set_xticklabels([f"PC{i}" for i in range(1, 5)])
    ax.set_ylabel("Explained variance (%)")
    ax.set_title("PCA explained variance")
    ax.legend()
    save(fig, "06_pca_variance.png")
    return X_pca


# =============================================================================
# 3d. ISOLATION FOREST
# =============================================================================

def run_isolation_forest(X_std: np.ndarray, df: pd.DataFrame, X_pca: np.ndarray,
                         contamination: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    section("3d. ISOLATION FOREST  (anomaly detection)")
    iso = IsolationForest(n_estimators=200, contamination=contamination,
                          random_state=0, n_jobs=-1)
    iso.fit(X_std)
    scores = iso.decision_function(X_std)   # higher = more normal
    pred   = iso.predict(X_std)             # +1 normal, -1 anomaly
    is_anom = (pred == -1).astype(int)

    n = int(is_anom.sum())
    print(f"  Flagged {n} anomalies ({n/len(df)*100:.2f}% of run, "
          f"contamination target {contamination:.0%})")

    # Inspect the anomalies
    anom_df = df.loc[is_anom == 1, [
        "LoRa_Datetime", "Session_ID", "Elapsed_s",
        "Speed_Kmph", "RPM", "Accel_m_s2",
        "Pack_Current_A", "Pack_Voltage_V", "Pack_Power_kW",
        "IMU_G_Magnitude", "IMU_Vibration", "Drive_State",
    ]].copy()
    anom_df["Anomaly_Score"] = scores[is_anom == 1].round(3)
    anom_df = anom_df.sort_values("Anomaly_Score").head(40)   # 40 most anomalous
    anom_df.to_csv(OUTDIR / "anomalies.csv", index=False)
    print(f"  -> top-40 anomalies in anomalies.csv")
    print("\n  Top 8 most anomalous moments:")
    print(anom_df.head(8).to_string(index=False))

    # ---- Visualize anomalies on the time series + PCA map ----
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    t = df["Elapsed_s"] / 60.0
    axes[0].plot(t, df["Speed_Kmph"], color="steelblue", linewidth=0.6)
    axes[0].scatter(t[is_anom == 1], df["Speed_Kmph"][is_anom == 1],
                    color="red", s=10, label="anomaly")
    axes[0].set_ylabel("Speed (km/h)")
    axes[0].legend()

    axes[1].plot(t, df["Pack_Current_A"], color="darkorange", linewidth=0.6)
    axes[1].scatter(t[is_anom == 1], df["Pack_Current_A"][is_anom == 1],
                    color="red", s=10)
    axes[1].set_ylabel("Pack current (A)")

    axes[2].plot(t, df["Pack_Voltage_V"], color="darkgreen", linewidth=0.6)
    axes[2].scatter(t[is_anom == 1], df["Pack_Voltage_V"][is_anom == 1],
                    color="red", s=10)
    axes[2].set_ylabel("Pack voltage (V)")
    axes[2].set_xlabel("Elapsed time (min)")
    fig.suptitle("Isolation Forest - anomalies overlaid on key channels", y=1.001)
    save(fig, "07_anomalies_timeseries.png")

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c="lightgray", s=6, alpha=0.5,
               label="normal")
    ax.scatter(X_pca[is_anom == 1, 0], X_pca[is_anom == 1, 1],
               c="red", s=20, alpha=0.9, label="anomaly")
    ax.set_title("Anomalies on the PCA map")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    save(fig, "08_anomalies_on_pca.png")
    return is_anom, scores


# =============================================================================

def main() -> None:
    df = pd.read_csv(SRC, parse_dates=["LoRa_Datetime"])
    print(f"Loaded {len(df):,} rows.")

    # Standardize features (fresh fit; no leakage from train/test as this is unsupervised).
    X = df[FEATURES].fillna(0).to_numpy(dtype=float)
    X_std = StandardScaler().fit_transform(X)
    print(f"Feature matrix: {X_std.shape}  (samples x features)")
    print(f"Features used: {FEATURES}")

    km_labels, best_k, _    = run_kmeans(X_std, df)
    db_labels               = run_dbscan(X_std, df)
    X_pca                   = run_pca(X_std, df, km_labels, best_k)
    iso_flag, iso_scores    = run_isolation_forest(X_std, df, X_pca)

    # ---- Persist enriched dataset ----
    out = df.copy()
    out["KMeans_Cluster"]   = km_labels
    out["DBSCAN_Cluster"]   = db_labels
    out["Anomaly_Flag"]     = iso_flag
    out["Anomaly_Score"]    = iso_scores.round(4)
    out["PC1"]              = X_pca[:, 0].round(3)
    out["PC2"]              = X_pca[:, 1].round(3)
    out.to_csv(DST, index=False)
    print(f"\nWrote enriched dataset -> {DST.name}  ({out.shape[1]} cols)")

    # ---- Cluster x DriveState confusion (sanity check) ----
    section("Cross-tab: K-Means cluster x rule-based Drive_State")
    cross = pd.crosstab(out["KMeans_Cluster"], out["Drive_State"], normalize="index")
    print((cross * 100).round(1).to_string())

    # ---- Final takeaways ----
    section("PHASE 3 TAKEAWAYS")
    print(f"  - K-Means picked k={best_k} as the best driving-regime count.")
    print(f"  - DBSCAN found {len(set(db_labels))-1 if -1 in db_labels else len(set(db_labels))} "
          f"dense clusters, {(db_labels==-1).sum():,} noise points.")
    pc1, pc2 = (out['PC1'].max()-out['PC1'].min()), (out['PC2'].max()-out['PC2'].min())
    print(f"  - PCA captured most variance in 2D; spans PC1={pc1:.1f}, PC2={pc2:.1f}.")
    print(f"  - Isolation Forest flagged {iso_flag.sum()} unusual moments "
          f"(see anomalies.csv).")
    print(f"\nAll Phase 3 outputs in: {OUTDIR}")


if __name__ == "__main__":
    main()
