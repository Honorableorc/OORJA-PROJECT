"""
Phase 1 - Exploratory Data Analysis (EDA)

Looks at the processed kart telemetry, produces:
  1. Time-series line plots (every channel)
  2. Histograms / distributions
  3. Scatter plots between key channels
  4. Correlation heatmap
  5. Per-session box plots
  6. Console summary of what we observe

All plots saved to analysis/phase1_eda/*.png so they can go straight into the report.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT    = Path(r"d:\OORJA_PROJECT")
SRC     = ROOT / "oorja_telemetry_processed.csv"
OUTDIR  = ROOT / "analysis" / "phase1_eda"
OUTDIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"]      = 110
plt.rcParams["savefig.dpi"]     = 140
plt.rcParams["figure.figsize"]  = (12, 5)


def save(fig, name: str) -> None:
    path = OUTDIR / name
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path.name}")


def section(title: str) -> None:
    print(f"\n{'='*60}\n{title}\n{'='*60}")


# =============================================================================
# 1. Time-series line plots
# =============================================================================

def plot_timeseries(df: pd.DataFrame) -> None:
    section("1. Time-series line plots")
    t = df["Elapsed_s"].to_numpy() / 60.0   # minutes for readability

    panels = [
        ("Speed_Kmph",      "Speed (km/h)",        "tab:blue"),
        ("RPM",             "Motor RPM",           "tab:orange"),
        ("Pack_Voltage_V",  "Pack Voltage (V)",    "tab:green"),
        ("Pack_Current_A",  "Pack Current (A)",    "tab:red"),
        ("Pack_Power_kW",   "Pack Power (kW)",     "tab:purple"),
        ("SoC_pct",         "SoC (%)",             "tab:brown"),
        ("Accel_m_s2",      "Acceleration (m/s^2)","tab:gray"),
        ("IMU_G_Magnitude", "IMU |G|",             "tab:olive"),
    ]

    fig, axes = plt.subplots(len(panels), 1, sharex=True, figsize=(13, 16))
    for ax, (col, label, color) in zip(axes, panels):
        ax.plot(t, df[col], color=color, linewidth=0.7)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        # Shade session boundaries
        for sid, sub in df.groupby("Session_ID"):
            ax.axvspan(sub["Elapsed_s"].iloc[0]/60, sub["Elapsed_s"].iloc[-1]/60,
                       alpha=0.04, color="black")
    axes[-1].set_xlabel("Elapsed time (minutes)")
    fig.suptitle("Telemetry channels over time (3 sessions)", y=1.001)
    save(fig, "01_timeseries_all_channels.png")


# =============================================================================
# 2. Histograms / distributions
# =============================================================================

def plot_histograms(df: pd.DataFrame) -> None:
    section("2. Distributions (histograms)")
    cols = ["Speed_Kmph", "RPM", "Pack_Current_A", "Pack_Voltage_V",
            "Pack_Power_kW", "Accel_m_s2", "IMU_G_Magnitude", "SoC_pct"]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, col in zip(axes.flat, cols):
        sns.histplot(df[col], bins=50, ax=ax, color="steelblue", edgecolor="white")
        ax.set_title(col)
        ax.axvline(df[col].mean(), color="red", linestyle="--", linewidth=1,
                   label=f"mean={df[col].mean():.2f}")
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Distribution of each channel", y=1.02)
    save(fig, "02_histograms.png")


# =============================================================================
# 3. Scatter plots between key channels
# =============================================================================

def plot_scatter_relationships(df: pd.DataFrame) -> None:
    section("3. Scatter relationships")

    # Sample down for cleaner scatter (3.6k pts is fine but 1k looks crisper)
    d = df.sample(min(len(df), 1500), random_state=0)

    pairs = [
        ("Speed_Kmph",      "Pack_Current_A",  "Speed vs Current"),
        ("Speed_Kmph",      "Pack_Power_kW",   "Speed vs Power"),
        ("RPM",             "Speed_Kmph",      "RPM vs Speed (gear ratio)"),
        ("Pack_Current_A",  "Pack_Voltage_V",  "Current vs Voltage (sag)"),
        ("Accel_m_s2",      "Pack_Current_A",  "Acceleration vs Current"),
        ("SoC_pct",         "Pack_Voltage_V",  "SoC vs Voltage"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, (x, y, title) in zip(axes.flat, pairs):
        sns.scatterplot(data=d, x=x, y=y, hue="Drive_State", s=12, alpha=0.6,
                        palette="tab10", ax=ax, legend=(ax is axes.flat[0]))
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Pairwise relationships (coloured by drive state)", y=1.01)
    save(fig, "03_scatter_relationships.png")


# =============================================================================
# 4. Correlation heatmap
# =============================================================================

def plot_correlation(df: pd.DataFrame) -> None:
    section("4. Correlation heatmap")
    cols = ["Speed_Kmph", "RPM", "Accel_m_s2",
            "MPU_Gx", "MPU_Gy", "MPU_Gz", "IMU_G_Magnitude", "IMU_Vibration",
            "Pack_Current_A", "Pack_Voltage_V", "Pack_Power_kW",
            "SoC_pct", "Energy_Wh_cum", "Distance_km"]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation matrix of telemetry channels")
    save(fig, "04_correlation_heatmap.png")

    # Print top correlations (besides diagonals)
    pairs = (corr.where(np.triu(np.ones_like(corr, dtype=bool), k=1))
                 .stack()
                 .sort_values(key=lambda s: s.abs(), ascending=False)
                 .head(8))
    print("\nTop 8 strongest pairwise correlations:")
    for (a, b), v in pairs.items():
        print(f"  {a:20s} <-> {b:20s}  r = {v:+.3f}")


# =============================================================================
# 5. Per-session box plots
# =============================================================================

def plot_per_session(df: pd.DataFrame) -> None:
    section("5. Per-session box plots")
    cols = ["Speed_Kmph", "Pack_Current_A", "Pack_Voltage_V", "Pack_Power_kW",
            "IMU_Vibration", "Accel_m_s2"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, c in zip(axes.flat, cols):
        sns.boxplot(data=df, x="Session_ID", y=c, ax=ax, palette="Set2",
                    hue="Session_ID", legend=False)
        ax.set_title(c)
    fig.suptitle("Distribution per session", y=1.01)
    save(fig, "05_per_session_boxplots.png")


# =============================================================================
# 6. Drive-state distribution donut
# =============================================================================

def plot_drive_state(df: pd.DataFrame) -> None:
    section("6. Drive-state distribution")
    counts = df["Drive_State"].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    counts.plot(kind="bar", ax=axes[0], color=["#4c72b0", "#dd8452", "#55a467", "#c44e52"])
    axes[0].set_ylabel("Seconds")
    axes[0].set_title("Time spent in each drive state")
    axes[0].tick_params(axis="x", rotation=0)

    pct = (counts / counts.sum() * 100).round(1)
    axes[1].pie(counts, labels=[f"{s}\n{p}%" for s, p in zip(counts.index, pct)],
                colors=["#4c72b0", "#dd8452", "#55a467", "#c44e52"],
                wedgeprops=dict(width=0.4), startangle=90)
    axes[1].set_title("Drive-state share")
    save(fig, "06_drive_state.png")

    print("\nDrive-state seconds:")
    for s, n in counts.items():
        print(f"  {s:8s} {n:5d} s   ({n/counts.sum()*100:5.1f}%)")


# =============================================================================
# Console summary
# =============================================================================

def print_summary(df: pd.DataFrame) -> None:
    section("EDA SUMMARY (numbers for the report)")
    duration_min = df["Elapsed_s"].iloc[-1] / 60.0
    print(f"Rows: {len(df):,}   Duration: {duration_min:.1f} min   "
          f"Sessions: {df['Session_ID'].nunique()}")

    print("\nKey channel ranges:")
    for c in ["Speed_Kmph", "RPM", "Pack_Voltage_V", "Pack_Current_A",
              "Pack_Power_kW", "SoC_pct"]:
        s = df[c]
        print(f"  {c:18s}  min={s.min():>7.2f}  mean={s.mean():>7.2f}"
              f"  max={s.max():>7.2f}  std={s.std():>6.2f}")

    print("\nPer-session highlights:")
    g = df.groupby("Session_ID").agg(
        rows=("Speed_Kmph", "size"),
        max_speed=("Speed_Kmph", "max"),
        avg_speed=("Speed_Kmph", "mean"),
        peak_kW=("Pack_Power_kW", "max"),
        peak_A=("Pack_Current_A", "max"),
        min_V=("Pack_Voltage_V", "min"),
        distance_km=("Distance_km", lambda s: s.max() - s.min()),
        energy_Wh=("Energy_Wh_cum", lambda s: s.max() - s.min()),
    ).round(2)
    g["Wh_per_km"] = (g["energy_Wh"] / g["distance_km"].replace(0, np.nan)).round(1)
    print(g.to_string())

    # Plain-language takeaways
    fastest = g["max_speed"].idxmax()
    most_eff = g["Wh_per_km"].idxmin()
    print(f"\nTakeaways:")
    print(f"  - Session {fastest} hit the highest top speed ({g.loc[fastest,'max_speed']:.0f} km/h).")
    print(f"  - Session {most_eff} was the most energy-efficient ({g.loc[most_eff,'Wh_per_km']:.1f} Wh/km).")
    print(f"  - Pack voltage stayed in {df['Pack_Voltage_V'].min():.1f}-"
          f"{df['Pack_Voltage_V'].max():.1f} V (nominal 72 V).")
    print(f"  - Peak current draw was {df['Pack_Current_A'].max():.1f} A "
          f"(BMS limit 150 A continuous, 300 A peak).")


# =============================================================================

def main() -> None:
    df = pd.read_csv(SRC, parse_dates=["LoRa_Datetime"])
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns from {SRC.name}")

    plot_timeseries(df)
    plot_histograms(df)
    plot_scatter_relationships(df)
    plot_correlation(df)
    plot_per_session(df)
    plot_drive_state(df)
    print_summary(df)

    print(f"\nAll plots saved to: {OUTDIR}")


if __name__ == "__main__":
    main()
