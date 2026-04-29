"""
Phase 2 - Performance Audit

Computes the headline KPIs that go into the report:
  - Top speed, average moving speed, max accel, max decel
  - 0-30 km/h, 0-60 km/h, 0-top launch times (best & worst per session)
  - Peak power (kW) & sustained cruise power
  - Voltage sag under load (effective pack internal resistance)
  - Energy efficiency (Wh/km) - overall + by speed band
  - Total distance, total energy

Outputs:
  analysis/phase2_performance/*.png  - charts
  analysis/phase2_performance/kpi_table.csv  - one row per session
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT   = Path(r"d:\OORJA_PROJECT")
SRC    = ROOT / "oorja_telemetry_processed.csv"
OUTDIR = ROOT / "analysis" / "phase2_performance"
OUTDIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"]  = 110
plt.rcParams["savefig.dpi"] = 140


def save(fig, name: str) -> None:
    p = OUTDIR / name
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    print(f"  -> {p.name}")


def section(title: str) -> None:
    print(f"\n{'='*60}\n{title}\n{'='*60}")


# =============================================================================
# Launch detection: 0 -> target km/h time
# =============================================================================

def find_launches(df: pd.DataFrame, target_kmph: int,
                  start_speed_kmph: float = 2.0) -> pd.DataFrame:
    """
    A 'launch' = a continuous segment where speed climbs from <start_speed
    up to >=target_kmph without ever dropping >5 km/h along the way.
    Returns one row per detected launch: time-to-target, session, start row.
    """
    speed = df["Speed_Kmph"].to_numpy(dtype=float)
    t     = df["Elapsed_s"].to_numpy(dtype=float)
    sess  = df["Session_ID"].to_numpy()

    launches = []
    n = len(speed)
    i = 0
    while i < n:
        if speed[i] < start_speed_kmph:
            # Find next sample where speed exceeds start
            j = i + 1
            while j < n and speed[j] < start_speed_kmph:
                j += 1
            # j is first moving sample; track climb until target or backslide
            launch_start = j
            peak = speed[j] if j < n else 0
            k = j
            while k < n and speed[k] >= start_speed_kmph:
                if speed[k] >= target_kmph:
                    launches.append({
                        "session":   int(sess[launch_start]),
                        "start_s":   float(t[launch_start]),
                        "reach_s":   float(t[k]),
                        "elapsed_s": float(t[k] - t[launch_start]),
                        "target":    target_kmph,
                    })
                    break
                if speed[k] < peak - 5:    # significant slowdown -> abort
                    break
                peak = max(peak, speed[k])
                k += 1
            i = max(k, launch_start) + 1
        else:
            i += 1
    return pd.DataFrame(launches)


# =============================================================================
# KPI table
# =============================================================================

def build_kpi_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, sub in df.groupby("Session_ID"):
        moving = sub[sub["Speed_Kmph"] > 0]
        cruise = sub[sub["Drive_State"] == "cruise"]
        accel_evts = sub[sub["Drive_State"] == "accel"]

        rows.append({
            "Session": sid,
            "Duration_min":      round(sub["Elapsed_s"].iloc[-1] - sub["Elapsed_s"].iloc[0], 1) / 60,
            "Distance_km":       round(sub["Distance_km"].max() - sub["Distance_km"].min(), 2),
            "Top_Speed_Kmph":    int(sub["Speed_Kmph"].max()),
            "Avg_Speed_Moving":  round(moving["Speed_Kmph"].mean() if len(moving) else 0, 2),
            "Max_Accel_m_s2":    round(accel_evts["Accel_m_s2"].max() if len(accel_evts) else 0, 2),
            "Max_Decel_m_s2":    round(sub["Accel_m_s2"].min(), 2),
            "Peak_Power_kW":     round(sub["Pack_Power_kW"].max(), 2),
            "Cruise_Power_kW":   round(cruise["Pack_Power_kW"].mean() if len(cruise) else 0, 2),
            "Peak_Current_A":    round(sub["Pack_Current_A"].max(), 1),
            "Min_Voltage_V":     round(sub["Pack_Voltage_V"].min(), 2),
            "Energy_Used_Wh":    round(sub["Energy_Wh_cum"].max() - sub["Energy_Wh_cum"].min(), 1),
            "Wh_per_km":         np.nan,  # filled below to avoid div-by-zero
            "SoC_Drop_pct":      round(sub["SoC_pct"].iloc[0] - sub["SoC_pct"].iloc[-1], 2),
        })

    kpi = pd.DataFrame(rows)
    dist = kpi["Distance_km"].replace(0, np.nan)
    kpi["Wh_per_km"] = (kpi["Energy_Used_Wh"] / dist).round(1)
    kpi["Duration_min"] = kpi["Duration_min"].round(2)
    return kpi


# =============================================================================
# Voltage sag analysis (effective internal resistance)
# =============================================================================

def voltage_sag(df: pd.DataFrame) -> dict:
    """
    Linear fit V = a + b*I per unit SoC band, then the slope -b is effective IR.
    We do a global fit too; pack datasheet says ~20 mOhm.
    """
    moving = df[df["Pack_Current_A"] > 5].copy()
    # Account for SoC by removing SoC trend first
    # voltage_residual = V - mean(V at this SoC bin)
    moving["SoC_bin"] = (moving["SoC_pct"] // 5 * 5).astype(int)
    moving["V_resid"] = moving.groupby("SoC_bin")["Pack_Voltage_V"].transform(
        lambda s: s - s.mean()
    )

    # Global slope fit of residual voltage vs current
    x = moving["Pack_Current_A"].to_numpy()
    y = moving["V_resid"].to_numpy()
    if len(x) > 50:
        slope, intercept = np.polyfit(x, y, 1)
    else:
        slope, intercept = 0.0, 0.0

    return {
        "effective_R_mOhm": round(-slope * 1000, 2),
        "intercept": round(intercept, 3),
        "samples": len(moving),
        "data": moving,   # for plotting
    }


# =============================================================================
# Plots
# =============================================================================

def plot_kpi_summary(kpi: pd.DataFrame) -> None:
    section("KPI bar charts")
    panels = [
        ("Top_Speed_Kmph",  "Top speed (km/h)"),
        ("Peak_Power_kW",   "Peak power (kW)"),
        ("Wh_per_km",       "Energy efficiency (Wh/km, lower=better)"),
        ("Distance_km",     "Distance (km)"),
        ("Peak_Current_A",  "Peak current (A)"),
        ("Min_Voltage_V",   "Min voltage (V, sag indicator)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (col, title) in zip(axes.flat, panels):
        sns.barplot(data=kpi, x="Session", y=col, hue="Session",
                    palette="Set2", legend=False, ax=ax)
        for i, v in enumerate(kpi[col]):
            ax.text(i, v, f"{v:.1f}", ha="center", va="bottom")
        ax.set_title(title)
    fig.suptitle("Performance KPIs by session", y=1.01)
    save(fig, "01_kpi_bars.png")


def plot_launches(df: pd.DataFrame) -> dict:
    section("Acceleration time analysis (launches)")
    targets = [30, 50, 60]
    all_launches = []
    summary = {}
    for t_kmph in targets:
        l = find_launches(df, t_kmph)
        l["target"] = t_kmph
        all_launches.append(l)
        if len(l):
            summary[f"0-{t_kmph}_best_s"] = round(l["elapsed_s"].min(), 2)
            summary[f"0-{t_kmph}_avg_s"]  = round(l["elapsed_s"].mean(), 2)
            summary[f"0-{t_kmph}_count"]  = len(l)

    launches_df = pd.concat(all_launches, ignore_index=True) if all_launches else pd.DataFrame()
    if launches_df.empty:
        print("  (no clean launches detected)")
        return summary

    print("Launch summary:")
    for k, v in summary.items():
        print(f"  {k:18s} {v}")

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.boxplot(data=launches_df, x="target", y="elapsed_s", ax=ax,
                hue="target", palette="Set2", legend=False)
    sns.stripplot(data=launches_df, x="target", y="elapsed_s", ax=ax,
                  color="black", size=4, alpha=0.6)
    ax.set_xlabel("Target speed (km/h)")
    ax.set_ylabel("Time to reach (seconds)")
    ax.set_title(f"Launch performance: 0-30 / 0-50 / 0-60 km/h "
                 f"({len(launches_df)} clean launches)")
    save(fig, "02_launch_times.png")

    launches_df.to_csv(OUTDIR / "launches.csv", index=False)
    print(f"  -> launches.csv")
    return summary


def plot_voltage_sag(df: pd.DataFrame, sag: dict) -> None:
    section("Voltage sag (effective pack internal resistance)")
    moving = sag["data"]
    print(f"Effective pack IR (after SoC detrend): {sag['effective_R_mOhm']} mOhm")
    print(f"  (datasheet target: ~20 mOhm)  samples used: {sag['samples']:,}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.scatterplot(data=moving.sample(min(2000, len(moving)), random_state=0),
                    x="Pack_Current_A", y="Pack_Voltage_V",
                    hue="SoC_pct", palette="viridis", s=10, alpha=0.6,
                    ax=axes[0])
    axes[0].set_title("Voltage vs current (coloured by SoC)")
    axes[0].set_xlabel("Pack current (A)")
    axes[0].set_ylabel("Pack voltage (V)")

    sns.scatterplot(data=moving.sample(min(2000, len(moving)), random_state=0),
                    x="Pack_Current_A", y="V_resid", s=10, alpha=0.5,
                    color="darkred", ax=axes[1])
    xline = np.linspace(0, moving["Pack_Current_A"].max(), 50)
    axes[1].plot(xline, sag["intercept"] - sag["effective_R_mOhm"]/1000 * xline,
                 color="black", linewidth=2,
                 label=f"fit: V_resid = {-sag['effective_R_mOhm']/1000:.4f}*I + {sag['intercept']:.3f}")
    axes[1].set_title(f"Residual voltage vs current  (effective IR ≈ {sag['effective_R_mOhm']:.1f} mΩ)")
    axes[1].set_xlabel("Pack current (A)")
    axes[1].set_ylabel("Voltage residual (V, SoC-detrended)")
    axes[1].legend()
    save(fig, "03_voltage_sag.png")


def plot_efficiency_curve(df: pd.DataFrame) -> None:
    section("Energy efficiency vs speed")
    moving = df[df["Speed_Kmph"] > 5].copy()
    moving["Speed_Bin"] = (moving["Speed_Kmph"] // 5 * 5).astype(int)
    eff = moving.groupby("Speed_Bin").agg(
        avg_power_kW=("Pack_Power_kW", "mean"),
        avg_speed=("Speed_Kmph", "mean"),
        n=("Speed_Kmph", "size"),
    )
    # Wh per km at this speed = avg_power_W / avg_speed_kmph (since W*h/km = W/(km/h))
    eff["Wh_per_km"] = eff["avg_power_kW"] * 1000.0 / eff["avg_speed"]
    eff = eff[eff["n"] >= 20]
    eff = eff.reset_index()
    print(eff.round(2).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=eff, x="Speed_Bin", y="Wh_per_km", ax=axes[0],
                color="steelblue")
    axes[0].set_title("Energy cost per km vs speed band")
    axes[0].set_xlabel("Speed band (km/h)")
    axes[0].set_ylabel("Wh/km (lower = more efficient)")
    axes[0].tick_params(axis="x", rotation=45)

    sns.barplot(data=eff, x="Speed_Bin", y="avg_power_kW", ax=axes[1],
                color="indianred")
    axes[1].set_title("Average power draw vs speed band")
    axes[1].set_xlabel("Speed band (km/h)")
    axes[1].set_ylabel("Average power (kW)")
    axes[1].tick_params(axis="x", rotation=45)
    save(fig, "04_efficiency_curve.png")

    eff.to_csv(OUTDIR / "efficiency_by_speed.csv", index=False)
    print(f"  -> efficiency_by_speed.csv")


def plot_power_distribution(df: pd.DataFrame) -> None:
    section("Power band: peak vs sustained")
    moving = df[df["Pack_Power_kW"] > 0.2]

    p99 = moving["Pack_Power_kW"].quantile(0.99)
    p95 = moving["Pack_Power_kW"].quantile(0.95)
    p50 = moving["Pack_Power_kW"].quantile(0.50)

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.histplot(moving["Pack_Power_kW"], bins=60, ax=ax, color="steelblue",
                 edgecolor="white", stat="percent")
    for q, c, lbl in [(p50, "green", f"median {p50:.2f} kW"),
                      (p95, "orange", f"95th pct {p95:.2f} kW"),
                      (p99, "red", f"99th pct {p99:.2f} kW")]:
        ax.axvline(q, color=c, linestyle="--", linewidth=1.5, label=lbl)
    ax.set_xlabel("Power (kW)")
    ax.set_ylabel("% of moving samples")
    ax.set_title("Power-draw distribution (moving only)")
    ax.legend()
    save(fig, "05_power_distribution.png")
    print(f"  Median (sustained) power: {p50:.2f} kW")
    print(f"  95th pct power:           {p95:.2f} kW")
    print(f"  99th pct (peak burst):    {p99:.2f} kW")


# =============================================================================

def main() -> None:
    df = pd.read_csv(SRC, parse_dates=["LoRa_Datetime"])
    print(f"Loaded {len(df):,} rows.")

    # ---- KPI table ----
    section("Per-session KPI table")
    kpi = build_kpi_table(df)
    print(kpi.to_string(index=False))
    kpi.to_csv(OUTDIR / "kpi_table.csv", index=False)
    print(f"\n-> {OUTDIR / 'kpi_table.csv'}")

    # ---- Plots & analyses ----
    plot_kpi_summary(kpi)
    launch_summary = plot_launches(df)
    sag = voltage_sag(df)
    plot_voltage_sag(df, sag)
    plot_efficiency_curve(df)
    plot_power_distribution(df)

    # ---- Final takeaways ----
    section("PHASE 2 TAKEAWAYS")
    best_eff_session = kpi.loc[kpi["Wh_per_km"].idxmin(), "Session"]
    worst_eff_session = kpi.loc[kpi["Wh_per_km"].idxmax(), "Session"]
    fastest_top  = kpi["Top_Speed_Kmph"].max()
    print(f"  - Top speed reached: {fastest_top} km/h")
    print(f"  - Best 0-30: {launch_summary.get('0-30_best_s', 'n/a')} s, "
          f"best 0-60: {launch_summary.get('0-60_best_s', 'n/a')} s")
    print(f"  - Most efficient session: #{best_eff_session} "
          f"({kpi.loc[kpi['Session']==best_eff_session,'Wh_per_km'].iloc[0]} Wh/km)")
    print(f"  - Effective pack IR: {sag['effective_R_mOhm']} mOhm "
          f"vs datasheet ~20 mOhm")
    print(f"\nAll Phase 2 outputs in: {OUTDIR}")


if __name__ == "__main__":
    main()
