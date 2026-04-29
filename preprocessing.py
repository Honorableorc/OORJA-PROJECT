"""
Preprocessing pipeline for oorja_telemetry_refined.csv

Stages:
  A. Time handling      -- parse timestamps, build dt, segment sessions
  B. Feature engineering -- accel, distance, gear ratio, IMU magnitude,
                            pitch/roll, vibration, power, energy, SoC,
                            drive-state label
  C. Filtering / smoothing -- low-pass Butterworth on IMU, idle flag
  D. Scaling / encoding   -- StandardScaler on continuous features,
                            one-hot encode Drive_State; saves scaler
                            artefact so test data can reuse it.

Outputs:
  oorja_telemetry_processed.csv   -- engineered features, unscaled (analysis-friendly)
  oorja_telemetry_scaled.csv      -- ML-ready (scaled + one-hot)
  preprocessing_scaler.joblib     -- fitted StandardScaler + column list
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
import joblib

ROOT = Path(r"d:\OORJA_PROJECT")
SRC          = ROOT / "oorja_telemetry_refined.csv"
DST_FEATS    = ROOT / "oorja_telemetry_processed.csv"
DST_SCALED   = ROOT / "oorja_telemetry_scaled.csv"
SCALER_PATH  = ROOT / "preprocessing_scaler.joblib"

SAMPLE_HZ        = 1.0
IDLE_GAP_SECONDS = 120     # >2 min stationary -> new session
LOWPASS_CUTOFF   = 0.3     # Hz; must be < Nyquist (0.5 Hz at 1 Hz sampling)
LOWPASS_ORDER    = 3
ACCEL_THRESH_KMPH_S = 0.5  # for drive-state classification


# =============================================================================
# STAGE A - Time handling
# =============================================================================

def stage_a_time(df: pd.DataFrame) -> pd.DataFrame:
    df["LoRa_Datetime"] = pd.to_datetime(df["LoRa_Timestamp"], unit="s", utc=True)
    df["LoRa_Datetime"] = df["LoRa_Datetime"].dt.tz_convert("Asia/Kolkata")

    t = df["LoRa_Timestamp"].astype("Int64").astype("float64").to_numpy()
    dt = np.diff(t, prepend=t[0])
    dt = np.clip(dt, 0.0, 5.0)
    df["dt_s"]      = np.round(dt, 3)
    df["Elapsed_s"] = np.cumsum(dt).round(2)

    # Session segmentation: a fresh session starts whenever the kart has been
    # idle (speed == 0) continuously for >IDLE_GAP_SECONDS, then starts moving.
    speed = df["Speed_Kmph"].to_numpy()
    moving = speed > 0
    idle_run = np.zeros(len(df), dtype=int)
    cur = 0
    for i, m in enumerate(moving):
        cur = 0 if m else cur + 1
        idle_run[i] = cur
    long_idle_break = (idle_run == 0) & (np.r_[0, idle_run[:-1]] > IDLE_GAP_SECONDS)
    df["Session_ID"] = (long_idle_break.cumsum() + 1).astype(int)
    return df


# =============================================================================
# STAGE B - Feature engineering
# =============================================================================

def stage_b_features(df: pd.DataFrame) -> pd.DataFrame:
    speed = df["Speed_Kmph"].to_numpy(dtype=float)
    rpm   = df["RPM"].to_numpy(dtype=float)
    gx    = df["MPU_Gx"].to_numpy(dtype=float)
    gy    = df["MPU_Gy"].to_numpy(dtype=float)
    gz    = df["MPU_Gz"].to_numpy(dtype=float)
    volt  = df["Pack_Voltage_V"].to_numpy(dtype=float)
    curr  = df["Pack_Current_A"].to_numpy(dtype=float)
    dt    = df["dt_s"].to_numpy(dtype=float)

    df["Accel_Kmph_s"] = np.round(np.gradient(speed) * SAMPLE_HZ, 3)
    df["Accel_m_s2"]   = np.round(df["Accel_Kmph_s"] / 3.6, 3)
    speed_mps = speed / 3.6
    df["Distance_m"]   = np.round(np.cumsum(speed_mps * dt), 2)
    df["Distance_km"]  = np.round(df["Distance_m"] / 1000.0, 4)

    with np.errstate(divide="ignore", invalid="ignore"):
        gear = np.where(speed > 1, rpm / speed, np.nan)
    df["Gear_Ratio"] = np.round(gear, 2)

    g_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    df["IMU_G_Magnitude"] = np.round(g_mag, 3)
    df["IMU_Pitch_deg"] = np.round(np.degrees(np.arctan2(gx, np.sqrt(gy**2 + gz**2))), 2)
    df["IMU_Roll_deg"]  = np.round(np.degrees(np.arctan2(gy, np.sqrt(gx**2 + gz**2))), 2)

    df["IMU_Vibration"] = (
        pd.Series(gz).rolling(3, center=True, min_periods=1).std().fillna(0).round(4)
    )

    power_w = volt * curr
    df["Pack_Power_W"]  = np.round(power_w, 1)
    df["Pack_Power_kW"] = np.round(power_w / 1000.0, 3)
    energy_wh = np.cumsum(power_w * dt) / 3600.0
    df["Energy_Wh_cum"] = np.round(energy_wh, 2)
    charge_ah = np.cumsum(curr * dt) / 3600.0
    df["Charge_Drawn_Ah"] = np.round(charge_ah, 3)
    INITIAL_SOC = 0.85
    PACK_AH     = 85.0
    soc = np.clip(INITIAL_SOC - charge_ah / PACK_AH, 0.0, 1.0)
    df["SoC_pct"] = np.round(soc * 100.0, 2)

    accel = df["Accel_Kmph_s"].to_numpy()
    state = np.full(len(df), "cruise", dtype=object)
    state[speed == 0] = "idle"
    state[(speed > 0) & (accel >  ACCEL_THRESH_KMPH_S)] = "accel"
    state[(speed > 0) & (accel < -ACCEL_THRESH_KMPH_S)] = "decel"
    df["Drive_State"] = state
    return df


# =============================================================================
# STAGE C - Filtering / smoothing
# =============================================================================

def _butter_lowpass(x: np.ndarray, fs: float, cutoff: float, order: int) -> np.ndarray:
    if len(x) < 3 * order:
        return x
    b, a = butter(order, cutoff / (fs / 2.0), btype="low")
    return filtfilt(b, a, x)


def stage_c_filtering(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["MPU_Gx", "MPU_Gy", "MPU_Gz"]:
        df[c + "_LP"] = np.round(
            _butter_lowpass(df[c].to_numpy(dtype=float),
                            fs=SAMPLE_HZ, cutoff=LOWPASS_CUTOFF, order=LOWPASS_ORDER),
            4,
        )
    df["Idle_Flag"] = (df["Speed_Kmph"] == 0).astype(int)
    return df


# =============================================================================
# STAGE D - Scaling / encoding
# =============================================================================

CONTINUOUS_FEATURES = [
    "Speed_Kmph", "RPM",
    "MPU_Gx_LP", "MPU_Gy_LP", "MPU_Gz_LP",
    "IMU_G_Magnitude", "IMU_Pitch_deg", "IMU_Roll_deg", "IMU_Vibration",
    "Accel_Kmph_s", "Gear_Ratio",
    "Pack_Voltage_V", "Pack_Current_A", "Pack_Power_kW",
    "SoC_pct",
]

PASSTHROUGH = [
    "LoRa_Datetime", "Elapsed_s", "Session_ID", "Distance_km",
    "Energy_Wh_cum", "Charge_Drawn_Ah", "Idle_Flag",
]


def stage_d_scaling(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    work = df.copy()
    work["Gear_Ratio"] = work["Gear_Ratio"].fillna(0.0)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(work[CONTINUOUS_FEATURES].to_numpy(dtype=float))
    scaled_df = pd.DataFrame(scaled, columns=[c + "_z" for c in CONTINUOUS_FEATURES],
                             index=work.index).round(4)

    one_hot = pd.get_dummies(work["Drive_State"], prefix="State").astype(int)

    out = pd.concat([work[PASSTHROUGH], scaled_df, one_hot], axis=1)
    return out, scaler


# =============================================================================

def main() -> None:
    df = pd.read_csv(SRC)
    print(f"Loaded {len(df):,} rows from {SRC.name}")

    df = stage_a_time(df)
    df = stage_b_features(df)
    df = stage_c_filtering(df)

    df.to_csv(DST_FEATS, index=False)
    print(f"Wrote engineered features  -> {DST_FEATS.name}  ({len(df.columns)} cols)")

    scaled, scaler = stage_d_scaling(df)
    scaled.to_csv(DST_SCALED, index=False)
    joblib.dump({"scaler": scaler, "feature_names": CONTINUOUS_FEATURES}, SCALER_PATH)
    print(f"Wrote scaled ML-ready set  -> {DST_SCALED.name}  ({len(scaled.columns)} cols)")
    print(f"Saved scaler artefact      -> {SCALER_PATH.name}")

    print("\n--- Sessions detected ---")
    print(df.groupby("Session_ID").agg(
        rows=("Speed_Kmph", "size"),
        max_speed=("Speed_Kmph", "max"),
        distance_km=("Distance_km", lambda s: s.max() - s.min()),
        energy_wh=("Energy_Wh_cum", lambda s: s.max() - s.min()),
    ))

    print("\n--- Drive-state distribution ---")
    print(df["Drive_State"].value_counts())

    print("\n--- Engineered feature stats (key columns) ---")
    cols = ["Accel_Kmph_s", "Distance_km", "Gear_Ratio", "IMU_G_Magnitude",
            "IMU_Vibration", "Pack_Power_kW", "Energy_Wh_cum", "SoC_pct"]
    print(df[cols].describe().round(3))


if __name__ == "__main__":
    main()
