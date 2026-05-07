"""
Refine oorja_telemetry.csv (electric-kart endurance telemetry):

Pipeline:
  STAGE 1 - Pre-process & refine
      a. Drop the "Endurance 2" banner; keep proper header.
      b. Type-cast all numeric columns; "ovf" / blanks -> NaN.
      c. Hard physical bounds (kart-realistic) -> NaN if violated.
      d. Rolling-median + MAD despiking on every numeric channel.
      e. Reject non-monotonic / wild LoRa epoch jumps.
      f. Interpolate gaps (limit 5 samples), then ffill/bfill edges.
      g. Smooth jittery signals with a small rolling mean.

  STAGE 2 - Add modelled Pack_Current_A and Pack_Voltage_V columns
    
      Voltage is kept around the 72 V nominal operating range, matching
      typical electric-kart race behaviour reported in EV-kart references.

Output -> oorja_telemetry_refined.csv (raw file untouched).
"""

from pathlib import Path
import numpy as np
import pandas as pd

SRC = Path(r"d:\OORJA_PROJECT\oorja_telemetry.csv")
DST = Path(r"d:\OORJA_PROJECT\oorja_telemetry_refined.csv")

# ---------- Datasheet constants  ----------
V_FULL           = 84.0     # max charge voltage (100% SoC)
V_NOMINAL        = 72.0     # nominal pack voltage
V_EMPTY          = 50.0     # min discharge voltage (0% SoC, BMS cutoff)
PACK_CAPACITY_AH = 85.0     # nominal Ah (NEW datasheet)
I_CONT_MAX       = 150.0    # BMS continuous current limit (A)
I_PEAK_MAX       = 300.0    # BMS peak current limit, <=10 s (A)
# Cell IR <=16 mOhm; pack 20S16P -> R_pack = 20 * 16e-3 / 16 = 20 mOhm
PACK_R_INTERNAL  = 0.020
IDLE_CURRENT_A   = 1.0      # BMS + electronics standby draw
SAMPLE_DT_S      = 1.0      # ~1 Hz logging
INITIAL_SOC      = 0.85     # race typically starts ~85% (slightly off full top-balance)

# Plausibility caps for an electric-kart drivetrain
SPEED_MAX_KMPH   = 80.0
RPM_MAX          = 3000.0
MPU_G_MAX        = 4.0      # karts can hit ~3 g in hard turns; >4 g is sensor noise

# Despiking knobs
MEDIAN_WINDOW    = 5        # rolling window for spike detection
MAD_THRESHOLD    = 6.0      # reject samples > 6 * MAD from local median
SMOOTH_WINDOW    = 3        # final smoothing window



_OCV_SOC = np.array([0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                     0.60, 0.70, 0.80, 0.90, 1.00])
_OCV_CELL = np.array([2.80, 3.30, 3.40, 3.55, 3.62, 3.66, 3.70,
                      3.76, 3.84, 3.94, 4.08, 4.20])
_OCV_PACK = _OCV_CELL * 20.0   # 20S


def ocv_pack(soc: np.ndarray) -> np.ndarray:
    """Pack open-circuit voltage as a function of SoC (0..1) using the NMC table."""
    return np.interp(np.clip(soc, 0.0, 1.0), _OCV_SOC, _OCV_PACK)


# =============================================================================
# STAGE 1 - Pre-process and refine
# =============================================================================

def load_raw(path: Path) -> pd.DataFrame:
    # Row 1 is the "Endurance 2" banner; row 2 is the real header.
    df = pd.read_csv(path, skiprows=1, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    return df


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def despike(s: pd.Series, window: int = MEDIAN_WINDOW,
            mad_k: float = MAD_THRESHOLD) -> pd.Series:
    """Replace samples that are >mad_k * MAD from local median with NaN."""
    med = s.rolling(window, center=True, min_periods=1).median()
    abs_dev = (s - med).abs()
    mad = abs_dev.rolling(window, center=True, min_periods=1).median()
    # Avoid divide-by-zero; if MAD is 0 (constant region), don't flag anything.
    mad = mad.replace(0, np.nan)
    spike = abs_dev > (mad_k * mad)
    cleaned = s.mask(spike)
    return cleaned


def refine(df: pd.DataFrame) -> pd.DataFrame:
    # ---- 1. Token-level cleanup: ovf, blanks -> NaN
    df = df.replace({"ovf": np.nan, "OVF": np.nan, "": np.nan, "nan": np.nan})

    df["Packet_ID"]      = to_num(df["Packet_ID"])
    df["Speed_Kmph"]     = to_num(df["Speed_Kmph"])
    df["RPM"]            = to_num(df["RPM"])
    df["LoRa_Timestamp"] = to_num(df["LoRa_Timestamp"])
    df["MPU_Gx"]         = to_num(df["MPU_Gx"])
    df["MPU_Gy"]         = to_num(df["MPU_Gy"])
    df["MPU_Gz"]         = to_num(df["MPU_Gz"])

    # ---- 2. Hard physical bounds
    df.loc[(df["Speed_Kmph"] < 0) | (df["Speed_Kmph"] > SPEED_MAX_KMPH), "Speed_Kmph"] = np.nan
    df.loc[(df["RPM"] < 0)        | (df["RPM"] > RPM_MAX),               "RPM"]        = np.nan
    for c in ["MPU_Gx", "MPU_Gy", "MPU_Gz"]:
        df.loc[df[c].abs() > MPU_G_MAX, c] = np.nan

    # LoRa epoch: must be in a sane absolute window AND monotonically increasing.
    lora_lo, lora_hi = 1_400_000_000, 2_000_000_000
    df.loc[(df["LoRa_Timestamp"] < lora_lo) | (df["LoRa_Timestamp"] > lora_hi),
           "LoRa_Timestamp"] = np.nan
    # Reject samples that go backwards by more than 5 s or jump forward >300 s.
    diff = df["LoRa_Timestamp"].diff()
    bad_jump = (diff < -5) | (diff > 300)
    df.loc[bad_jump, "LoRa_Timestamp"] = np.nan

    # ---- 3. Statistical despiking (rolling-median + MAD)
    for c in ["Speed_Kmph", "RPM", "MPU_Gx", "MPU_Gy", "MPU_Gz"]:
        df[c] = despike(df[c])

    # Speed/RPM coherence: if RPM is essentially zero but speed is moving fast,
    # one of them is wrong. Trust whichever is more plausible by flagging the
    # outlier of the pair (cheap heuristic: drop both, interpolate).
    incoherent = (df["RPM"] < 30) & (df["Speed_Kmph"] > 10)
    df.loc[incoherent, ["Speed_Kmph", "RPM"]] = np.nan

    # ---- 4. Gap fill: short gaps -> interpolate, edges -> ffill/bfill
    fill_cols = ["Speed_Kmph", "RPM", "MPU_Gx", "MPU_Gy", "MPU_Gz", "LoRa_Timestamp"]
    df[fill_cols] = df[fill_cols].interpolate(limit=5).ffill().bfill()

    # ---- 5. Light smoothing on noisy signals (kart vibration -> IMU jitter)
    for c in ["MPU_Gx", "MPU_Gy", "MPU_Gz"]:
        df[c] = df[c].rolling(SMOOTH_WINDOW, center=True, min_periods=1).mean()

    # ---- 6. Type & rounding cleanup
    df["Speed_Kmph"]     = df["Speed_Kmph"].round(0).astype(int)
    df["RPM"]            = df["RPM"].round(2)
    df["MPU_Gx"]         = df["MPU_Gx"].round(3)
    df["MPU_Gy"]         = df["MPU_Gy"].round(3)
    df["MPU_Gz"]         = df["MPU_Gz"].round(3)
    df["LoRa_Timestamp"] = df["LoRa_Timestamp"].round(0).astype("Int64")
    df["Packet_ID"]      = df["Packet_ID"].astype("Int64")
    return df


# =============================================================================
# STAGE 2 - Battery model (current + voltage)
# =============================================================================

def model_battery(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pack_Current_A: blend of cruise load (speed), acceleration demand (dRPM/dt),
                    and forward IMU pull (MPU_Gx). Clipped per BMS limits.
    Pack_Voltage_V: NMC OCV(SoC) - I*R_internal, SoC depleted by Coulomb counting,
                    starting at INITIAL_SOC so the operating band sits near the
                    72 V nominal rather than the 84 V top-of-charge.
    """
    speed = df["Speed_Kmph"].astype(float).to_numpy()
    rpm   = df["RPM"].astype(float).to_numpy()
    gx    = df["MPU_Gx"].astype(float).to_numpy()

    drpm = np.gradient(rpm) / SAMPLE_DT_S  # RPM/s

    # Tuning -- targets:
    #   stationary       -> ~1 A
    #   cruise @ 30 km/h -> ~30 A
    #   cruise @ 60 km/h -> ~60 A
    #   hard accel       -> 120-180 A (briefly above continuous, BMS soft-clip)
    K_SPEED = 1.0
    K_ACCEL = 0.05
    K_GX    = 80.0

    current = (
        IDLE_CURRENT_A
        + K_SPEED * speed
        + K_ACCEL * np.clip(drpm, 0, None)
        + K_GX    * np.clip(gx,   0, None)
    )

    # Force near-idle when truly stationary
    stationary = (speed == 0) & (rpm < 30) & (np.abs(gx) < 0.05)
    current = np.where(stationary, IDLE_CURRENT_A, current)

    # BMS soft-clip above continuous, hard-clip at peak
    over_cont = current > I_CONT_MAX
    current = np.where(over_cont,
                       I_CONT_MAX + (current - I_CONT_MAX) * 0.5,
                       current)
    current = np.clip(current, 0.0, I_PEAK_MAX)

    # Coulomb counting from INITIAL_SOC
    charge_drawn_ah = np.cumsum(current) * SAMPLE_DT_S / 3600.0
    soc = np.clip(INITIAL_SOC - charge_drawn_ah / PACK_CAPACITY_AH, 0.0, 1.0)

    voltage = ocv_pack(soc) - current * PACK_R_INTERNAL
    voltage = np.clip(voltage, V_EMPTY, V_FULL)

    df["Pack_Current_A"] = np.round(current, 2)
    df["Pack_Voltage_V"] = np.round(voltage, 2)
    return df


# =============================================================================

def main() -> None:
    raw = load_raw(SRC)
    print(f"Loaded {len(raw):,} rows from {SRC.name}")

    refined = refine(raw)

    # Quick refinement audit
    n_zero_speed = int((refined["Speed_Kmph"] == 0).sum())
    print(f"Refined: {len(refined):,} rows, {n_zero_speed:,} idle "
          f"({n_zero_speed/len(refined)*100:.1f}%)")

    out = model_battery(refined)
    out.to_csv(DST, index=False)
    print(f"\nWrote -> {DST}")

    print("\n--- Summary stats (refined + modelled) ---")
    print(out[["Speed_Kmph", "RPM", "MPU_Gx", "MPU_Gy", "MPU_Gz",
               "Pack_Current_A", "Pack_Voltage_V"]].describe().round(2))

    print("\n--- First 3 rows ---")
    print(out.head(3).to_string(index=False))

    print("\n--- A racing burst (rows ~127-138) ---")
    print(out.iloc[126:138].to_string(index=False))

    print("\n--- End of session ---")
    print(out.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
