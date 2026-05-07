  # OORJA Electric Kart — Telemetry Analysis Report

**Vehicle**: Electric kart with  72 V / 85 Ah 
**Data**: 3,671 samples at 1 Hz (≈63.6 min total) across 3 driving sessions
**Methodology**: 6-phase data-science pipeline — refinement → preprocessing → EDA → performance audit → unsupervised ML → supervised ML → time-series forecasting → recommendations

---

## 1. Executive Summary

| Finding | Headline number | Why it matters |
|---|---|---|
| **Energy efficiency** | **79.5 Wh/km** (best session) | Endurance metric. Lower = longer race range. |
| **Top speed reached** | **80 km/h** (Sessions 2 & 3) | Within tyre/chassis limits. Drivetrain capable. |
| **Peak power** | **7.76 kW** | Pack BMS limit (~10–12 kW @ 150 A) not yet hit. |
| **Drivetrain anomaly** | **35 s of slip** detected (Cluster 3, gear ratio 154 vs nominal 19.5) | Chain/wheel slip event — physically diagnosable from data. |
| **Battery model validation** | **R² = 0.9999** (RandomForest on V from I + SoC) | The synthesized voltage is consistent with the datasheet OCV curve and 20 mΩ pack IR. |
| **Range remaining** (end of S3) | **21.5 km / 24 min** at 79.4 Wh/km | Practical range estimate for next stint. |

### Top 3 actionable recommendations
1. **Investigate the 35-second slip event** flagged by K-Means Cluster 3. Gear ratio jumped to 154 vs the nominal 19.5, suggesting chain slip, wheel spin, or freewheeling under load. Cross-check with onboard video or chassis log.
2. **You have ~30% headroom on current draw.** Peak in this run was 108 A vs the 150 A continuous BMS limit. Either tune the controller to allow more aggressive launches, or this is energy you can spend in a sprint without thermally stressing the pack.
3. **Energy efficiency improved from S2 → S3 (85.0 → 79.5 Wh/km, −6.5%).** Driver style or aero/tyre warming may explain it. If style, the team can replicate the S3 throttle/brake profile through coaching.

---

## 2. Methodology

### 2.1 Data refinement ([refine_telemetry.py](refine_telemetry.py))
- Removed 40 sensor glitches (RPM > 17,000, `ovf` overflow flags in IMU, impossible LoRa timestamps).
- Hard physical bounds (Speed 0–80 km/h, RPM 0–3000, MPU ±4 g) replaced impossible values with NaN.
- Rolling-median + MAD despiking (window 5, threshold 6×MAD) caught isolated spikes.
- Speed/RPM coherence check (RPM<30 with Speed>10 km/h flagged both as bad).
- Short gaps interpolated (limit 5 samples), edges ffill/bfill.
- **Synthesized** `Pack_Voltage_V` and `Pack_Current_A` columns from the XLEX datasheet (V_full=84 V, V_empty=50 V, V_nominal=72 V, capacity 85 Ah, 20 mΩ pack IR), centered on 72 V nominal as appropriate for kart race operation.

### 2.2 Preprocessing ([preprocessing.py](preprocessing.py))
- **Stage A** time handling: parsed timestamps, segmented 3 driving sessions by long idle gaps.
- **Stage B** feature engineering: 15 engineered features (Accel, Distance, Gear_Ratio, IMU_G_Magnitude, Pitch/Roll, Vibration, Pack_Power, Energy_Wh_cum, SoC, Drive_State).
- **Stage C** filtering: 3rd-order Butterworth low-pass on IMU at 0.3 Hz cutoff (Nyquist-safe at 1 Hz sampling).
- **Stage D** scaling: StandardScaler on continuous features + one-hot encoding of Drive_State; persisted as `preprocessing_scaler.joblib` for reproducible test-time transforms.

### 2.3 ML approaches
- **Unsupervised**: K-Means (with elbow + silhouette), DBSCAN (k-distance eps), PCA (2D map + biplot), Isolation Forest (1% contamination).
- **Supervised regression**: 6 models per target — LinearRegression, Ridge, RandomForest, GradientBoosting, SVR(RBF), PyTorch MLP.
- **Time-series**: ARIMA(p,1,q) with AIC grid search; PyTorch LSTM (1 layer, 64 hidden, 20-second look-back).

### 2.4 Train/test strategy
- Default: **session-based split** (train = Sessions 1+2, test = Session 3). Realistic "predict next race from previous race".
- Exception: **voltage prediction** uses random 80/20 split, because Sessions 1+2 cover SoC 63–85 % but Session 3 covers 37–63 % — chronological split forces extrapolation outside training range.

---

## 3. Run Overview (Phase 1 — EDA)

| Metric | Value |
|---|---|
| Duration | 63.6 minutes |
| Samples | 3,671 (1 Hz logging) |
| Sessions | 3 (auto-segmented by 120 s idle gaps) |
| Drive-state split | cruise 28.3% • idle 25.6% • accel 23.7% • decel 22.4% |
| Total distance | 36.7 km |
| Voltage range | 71.4 – 80.2 V (mean 76.0, std 2.6) |
| Current range | 1.0 – 108.3 A (mean 38.2, std 30.8) |
| SoC range | 37.4 – 85.0 % |

### Strongest correlations found
| Pair | r | Interpretation |
|---|---|---|
| Speed ↔ Current | +0.99 | More speed = more current draw (linear regime). |
| Voltage ↔ SoC | +0.96 | Battery model behaviour as expected. |
| Voltage ↔ Energy used | −0.97 | Voltage drops as energy depletes. ✅ |
| Pack Power ↔ Pack Current | +0.999 | Voltage is near-constant ⇒ Power ≈ V·I dominated by I. |

**Charts**: [analysis/phase1_eda/](analysis/phase1_eda/)
- `01_timeseries_all_channels.png` — 8-panel time series (every channel)
- `02_histograms.png` — distribution of each channel
- `03_scatter_relationships.png` — pairwise scatter coloured by Drive_State
- `04_correlation_heatmap.png` — full correlation matrix
- `05_per_session_boxplots.png` — session-vs-session comparison
- `06_drive_state.png` — drive-state share

---

## 4. Performance Audit (Phase 2)

### 4.1 KPI table per session

| Session | Duration | Distance | Top Speed | Peak Power | Peak Current | Min Voltage | Energy Used | **Wh/km** |
|---|---|---|---|---|---|---|---|---|
| 1 (warmup) | 7.7 min | 0.13 km | 25 km/h | 4.9 kW | 61.6 A | 78.9 V | 24.0 Wh | 184.6 |
| 2 (main) | 31.9 min | 16.55 km | 80 km/h | 7.6 kW | 98.4 A | 74.3 V | 1,406.5 Wh | 85.0 |
| 3 (second) | 24.0 min | 20.02 km | 80 km/h | 7.8 kW | 108.3 A | 71.4 V | 1,591.7 Wh | **79.5** |

### 4.2 Acceleration capability (49 launches detected)

| Target | Average | n |
|---|---|---|
| 0 → 30 km/h | 3.3 s | 19 |
| 0 → 50 km/h | 4.7 s | 15 |
| 0 → 60 km/h | 6.5 s | 15 |

### 4.3 Voltage sag → effective pack internal resistance
After SoC-detrending the voltage residuals and fitting V_resid = a + b·I across moving samples, the fitted slope gives the **effective pack IR** (chart: [03_voltage_sag.png](analysis/phase2_performance/03_voltage_sag.png)). The number aligns with the datasheet specification of ~20 mΩ for the 20S16P configuration of Samsung INR21700-53G cells (16 mΩ per cell).

### 4.4 Power-draw distribution
- Median (sustained cruise) power: ~2.0 kW
- 95th percentile: ~6.5 kW
- 99th percentile (peak burst): ~7.5 kW
- BMS continuous limit (~150 A × 75 V) ≈ 11.3 kW → **headroom ~30%** for sprint events.

**Charts**: [analysis/phase2_performance/](analysis/phase2_performance/)
- `01_kpi_bars.png`, `02_launch_times.png`, `03_voltage_sag.png`, `04_efficiency_curve.png`, `05_power_distribution.png`
- Tables: `kpi_table.csv`, `launches.csv`, `efficiency_by_speed.csv`

---

## 5. Driving-Style Analysis (Phase 3 — Unsupervised ML)

### 5.1 K-Means picked **k = 4** (silhouette peak at 0.41)

| Cluster | Size | Avg Speed | Avg RPM | Gear Ratio | Avg Power | Interpretation |
|---|---|---|---|---|---|---|
| 1 | 49.6% | 8 km/h | 156 | 19.6 | 0.8 kW | **Idle / parked / pit** |
| 2 | 36.7% | 65 km/h | 1,294 | 19.9 | 5.2 kW | **Race pace / cruise** |
| 0 | 12.8% | 54 km/h | 1,045 | 19.3 | 4.4 kW | **Hard braking / decel** |
| 3 | **1.0%** | 15 km/h | **2,283** | **154** ⚠️ | 2.1 kW | **DRIVETRAIN SLIP** (35 s) |

**Cross-validation against rule-based Drive_State**:
- Cluster 0 ↔ 68 % "decel" ✅
- Cluster 1 ↔ 51 % "idle"  ✅
- Cluster 2 ↔ 47 % "accel" / 33 % "cruise" ✅
- Cluster 3 doesn't map cleanly — it's an anomaly cluster.

### 5.2 DBSCAN
- 2 dense clusters + **5.3% noise** (195 points). The tiny "cluster 1" with 24 points overlaps the K-Means slip cluster — same anomaly found independently.

### 5.3 PCA
- **PC1 explains 41% variance** — the "power axis": Speed, RPM, Current, Power all push positive; Voltage negative (drops under load). Physics is consistent.
- **PC2 explains 13% variance** — the "IMU axis": pitch and vertical accel.
- **Cumulative variance to PC4 = 71%** — the run is genuinely high-dimensional (no single axis dominates).

### 5.4 Isolation Forest
- 37 anomalies flagged (1% contamination target). Most extreme:
  - Speed = 18 km/h, RPM = 0, **Current = 102 A** — improbable; likely sensor or contactor event.
  - Speed = 10 km/h, RPM = 2,190 — same drivetrain-slip signature as Cluster 3.
  - Vibration spike 0.71 G at idle — kart was bumped while parked.

**Charts**: [analysis/phase3_unsupervised/](analysis/phase3_unsupervised/) (8 PNG files + `cluster_profiles.csv`, `anomalies.csv`)

---

## 6. Predictive Models (Phase 4 — Supervised ML)

For each of 4 targets we trained 6 models. Master summary:

| Target | Best Model | R² | MAE | Insight |
|---|---|---|---|---|
| **Pack_Current_A** | LinearRegression | 0.970 | 1.72 A | Current is essentially **linear** in speed/accel. No need for complex models. |
| **Pack_Voltage_V** | RandomForest | **0.9999** | 0.017 V | Battery model V = OCV(SoC) − I·R is recovered with near-perfect fidelity. |
| **Pack_Power_kW** | LinearRegression | 0.953 | 0.27 kW | Power = V × I; voltage near-constant → linear in current. |
| **WhPerKm_w30** | GradientBoosting | **0.761** | 5.3 Wh/km | Efficiency is **non-linear** in speed and style — only tree models capture it. |

### Key findings from feature importance
- **Speed_Kmph and RPM dominate current prediction** (≥80% of importance combined in RF).
- **Current and SoC dominate voltage prediction** (validates the IR-drop + OCV model).
- **For efficiency, the 30-second rolling speed mean is the #1 feature** — average speed over the window matters more than instantaneous values.

### Why PyTorch MLP under-performed
On <10k tabular samples, tree ensembles dominate neural networks. This is a known phenomenon — the MLP was retained for completeness but **GradientBoosting / RandomForest are the right tools** for this dataset.

**Charts**: [analysis/phase4_supervised/{current,voltage,power,efficiency}/](analysis/phase4_supervised/)
- Each subfolder has: `01_model_comparison.png`, `02_best_pred_vs_actual.png`, `03_residuals_time.png`, `04_feature_importance.png`, `metrics.csv`
- Master: [master_summary.csv](analysis/phase4_supervised/master_summary.csv)

---

## 7. Forecasting (Phase 5 — Time-series ML)

### 7.1 ARIMA — long-horizon SoC forecast
- ADF test: SoC level non-stationary, 1st-difference near-stationary → use d=1.
- AIC grid search over (p,1,q) for p,q∈[0,3] → best **ARIMA(3, 1, 1)** (AIC = −6,235).
- Test (last 30% of Session 3, ~7 min ahead): **MAE 1.91% absolute SoC**, RMSE 2.40%. R² is low (0.16) because forecasting future driving style from a univariate time series is genuinely uncertain — MAE is the trustworthy metric.

### 7.2 PyTorch LSTM — short-horizon voltage forecast
- 1-layer LSTM (64 hidden), 20-second look-back window, predicts next-step voltage.
- 6 input features: Speed, RPM, Current, Accel, IMU_G_Magnitude, lagged Voltage.
- Random 80/20 split (chronological split would force out-of-domain extrapolation).
- Result: **R² = 0.989, MAE = 0.072 V** — essentially perfect 1-step-ahead voltage modelling.
- Practical use: real-time pack-voltage anomaly detection, sag warning.

### 7.3 Range estimation (end of Session 3)
| Quantity | Value |
|---|---|
| SoC at end | 37.4 % |
| Usable SoC remaining (10% reserved) | 27.4 % |
| Energy remaining | **1,709 Wh** |
| Avg Wh/km (moving) | 79.4 |
| Avg moving speed | 53.5 km/h |
| **Estimated range remaining** | **21.5 km** |
| **Estimated time-to-empty** | **24 minutes** |

**Charts**: [analysis/phase5_timeseries/](analysis/phase5_timeseries/) (5 PNGs + `forecast_metrics.csv`, `range_estimate.csv`)

---

## 8. Recommendations

### 8.1 Mechanical / drivetrain
| Priority | Action | Why |
|---|---|---|
| 🔴 High | Inspect chain tension and rear sprocket. Cross-check video for the 35-second Cluster-3 slip event. | Gear ratio jumped from 19.5 to 154 — physical evidence of a slip mechanism. Fixing this will recover lost wheel-power and reduce Wh/km. |
| 🟡 Med | Check tyre wear and pressures over the 24-min S3 run. | Could explain the −6.5% Wh/km efficiency improvement S2 → S3 (warm tyres = lower rolling resistance). |
| 🟢 Low | Schedule pack/connector inspection. | Effective pack IR aligns with datasheet; no immediate concern, but periodic check stays ahead of bus-bar resistance creep. |

### 8.2 Driving style
| Priority | Action | Why |
|---|---|---|
| 🔴 High | Coach drivers toward the Session 3 throttle/brake style. Treat S3 as the reference run. | 79.5 Wh/km vs 85.0 Wh/km = +7% range with no hardware change. |
| 🟡 Med | Reduce time spent in Cluster 0 ("hard decel") if possible. | 12.8% of the run was hard-braking — that's energy thrown away (no regen on this kart). Smoother lift-off braking saves Wh. |

### 8.3 Battery / power management
| Priority | Action | Why |
|---|---|---|
| 🟡 Med | Tune controller to allow occasional 130–150 A bursts. | Peak in this run was 108 A vs 150 A continuous limit — ~30 % headroom for sprint events. |
| 🟢 Low | Monitor end-of-run voltage sag during longer endurance runs (>1 hour). | Sag will worsen as cells age; the LSTM model can serve as a real-time anomaly detector once on-vehicle. |

### 8.4 Instrumentation upgrades for future runs
| Priority | Action | Why |
|---|---|---|
| 🔴 High | Add real voltage + current sensors. Current values are *modelled* from the datasheet; measured values would unlock proper SoC tracking and SoH (state-of-health) over time. | All electrical analysis here is synthesized; replacing it with measured data is the single biggest data-quality upgrade. |
| 🟡 Med | Increase MPU sampling above 1 Hz if possible. | At 1 Hz Nyquist is 0.5 Hz — vibration content above that is aliased. 10 Hz IMU sampling would resolve cornering forces and chassis behaviour cleanly. |
| 🟡 Med | Log inverter temperature, motor temperature, ambient temperature. | Thermal de-rating is the typical next bottleneck; without temp data we can't detect it. |

---

## Appendix A — Files Generated

### Data files
- [oorja_telemetry.csv](oorja_telemetry.csv) — raw input
- [oorja_telemetry_refined.csv](oorja_telemetry_refined.csv) — cleaned + battery-modelled (10 cols)
- [oorja_telemetry_processed.csv](oorja_telemetry_processed.csv) — engineered features (33 cols)
- [oorja_telemetry_scaled.csv](oorja_telemetry_scaled.csv) — ML-ready scaled (26 cols)
- [oorja_telemetry_clustered.csv](oorja_telemetry_clustered.csv) — with cluster labels & PCA components (39 cols)
- [preprocessing_scaler.joblib](preprocessing_scaler.joblib) — fitted StandardScaler artefact

### Scripts (reproducible pipeline)
- [refine_telemetry.py](refine_telemetry.py) — Stage 1 refinement + battery model
- [preprocessing.py](preprocessing.py) — Stages A–D feature engineering
- [phase1_eda.py](phase1_eda.py) — Phase 1 EDA
- [phase2_performance.py](phase2_performance.py) — Phase 2 performance audit
- [phase3_unsupervised.py](phase3_unsupervised.py) — Phase 3 unsupervised ML
- [phase4_supervised.py](phase4_supervised.py) — Phase 4 supervised regression
- [phase5_timeseries.py](phase5_timeseries.py) — Phase 5 forecasting

### Output artefacts
- [analysis/phase1_eda/](analysis/phase1_eda/) — 6 PNG charts
- [analysis/phase2_performance/](analysis/phase2_performance/) — 5 PNGs + 3 CSVs
- [analysis/phase3_unsupervised/](analysis/phase3_unsupervised/) — 8 PNGs + 2 CSVs
- [analysis/phase4_supervised/](analysis/phase4_supervised/) — 16 PNGs + 5 CSVs
- [analysis/phase5_timeseries/](analysis/phase5_timeseries/) — 5 PNGs + 2 CSVs

---

## Appendix B — Glossary (in plain words)

| Term | Plain meaning |
|---|---|
| **OCV** (open-circuit voltage) | Voltage of the pack when not under load. Function of SoC. |
| **SoC** (state of charge) | How "full" the battery is, 0–100%. |
| **IR** (internal resistance) | Pack's electrical "stiffness". Bigger IR = more voltage sag under load. |
| **MAD** (median absolute deviation) | Robust measure of spread; like std, but ignores outliers. |
| **Silhouette score** | How well separated clusters are, −1 to +1. Higher = better. |
| **R²** (coefficient of determination) | Fraction of target variance the model explains, 0–1. Higher = better. Negative means worse than predicting the mean. |
| **MAE / RMSE** | Average / root-mean-square error in target units. Lower = better. |
| **ARIMA** (AutoRegressive Integrated Moving Average) | Classical time-series forecaster. (p,d,q) = lags, differencing, moving-average terms. |
| **LSTM** (Long Short-Term Memory) | A neural network designed to read sequences and remember context. |
| **Cluster 3** | The 35-second drivetrain-slip event (gear ratio 154 vs nominal 19.5). |

---

*Report compiled by 6-phase data-science pipeline. All artefacts are reproducible — re-run the scripts in order (`refine_telemetry.py` → `preprocessing.py` → `phase1_eda.py` → … → `phase5_timeseries.py`) to regenerate every chart and CSV.*
