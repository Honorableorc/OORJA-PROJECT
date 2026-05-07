# OORJA — Electric Kart Telemetry Analysis

End-to-end data-science pipeline for the **OORJA electric kart**, applied to 1 Hz onboard telemetry over 3 driving sessions (~64 minutes, 36.7 km, 3,671 samples).

The project takes raw sensor logs (speed, RPM, IMU, LoRa timestamps), refines them, synthesises a battery model from the  72 V / 85 Ah , engineers features, runs unsupervised + supervised + time-series ML, and produces a full report with concrete performance recommendations.

---

## Quickstart

```bash
# 1. Install Python 3.10+ dependencies
pip install pandas numpy scipy scikit-learn matplotlib seaborn statsmodels joblib torch

# 2. Run the pipeline in order
python refine_telemetry.py     # cleans data + applies battery model
python preprocessing.py         # engineers features, creates ML-ready CSVs
python phase1_eda.py            # exploratory data analysis
python phase2_performance.py    # KPI + performance audit
python phase3_unsupervised.py   # K-Means / DBSCAN / PCA / IsolationForest
python phase4_supervised.py     # 6 regression models x 4 targets
python phase5_timeseries.py     # ARIMA + LSTM forecasting
```

All charts and CSV outputs land in `analysis/<phase>/`.

---

## Pipeline overview

| Stage | Script | What it does |
|---|---|---|
| **Refinement** | [refine_telemetry.py](refine_telemetry.py) | Cleans `ovf` flags, hard physical bounds, rolling-median + MAD despiking, speed/RPM coherence; synthesises `Pack_Voltage_V` and `Pack_Current_A` per the XLEX datasheet |
| **Preprocessing** | [preprocessing.py](preprocessing.py) | Time handling → 15 engineered features → Butterworth low-pass on IMU → StandardScaler + one-hot encoding |
| **Phase 1 — EDA** | [phase1_eda.py](phase1_eda.py) | 6 charts: time series, histograms, scatters, correlation heatmap, box plots, drive-state share |
| **Phase 2 — Performance** | [phase2_performance.py](phase2_performance.py) | Per-session KPI table, 0–30 / 0–60 km/h launch times, voltage-sag IR fit, efficiency curve |
| **Phase 3 — Unsupervised ML** | [phase3_unsupervised.py](phase3_unsupervised.py) | K-Means (elbow + silhouette), DBSCAN (k-distance), PCA (2D + biplot), Isolation Forest |
| **Phase 4 — Supervised ML** | [phase4_supervised.py](phase4_supervised.py) | 6 models × 4 targets (current, voltage, power, efficiency); LinearRegression / Ridge / RandomForest / GradientBoosting / SVR / PyTorch MLP |
| **Phase 5 — Forecasting** | [phase5_timeseries.py](phase5_timeseries.py) | ARIMA(p,1,q) AIC search on SoC + PyTorch LSTM next-step voltage + range projection |

---

## Headline findings

| Finding | Number |
|---|---|
| Best energy efficiency | **79.5 Wh/km** (Session 3) |
| Top speed reached | **80 km/h** |
| Peak power draw | **7.76 kW** (108 A peak vs 150 A BMS limit — 30% headroom) |
| Drivetrain anomaly detected | **35 s of slip** (gear ratio 154 vs nominal 19.5) |
| Battery model validation | RandomForest fits V = OCV(SoC) − I·R with **R² = 0.9999** |
| Range remaining at end of S3 | **21.5 km / 24 min** at 79.4 Wh/km |

See [REPORT.md](REPORT.md) for the full analysis and [RECOMMENDATIONS.md](RECOMMENDATIONS.md) for the 18-action improvement plan.

---

## Data files

| File | Description |
|---|---|
| `oorja_telemetry.csv` | Raw sensor log (1 Hz × 3,672 rows × 8 channels) |
| `oorja_telemetry_refined.csv` | Cleaned + battery-modelled (10 cols) |
| `oorja_telemetry_processed.csv` | Engineered features (33 cols) |
| `oorja_telemetry_scaled.csv` | ML-ready scaled + one-hot (26 cols) |
| `oorja_telemetry_clustered.csv` | With cluster labels + PCA components (39 cols) |
| `preprocessing_scaler.joblib` | Fitted StandardScaler artefact for inference |

---

## Output artefacts

| Folder | Charts | CSVs |
|---|---|---|
| [analysis/phase1_eda/](analysis/phase1_eda/) | 6 | — |
| [analysis/phase2_performance/](analysis/phase2_performance/) | 5 | 3 |
| [analysis/phase3_unsupervised/](analysis/phase3_unsupervised/) | 8 | 2 |
| [analysis/phase4_supervised/](analysis/phase4_supervised/) | 16 | 5 |
| [analysis/phase5_timeseries/](analysis/phase5_timeseries/) | 5 | 2 |
| **Total** | **40 charts** | **12 tables** |

---

## Documents

- [REPORT.md](REPORT.md) — full analysis report with executive summary
- [RECOMMENDATIONS.md](RECOMMENDATIONS.md) — 18 prioritised actions with expected impact

---

## Notes

- **Battery datasheets** ( 72 V / 80 Ah and 85 Ah PDFs) are excluded from this repo for IP reasons. The synthesised battery model is fully documented inside [refine_telemetry.py](refine_telemetry.py).
- **Voltage and current** in the dataset are *modelled* from the datasheet, not measured. See [RECOMMENDATIONS.md § D1](RECOMMENDATIONS.md) — adding real V/I sensors is the highest-priority data-quality upgrade.
- **Data is at 1 Hz** so IMU vibration content is aliased; see [§ D2](RECOMMENDATIONS.md) for the recommended upgrade.

---

## License

Code is licensed under MIT (see [LICENSE](LICENSE)). Data is provided as-is for research/educational use.
