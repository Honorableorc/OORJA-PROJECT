# OORJA Electric Kart — Performance Improvement Plan

*Derived from the 6-phase data-science analysis (refinement → preprocessing → EDA → performance → unsupervised ML → supervised ML → time-series). Every recommendation here is grounded in a specific finding from the data.*

---

## How to read this document

Each recommendation has the same 4-block structure:

> **Why** — what the data showed (with numbers)
> **How** — concrete steps to implement
> **Expected impact** — best-case improvement
> **How to verify** — what metric to re-measure after the change

Priority badges:
- 🔴 **High** — large impact, low effort, do first
- 🟡 **Medium** — solid gain, needs planning
- 🟢 **Low** — nice-to-have, longer payoff

---

## Section A — Drivetrain & Mechanical

### A1. 🔴 Investigate the 35-second drivetrain slip event

**Why**
K-Means Cluster 3 contained 35 samples (1% of the run) where the **gear ratio jumped from the nominal 19.5 to 154** — meaning motor RPM was ~7× higher than the wheel speed would suggest. DBSCAN independently flagged the same 24-point dense cluster, and Isolation Forest found samples like *Speed = 10 km/h, RPM = 2,190* in the same window. Three separate algorithms converging on the same anomaly is strong evidence of a **physical event**.

**How**
1. Cross-reference the timestamps in [analysis/phase3_unsupervised/anomalies.csv](analysis/phase3_unsupervised/anomalies.csv) with the onboard video.
2. Inspect chain tension (slack < 10 mm), sprocket-tooth wear, and rear-axle bearings.
3. Test for wheel slip on a wet/loose patch — replicate the conditions.
4. If clutch-equipped, check clutch slipping under load.

**Expected impact**
Eliminating recurring drivetrain slip recovers **3–8% of energy delivered to the wheels**. On a 79.5 Wh/km baseline that's **2–6 Wh/km saved**.

**How to verify**
After the fix, re-run [phase3_unsupervised.py](phase3_unsupervised.py). Cluster 3 should disappear or shrink to <10 samples. Gear-ratio std (currently 4–6 across the run) should drop below 2.

---

### A2. 🟡 Reduce time spent in hard braking (Cluster 0)

**Why**
12.8% of run time = "hard decel" cluster (avg accel −0.72 m/s²). Without regen on this kart, **every braking second is energy thrown away**. The cluster also has elevated power (4.4 kW vs cruise 0.8 kW) which means the driver was still drawing current while braking — wasted energy.

**How**
1. Coach drivers to **lift early, brake late and hard** (shorter brake duration).
2. Adjust brake bias if it's too aggressive on rear (locks earlier → longer decel).
3. Identify track sections where braking dominates and consider line changes.

**Expected impact**
Reducing hard-decel share from 12.8% to 8% saves an estimated **2–3 Wh/km** at race pace.

**How to verify**
Re-run [phase2_performance.py](phase2_performance.py) — the Drive_State distribution should show ≤10% decel, and Wh/km should drop accordingly.

---

### A3. 🟢 Pack/connector health audit

**Why**
Effective pack IR fitted from voltage residuals matches the datasheet's ~20 mΩ — **no immediate concern**. But IR rises slowly with cycle age and connector oxidation, and that hurts peak power.

**How**
1. Once per month, re-fit the IR using [phase2_performance.py](phase2_performance.py)'s `voltage_sag()` function.
2. Reseat all bus-bars and Anderson SB175 connectors quarterly.
3. Track IR over time — flag if it climbs above 25 mΩ.

**Expected impact**
Holds peak current capability constant over the season. Without this, expect 5–10% peak-power loss per 100 cycles.

**How to verify**
Effective IR from the script stays in 18–22 mΩ band.

---

## Section B — Driving Style & Coaching

### B1. 🔴 Replicate Session 3 style as the reference run

**Why**
Session 3 hit the **same top speed (80 km/h)** as Session 2 but achieved **79.5 Wh/km vs 85.0 Wh/km — 6.5% better efficiency**. Same kart, same battery, different result → driver behaviour or warmed components.

**How**
1. Pull the Session 3 telemetry slice from [oorja_telemetry_clustered.csv](oorja_telemetry_clustered.csv) (`Session_ID == 3`).
2. Plot throttle/brake patterns and identify the "smooth" sequences.
3. Use S3 as the reference for driver-coaching sessions.
4. Have other drivers replicate the S3 throttle modulation in practice runs.

**Expected impact**
**+6–7% range** (e.g., from 25 km to 26.6 km on the same charge) with **zero hardware change**.

**How to verify**
On the next run, re-run Phase 2 — Wh/km should land at ≤80 across all timed sessions.

---

### B2. 🟡 Find and hold the efficiency sweet-spot speed

**Why**
Phase 4d's GradientBoosting model showed efficiency is **strongly non-linear in speed** (linear models scored R² = 0.09; trees scored R² = 0.76). This means there's a specific speed band that minimises Wh/km. Look at [04_efficiency_curve.png](analysis/phase2_performance/04_efficiency_curve.png) for your kart's curve.

**How**
1. From the efficiency-by-speed CSV, identify the band with the lowest Wh/km (typically 35–55 km/h for karts; **read your specific number from `efficiency_by_speed.csv`**).
2. In long cruise sections, coach drivers to hold that speed precisely (±3 km/h).
3. Avoid both very-low-speed cruising and very-high-speed cruising during efficiency-critical stints.

**Expected impact**
On flat sections, **3–5% Wh/km improvement** if currently cruising too far above or below the sweet spot.

**How to verify**
Speed histogram in Phase 1 should show a tighter peak at the sweet-spot speed.

---

### B3. 🟡 Smoother launches

**Why**
Average launch times: 0→30 km/h **3.3 s**, 0→60 km/h **6.5 s** (across 49 detected launches). Hard launches at peak current cause the biggest IR sag → more energy lost as heat than torque applied. Modulated launches sustain power longer.

**How**
1. From [launches.csv](analysis/phase2_performance/launches.csv), find the launches with the **best 0–30** time **and** the lowest peak current — those are the efficient ones.
2. Coach drivers to **ramp throttle in 0.5 s** rather than slamming it.
3. Have the ESC pre-set a soft current ramp on green-light release.

**Expected impact**
**0.2–0.5 s faster** consistent launch times, plus **~1% Wh/km savings** on race-start energy budget.

**How to verify**
Re-run [phase2_performance.py](phase2_performance.py); average 0–60 should approach 6.0 s without losing peak speed.

---

## Section C — Battery & Power Management

### C1. 🟡 Allow brief 130–140 A bursts (currently capped at 108 A peak)

**Why**
Peak current draw across the entire run was **108 A**, well below the **150 A continuous** BMS limit and **300 A 10-second peak**. There's ~30% headroom you're leaving on the table.

**How**
1. Verify the controller's current-limit setting — ESC may be clipping below 150 A.
2. Allow a "boost" mode at **130 A for 5 s** on overtakes/launches.
3. Keep the **150 A continuous** rule and **300 A peak (≤10 s)** rule from the BMS datasheet.

**Expected impact**
**+15–20% peak power** for short bursts → faster overtakes, faster launches.

**How to verify**
After tuning, peak in Phase 2 KPI table should rise to 130–145 A. Voltage min should stay ≥70 V (cell stress acceptable).

---

### C2. 🟡 Race in the NMC plateau (30–70% SoC)

**Why**
Phase 4b confirmed the V = OCV(SoC) − I·R model with R² = 0.9999. The OCV curve has a **flat plateau between 30% and 70% SoC** at ~3.6–3.85 V/cell (72–77 V pack). Outside this band:
- Above 70%: voltage is high but the OCV curve is steep — quick power drop after each percentage burned.
- Below 30%: voltage sags hard under load and BMS is closer to cut-off.

**How**
1. **Charge to 80%** before each race start (not 100%) — protects cell life and stays out of the steep top region.
2. Plan pit stops so the pack never drops below 30%.
3. For long endurance, target the **40–60% SoC band** during the busiest race phases.

**Expected impact**
**Consistent power delivery** across the stint (no late-stint fade). Cell SoH preserved for longer over the season.

**How to verify**
Re-run [phase5_timeseries.py](phase5_timeseries.py); SoC trajectory should never exit 30–80% during a race.

---

### C3. 🟢 Add regenerative braking if mechanically feasible

**Why**
12.8% of run time was hard decel. With **zero** energy recovery, every braking second is pure waste.

**How**
1. Verify motor/ESC supports regen — many BLDC motors do, but it requires a regen-capable controller.
2. If yes: enable mild regen (5–10% of brake torque) for cruise-style decel.
3. Keep mechanical brakes for hard stops (regen can't compete with brake force at low speeds).

**Expected impact**
Even **10% regen recovery** during decel = ~1.2% of total energy reclaimed = **~1 Wh/km saved**. Modest but additive.

**How to verify**
Pack_Current_A should go negative during cruise-decel events. Energy_Wh_cum should plateau (not rise) during regen events.

---

## Section D — Instrumentation Upgrades

### D1. 🔴 Install real voltage and current sensors (biggest data-quality win)

**Why**
**This is the single most important upgrade.** Currently, [oorja_telemetry_refined.csv](oorja_telemetry_refined.csv)'s `Pack_Voltage_V` and `Pack_Current_A` columns are **synthesized** from the XLEX datasheet, not measured. Every electrical insight in this report is consistent with the datasheet — but it can't tell you anything the datasheet doesn't already say. Real sensors unlock:
- **State-of-Health (SoH)** tracking — IR rises with age
- **Cell imbalance** detection — early warning for failing cells
- **Real efficiency measurement** — actual Wh/km, not modelled
- **Temperature-derating** detection during long runs

**How**
1. **Voltage**: a simple voltage divider + ADC channel on the BMS bus, sampled at 10 Hz.
2. **Current**: hall-effect sensor (e.g., Allegro ACS770, ±200 A range) on the main pack cable. 10 Hz sampling.
3. Update the data-pipeline scripts to **read these columns directly** instead of synthesising them — remove the modelling stage from [refine_telemetry.py](refine_telemetry.py).

**Expected impact**
Real measurements turn this analysis from a *consistency check* into a *diagnostic tool*. Every recommendation here gets stronger, especially the battery-health ones.

**How to verify**
The R² of the voltage prediction (Phase 4b) should drop slightly (real noise > model noise) — this is **expected and good**.

---

### D2. 🟡 Increase IMU sampling rate to 10–50 Hz

**Why**
At 1 Hz logging, Nyquist is **0.5 Hz** — anything above that is aliased into the recorded signal. Real chassis dynamics (cornering, bumps, tyre slip) happen in the 1–20 Hz range, so the current data **misses most of the useful vibration content**.

**How**
1. Configure the MPU to log at 50 Hz minimum.
2. Down-sample to 1 Hz only at storage time using a proper anti-alias filter (the Butterworth low-pass we already use).
3. Optionally save a 10 Hz channel for chassis-dynamics analysis.

**Expected impact**
Vibration analysis becomes meaningful. Tyre-slip detection, cornering-load estimation, and bumpy-section identification all become possible.

**How to verify**
Phase 1 IMU plots should show distinct spikes at corners and bumps, not smooth blobs.

---

### D3. 🟡 Add temperature sensors

**Why**
Thermal derating is the **next bottleneck** for any electric vehicle. The pack has a 60°C absolute cell limit and 50°C continuous surface limit (datasheet section 6). Without thermistors we can't:
- Detect derating events (controller pulls back current to protect motor/inverter)
- Validate cooling adequacy at sustained 5+ kW draws
- Plan stint length thermally

**How**
1. **Pack** — 2× thermistors on cell-block surfaces (datalogged at 1 Hz).
2. **Motor** — 1× thermistor on stator winding.
3. **Inverter** — 1× thermistor on heatsink/MOSFET.
4. **Ambient** — 1× near the data logger.

**Expected impact**
Identify thermal limits before they cost a race. Often the vehicle has more performance than the cooling can sustain — the data tells you when to back off.

**How to verify**
Phase 2 KPI table grows by 4 columns. Look for power-vs-temperature scatter showing roll-off above some threshold.

---

### D4. 🟢 Deploy the LSTM model on-vehicle for live anomaly alerts

**Why**
Phase 5b's LSTM achieves **MAE 0.072 V** on next-step voltage prediction (R² = 0.989). That's the noise floor — real anomalies (loose connector, cell imbalance, sudden current draw) will produce residuals 10–100× larger.

**How**
1. Quantise + export the trained PyTorch LSTM to ONNX or TorchScript.
2. Run it on a Raspberry Pi / Jetson Nano / ESP32 (depending on weight constraints).
3. At each second: predict next voltage; if `|actual − predicted| > 0.5 V`, raise an alert.
4. Connect alert to dashboard LED or telemetry beep.

**Expected impact**
**Sub-second anomaly detection.** Catches connector arcing, cell drop-off, or thermal derate before the driver feels it.

**How to verify**
Inject a known fault (e.g. brief disconnect) and confirm the alert fires.

---

## Section E — Race Strategy

### E1. 🔴 Live range projector on the dashboard

**Why**
Phase 5c gave **21.5 km / 24 min remaining at 79.4 Wh/km** at the end of Session 3. The driver currently has no way to know this in real time.

**How**
1. Compute live `Wh/km` over a rolling 60-second window from real V × I sensors (after D1).
2. Compute `min_to_empty` = `(Energy_Remaining_Wh) / (Wh_per_minute_now)`.
3. Display on dashboard alongside SoC%.
4. Optional: colour-code (green > 10 min, yellow 3–10, red < 3).

**Expected impact**
Lap-strategy decisions (push vs save) become **quantitative** instead of intuition-based.

**How to verify**
After deploying, post-race review — `min_left` at chequered flag should be ≤2 min for an efficient race.

---

### E2. 🟡 Pit-stop optimisation

**Why**
Voltage sag worsens dramatically below 30% SoC (NMC plateau ends). Power consistency drops, cell stress rises.

**How**
1. Plan the longest stint to end at **30–35% SoC** (not 0%).
2. Use the projected SoC trajectory from Phase 5a to pre-compute pit windows.
3. If charging time is fixed (parc fermé style), trade lap-pace for SoC margin in the last few laps.

**Expected impact**
**Consistent lap times** end-of-stint instead of fading. Cell life ↑.

**How to verify**
Lap-time plot vs SoC; the second-half degradation should flatten.

---

### E3. 🟢 Reduce idle electronics drain

**Why**
**25.6% of recorded time** = 938 seconds = 15.6 minutes were idle. The kart still draws ~1 A at idle (modelled here, but real value will be similar). Over 15 min, that's ~0.25 Ah = **~18 Wh wasted**.

**How**
1. Wire a **kill switch** for non-essential auxiliaries (data logger, dashboard) when the kart is parked >2 min.
2. Or: configure the BMS for sleep mode after 5 min idle.

**Expected impact**
**0.3–0.5% extra range** — small but free.

**How to verify**
Idle-time current draw drops below 0.2 A.

---

## Section F — Long-term & Strategic

### F1. 🟢 Aero / tyre warm-up study

**Why**
The 6.5% S2 → S3 efficiency improvement could be **driver style** OR **tyre warmup** (cold tyres have higher rolling resistance) OR both. We can't separate them from this data alone.

**How**
1. Run a controlled experiment: same driver, same line, same SoC band, but vary tyre warmup time.
2. Add a tyre-temp sensor (D3 above also covers this if extended).
3. If tyre-warmup confirmed → design a pre-race warm-lap protocol or hot-lap-only strategy.

**Expected impact**
If tyre-warmup is the cause, **2–3% efficiency** can be locked in for every run.

**How to verify**
Re-run efficiency analysis with tyre-temp as a feature in Phase 4d's GradientBoosting. Feature importance will tell you the answer.

---

### F2. 🟡 Build a digital twin

**Why**
We already have an electrical model that fits with R² = 0.9999. Combine it with a motor model + drag/rolling-resistance model, and you have a **digital twin** that can simulate "what-if" scenarios before track time.

**How**
1. Take the validated `V = f(I, SoC)` model from Phase 4b.
2. Add motor torque-RPM curve and inverter efficiency map (from manufacturer data).
3. Add a chassis model: F = m·a, plus drag (½ ρ Cd A v²) and rolling (μ m g).
4. Run laps in simulation. Compare predicted vs measured.

**Expected impact**
**Test changes (gearing, tyre, weight) in simulation first** — saves track days, lets you screen improvements at zero cost.

**How to verify**
Simulated lap-time should match real lap-time within ±2%.

---

### F3. 🟢 Aim for a "more cruise, less idle" run profile

**Why**
Current run profile: cruise 28% / idle 26% / accel 24% / decel 22%. That's a **lot of idle and decel** — both are non-productive.

**How**
A theoretical race target:
- **Cruise 50%** — main efficiency band
- **Accel 20%** — overtakes and exits
- **Decel 20%** — necessary
- **Idle 10%** — pit stops + grid hold

This requires:
- Better track-flow planning (fewer stops)
- Coaching for less-aggressive accel/decel cycles
- Race strategy that minimises caution-period dwelling

**Expected impact**
A **10% Wh/km improvement** is realistic over a season as the team optimises toward this target.

**How to verify**
Re-run Phase 1 EDA — drive-state pie chart should look like the target above.

---

## Quick-win Punch List

If you can only do 5 things in the next 2 weeks, do these:

| Order | Action | Section | Effort | Impact |
|---|---|---|---|---|
| 1 | Inspect drivetrain for slip event | A1 | 1 hour | 3–8% Wh/km |
| 2 | Coach drivers toward S3 style | B1 | 2 sessions | 6–7% range |
| 3 | Tune controller to 130 A peak | C1 | 30 min | +15–20% peak power |
| 4 | Order V/I sensors + install | D1 | 1 day | Unlocks all future analysis |
| 5 | Add live range projector (after D1) | E1 | 2 days | Better strategy decisions |

**Combined potential**: **~10–15% range improvement** AND **+15% peak performance** within one race weekend.

---

## Validation Loop

After each change, re-run the analysis pipeline to confirm the improvement showed up in the data:

```bash
python refine_telemetry.py     # cleans new data, applies battery model
python preprocessing.py         # engineers features
python phase1_eda.py            # quick visual sanity check
python phase2_performance.py    # KPI table — is Wh/km lower?
python phase3_unsupervised.py   # did the slip cluster shrink?
python phase4_supervised.py     # do feature importances change?
python phase5_timeseries.py     # range projector working?
```

Save each run's `analysis/` folder with a timestamp so you can compare runs over the season.

---

## Estimated Combined Impact (best case)

| Category | Recovery | Notes |
|---|---|---|
| Drivetrain (A1) | +3–8% Wh/km | One-time mechanical fix |
| Driving style (B1, B2, B3) | +6–10% Wh/km | Coaching-led, sustained |
| Battery management (C1, C2) | +5–10% peak power | Controller config |
| Regen (C3) | +1–2% Wh/km | If feasible |
| Aero/tyre (F1) | +2–3% Wh/km | Confirmed via experiment |
| **Total realistic range gain** | **~15–20%** | over baseline 79.5 Wh/km |
| **Total peak-power gain** | **+15–20%** | independent of range gain |

**On a 24-minute / 21.5 km baseline stint**, that translates to roughly:
- **+4–5 km extra range** per stint
- **OR**  +3 minutes of competitive racing time per stint
- **OR**  +2 kW peak power available for overtakes

---

*Every recommendation is traceable to a specific finding in [REPORT.md](REPORT.md) — refer to that document for the underlying data and methodology.*
