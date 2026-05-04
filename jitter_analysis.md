# Integration-Time Jitter: Impact on NUC Calibration Uncertainty

## Background

During characterization of the MCT/HgCdTe SWIR detector array, an anomalous
behavior was identified: the detector's integration time shifts by a small
amount on a periodic cadence, alternating every two frames (AABB pattern —
two frames at nominal duration, two frames at a slightly longer duration,
repeating). At the default operating point, this shift corresponds to
approximately **36 ADU** of additional signal on the hot calibration source
per affected frame.

This analysis quantifies the impact of this anomaly on two-point NUC
calibration uncertainty.

---

## Simulation Setup

| Parameter | Value |
|---|---|
| Detector | 512 × 512 MCT/HgCdTe, 20 µm pitch |
| Band | SWIR 1.0–2.5 µm |
| Optics | 20 mm aperture, 40 mm EFL (f/2) |
| Integration time (nominal) | 10 ms |
| Read noise | 30 e⁻ RMS |
| Dark current | 100 e⁻/s |
| PRNU σ | 1% |
| DSNU σ | 10% |
| Hot source | 300 K blackbody |
| Cold source | 140 K blackbody |
| Frames per source | 100 |
| Jitter fraction | 5% of nominal t_int |

The hot source produces a mean signal of approximately **716 ADU** above the
500 ADU digital pedestal. The cold source at 140 K produces negligible photon
flux in SWIR and sits at the noise floor. The 5% jitter on a 10 ms integration
time shifts affected frames by **~36 ADU** on the hot channel and is
imperceptible on the cold channel.

---

## Results

### Signal statistics

| | Baseline | With Jitter |
|---|---|---|
| Mean Δμ (hot − cold) | 716.0 ADU | 733.9 ADU |
| Mean NUC gain G | 1.3971 × 10⁻³ ADU⁻¹ | 1.3630 × 10⁻³ ADU⁻¹ |

The jittered mean is slightly elevated because frames with longer integration
time accumulate more signal. With an equal split between nominal and jittered
frames, the measured mean converges to the signal at `t_int × (1 + fraction/2)`
rather than `t_int`, introducing a **systematic bias of ~2.5%** in the estimated
NUC gain.

### Calibration uncertainty

| Metric | Baseline | With Jitter | Change |
|---|---|---|---|
| Mean relative uncertainty σ_G/\|G\| | 0.157% | 0.290% | +0.133 pp |
| Minimum (center pixels) | 0.112% | 0.232% | +0.120 pp |
| Maximum (corner pixels) | 0.205% | 0.353% | +0.148 pp |
| Degradation | — | — | **85% worse** |

---

## Root Cause

The Welford running-mean algorithm accumulates per-pixel temporal variance
across the frame stack. In the baseline case this variance reflects only
shot noise and read noise:

```
σ²_pixel ≈ μ_total + σ²_read
```

When integration-time jitter is present, the per-frame signal alternates
between two levels separated by `Δsignal = μ_hot × jitter_fraction ≈ 36 ADU`.
This periodic deterministic variation is indistinguishable from noise to the
accumulator and inflates the estimated variance by an additional term:

```
σ²_jitter ≈ (Δsignal / 2)² = (716 × 0.05 / 2)² ≈ 321 e²
```

This extra variance propagates directly into σ_G through the error propagation
formula `σ_G = sqrt(σ²_μH + σ²_μC) / (μ_H − μ_C)²`, increasing the NUC
uncertainty independent of how many frames are collected.

---

## Frame Count Implications

Because the jitter-induced variance is systematic rather than reducible by
averaging (the per-frame signal really does alternate), the calibration
uncertainty tracks `1/√N` but from a higher floor. To achieve the same
uncertainty as the clean baseline at 100 frames, a jittered acquisition
requires approximately **341 frames** — a **3.4× increase** in acquisition time.

| Target uncertainty | Baseline frames needed | Jittered frames needed |
|---|---|---|
| 0.157% (baseline @ N=100) | 100 | ~341 |
| 0.100% | ~246 | ~840 |
| 0.050% | ~983 | ~3,360 |

---

## Conclusions

1. **The jitter degrades NUC calibration uncertainty by ~85%** at the tested
   operating point (100 frames, 5% fractional jitter).

2. **A systematic gain bias of ~2.5% is introduced** because the running mean
   converges to a weighted average of the two integration-time states rather
   than the true nominal value.

3. **The effect is not mitigated by collecting more frames.** The variance floor
   set by the jitter amplitude is fixed; additional frames only reduce the
   random (shot + read) component. The jitter-induced uncertainty dominates
   once enough frames have been collected to suppress shot and read noise below
   the jitter level.

4. **Corrective options** (in order of preference):
   - Identify and fix the root cause of the integration-time instability in
     the detector clock or ROIC control logic.
   - If the jitter period is known and stable, correct for it in preprocessing
     by sorting frames into their two integration-time classes and computing
     separate means before combining.
   - Accept the degraded uncertainty and collect ~3.4× more calibration frames
     to compensate.
