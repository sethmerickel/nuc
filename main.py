import matplotlib
matplotlib.use("Agg")  # headless-safe; remove or override for interactive use
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import replace

from nuc.config import SimConfig
from nuc.detector import _gain_e_per_adu
from nuc import simulate
from nuc import nuc_cal

# ── Simulation parameters ────────────────────────────────────────────────────

N_FRAMES = 100

# Integration-time jitter: fractional change in t_int for jittered frames.
# 0.05 → ~5% change, ~36 ADU shift on the hot source at default settings.
# Set to 0.0 to disable entirely.
JITTER_FRACTION = 0.05


# ── Helpers ──────────────────────────────────────────────────────────────────

def print_cal_summary(config: SimConfig, frames: simulate.CalibrationFrames) -> None:
    det = config.detector
    opt = config.optics
    max_adu = (1 << det.bit_depth) - 1
    gain = _gain_e_per_adu(det)
    ri_corner = frames.ri_map[0, 0]

    print("=== Calibration Frames ===")
    print(f"  Band              : {config.band.lambda_min_um:.1f}–{config.band.lambda_max_um:.1f} µm (SWIR)")
    print(f"  Array             : {det.n_rows}×{det.n_cols} pixels, {det.pixel_pitch_um:.0f} µm pitch")
    print(f"  Optics            : {opt.aperture_diameter_mm:.0f} mm aperture, "
          f"{opt.focal_length_mm:.0f} mm EFL  "
          f"(f/{opt.focal_length_mm / opt.aperture_diameter_mm:.1f}), τ={opt.optical_transmission:.2f}")
    print(f"  Integration time  : {config.integration_time_s * 1e3:.1f} ms")
    print(f"  ADC               : {det.bit_depth}-bit, {max_adu} max ADU, "
          f"offset={det.digital_offset_adu} ADU, gain={gain:.2f} e⁻/ADU")
    print(f"  Read noise        : {det.read_noise_electrons:.0f} e⁻ RMS   "
          f"Dark: {det.dark_current_electrons_per_s:.0f} e⁻/s")
    print(f"  PRNU σ={det.prnu_sigma*100:.1f}%   DSNU σ={det.dsnu_sigma*100:.1f}%")
    print()
    print(f"  Hot  BB ({config.scene.hot_temperature_K:.0f} K)  "
          f"radiance: {frames.hot_photon_radiance:.3e} ph/s/m²/sr  "
          f"→ mean {np.mean(frames.hot):.0f} ADU")
    print(f"  Cold BB ({config.scene.cold_temperature_K:.0f} K)  "
          f"radiance: {frames.cold_photon_radiance:.3e} ph/s/m²/sr  "
          f"→ mean {np.mean(frames.cold):.0f} ADU")
    print(f"  RI rolloff        : {(1 - ri_corner) * 100:.1f}% at corner")


def print_nuc_summary(label: str, result: nuc_cal.NUCResult) -> None:
    rel_pct = result.relative_uncertainty * 100
    diff_mean = np.nanmean(result.hot_acc.mean - result.cold_acc.mean)
    print(f"\n--- NUC Uncertainty  [{label}] ---")
    print(f"  Frames / source   : {result.hot_acc.n}")
    print(f"  Mean Δμ (H−C)     : {diff_mean:.1f} ADU")
    print(f"  Mean G            : {np.nanmean(result.nuc_gain):.4e} ADU⁻¹")
    print(f"  Relative unc.     : mean={np.nanmean(rel_pct):.3f}%  "
          f"min={np.nanmin(rel_pct):.3f}%  max={np.nanmax(rel_pct):.3f}%")


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_calibration_frames(config: SimConfig, frames: simulate.CalibrationFrames) -> None:
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(
        f"Calibration Frames — SWIR {config.band.lambda_min_um:.1f}–"
        f"{config.band.lambda_max_um:.1f} µm  |  "
        f"f/{config.optics.focal_length_mm / config.optics.aperture_diameter_mm:.1f}  |  "
        f"{config.detector.bit_depth}-bit  |  "
        f"{config.detector.n_rows}×{config.detector.n_cols}",
        fontsize=12,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    vmin, vmax = int(frames.cold.min()), int(frames.hot.max())

    for ax, img, title in [
        (fig.add_subplot(gs[0, 0]), frames.hot,
         f"Hot BB  ({config.scene.hot_temperature_K:.0f} K)"),
        (fig.add_subplot(gs[0, 1]), frames.cold,
         f"Cold BB  ({config.scene.cold_temperature_K:.0f} K)"),
    ]:
        im = ax.imshow(img, cmap="inferno", origin="upper", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Column [px]")
        ax.set_ylabel("Row [px]")
        plt.colorbar(im, ax=ax, label="ADU")

    ax_ri = fig.add_subplot(gs[0, 2])
    im = ax_ri.imshow(frames.ri_map, cmap="viridis", origin="upper",
                      vmin=frames.ri_map.min(), vmax=1.0)
    ax_ri.set_title("Relative Illumination  cos⁴(θ)")
    ax_ri.set_xlabel("Column [px]")
    ax_ri.set_ylabel("Row [px]")
    plt.colorbar(im, ax=ax_ri, label="Normalized")

    for ax, img, title, label in [
        (fig.add_subplot(gs[1, 0]), frames.prnu,
         "PRNU  (QE non-uniformity)", "QE scale factor"),
        (fig.add_subplot(gs[1, 1]), frames.dsnu,
         "DSNU  (dark current non-uniformity)", "Dark scale factor"),
    ]:
        im = ax.imshow(img, cmap="RdBu_r", origin="upper")
        ax.set_title(title)
        ax.set_xlabel("Column [px]")
        ax.set_ylabel("Row [px]")
        plt.colorbar(im, ax=ax, label=label)

    ax_prof = fig.add_subplot(gs[1, 2])
    mid = config.detector.n_rows // 2
    cols = np.arange(config.detector.n_cols)
    ax_prof.plot(cols, frames.hot[mid, :], linewidth=0.8,
                 label=f"Hot  ({config.scene.hot_temperature_K:.0f} K)", alpha=0.85)
    ax_prof.plot(cols, frames.cold[mid, :], linewidth=0.8,
                 label=f"Cold ({config.scene.cold_temperature_K:.0f} K)", alpha=0.85)
    ri_row = frames.ri_map[mid, :]
    ax_prof.plot(cols, frames.hot[mid, mid] * ri_row, "k--",
                 linewidth=1.0, alpha=0.5, label="Ideal RI")
    ax_prof.plot(cols, frames.cold[mid, mid] * ri_row, "k--", linewidth=1.0, alpha=0.5)
    ax_prof.set_xlabel("Column [px]")
    ax_prof.set_ylabel("ADU")
    ax_prof.set_title("Center Row Profile")
    ax_prof.legend(fontsize=8)
    ax_prof.grid(True, alpha=0.3)

    out = "calibration_simulation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {out}")
    plt.close(fig)


def plot_nuc_comparison(
    config: SimConfig,
    baseline: nuc_cal.NUCResult,
    jittered: nuc_cal.NUCResult,
    jitter_fraction: float,
) -> None:
    """2×3 figure comparing NUC uncertainty with and without integration-time jitter."""

    rel_base = baseline.relative_uncertainty * 100
    rel_jit = jittered.relative_uncertainty * 100

    # Shared color scale across both relative-uncertainty maps
    vmax_rel = float(np.nanpercentile(np.concatenate([rel_base.ravel(), rel_jit.ravel()]), 99))

    fig = plt.figure(figsize=(16, 10))
    delta_adu = jitter_fraction * np.nanmean(
        jittered.hot_acc.mean - jittered.cold_acc.mean
    )
    fig.suptitle(
        f"NUC Uncertainty — Jitter Impact  |  {baseline.hot_acc.n} frames/source  |  "
        f"jitter fraction = {jitter_fraction:.3f}  (≈{delta_adu:.0f} ADU on hot signal)",
        fontsize=12,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.40)

    # ── Relative uncertainty maps ────────────────────────────────────────────
    for col, rel, label in [
        (0, rel_base, "Baseline  (no jitter)"),
        (1, rel_jit, f"Jitter  (fraction = {jitter_fraction:.3f})"),
    ]:
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(rel, cmap="magma", origin="upper", vmin=0, vmax=vmax_rel)
        ax.set_title(f"Relative Uncertainty  (%)\n{label}")
        ax.set_xlabel("Column [px]")
        ax.set_ylabel("Row [px]")
        plt.colorbar(im, ax=ax, label="%")

    # ── Convergence comparison ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    for result, label, color in [
        (baseline, "Baseline", "steelblue"),
        (jittered, f"Jitter {jitter_fraction:.3f}", "tomato"),
    ]:
        n_arr = result.convergence_n
        unc_arr = result.convergence_rel_unc * 100
        ax.loglog(n_arr, unc_arr, "o-", color=color, markersize=3, label=label)

    # 1/√N reference anchored to the baseline final point
    n_ref = baseline.convergence_n[-1]
    unc_ref = baseline.convergence_rel_unc[-1] * 100
    n_theory = np.geomspace(baseline.convergence_n[0], n_ref, 200)
    ax.loglog(n_theory, unc_ref * np.sqrt(n_ref / n_theory),
              "--", color="gray", linewidth=1, label="1/√N ref.")

    ax.set_xlabel("Frames per source  N")
    ax.set_ylabel("Mean relative uncertainty (%)")
    ax.set_title("Convergence Comparison")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    # ── NUC gain maps (show bias introduced by jitter) ───────────────────────
    for col, result, label in [
        (0, baseline, "NUC Gain  (baseline)"),
        (1, jittered, "NUC Gain  (jittered)"),
    ]:
        ax = fig.add_subplot(gs[1, col])
        im = ax.imshow(result.nuc_gain, cmap="viridis", origin="upper")
        ax.set_title(label)
        ax.set_xlabel("Column [px]")
        ax.set_ylabel("Row [px]")
        plt.colorbar(im, ax=ax, label="ADU⁻¹")

    # ── Histogram overlay ────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    for rel, label, color in [
        (rel_base, "Baseline", "steelblue"),
        (rel_jit, f"Jitter {jitter_fraction:.3f}", "tomato"),
    ]:
        flat = rel[np.isfinite(rel)].ravel()
        ax.hist(flat, bins=80, color=color, alpha=0.55, edgecolor="none",
                label=f"{label}  μ={np.nanmean(rel):.3f}%")
        ax.axvline(float(np.nanmean(rel)), color=color, linestyle="--", linewidth=1.2)

    ax.set_xlabel("Relative uncertainty  σ_G / |G|  (%)")
    ax.set_ylabel("Pixel count")
    ax.set_title("Pixel Distribution of Relative Uncertainty")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    out = "nuc_uncertainty.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {out}")
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    config = SimConfig()

    # ── Single-frame calibration preview ─────────────────────────────────────
    print("Simulating calibration frames...")
    frames = simulate.run(config)
    print_cal_summary(config, frames)
    plot_calibration_frames(config, frames)

    # ── NUC uncertainty: baseline (no jitter) ────────────────────────────────
    config_baseline = replace(config, integration_time_jitter_fraction=0.0)
    print("\n[Baseline — no jitter]")
    result_baseline = nuc_cal.run_nuc_uncertainty(n_frames=N_FRAMES, config=config_baseline)
    print_nuc_summary("baseline", result_baseline)

    # ── NUC uncertainty: with jitter ─────────────────────────────────────────
    config_jitter = replace(config, integration_time_jitter_fraction=JITTER_FRACTION)
    print(f"\n[Jitter — fraction={JITTER_FRACTION:.3f}]")
    result_jitter = nuc_cal.run_nuc_uncertainty(n_frames=N_FRAMES, config=config_jitter)
    print_nuc_summary(f"jitter {JITTER_FRACTION:.3f}", result_jitter)

    # ── Impact summary ────────────────────────────────────────────────────────
    mean_base = np.nanmean(result_baseline.relative_uncertainty) * 100
    mean_jit = np.nanmean(result_jitter.relative_uncertainty) * 100
    print(f"\n=== Jitter Impact ===")
    print(f"  Baseline mean rel. unc.  : {mean_base:.3f}%")
    print(f"  Jittered mean rel. unc.  : {mean_jit:.3f}%")
    print(f"  Increase                 : {mean_jit - mean_base:.3f} pp  "
          f"({(mean_jit / mean_base - 1) * 100:.1f}% worse)")
    equiv_n = int(np.round(N_FRAMES * (mean_jit / mean_base) ** 2))
    print(f"  Frames needed (jittered) to match baseline unc.: ~{equiv_n}")

    # ── Figures ───────────────────────────────────────────────────────────────
    plot_nuc_comparison(config, result_baseline, result_jitter, JITTER_FRACTION)


if __name__ == "__main__":
    main()
