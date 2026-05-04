import matplotlib
matplotlib.use("Agg")  # headless-safe; remove or override for interactive use
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from nuc.config import SimConfig
from nuc.detector import _gain_e_per_adu
from nuc import simulate
from nuc import nuc_cal

N_FRAMES = 100   # frame pairs per source for NUC uncertainty analysis


def print_cal_summary(config: SimConfig, frames: simulate.CalibrationFrames) -> None:
    det = config.detector
    opt = config.optics
    max_adu = (1 << det.bit_depth) - 1
    gain = _gain_e_per_adu(det)
    f_number = opt.focal_length_mm / opt.aperture_diameter_mm
    ri_corner = frames.ri_map[0, 0]

    print("=== NUC Calibration Simulation ===")
    print(f"  Band              : {config.band.lambda_min_um:.1f}–{config.band.lambda_max_um:.1f} µm (SWIR)")
    print(f"  Array             : {det.n_rows}×{det.n_cols} pixels, {det.pixel_pitch_um:.0f} µm pitch")
    print(f"  Optics            : {opt.aperture_diameter_mm:.0f} mm aperture, "
          f"{opt.focal_length_mm:.0f} mm EFL  (f/{f_number:.1f}), τ={opt.optical_transmission:.2f}")
    print(f"  Integration time  : {config.integration_time_s * 1e3:.1f} ms")
    print(f"  ADC               : {det.bit_depth}-bit, {max_adu} max ADU, "
          f"offset={det.digital_offset_adu} ADU, gain={gain:.2f} e⁻/ADU")
    print(f"  Full well         : {det.full_well_electrons:,.0f} e⁻")
    print(f"  Read noise        : {det.read_noise_electrons:.0f} e⁻ RMS")
    print(f"  Dark current      : {det.dark_current_electrons_per_s:.0f} e⁻/s mean")
    print(f"  PRNU σ            : {det.prnu_sigma * 100:.1f}%")
    print(f"  DSNU σ            : {det.dsnu_sigma * 100:.1f}%")
    print()
    print(f"  Hot  BB ({config.scene.hot_temperature_K:.0f} K)  "
          f"photon radiance : {frames.hot_photon_radiance:.4e} ph/s/m²/sr")
    print(f"  Cold BB ({config.scene.cold_temperature_K:.0f} K)  "
          f"photon radiance : {frames.cold_photon_radiance:.4e} ph/s/m²/sr")
    print(f"  Radiance ratio (hot/cold) : {frames.hot_photon_radiance / frames.cold_photon_radiance:.1f}×")
    print()
    hot_pct = (np.mean(frames.hot) - det.digital_offset_adu) / (max_adu - det.digital_offset_adu) * 100
    cold_pct = (np.mean(frames.cold) - det.digital_offset_adu) / (max_adu - det.digital_offset_adu) * 100
    print(f"  Hot  image  : mean={np.mean(frames.hot):.0f}  std={np.std(frames.hot):.1f}  ADU  ({hot_pct:.1f}% full scale)")
    print(f"  Cold image  : mean={np.mean(frames.cold):.0f}  std={np.std(frames.cold):.1f}  ADU  ({cold_pct:.1f}% full scale)")
    print(f"  RI rolloff  : corner/center = {ri_corner:.4f}  ({(1 - ri_corner) * 100:.1f}% drop at corner)")


def print_nuc_summary(result: nuc_cal.NUCResult) -> None:
    rel_pct = result.relative_uncertainty * 100
    print("\n=== NUC Uncertainty Analysis ===")
    print(f"  Frames per source : {result.hot_acc.n}")
    print(f"  Mean  Δμ (hot−cold)  : {np.nanmean(result.hot_acc.mean - result.cold_acc.mean):.1f} ADU")
    print(f"  Mean  G              : {np.nanmean(result.nuc_gain):.4e} ADU⁻¹")
    print(f"  Mean  σ_G            : {np.nanmean(result.nuc_uncertainty):.4e} ADU⁻¹")
    print(f"  Relative uncertainty : mean={np.nanmean(rel_pct):.3f}%  "
          f"min={np.nanmin(rel_pct):.3f}%  max={np.nanmax(rel_pct):.3f}%")
    print(f"  Corner/center ratio  : {np.nanmean(rel_pct[0:5, 0:5]):.3f}% / "
          f"{np.nanmean(rel_pct[252:260, 252:260]):.3f}%")


def plot_calibration_frames(config: SimConfig, frames: simulate.CalibrationFrames) -> None:
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(
        f"NUC Calibration Simulation — SWIR {config.band.lambda_min_um:.1f}–{config.band.lambda_max_um:.1f} µm  |  "
        f"f/{config.optics.focal_length_mm / config.optics.aperture_diameter_mm:.1f}  |  "
        f"{config.detector.bit_depth}-bit  |  {config.detector.n_rows}×{config.detector.n_cols}",
        fontsize=12,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    vmin = int(frames.cold.min())
    vmax = int(frames.hot.max())

    ax_hot = fig.add_subplot(gs[0, 0])
    im = ax_hot.imshow(frames.hot, cmap="inferno", origin="upper", vmin=vmin, vmax=vmax)
    ax_hot.set_title(f"Hot BB  ({config.scene.hot_temperature_K:.0f} K)")
    ax_hot.set_xlabel("Column [px]")
    ax_hot.set_ylabel("Row [px]")
    plt.colorbar(im, ax=ax_hot, label="ADU")

    ax_cold = fig.add_subplot(gs[0, 1])
    im = ax_cold.imshow(frames.cold, cmap="inferno", origin="upper", vmin=vmin, vmax=vmax)
    ax_cold.set_title(f"Cold BB  ({config.scene.cold_temperature_K:.0f} K)")
    ax_cold.set_xlabel("Column [px]")
    ax_cold.set_ylabel("Row [px]")
    plt.colorbar(im, ax=ax_cold, label="ADU")

    ax_ri = fig.add_subplot(gs[0, 2])
    im = ax_ri.imshow(frames.ri_map, cmap="viridis", origin="upper",
                      vmin=frames.ri_map.min(), vmax=1.0)
    ax_ri.set_title("Relative Illumination  cos⁴(θ)")
    ax_ri.set_xlabel("Column [px]")
    ax_ri.set_ylabel("Row [px]")
    plt.colorbar(im, ax=ax_ri, label="Normalized")

    ax_prnu = fig.add_subplot(gs[1, 0])
    im = ax_prnu.imshow(frames.prnu, cmap="RdBu_r", origin="upper")
    ax_prnu.set_title("PRNU  (QE non-uniformity)")
    ax_prnu.set_xlabel("Column [px]")
    ax_prnu.set_ylabel("Row [px]")
    plt.colorbar(im, ax=ax_prnu, label="QE scale factor")

    ax_dsnu = fig.add_subplot(gs[1, 1])
    im = ax_dsnu.imshow(frames.dsnu, cmap="RdBu_r", origin="upper")
    ax_dsnu.set_title("DSNU  (dark current non-uniformity)")
    ax_dsnu.set_xlabel("Column [px]")
    ax_dsnu.set_ylabel("Row [px]")
    plt.colorbar(im, ax=ax_dsnu, label="Dark scale factor")

    ax_prof = fig.add_subplot(gs[1, 2])
    mid_row = config.detector.n_rows // 2
    cols = np.arange(config.detector.n_cols)
    ax_prof.plot(cols, frames.hot[mid_row, :],
                 label=f"Hot  ({config.scene.hot_temperature_K:.0f} K)", alpha=0.85, linewidth=0.8)
    ax_prof.plot(cols, frames.cold[mid_row, :],
                 label=f"Cold ({config.scene.cold_temperature_K:.0f} K)", alpha=0.85, linewidth=0.8)
    ri_row = frames.ri_map[mid_row, :]
    hot_center = float(frames.hot[mid_row, config.detector.n_cols // 2])
    cold_center = float(frames.cold[mid_row, config.detector.n_cols // 2])
    ax_prof.plot(cols, hot_center * ri_row, "k--", linewidth=1.0, alpha=0.5, label="Ideal RI (no noise)")
    ax_prof.plot(cols, cold_center * ri_row, "k--", linewidth=1.0, alpha=0.5)
    ax_prof.set_xlabel("Column [px]")
    ax_prof.set_ylabel("ADU")
    ax_prof.set_title("Center Row Profile")
    ax_prof.legend(fontsize=8)
    ax_prof.grid(True, alpha=0.3)

    out = "calibration_simulation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {out}")
    plt.close(fig)


def plot_nuc_uncertainty(config: SimConfig, result: nuc_cal.NUCResult) -> None:
    rel_pct = result.relative_uncertainty * 100
    mid_row = config.detector.n_rows // 2
    cols = np.arange(config.detector.n_cols)

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(
        f"NUC Uncertainty — {result.hot_acc.n} frames/source  |  "
        f"SWIR {config.band.lambda_min_um:.1f}–{config.band.lambda_max_um:.1f} µm  |  "
        f"f/{config.optics.focal_length_mm / config.optics.aperture_diameter_mm:.1f}",
        fontsize=12,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── NUC gain map ────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(result.nuc_gain, cmap="viridis", origin="upper")
    ax.set_title("NUC Gain  G = 1/(μ_H − μ_C)")
    ax.set_xlabel("Column [px]")
    ax.set_ylabel("Row [px]")
    plt.colorbar(im, ax=ax, label="ADU⁻¹")

    # ── Absolute uncertainty map ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(result.nuc_uncertainty, cmap="plasma", origin="upper")
    ax.set_title("NUC Gain Uncertainty  σ_G")
    ax.set_xlabel("Column [px]")
    ax.set_ylabel("Row [px]")
    plt.colorbar(im, ax=ax, label="ADU⁻¹")

    # ── Relative uncertainty map ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(rel_pct, cmap="magma", origin="upper")
    ax.set_title("Relative Uncertainty  σ_G / |G|  (%)")
    ax.set_xlabel("Column [px]")
    ax.set_ylabel("Row [px]")
    plt.colorbar(im, ax=ax, label="%")

    # ── Convergence curve ────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    n_arr = result.convergence_n
    unc_arr = result.convergence_rel_unc * 100

    ax.loglog(n_arr, unc_arr, "o-", markersize=3, label="Simulated")

    # Theoretical 1/√N curve anchored to the final measured point
    n_theory = np.geomspace(n_arr[0], n_arr[-1], 200)
    unc_theory = unc_arr[-1] * np.sqrt(n_arr[-1] / n_theory)
    ax.loglog(n_theory, unc_theory, "--", color="gray", label="1/√N")

    ax.set_xlabel("Frames per source  N")
    ax.set_ylabel("Mean relative uncertainty (%)")
    ax.set_title("Convergence of σ_G with Frame Count")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    # ── Histogram of relative uncertainty ───────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    flat = rel_pct[np.isfinite(rel_pct)].ravel()
    ax.hist(flat, bins=80, color="steelblue", edgecolor="none", alpha=0.85)
    ax.axvline(float(np.nanmean(rel_pct)), color="k", linestyle="--",
               linewidth=1.2, label=f"Mean = {np.nanmean(rel_pct):.3f}%")
    ax.set_xlabel("Relative uncertainty  σ_G / |G|  (%)")
    ax.set_ylabel("Pixel count")
    ax.set_title("Pixel Distribution of Relative Uncertainty")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Center-row profile of relative uncertainty ───────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(cols, rel_pct[mid_row, :], linewidth=0.9, label="Simulated")

    # Overlay the mean-difference profile to show why uncertainty rises at edges
    diff_row = result.hot_acc.mean[mid_row, :] - result.cold_acc.mean[mid_row, :]
    diff_norm = diff_row / float(np.nanmax(diff_row))
    ax2 = ax.twinx()
    ax2.plot(cols, diff_norm, color="orange", linewidth=0.9, alpha=0.7, label="Δμ (norm.)")
    ax2.set_ylabel("Δμ / max(Δμ)  [normalized]", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")

    ax.set_xlabel("Column [px]")
    ax.set_ylabel("Relative uncertainty (%)")
    ax.set_title("Center Row: Uncertainty vs. Signal Difference")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax.grid(True, alpha=0.3)

    out = "nuc_uncertainty.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {out}")
    plt.close(fig)


def main() -> None:
    config = SimConfig()

    # ── Single-frame calibration images ─────────────────────────────────────
    print("Simulating calibration frames...")
    frames = simulate.run(config)
    print_cal_summary(config, frames)
    plot_calibration_frames(config, frames)

    # ── NUC uncertainty analysis ─────────────────────────────────────────────
    result = nuc_cal.run_nuc_uncertainty(n_frames=N_FRAMES, config=config)
    print_nuc_summary(result)
    plot_nuc_uncertainty(config, result)


if __name__ == "__main__":
    main()
