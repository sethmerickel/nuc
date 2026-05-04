import matplotlib
matplotlib.use("Agg")  # headless-safe; remove or override for interactive use
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from nuc.config import SimConfig
from nuc.detector import _gain_e_per_adu
from nuc import simulate


def print_summary(config: SimConfig, frames: simulate.CalibrationFrames) -> None:
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


def main() -> None:
    config = SimConfig()

    print("Running simulation...")
    frames = simulate.run(config)
    print_summary(config, frames)

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(
        f"NUC Calibration Simulation — SWIR {config.band.lambda_min_um:.1f}–{config.band.lambda_max_um:.1f} µm  |  "
        f"f/{config.optics.focal_length_mm / config.optics.aperture_diameter_mm:.1f}  |  "
        f"{config.detector.bit_depth}-bit  |  {config.detector.n_rows}×{config.detector.n_cols}",
        fontsize=12,
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Row 0: calibration images + RI map ──────────────────────────────────

    ax_hot = fig.add_subplot(gs[0, 0])
    vmin = int(frames.cold.min())
    vmax = int(frames.hot.max())
    im_hot = ax_hot.imshow(frames.hot, cmap="inferno", origin="upper", vmin=vmin, vmax=vmax)
    ax_hot.set_title(f"Hot BB  ({config.scene.hot_temperature_K:.0f} K)")
    ax_hot.set_xlabel("Column [px]")
    ax_hot.set_ylabel("Row [px]")
    plt.colorbar(im_hot, ax=ax_hot, label="ADU")

    ax_cold = fig.add_subplot(gs[0, 1])
    im_cold = ax_cold.imshow(frames.cold, cmap="inferno", origin="upper", vmin=vmin, vmax=vmax)
    ax_cold.set_title(f"Cold BB  ({config.scene.cold_temperature_K:.0f} K)")
    ax_cold.set_xlabel("Column [px]")
    ax_cold.set_ylabel("Row [px]")
    plt.colorbar(im_cold, ax=ax_cold, label="ADU")

    ax_ri = fig.add_subplot(gs[0, 2])
    im_ri = ax_ri.imshow(frames.ri_map, cmap="viridis", origin="upper",
                         vmin=frames.ri_map.min(), vmax=1.0)
    ax_ri.set_title("Relative Illumination  cos⁴(θ)")
    ax_ri.set_xlabel("Column [px]")
    ax_ri.set_ylabel("Row [px]")
    plt.colorbar(im_ri, ax=ax_ri, label="Normalized")

    # ── Row 1: FPN maps + center-row profile ────────────────────────────────

    ax_prnu = fig.add_subplot(gs[1, 0])
    im_prnu = ax_prnu.imshow(frames.prnu, cmap="RdBu_r", origin="upper")
    ax_prnu.set_title("PRNU  (QE non-uniformity)")
    ax_prnu.set_xlabel("Column [px]")
    ax_prnu.set_ylabel("Row [px]")
    plt.colorbar(im_prnu, ax=ax_prnu, label="QE scale factor")

    ax_dsnu = fig.add_subplot(gs[1, 1])
    im_dsnu = ax_dsnu.imshow(frames.dsnu, cmap="RdBu_r", origin="upper")
    ax_dsnu.set_title("DSNU  (dark current non-uniformity)")
    ax_dsnu.set_xlabel("Column [px]")
    ax_dsnu.set_ylabel("Row [px]")
    plt.colorbar(im_dsnu, ax=ax_dsnu, label="Dark scale factor")

    ax_prof = fig.add_subplot(gs[1, 2])
    mid_row = config.detector.n_rows // 2
    cols = np.arange(config.detector.n_cols)
    ax_prof.plot(cols, frames.hot[mid_row, :],
                 label=f"Hot  ({config.scene.hot_temperature_K:.0f} K)", alpha=0.85, linewidth=0.8)
    ax_prof.plot(cols, frames.cold[mid_row, :],
                 label=f"Cold ({config.scene.cold_temperature_K:.0f} K)", alpha=0.85, linewidth=0.8)

    # Overlay noiseless RI curve scaled to the center pixel of each image
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

    out_path = "calibration_simulation.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {out_path}")
    # plt.show()  # uncomment when running with an interactive backend


if __name__ == "__main__":
    main()
