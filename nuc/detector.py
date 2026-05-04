"""
MCT/HgCdTe detector noise model and signal chain.

Signal chain (in order):
  scene photon radiance
  → aperture solid angle + optical transmission
  → pixel area × integration time
  → quantum efficiency (modulated by PRNU)
  → relative illumination rolloff
  ─────────────────────────────────────
  = mean signal electrons  [Poisson]
  + mean dark electrons (modulated by DSNU)  [Poisson]
  → joint Poisson shot noise sample         (charge in well: clip to [0, FW])
  + Gaussian read noise                     (added in ROIC amplifier; can go negative)
  → ADC: electrons → ADU (gain + digital offset)
  → round + clamp to [0, max_ADU]
"""

import numpy as np

from .config import DetectorConfig, SimConfig


def _gain_e_per_adu(det: DetectorConfig) -> float:
    """
    Electrons per ADU such that full-well maps to max ADU and 0 e⁻ maps to
    digital_offset_adu.  The offset reserves headroom below the signal for
    negative-going read-noise excursions without bottom-clipping.
    """
    max_adu = (1 << det.bit_depth) - 1
    usable_range = max_adu - det.digital_offset_adu
    return det.full_well_electrons / usable_range


def make_prnu(rng: np.random.Generator, config: SimConfig) -> np.ndarray:
    """
    Fixed pixel response non-uniformity: multiplicative map of relative QE.

    Drawn once per simulated sensor from N(1, prnu_sigma²), representing
    pixel-to-pixel variation in cut-off wavelength and fill factor.
    """
    shape = (config.detector.n_rows, config.detector.n_cols)
    return rng.normal(1.0, config.detector.prnu_sigma, shape).clip(0.0, None)


def make_dsnu(rng: np.random.Generator, config: SimConfig) -> np.ndarray:
    """
    Fixed dark signal non-uniformity: multiplicative scale on dark current.

    Drawn once per simulated sensor from N(1, dsnu_sigma²), representing
    pixel-to-pixel variation in bulk trap density and surface leakage.
    """
    shape = (config.detector.n_rows, config.detector.n_cols)
    return rng.normal(1.0, config.detector.dsnu_sigma, shape).clip(0.0, None)


def simulate_frame(
    scene_photon_radiance: float,
    ri_map: np.ndarray,
    prnu: np.ndarray,
    dsnu: np.ndarray,
    rng: np.random.Generator,
    config: SimConfig,
    t_int_s: float | None = None,
) -> np.ndarray:
    """
    Simulate one detector frame and return a uint16 array of digital counts.

    Parameters
    ----------
    scene_photon_radiance : float
        In-band photon radiance of the extended source [photons/s/m²/sr].
    ri_map : ndarray
        Relative illumination map [0, 1], shape (n_rows, n_cols).
    prnu : ndarray
        Pixel response non-uniformity map, shape (n_rows, n_cols).
    dsnu : ndarray
        Dark signal non-uniformity map, shape (n_rows, n_cols).
    rng : Generator
        NumPy random generator (caller owns seeding).
    config : SimConfig
        Full simulation configuration.
    t_int_s : float | None
        Integration time override [s].  Uses config.integration_time_s when None.
        Pass an effective value from the jitter model to simulate anomalous frames.
    """
    det = config.detector
    opt = config.optics

    f_m = opt.focal_length_mm * 1e-3
    D_m = opt.aperture_diameter_mm * 1e-3
    pitch_m = det.pixel_pitch_um * 1e-6

    # Solid angle subtended by the entrance pupil at an on-axis detector pixel [sr].
    # Paraxial approximation: Ω = π (D/2)² / f²
    omega_sr = np.pi * (D_m / 2.0) ** 2 / f_m**2

    t_int = t_int_s if t_int_s is not None else config.integration_time_s

    # Mean signal electrons: radiance × étendue × QE × relative illumination
    signal_e: np.ndarray = (
        scene_photon_radiance  # photons/s/m²/sr
        * omega_sr             # sr  (aperture solid angle)
        * opt.optical_transmission
        * pitch_m**2           # m²  (pixel area)
        * t_int                # s
        * det.quantum_efficiency
        * prnu
        * ri_map
    )

    # Mean dark electrons per pixel over integration period
    dark_e: np.ndarray = (
        det.dark_current_electrons_per_s
        * t_int
        * dsnu
    )

    # Shot noise: Poisson sample of combined signal + dark electrons.
    # Signal and dark share the same Poisson draw because both represent
    # randomly arriving carriers in the same integration well.
    mean_total_e = np.maximum(signal_e + dark_e, 0.0)
    charge_e = rng.poisson(mean_total_e).astype(np.float64)

    # Charge saturates at full well; lower bound is 0 (no negative charge).
    charge_e = np.clip(charge_e, 0.0, det.full_well_electrons)

    # Read noise is added in the ROIC source-follower / amplifier chain, after
    # charge collection.  It is symmetric and can push the readout below the
    # pedestal — the digital offset is sized to catch this without bottom-clipping.
    readout_e = charge_e + rng.normal(0.0, det.read_noise_electrons, charge_e.shape)

    # ADC conversion: electrons → digital counts, then clamp to bit depth.
    gain = _gain_e_per_adu(det)
    adu = readout_e / gain + det.digital_offset_adu

    max_adu = (1 << det.bit_depth) - 1
    return np.clip(np.round(adu), 0, max_adu).astype(np.uint16)
