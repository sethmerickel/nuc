"""Planck blackbody radiation formulas in photon-count units."""

import numpy as np

# CODATA 2018 exact values
H = 6.62607015e-34    # Planck constant [J·s]
C = 2.99792458e8      # Speed of light in vacuum [m/s]
K_B = 1.380649e-23    # Boltzmann constant [J/K]


def spectral_photon_radiance(wavelengths_m: np.ndarray, temp_K: float) -> np.ndarray:
    """
    Spectral photon radiance of an ideal blackbody [photons/s/m²/sr/m].

    Uses expm1 for numerical stability at small x.  Returns 0 where the
    exponential argument exceeds 500 (physically negligible contribution).
    """
    x = (H * C) / (wavelengths_m * K_B * temp_K)
    prefactor = 2.0 * C / wavelengths_m**4
    result = np.where(x < 500.0, prefactor / np.expm1(np.minimum(x, 500.0)), 0.0)
    return result


def band_integrated_photon_radiance(
    lambda_min_m: float,
    lambda_max_m: float,
    temp_K: float,
    n_points: int = 1000,
) -> float:
    """
    In-band photon radiance integrated over [lambda_min_m, lambda_max_m] [photons/s/m²/sr].

    Uses trapezoidal quadrature on a uniformly spaced wavelength grid.
    Increase n_points for higher accuracy at the cost of compute time.
    """
    wavelengths = np.linspace(lambda_min_m, lambda_max_m, n_points)
    L = spectral_photon_radiance(wavelengths, temp_K)
    return float(np.trapezoid(L, wavelengths))
