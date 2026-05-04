"""Optical model: relative illumination for a rotationally symmetric imaging system."""

import numpy as np

from .config import DetectorConfig, OpticsConfig


def cos4_map(det: DetectorConfig, opt: OpticsConfig) -> np.ndarray:
    """
    Relative illumination map using the cos⁴(θ) model, normalized to 1 at center.

    For an extended Lambertian source imaged through a circular aperture the
    irradiance at a focal-plane position falls as cos⁴(θ) where θ is the
    chief-ray angle.  The four factors are: projected aperture area (cos θ),
    pixel foreshortening (cos θ), inverse-square path length (cos² θ).

    Returns an (n_rows × n_cols) float64 array in [0, 1].
    """
    f_m = opt.focal_length_mm * 1e-3
    pitch_m = det.pixel_pitch_um * 1e-6

    cx = (det.n_cols - 1) / 2.0
    cy = (det.n_rows - 1) / 2.0

    col_idx, row_idx = np.meshgrid(
        np.arange(det.n_cols, dtype=np.float64),
        np.arange(det.n_rows, dtype=np.float64),
    )
    r_m = np.hypot(col_idx - cx, row_idx - cy) * pitch_m
    theta = np.arctan(r_m / f_m)
    return np.cos(theta) ** 4
