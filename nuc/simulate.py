"""Top-level orchestration for the NUC calibration simulation."""

from dataclasses import dataclass

import numpy as np

from .config import SimConfig
from .detector import make_dsnu, make_prnu, simulate_frame
from .optics import cos4_map
from .physics import band_integrated_photon_radiance


@dataclass
class CalibrationFrames:
    hot: np.ndarray               # [ADU] image of hot blackbody
    cold: np.ndarray              # [ADU] image of cold blackbody
    ri_map: np.ndarray            # relative illumination map [0, 1]
    prnu: np.ndarray              # pixel response non-uniformity map
    dsnu: np.ndarray              # dark signal non-uniformity map
    hot_photon_radiance: float    # in-band radiance of hot source [photons/s/m²/sr]
    cold_photon_radiance: float   # in-band radiance of cold source [photons/s/m²/sr]


def run(config: SimConfig | None = None) -> CalibrationFrames:
    """
    Simulate a two-point radiometric calibration acquisition.

    The same PRNU and DSNU maps are used for both frames, as they represent
    fixed sensor characteristics that persist across an image sequence.
    """
    if config is None:
        config = SimConfig()

    rng = np.random.default_rng(config.random_seed)

    lambda_min_m = config.band.lambda_min_um * 1e-6
    lambda_max_m = config.band.lambda_max_um * 1e-6

    hot_L = band_integrated_photon_radiance(
        lambda_min_m, lambda_max_m,
        config.scene.hot_temperature_K,
        config.band.n_quadrature_points,
    )
    cold_L = band_integrated_photon_radiance(
        lambda_min_m, lambda_max_m,
        config.scene.cold_temperature_K,
        config.band.n_quadrature_points,
    )

    ri_map = cos4_map(config.detector, config.optics)
    prnu = make_prnu(rng, config)
    dsnu = make_dsnu(rng, config)

    hot_image = simulate_frame(hot_L, ri_map, prnu, dsnu, rng, config)
    cold_image = simulate_frame(cold_L, ri_map, prnu, dsnu, rng, config)

    return CalibrationFrames(
        hot=hot_image,
        cold=cold_image,
        ri_map=ri_map,
        prnu=prnu,
        dsnu=dsnu,
        hot_photon_radiance=hot_L,
        cold_photon_radiance=cold_L,
    )
