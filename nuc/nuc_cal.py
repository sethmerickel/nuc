"""
Two-point NUC gain computation and uncertainty analysis.

The NUC gain for pixel (r, c) is:

    G(r, c) = 1 / (μ_H(r,c) − μ_C(r,c))

where μ_H and μ_C are the estimated mean responses to the hot and cold
blackbody sources.  Uncertainty in G is propagated from the uncertainty in
each mean estimate via first-order error propagation.
"""

from dataclasses import dataclass

import numpy as np

from .accumulator import WelfordAccumulator
from .config import SimConfig
from .detector import make_dsnu, make_prnu, simulate_frame
from .optics import cos4_map
from .physics import band_integrated_photon_radiance


@dataclass
class NUCResult:
    nuc_gain: np.ndarray              # G(r,c) = 1 / (μ_H − μ_C)  [1/ADU]
    nuc_uncertainty: np.ndarray       # σ_G(r,c) from error propagation  [1/ADU]
    relative_uncertainty: np.ndarray  # σ_G / |G|  [dimensionless]
    hot_acc: WelfordAccumulator
    cold_acc: WelfordAccumulator
    convergence_n: np.ndarray         # frame counts at each checkpoint
    convergence_rel_unc: np.ndarray   # array-mean relative uncertainty at each checkpoint


def compute_nuc(
    hot_acc: WelfordAccumulator,
    cold_acc: WelfordAccumulator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute NUC gain and uncertainty from two Welford accumulators.

    Error propagation:

        G = 1 / (μ_H − μ_C)

        ∂G/∂μ_H = −1 / (μ_H − μ_C)²
        ∂G/∂μ_C = +1 / (μ_H − μ_C)²

        σ²_G = (σ²_{μH} + σ²_{μC}) / (μ_H − μ_C)⁴

    where σ²_{μX} = σ²_X / N is the variance of the mean estimate for source X.

    Returns (nuc_gain, nuc_uncertainty, relative_uncertainty).
    Pixels where |μ_H − μ_C| < 0.5 ADU are set to NaN to avoid division noise.
    """
    diff = hot_acc.mean - cold_acc.mean
    safe_diff = np.where(np.abs(diff) >= 0.5, diff, np.nan)

    nuc_gain = 1.0 / safe_diff

    var_mean_hot = hot_acc.variance / hot_acc.n
    var_mean_cold = cold_acc.variance / cold_acc.n
    nuc_uncertainty = np.sqrt(var_mean_hot + var_mean_cold) / safe_diff**2

    relative_uncertainty = np.abs(nuc_uncertainty / nuc_gain)

    return nuc_gain, nuc_uncertainty, relative_uncertainty


def run_nuc_uncertainty(
    n_frames: int = 100,
    config: SimConfig | None = None,
    n_checkpoints: int = 60,
) -> NUCResult:
    """
    Simulate n_frames hot/cold frame pairs and compute NUC uncertainty.

    PRNU and DSNU are fixed for the sensor instance; only shot noise and read
    noise vary frame-to-frame.  Hot and cold frames alternate draws from the
    same RNG so the two accumulators are statistically independent.

    Convergence checkpoints are recorded at logarithmically spaced intervals
    so the σ_G ∝ 1/√N curve can be plotted.

    Parameters
    ----------
    n_frames : int
        Number of frames to simulate for each source.
    config : SimConfig | None
        Simulation configuration; uses defaults if None.
    n_checkpoints : int
        Number of points to record on the convergence curve.
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

    hot_acc = WelfordAccumulator()
    cold_acc = WelfordAccumulator()

    # Logarithmically spaced checkpoints; need n ≥ 2 for variance estimate
    checkpoint_set = set(
        np.unique(
            np.round(np.geomspace(2, n_frames, min(n_checkpoints, n_frames - 1))).astype(int)
        ).tolist()
    )

    convergence_n: list[int] = []
    convergence_rel_unc: list[float] = []

    print(f"Simulating {n_frames} frame pairs...")
    report_interval = max(1, n_frames // 10)

    for i in range(1, n_frames + 1):
        hot_acc.update(simulate_frame(hot_L, ri_map, prnu, dsnu, rng, config))
        cold_acc.update(simulate_frame(cold_L, ri_map, prnu, dsnu, rng, config))

        if i in checkpoint_set:
            _, _, rel_unc = compute_nuc(hot_acc, cold_acc)
            convergence_n.append(i)
            convergence_rel_unc.append(float(np.nanmean(rel_unc)))

        if i % report_interval == 0:
            print(f"  {i}/{n_frames} frames")

    nuc_gain, nuc_uncertainty, relative_uncertainty = compute_nuc(hot_acc, cold_acc)

    return NUCResult(
        nuc_gain=nuc_gain,
        nuc_uncertainty=nuc_uncertainty,
        relative_uncertainty=relative_uncertainty,
        hot_acc=hot_acc,
        cold_acc=cold_acc,
        convergence_n=np.array(convergence_n),
        convergence_rel_unc=np.array(convergence_rel_unc),
    )
