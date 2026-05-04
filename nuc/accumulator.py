"""
Online mean and variance accumulator using Welford's single-pass algorithm.

Reference: Welford, B.P. (1962). "Note on a method for calculating corrected
sums of squares and products." Technometrics 4(3):419-420.
"""

import numpy as np


class WelfordAccumulator:
    """
    Accumulate per-pixel mean and variance across frames without storing the stack.

    Memory cost is O(pixels) regardless of how many frames are added.  For a
    512×512 float64 array that is ~2 MB for mean and ~2 MB for M2 — independent
    of frame count.

    The variance estimate is Bessel-corrected (divided by n-1) and converges to
    the true per-pixel temporal variance as n → ∞.  Fixed-pattern structure
    (PRNU, DSNU) appears in the mean but not in the temporal variance, which is
    exactly the quantity needed for NUC uncertainty propagation.
    """

    def __init__(self) -> None:
        self._n: int = 0
        self._mean: np.ndarray | None = None
        self._M2: np.ndarray | None = None

    def update(self, frame: np.ndarray) -> None:
        """Ingest one frame.  Frame may be any numeric dtype; stored as float64."""
        x = frame.astype(np.float64)
        self._n += 1
        if self._mean is None:
            self._mean = x.copy()
            self._M2 = np.zeros_like(x)
        else:
            delta = x - self._mean
            self._mean += delta / self._n
            # Second delta uses the *updated* mean — this is the Welford correction
            # that keeps M2 numerically stable even for large n.
            self._M2 += delta * (x - self._mean)

    @property
    def n(self) -> int:
        """Number of frames accumulated."""
        return self._n

    @property
    def mean(self) -> np.ndarray:
        if self._mean is None:
            raise RuntimeError("No frames accumulated yet.")
        return self._mean

    @property
    def variance(self) -> np.ndarray:
        """Per-pixel sample variance (Bessel-corrected).  Requires n ≥ 2."""
        if self._n < 2:
            raise RuntimeError("Variance requires at least 2 frames.")
        return self._M2 / (self._n - 1)

    @property
    def std_error_of_mean(self) -> np.ndarray:
        """Per-pixel standard error of the mean: sqrt(variance / n)."""
        return np.sqrt(self.variance / self._n)
