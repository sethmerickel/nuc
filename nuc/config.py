from dataclasses import dataclass, field


@dataclass
class BandConfig:
    lambda_min_um: float = 1.0    # SWIR lower cutoff [µm]
    lambda_max_um: float = 2.5    # SWIR upper cutoff [µm]
    n_quadrature_points: int = 1000


@dataclass
class OpticsConfig:
    aperture_diameter_mm: float = 20.0   # entrance pupil diameter
    focal_length_mm: float = 40.0        # effective focal length (f/2)
    optical_transmission: float = 0.85   # in-band throughput [0, 1]


@dataclass
class DetectorConfig:
    n_rows: int = 512
    n_cols: int = 512
    pixel_pitch_um: float = 20.0          # center-to-center pixel spacing
    bit_depth: int = 14                   # ADC resolution
    full_well_electrons: float = 120_000.0
    quantum_efficiency: float = 0.80      # mean QE over band
    read_noise_electrons: float = 30.0    # per-read Gaussian noise [e⁻ RMS]
    dark_current_electrons_per_s: float = 100.0   # mean dark rate at operating temp
    digital_offset_adu: int = 500         # bias pedestal; maps 0 e⁻ → offset ADU
    prnu_sigma: float = 0.01              # pixel-to-pixel QE spread [1σ fraction]
    dsnu_sigma: float = 0.10             # pixel-to-pixel dark current spread [1σ fraction]


@dataclass
class SceneConfig:
    hot_temperature_K: float = 300.0
    cold_temperature_K: float = 140.0


@dataclass
class SimConfig:
    band: BandConfig = field(default_factory=BandConfig)
    optics: OpticsConfig = field(default_factory=OpticsConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    integration_time_s: float = 0.010
    random_seed: int | None = 42
    # Integration-time jitter anomaly.  When non-zero the detector alternates
    # between t_int and t_int*(1+fraction) in an AABB pattern (2 nominal frames,
    # 2 jittered frames, repeating).  Set to 0.0 to disable entirely.
    integration_time_jitter_fraction: float = 0.0
