# NUC Calibration Simulation

Simulation of a two-point radiometric calibration acquisition for a cooled MCT/HgCdTe SWIR detector array. The simulator generates synthetic flat-field images of a hot and a cold extended blackbody source, including a physics-based noise model and optical relative illumination rolloff.

## Quick start

```bash
uv run main.py
```

Output is written to `calibration_simulation.png`.

## Configuration

All parameters are defined in `nuc/config.py` as nested dataclasses under `SimConfig`. Nothing is hardcoded in the computation modules.

| Config class | Key parameters |
|---|---|
| `BandConfig` | `lambda_min_um`, `lambda_max_um`, `n_quadrature_points` |
| `OpticsConfig` | `aperture_diameter_mm`, `focal_length_mm`, `optical_transmission` |
| `DetectorConfig` | `n_rows`, `n_cols`, `pixel_pitch_um`, `bit_depth`, `full_well_electrons`, `quantum_efficiency`, `read_noise_electrons`, `dark_current_electrons_per_s`, `digital_offset_adu`, `prnu_sigma`, `dsnu_sigma` |
| `SceneConfig` | `hot_temperature_K`, `cold_temperature_K` |
| `SimConfig` | `integration_time_s`, `random_seed` |

---

## Mathematical description

### 1. Blackbody spectral photon radiance

The scene is an ideal (emissivity = 1) extended Lambertian blackbody. Its
spectral photon radiance — photons per second per unit source area per steradian
per unit wavelength — is given by the Planck function in photon-count form:

$$N(\lambda, T) = \frac{2c}{\lambda^4} \cdot \frac{1}{e^{hc / \lambda k_B T} - 1} \quad \left[\frac{\text{ph}}{\text{s} \cdot \text{m}^2 \cdot \text{sr} \cdot \text{m}}\right]$$

where $c$ is the speed of light, $h$ is Planck's constant, $k_B$ is the
Boltzmann constant, $\lambda$ is wavelength, and $T$ is the source temperature.

> **Implementation note (`nuc/physics.py`):** The argument $x = hc / \lambda k_B T$
> can exceed floating-point range at short wavelengths or low temperatures. The
> code uses `numpy.expm1(x)` (equivalent to $e^x - 1$, numerically stable for
> small $x$) and clamps to zero for $x > 500$, where the contribution is
> physically negligible.

### 2. In-band photon radiance

The in-band photon radiance is the integral of $N(\lambda, T)$ over the
detector's spectral response band $[\lambda_{\min}, \lambda_{\max}]$:

$$L(T) = \int_{\lambda_{\min}}^{\lambda_{\max}} N(\lambda, T) \, d\lambda \quad \left[\frac{\text{ph}}{\text{s} \cdot \text{m}^2 \cdot \text{sr}}\right]$$

The integral is evaluated numerically with the trapezoidal rule on a uniform
wavelength grid of `n_quadrature_points` points.

### 3. Aperture solid angle

The entrance pupil subtends a solid angle $\Omega$ at an on-axis detector pixel
in the paraxial approximation:

$$\Omega = \frac{\pi (D/2)^2}{f^2} \quad [\text{sr}]$$

where $D$ is the aperture diameter and $f$ is the effective focal length.

#### Paraxial approximation

The formula above assumes all field angles are small enough that
$\sin\theta \approx \tan\theta \approx \theta$. The exact solid angle
subtended by a circular aperture is $\Omega = \pi \sin^2\alpha$ where
$\alpha = \arctan(D/2f)$ is the half-angle. The paraxial form replaces
$\sin\alpha$ with $\tan\alpha = D/2f$, which is exact only at normal
incidence. At f/2 (the default), $\alpha \approx 14°$ and the error between
$\sin\alpha$ and $\tan\alpha$ is about 3% — well within the other
uncertainties in the model. At f/1 or faster the approximation would begin
to matter and the exact form should be used instead.

The same approximation implicitly treats the lens-to-pixel distance as $f$
for all pixels regardless of field angle. The true distance to an off-axis
pixel is $f / \cos\theta$. Carrying that correction through consistently is
exactly what produces the cos⁴ rolloff described in the next section — so
the two approximations are interrelated, and the overall model remains
self-consistent.

### 4. Relative illumination — the cos⁴(θ) law

For an extended Lambertian source imaged by a rotationally symmetric lens, the
irradiance at a focal-plane pixel located at field angle $\theta$ from the
optical axis falls as $\cos^4 \theta$. The four factors contributing to this
rolloff are:

| Factor | Exponent | Physical origin |
|---|---|---|
| Projected aperture area | $\cos \theta$ | The aperture foreshortens when viewed off-axis |
| Pixel foreshortening | $\cos \theta$ | The pixel's effective collecting area decreases |
| Inverse-square path length | $\cos^2 \theta$ | The oblique ray path from source to pixel is longer by $1/\cos\theta$ |

The field angle for pixel $(r, c)$ relative to the array center $(r_0, c_0)$ is:

$$\theta_{r,c} = \arctan\!\left(\frac{\sqrt{(c - c_0)^2 + (r - r_0)^2} \cdot p}{f}\right)$$

where $p$ is the pixel pitch. The relative illumination map is:

$$\text{RI}_{r,c} = \cos^4(\theta_{r,c})$$

normalized to 1.0 at the array center. At the default settings (f/2, 20 µm
pitch, 512×512) the corner-to-center rolloff is approximately **6%**.

### 5. Mean signal electrons

The mean number of photoelectrons collected in pixel $(r, c)$ over integration
time $t_\text{int}$ is:

$$\mu_\text{sig}(r,c) = L(T) \cdot \Omega \cdot \tau \cdot p^2 \cdot t_\text{int} \cdot \eta \cdot \text{PRNU}_{r,c} \cdot \text{RI}_{r,c}$$

where:

| Symbol | Quantity |
|---|---|
| $L(T)$ | In-band photon radiance [ph/s/m²/sr] |
| $\Omega$ | Aperture solid angle [sr] |
| $\tau$ | Optical transmission [0, 1] |
| $p^2$ | Pixel area [m²] |
| $t_\text{int}$ | Integration time [s] |
| $\eta$ | Mean quantum efficiency [e⁻/photon] |
| $\text{PRNU}_{r,c}$ | Pixel response non-uniformity scale factor |
| $\text{RI}_{r,c}$ | Relative illumination (cos⁴ rolloff) |

### 6. Fixed pattern noise maps

Fixed pattern noise (FPN) is a spatially correlated, temporally stable offset
that arises from pixel-to-pixel variation in detector properties. It is the
primary target of the NUC calibration process.

**Pixel Response Non-Uniformity (PRNU)** is a multiplicative variation in
effective quantum efficiency across the array, caused by differences in
absorber thickness, cut-off wavelength, and fill factor:

$$\text{PRNU}_{r,c} \sim \mathcal{N}(1,\, \sigma_\text{PRNU}^2), \quad \text{PRNU}_{r,c} \geq 0$$

**Dark Signal Non-Uniformity (DSNU)** is a multiplicative variation in dark
current across the array, caused by differences in bulk trap density and
surface leakage current at the HgCdTe/CdZnTe interface:

$$\text{DSNU}_{r,c} \sim \mathcal{N}(1,\, \sigma_\text{DSNU}^2), \quad \text{DSNU}_{r,c} \geq 0$$

Both maps are drawn once per simulated sensor instance and held fixed across all
frames, matching the physical reality that these are static device properties.

### 7. Mean dark electrons

The mean number of dark electrons per pixel over the integration period is:

$$\mu_\text{dark}(r,c) = I_d \cdot t_\text{int} \cdot \text{DSNU}_{r,c}$$

where $I_d$ is the mean dark current rate [e⁻/s]. For cooled MCT/HgCdTe
detectors, $I_d$ is set by the Auger-1 generation-recombination mechanism and
is a strong function of detector operating temperature and cut-off wavelength.

### 8. Shot noise

Photon arrival and dark-carrier generation are independent Poisson processes.
Because both populate the same integration well, they share a single Poisson
draw with combined mean:

$$\mu_\text{total}(r,c) = \mu_\text{sig}(r,c) + \mu_\text{dark}(r,c)$$

$$S_\text{shot}(r,c) \sim \text{Poisson}\!\left(\mu_\text{total}(r,c)\right)$$

The shot noise standard deviation is $\sqrt{\mu_\text{total}}$, so the
signal-to-noise ratio for a shot-noise-limited pixel is
$\text{SNR} = \mu_\text{sig} / \sqrt{\mu_\text{total}}$.

### 9. Read noise

Read noise arises from the ROIC input transistor (typically a source-follower
or a direct-injection stage for MCT) and subsequent analog chain. It is
modeled as zero-mean Gaussian, independent per pixel and per frame:

$$S_\text{read}(r,c) \sim \mathcal{N}(0,\, \sigma_\text{read}^2)$$

The total electron count before the ADC is:

$$S_\text{total}(r,c) = \text{clip}\!\left(S_\text{shot}(r,c) + S_\text{read}(r,c),\; 0,\; N_\text{FW}\right)$$

where $N_\text{FW}$ is the full-well capacity. Clipping to zero prevents
unphysical negative charges; clipping to $N_\text{FW}$ represents charge
overflow (blooming).

### 10. Analog-to-digital conversion

The charge-to-voltage gain and ADC are modeled as a linear, noiseless
conversion with a digital offset (bias pedestal):

$$\text{ADU}(r,c) = \text{round}\!\left(\frac{S_\text{total}(r,c)}{g} + \delta\right)$$

where the gain $g$ [e⁻/ADU] maps the full-well capacity to the top of the
usable ADC range:

$$g = \frac{N_\text{FW}}{N_\text{max} - \delta}$$

and $\delta$ is the digital offset in ADU. The offset reserves headroom below
the signal so that negative-going read-noise excursions are not lost to
bottom-clipping. The final digital output is clamped to $[0, N_\text{max}]$
where $N_\text{max} = 2^b - 1$ and $b$ is the ADC bit depth.

Quantization noise, i.e., the rounding error, is implicitly $\pm 0.5$ LSB.

### 11. Complete noise budget

Summing the independent noise contributions in quadrature gives the total
noise standard deviation in electrons:

$$\sigma_\text{total} = \sqrt{\mu_\text{total} + \sigma_\text{read}^2}$$

At high signal levels shot noise dominates ($\sigma \approx \sqrt{\mu_\text{sig}}$).
At low signal levels or short integration times read noise dominates
($\sigma \approx \sigma_\text{read}$). PRNU and DSNU contribute fixed spatial
offsets rather than random noise; they do not appear in $\sigma_\text{total}$
but are the primary structured artifacts that NUC calibration removes.

---

## The two-point NUC calibration model

This simulation generates the raw data needed to derive a two-point
(gain + offset) NUC correction. For each pixel $(r,c)$, the measured response
to a uniform source of in-band radiance $L$ is modeled as:

$$\text{ADU}(r,c) = G(r,c) \cdot L + O(r,c)$$

where $G(r,c)$ is the pixel gain (encompassing $\Omega$, $\tau$, $p^2$,
$t_\text{int}$, $\eta$, $g^{-1}$, PRNU, and RI rolloff) and $O(r,c)$ is the
pixel offset (dark current, digital offset, and DSNU). The two calibration
frames — hot at $L_H = L(T_H)$ and cold at $L_C = L(T_C)$ — allow $G$ and
$O$ to be solved for pixel-by-pixel, which is the subject of the next
simulation stage.

---

## Module summary

| Module | Responsibility |
|---|---|
| `nuc/config.py` | All simulation parameters as `@dataclass` types |
| `nuc/physics.py` | Planck photon radiance; trapezoidal band integration |
| `nuc/optics.py` | cos⁴(θ) relative illumination map |
| `nuc/detector.py` | PRNU/DSNU generation; full signal chain → digital counts |
| `nuc/simulate.py` | Orchestration; returns `CalibrationFrames` dataclass |
| `main.py` | Entry point; prints noise budget summary; saves figure |
