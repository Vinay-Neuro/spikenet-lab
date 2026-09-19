"""
Read-out layer
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


# --------------------------------------------------------------------------
# Spectrum
# --------------------------------------------------------------------------


def rate_spectrum(
    t: np.ndarray,
    rate: np.ndarray,
    transient: float = 0.0,
    normalise: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Amplitude spectrum of a population rate trace.

    Args:
        t: sample times in seconds.
        rate: population rate in Hz.
        transient: seconds to discard from the start, so the initial swing
            into the attractor does not smear the peak.
        normalise: divide by total amplitude, making spectra from runs with
            different mean rates directly comparable.

    Returns:
        (frequencies in Hz, amplitudes).
    """
    t = np.asarray(t, dtype=float)
    rate = np.asarray(rate, dtype=float)

    mask = t >= transient
    t, rate = t[mask], rate[mask]
    if t.size < 8:
        raise ValueError("Not enough samples left after the transient.")

    # Removing the mean matters: the DC term otherwise dwarfs everything and
    # the normalisation below would be dominated by it.
    centred = rate - rate.mean()
    dt = float(np.mean(np.diff(t)))

    spectrum = np.fft.rfft(centred)
    freqs = np.fft.rfftfreq(centred.size, d=dt)
    amps = (2.0 / centred.size) * np.abs(spectrum)

    if normalise:
        total = amps.sum()
        if total > 0:
            amps = amps / total

    return freqs, amps


def spectral_entropy(
    freqs: np.ndarray, amps: np.ndarray, low: float = 0.0, high: float = 40.0
) -> float:
    """Shannon entropy of the power distribution over a band, normalised to [0, 1].

    1.0 means power is spread evenly across the band (asynchronous, no preferred
    rhythm). Near 0 means power is concentrated at one frequency (strongly
    oscillatory). This is the single most useful scalar for sweeping a parameter
    and asking "where does the network start oscillating?".
    """
    band = (freqs > low) & (freqs <= high)
    if not band.any():
        raise ValueError(f"No frequency bins in {low}-{high} Hz.")

    power = np.asarray(amps)[band] ** 2
    total = power.sum()
    if total <= 0:
        return float("nan")

    p = power / total
    H = -np.sum(p * np.log(p + 1e-12))
    return float(H / np.log(len(p)))


def amplitude_at_frequency(
    freqs: np.ndarray, amps: np.ndarray, target: float
) -> tuple[float, float]:
    """Amplitude in the bin nearest `target` Hz.

    Returns (actual_bin_frequency, amplitude). The bin frequency is returned
    because FFT resolution is 1/T: asking for 23 Hz from a 1 s run gets you the
    23 Hz bin, but from a 0.4 s run the nearest bin is 22.5 Hz. Silently
    answering a different question than the one asked is how sweeps end up
    looking noisier than the underlying physics.
    """
    freqs = np.asarray(freqs, dtype=float)
    amps = np.asarray(amps, dtype=float)
    if freqs.size == 0:
        raise ValueError("Empty spectrum.")
    k = int(np.argmin(np.abs(freqs - target)))
    return float(freqs[k]), float(amps[k])


def isi_histogram(
    spike_times: np.ndarray,
    spike_indices: np.ndarray,
    n_neurons: int,
    transient: float = 0.0,
    bins: int = 40,
    max_isi: float = 0.2,
) -> dict:
    """Distribution of inter-spike intervals, pooled across neurons.

    The shape is the quickest read on firing regime: exponential means
    Poisson-like, a peak away from zero means the cells have a preferred
    interval, i.e. they are oscillating.
    """
    spike_times = np.asarray(spike_times, dtype=float)
    spike_indices = np.asarray(spike_indices, dtype=int)
    mask = spike_times >= transient
    spike_times, spike_indices = spike_times[mask], spike_indices[mask]

    order = np.argsort(spike_indices, kind="stable")
    sorted_idx, sorted_t = spike_indices[order], spike_times[order]
    boundaries = np.searchsorted(sorted_idx, np.arange(n_neurons + 1))

    intervals: list[np.ndarray] = []
    for n in range(n_neurons):
        ts = np.sort(sorted_t[boundaries[n]:boundaries[n + 1]])
        if ts.size >= 2:
            intervals.append(np.diff(ts))

    if not intervals:
        edges = np.linspace(0.0, max_isi, bins + 1)
        return {"edges": edges.tolist(), "counts": [0] * bins}

    pooled = np.concatenate(intervals)
    counts, edges = np.histogram(pooled, bins=bins, range=(0.0, max_isi))
    return {"edges": edges.tolist(), "counts": counts.tolist()}


def rate_distribution(
    spike_indices: np.ndarray, n_neurons: int, window: float, bins: int = 30
) -> dict:
    """Histogram of per-neuron firing rates.

    A population mean of 10 Hz can mean every cell at 10 Hz or a tenth of them
    at 100 Hz and the rest silent. Only the distribution distinguishes those.
    """
    counts = np.bincount(np.asarray(spike_indices, dtype=int), minlength=n_neurons)
    rates = counts / window if window > 0 else counts * 0.0
    hist, edges = np.histogram(rates, bins=bins)
    return {"edges": edges.tolist(), "counts": hist.tolist()}


@dataclass
class Resonance:
    peak_frequency: float          # Hz
    peak_amplitude: float
    bandwidth_fwhm: float          # Hz, full width at half maximum *amplitude*
    bandwidth_half_power: float    # Hz, width at peak/sqrt(2), the -3 dB width
    q_factor: float                # peak_frequency / bandwidth_half_power
    truncated: bool                # a crossing fell outside the analysis band


def _width_at(
    f_band: np.ndarray, a_band: np.ndarray, k: int, threshold: float
) -> tuple[float, bool]:
    """Width of the peak at a given amplitude threshold.

    Returns (width, truncated). `truncated` is True when the curve never drops
    below the threshold before the edge of the analysis band, in which case the
    width is a lower bound and is reported as NaN rather than as a number that
    looks measured.
    """
    left = k
    while left > 0 and a_band[left] > threshold:
        left -= 1
    right = k
    while right < len(a_band) - 1 and a_band[right] > threshold:
        right += 1

    hit_edge = (left == 0 and a_band[0] > threshold) or (
        right == len(a_band) - 1 and a_band[-1] > threshold
    )
    if hit_edge:
        return float("nan"), True
    return float(f_band[right] - f_band[left]), False


def resonant_frequency(
    freqs: np.ndarray, amps: np.ndarray, low: float = 1.0, high: float = 100.0
) -> Resonance:
    """Locate the spectral peak and measure how sharply it is tuned.

    The band deliberately excludes DC: without `low > 0` the peak is almost
    always bin 0 and the answer is meaningless.

    Two widths are reported because they are not the same thing and the
    difference is easy to get wrong on an *amplitude* spectrum:

    * `bandwidth_fwhm` is the full width at half maximum amplitude, the
      threshold being `A_peak / 2`.
    * `bandwidth_half_power` is the width where the *power* has halved. Power
      goes as amplitude squared, so the amplitude threshold is
      `A_peak / sqrt(2)`, not `A_peak / 2`. This is the -3 dB width.

    Q factor is defined against the half-power width, which is the convention
    everywhere.

    Either width is NaN when the peak never falls below its threshold inside
    `[low, high]`: the honest answer there is "wider than the window", not a
    number obtained by clipping at the edge.
    """
    freqs = np.asarray(freqs, dtype=float)
    amps = np.asarray(amps, dtype=float)

    band = (freqs >= low) & (freqs <= high)
    if not band.any():
        raise ValueError(f"No frequency bins in {low}-{high} Hz.")

    f_band, a_band = freqs[band], amps[band]
    k = int(np.argmax(a_band))
    peak_f, peak_a = float(f_band[k]), float(a_band[k])

    fwhm, trunc_fwhm = _width_at(f_band, a_band, k, peak_a / 2.0)
    half_power, trunc_hp = _width_at(f_band, a_band, k, peak_a / np.sqrt(2.0))

    q = float("nan") if (np.isnan(half_power) or half_power <= 0) else peak_f / half_power

    return Resonance(peak_f, peak_a, fwhm, half_power, float(q),
                     bool(trunc_fwhm or trunc_hp))


# --------------------------------------------------------------------------
# Spike statistics
# --------------------------------------------------------------------------


@dataclass
class SpikeStats:
    n_spikes: int
    mean_rate: float          # Hz per neuron
    cv_isi: float             # irregularity: ~1 Poisson-like, ~0 clock-like
    # Count Fano factor ACROSS NEURONS over the whole analysis window:
    # var(spike count per neuron) / mean(spike count per neuron). This is not
    # the temporal Fano factor (variance of one neuron's count across repeated
    # windows or trials), which is the more common meaning in the literature
    # and would need repeats this tool does not run.
    fano_factor: float
    active_fraction: float    # fraction of neurons that spiked at all


def spike_statistics(
    spike_times: np.ndarray,
    spike_indices: np.ndarray,
    n_neurons: int,
    duration: float,
    transient: float = 0.0,
) -> SpikeStats:
    """Per-neuron rate, ISI irregularity and count variability."""
    spike_times = np.asarray(spike_times, dtype=float)
    spike_indices = np.asarray(spike_indices, dtype=int)

    mask = spike_times >= transient
    spike_times, spike_indices = spike_times[mask], spike_indices[mask]
    window = duration - transient
    if window <= 0:
        raise ValueError("The transient is at least as long as the run.")

    counts = np.bincount(spike_indices, minlength=n_neurons)
    rates = counts / window

    # CV of the inter-spike interval, pooled over neurons with >= 3 spikes.
    cvs = []
    order = np.argsort(spike_indices, kind="stable")
    sorted_idx, sorted_t = spike_indices[order], spike_times[order]
    boundaries = np.searchsorted(sorted_idx, np.arange(n_neurons + 1))
    for n in range(n_neurons):
        ts = np.sort(sorted_t[boundaries[n]:boundaries[n + 1]])
        if ts.size >= 3:
            isi = np.diff(ts)
            if isi.mean() > 0:
                cvs.append(isi.std() / isi.mean())

    cv = float(np.mean(cvs)) if cvs else float("nan")
    fano = float(counts.var() / counts.mean()) if counts.mean() > 0 else float("nan")

    return SpikeStats(
        n_spikes=int(spike_times.size),
        mean_rate=float(rates.mean()),
        cv_isi=cv,
        fano_factor=fano,
        active_fraction=float((counts > 0).mean()),
    )


def _rebin(t: np.ndarray, rate: np.ndarray, bin_width: float):
    """Average a rate trace into wider bins. Returns (new_dt, binned_rate)."""
    dt = float(np.mean(np.diff(t)))
    k = max(1, int(round(bin_width / dt)))
    if k == 1:
        return dt, rate
    usable = (rate.size // k) * k
    binned = rate[:usable].reshape(-1, k).mean(axis=1)
    return dt * k, binned


def synchrony_index(
    t: np.ndarray,
    rate: np.ndarray,
    n_neurons: int,
    transient: float = 0.0,
    finite_size_correction: bool = True,
    bin_width: float = 1e-3,
) -> float:
    """Coefficient of variation of the population rate, corrected for shot noise.


    The trace is rebinned to `bin_width` first. PopulationRateMonitor bins at the simulation timestep, typically
    0.1 ms, where a 1000-neuron population firing at 16 Hz averages under two
    spikes per bin. 
    
    Returns 0.0 for a fully asynchronous population; values approaching and
    exceeding 1 indicate population-wide volleys.
    """
    t = np.asarray(t, dtype=float)
    rate = np.asarray(rate, dtype=float)

    mask = t >= transient
    t, rate = t[mask], rate[mask]
    if rate.size < 2:
        return float("nan")

    dt, rate = _rebin(t, rate, bin_width)
    if rate.size < 2:
        return float("nan")

    mean_rate = rate.mean()
    if mean_rate <= 0:
        return float("nan")

    cv2 = float(rate.var() / mean_rate**2)

    if finite_size_correction:
        noise_floor = 1.0 / (mean_rate * n_neurons * dt)
        cv2 = max(0.0, cv2 - noise_floor)

    return float(np.sqrt(cv2))


def classify_regime(
    cv_isi: float, sync: float, cv_threshold: float = 0.5,
    sync_threshold: float = 0.3,
) -> str:
    """Coarse Brunel-style label.

    A hint for the UI, not ground truth. The two thresholds are conventions,
    not measured boundaries -- real transitions are gradual, and where you put
    the line depends on the synaptic model. Exponential synapses low-pass the
    input and systematically reduce CV_ISI relative to the delta-synapse
    networks the AI/SI terminology was coined for, so expect to retune these
    for your own model. 
    """
    if np.isnan(cv_isi) or np.isnan(sync):
        return "insufficient activity"
    irregular = cv_isi > cv_threshold
    synchronous = sync > sync_threshold
    if irregular and not synchronous:
        return "asynchronous irregular (AI)"
    if irregular and synchronous:
        return "synchronous irregular (SI)"
    if not irregular and synchronous:
        return "synchronous regular (SR)"
    return "asynchronous regular (AR)"


# --------------------------------------------------------------------------


def summarise(
    t: np.ndarray,
    rate: np.ndarray,
    spike_times: np.ndarray,
    spike_indices: np.ndarray,
    n_neurons: int,
    duration: float,
    transient: float = 0.0,
    band: tuple[float, float] = (1.0, 100.0),
) -> dict:
    """Everything above in one JSON-serialisable dict, ready for the dashboard."""
    freqs, amps = rate_spectrum(t, rate, transient=transient)
    res = resonant_frequency(freqs, amps, low=band[0], high=band[1])
    stats = spike_statistics(spike_times, spike_indices, n_neurons, duration, transient)
    sync = synchrony_index(t, rate, n_neurons, transient)
    window = max(duration - transient, 1e-9)

    return {
        "spectrum": {
            "frequencies": freqs.tolist(),
            "amplitudes": amps.tolist(),
            "entropy": spectral_entropy(freqs, amps, band[0], band[1]),
            "normalised": True,
        },
        "resonance": asdict(res),
        "spikes": asdict(stats),
        "synchrony": sync,
        "regime": classify_regime(stats.cv_isi, sync),
        "isi": isi_histogram(spike_times, spike_indices, n_neurons, transient),
        "rate_distribution": rate_distribution(spike_indices, n_neurons, window),
    }
