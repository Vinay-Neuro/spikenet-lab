"""
Parameter sweeps
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from collections.abc import Callable

import analysis
from schema import NetworkGraph

#: metric name -> (label, extractor). The extractor receives the metrics dict
#: that `analysis.summarise` produced for the target population, plus the sweep
#: options, and returns a single float.
MetricFn = Callable[[dict, dict], float]


def _amplitude_at(metrics: dict, options: dict) -> float:
    """
    """
    spectrum = metrics["spectrum"]
    target = options.get("at_frequency", "resonant")
    if target == "resonant" or target is None:
        return float(metrics["resonance"]["peak_amplitude"])
    _, amplitude = analysis.amplitude_at_frequency(
        spectrum["frequencies"], spectrum["amplitudes"], float(target)
    )
    return amplitude


def _power_at(metrics: dict, options: dict) -> float:
    """Power is amplitude squared. Offered separately because people ask for it
    by name, and squaring by hand after the fact is an easy place to slip."""
    return _amplitude_at(metrics, options) ** 2


METRICS: dict[str, tuple[str, MetricFn]] = {
    "peak_frequency":  ("Resonant frequency (Hz)", lambda m, o: m["resonance"]["peak_frequency"]),
    "peak_amplitude":  ("Peak amplitude (norm.)",  lambda m, o: m["resonance"]["peak_amplitude"]),
    "amplitude_at":    ("Amplitude at frequency",  _amplitude_at),
    "power_at":        ("Power at frequency",      _power_at),
    "q_factor":        ("Q factor",                lambda m, o: m["resonance"]["q_factor"]),
    "bandwidth_half_power": ("-3 dB bandwidth (Hz)", lambda m, o: m["resonance"]["bandwidth_half_power"]),
    "bandwidth_fwhm":  ("FWHM amplitude (Hz)",    lambda m, o: m["resonance"]["bandwidth_fwhm"]),
    "entropy":         ("Spectral entropy",        lambda m, o: m["spectrum"]["entropy"]),
    "mean_rate":       ("Mean rate (Hz)",          lambda m, o: m["spikes"]["mean_rate"]),
    "cv_isi":          ("CV of ISI",               lambda m, o: m["spikes"]["cv_isi"]),
    "fano_factor":     ("Fano factor",             lambda m, o: m["spikes"]["fano_factor"]),
    "active_fraction": ("Active fraction",         lambda m, o: m["spikes"]["active_fraction"]),
    "synchrony":       ("Synchrony index",         lambda m, o: m["synchrony"]),
}


def _finite(value):
    """None instead of NaN or infinity, because JSON cannot express either."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass
class SweepPoint:
    value: float
    metric: float | None = None
    error: str | None = None
    extra: dict = field(default_factory=dict)


def _apply(graph: NetworkGraph, scope: str, node_id: str | None,
           key: str, text: str) -> None:
    """Write one constant into the graph, wherever it lives."""
    if scope == "globals":
        graph.globals[key] = text
        return
    if scope == "population":
        for pop in graph.populations:
            if pop.id == node_id:
                pop.params[key] = text
                return
        raise ValueError(f"No population {node_id!r}.")
    if scope == "synapse":
        for syn in graph.synapses:
            if syn.id == node_id:
                syn.params[key] = text
                return
        raise ValueError(f"No synapse {node_id!r}.")
    if scope == "stimulus":
        for stim in graph.stimuli:
            if stim.id == node_id:
                stim.params[key] = text
                return
        raise ValueError(f"No stimulus {node_id!r}.")
    raise ValueError(f"Unknown scope {scope!r}.")


def run_sweep(graph: NetworkGraph, spec: dict) -> dict:
    """Run one simulation per parameter value and collect one metric each.

    spec:
      scope       "globals" | "population" | "synapse" | "stimulus"
      node_id     required unless scope is "globals"
      key         the constant's name, e.g. "g"
      unit        unit suffix to append, e.g. "mV"; "" for dimensionless
      values      list of numbers
      target      population id whose metrics are read
      metric      a key of METRICS
      options     e.g. {"at_frequency": 20} or {"at_frequency": "resonant"}
    """
    from runner import run_graph

    metric_name = spec.get("metric", "peak_frequency")
    if metric_name not in METRICS:
        raise ValueError(f"Unknown metric {metric_name!r}.")
    label, extract = METRICS[metric_name]

    scope = spec.get("scope", "globals")
    node_id = spec.get("node_id")
    key = spec.get("key")
    if not key:
        raise ValueError("The sweep does not say which parameter to vary.")
    unit = spec.get("unit", "") or ""
    options = spec.get("options", {}) or {}

    values = [float(v) for v in (spec.get("values") or [])]
    if not values:
        raise ValueError("The sweep has no values.")
    if any(not math.isfinite(v) for v in values):
        raise ValueError("Sweep values must all be finite numbers.")

    target = spec.get("target") or (graph.populations[0].id if graph.populations else None)
    if target is None:
        raise ValueError("The graph has no populations to measure.")

    points: list[SweepPoint] = []
    for value in values:
        # Format through the same "number*unit" convention the UI uses, so a
        # swept parameter is indistinguishable from a typed one.
        text = f"{value:.6g}*{unit}" if unit else f"{value:.6g}"

        try:
            trial = copy.deepcopy(graph)
            # Inside the try: a stale panel can name a node the user has since
            # deleted, and that should show up as a labelled point rather than
            # taking down the request.
            _apply(trial, scope, node_id, key, text)
            result = run_graph(trial)
            if not result.ok:
                points.append(SweepPoint(value, None,
                    result.validation.errors[0].message[:160]))
                continue
            metrics = result.metrics.get(target)
            if not metrics or "spectrum" not in metrics:
                points.append(SweepPoint(value, None,
                    f"No rate monitor on {target!r}, so there is nothing to measure."))
                continue
            value_out = float(extract(metrics, options))
            if not math.isfinite(value_out):
                # NaN is a real outcome, not a crash: a population too quiet to
                # give any neuron two spikes has an undefined CV of ISI. Report
                # it as a labelled gap so the curve shows a hole rather than
                # silently plotting nonsense, and so it survives JSON, which
                # cannot represent NaN at all.
                points.append(SweepPoint(value, None,
                    f"{metric_name} is undefined here "
                    f"(mean rate {metrics['spikes']['mean_rate']:.2f} Hz)."))
                continue
            points.append(SweepPoint(value, value_out, None, {
                "mean_rate": _finite(metrics["spikes"]["mean_rate"]),
                "regime": metrics["regime"],
                "peak_frequency": _finite(metrics["resonance"]["peak_frequency"]),
            }))
        except Exception as exc:  # one bad point must not lose the whole sweep
            points.append(SweepPoint(value, None, f"{type(exc).__name__}: {exc}"[:160]))

    return {
        "label": label,
        "metric": metric_name,
        "parameter": key,
        "unit": unit,
        "scope": scope,
        "node_id": node_id,
        "target": target,
        "options": options,
        "points": [
            {"value": p.value, "metric": p.metric, "error": p.error, "extra": p.extra}
            for p in points
        ],
    }
