"""
The one function the API layer calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from brian2 import Hz, second

import analysis
from builder import build_network
from safe_units import safe_eval_quantity
from schema import NetworkGraph
from validation import ValidationResult, validate


@dataclass
class SimulationResult:
    ok: bool
    validation: ValidationResult
    duration: float = 0.0
    wall_time: float = 0.0
    rasters: dict[str, dict] = field(default_factory=dict)
    rates: dict[str, dict] = field(default_factory=dict)
    traces: dict[str, dict] = field(default_factory=dict)
    metrics: dict[str, dict] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "ok": self.ok,
            "diagnostics": [
                {
                    "severity": d.severity,
                    "message": d.message,
                    "node_id": d.node_id,
                    "field": d.field,
                }
                for d in self.validation.diagnostics
            ],
            "duration": self.duration,
            "wall_time": self.wall_time,
            "rasters": self.rasters,
            "rates": self.rates,
            "traces": self.traces,
            "metrics": self.metrics,
        }


def run_graph(graph: NetworkGraph, analyse: bool = True) -> SimulationResult:
    """Validate, build, simulate, and read the monitors back."""
    import time

    from brian2 import seed as brian_seed

    report = validate(graph, dry_run=True)
    if not report.ok:
        return SimulationResult(ok=False, validation=report)

    if graph.run.seed is not None:
        brian_seed(graph.run.seed)
        np.random.seed(graph.run.seed)

    built = build_network(graph)
    duration = safe_eval_quantity(graph.run.duration)
    transient = float(safe_eval_quantity(graph.run.transient) / second)

    t0 = time.time()
    built.network.run(duration)
    wall = time.time() - t0

    result = SimulationResult(
        ok=True,
        validation=report,
        duration=float(duration / second),
        wall_time=wall,
    )

    pop_size = {p.id: p.n for p in graph.populations}
    spike_data: dict[str, tuple] = {}
    rate_data: dict[str, tuple] = {}

    for mon in graph.monitors:
        obj = built.monitors[mon.id]

        if mon.kind == "spike":
            t = np.asarray(obj.t / second)
            i = np.asarray(obj.i, dtype=int)
            spike_data[mon.target] = (t, i)
            result.rasters[mon.id] = {
                "target": mon.target,
                "t": t.tolist(),
                "i": i.tolist(),
            }

        elif mon.kind == "rate":
            t = np.asarray(obj.t / second)
            r = np.asarray(obj.rate / Hz)
            rate_data[mon.target] = (t, r)
            result.rates[mon.id] = {
                "target": mon.target,
                "t": t.tolist(),
                "rate": r.tolist(),
            }

        else:
            entry = {"target": mon.target, "t": np.asarray(obj.t / second).tolist()}
            for var in mon.variables:
                values = np.asarray(getattr(obj, var))
                entry[var] = values.tolist()
            result.traces[mon.id] = entry

    if analyse:
        for pop_id, (t, r) in rate_data.items():
            st, si = spike_data.get(pop_id, (np.array([]), np.array([], dtype=int)))
            try:
                result.metrics[pop_id] = analysis.summarise(
                    t, r, st, si,
                    n_neurons=pop_size[pop_id],
                    duration=result.duration,
                    transient=transient,
                )
            except ValueError as exc:
                result.metrics[pop_id] = {"error": str(exc)}

    return result
