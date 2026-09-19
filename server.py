"""
HTTP layer. Thin by design: endpoints wrap library calls so the core remains
testable without the server.

Simulations use separate processes for isolation and hard timeouts.
Validation stays inline because it is fast enough to run on every edit.
"""

from __future__ import annotations

import os
import contextlib
import math
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from builder import export_script
from schema import NetworkGraph
from validation import validate


HERE = Path(__file__).resolve().parent
INDEX_HTML = HERE / "index.html"


def _example_path() -> Path | None:
    path = HERE / "brunel.json"
    return path if path.is_file() else None

# Guard rails
MAX_NEURON_SECONDS = float(os.environ.get("SPIKENET_MAX_NEURON_SECONDS", 2_000_000))
SIM_TIMEOUT = float(os.environ.get("SPIKENET_TIMEOUT", 120))
MAX_SPIKES_RETURNED = 20_000


# --------------------------------------------------------------------------
# Worker process
# --------------------------------------------------------------------------


def _json_safe(obj):
    """Replace NaN and infinity with null on the way out.

    JSON has no literal for either, and Starlette's JSONResponse serialises
    with allow_nan=False, so a single NaN turns a successful simulation into an
    opaque HTTP 500. NaN is not exotic here: a network too quiet to produce two
    spikes in any neuron has an undefined CV of ISI, which is a legitimate
    result the user needs to see, not a crash.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def _quiet() -> None:
    from brian2 import BrianLogger

    BrianLogger.suppress_name("resolution_conflict")
    BrianLogger.suppress_name("method_choice")


def _sweep_worker(payload: dict) -> dict:
    """Runs in the child process. One simulation per parameter value."""
    from schema import NetworkGraph as _NG
    from sweep import run_sweep

    _quiet()
    return _json_safe(run_sweep(_NG.model_validate(payload["graph"]), payload["spec"]))


def _worker(payload: dict) -> dict:
    """Runs in the child process. Returns plain JSON, never live objects."""
    from runner import run_graph
    from schema import NetworkGraph as _NG

    _quiet()

    result = run_graph(_NG.model_validate(payload))
    data = result.to_json()

    # Subsample rasters before they cross the process boundary and again before
    # they cross the network. 
    for raster in data.get("rasters", {}).values():
        times, indices = raster["t"], raster["i"]
        if len(times) > MAX_SPIKES_RETURNED:
            step = len(times) // MAX_SPIKES_RETURNED + 1
            raster["t"] = times[::step]
            raster["i"] = indices[::step]
            raster["subsampled"] = True

    return _json_safe(data)


def _init_worker() -> None:
    """Restore default signal handling in the child.
    """
    import contextlib
    import signal

    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(ValueError, OSError):
            signal.signal(sig, signal.SIG_DFL)


class _Pool:
    """"""

    def __init__(self) -> None:
        self._pool: ProcessPoolExecutor | None = None

    def _ensure(self) -> ProcessPoolExecutor:
        if self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=1, initializer=_init_worker)
        return self._pool

    def run(self, payload: dict, timeout: float, fn=None) -> dict:
        future = self._ensure().submit(fn or _worker, payload)
        try:
            return future.result(timeout=timeout)
        except FutureTimeout:
            # The worker is still busy and cannot be interrupted, so discard the
            # whole pool. The next request gets a fresh one.
            self.reset()
            raise
        except Exception:
            self.reset()
            raise

    def reset(self) -> None:
        """
        """
        if self._pool is None:
            return
        for proc in list(getattr(self._pool, "_processes", {}).values()):
            with contextlib.suppress(Exception):
                proc.kill()
        self._pool.shutdown(wait=False, cancel_futures=True)
        self._pool = None


POOL = _Pool()


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------

app = FastAPI(title="SpikeNet Lab", version="0.1.0")


class GraphPayload(BaseModel):
    graph: dict[str, Any]


def _parse(payload: GraphPayload) -> NetworkGraph:
    try:
        return NetworkGraph.model_validate(payload.graph)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Malformed graph: {exc}") from exc


def _diagnostics(report) -> list[dict]:
    return [
        {
            "severity": d.severity,
            "message": d.message,
            "node_id": d.node_id,
            "field": d.field,
        }
        for d in report.diagnostics
    ]


def _cost(graph: NetworkGraph) -> float:
    from safe_units import safe_eval_quantity
    from brian2 import second

    try:
        duration = float(safe_eval_quantity(graph.run.duration) / second)
    except Exception:
        return 0.0
    return sum(p.n for p in graph.populations) * duration


@app.get("/api/example")
def example() -> dict:
    path = _example_path()
    if path is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "brunel.json was not found next to server.py. "
                "Start from an empty canvas instead."
            ),
        )
    import json

    # Round-trip through the schema rather than serving the file verbatim, so
    # the browser receives a *complete* graph with every default filled in.
    # Serving raw JSON meant optional fields the file happened to omit arrived
    # as undefined, and the frontend then wrote NaN into them.
    try:
        graph = NetworkGraph.model_validate(json.loads(path.read_text()))
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"The bundled example is invalid: {exc}"
        ) from exc
    return graph.model_dump(mode="json")


@app.post("/api/validate")
def api_validate(payload: GraphPayload) -> dict:
    """Structure, syntax, allowlist and units. No simulation. Safe to spam."""
    graph = _parse(payload)
    try:
        report = validate(graph, dry_run=True)
    except Exception as exc:
        return {"ok": False, "diagnostics": [{
            "severity": "error",
            "message": f"Validation failed unexpectedly: {type(exc).__name__}: {exc}",
            "node_id": None, "field": None,
        }]}
    return {"ok": report.ok, "diagnostics": _diagnostics(report)}


@app.post("/api/simulate")
def api_simulate(payload: GraphPayload) -> JSONResponse:
    graph = _parse(payload)

    cost = _cost(graph)
    if cost > MAX_NEURON_SECONDS:
        raise HTTPException(
            status_code=413,
            detail=(
                f"This run is {cost:,.0f} neuron-seconds, over the "
                f"{MAX_NEURON_SECONDS:,.0f} limit. Reduce the population sizes "
                f"or the duration, or raise SPIKENET_MAX_NEURON_SECONDS."
            ),
        )

    try:
        data = POOL.run(graph.model_dump(mode="json"), timeout=SIM_TIMEOUT)
    except FutureTimeout:
        raise HTTPException(
            status_code=504,
            detail=(
                f"Simulation exceeded {SIM_TIMEOUT:.0f}s and was abandoned. "
                f"Shorten the run or switch codegen_target to 'cython'."
            ),
        ) from None
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Worker failed: {exc}") from exc

    return JSONResponse(data)


class SweepPayload(BaseModel):
    graph: dict[str, Any]
    spec: dict[str, Any]


@app.get("/api/metrics")
def api_metrics() -> dict:
    """What a sweep can measure. The UI builds its dropdown from this, so the
    two can never drift apart."""
    from sweep import METRICS

    return {"metrics": [{"id": k, "label": v[0]} for k, v in METRICS.items()]}


@app.post("/api/sweep")
def api_sweep(payload: SweepPayload) -> JSONResponse:
    """Run one simulation per parameter value. Slow by construction."""
    graph = _parse(GraphPayload(graph=payload.graph))

    values = payload.spec.get("values") or []
    if not values:
        raise HTTPException(status_code=422, detail="The sweep has no values.")
    if len(values) > 40:
        raise HTTPException(
            status_code=413,
            detail=f"{len(values)} points is too many. Keep it under 40.",
        )

    cost = _cost(graph) * len(values)
    if cost > MAX_NEURON_SECONDS * 4:
        raise HTTPException(
            status_code=413,
            detail=(
                f"That sweep is {cost:,.0f} neuron-seconds across {len(values)} "
                f"runs. Shorten the duration, shrink the populations, or use "
                f"fewer points."
            ),
        )

    # A sweep is N simulations, so it gets N times the patience.
    timeout = SIM_TIMEOUT * max(2, len(values))
    try:
        data = POOL.run(
            {"graph": graph.model_dump(mode="json"), "spec": payload.spec},
            timeout=timeout,
            fn=_sweep_worker,
        )
    except FutureTimeout:
        raise HTTPException(
            status_code=504,
            detail=f"The sweep exceeded {timeout:.0f}s and was abandoned.",
        ) from None
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Sweep failed: {exc}") from exc

    return JSONResponse(data)


@app.post("/api/export")
def api_export(payload: GraphPayload) -> dict:
    graph = _parse(payload)
    return {"script": export_script(graph)}


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "version": app.version}


@app.get("/")
def index() -> FileResponse:
    """
    """
    if not INDEX_HTML.is_file():
        raise HTTPException(status_code=404, detail="index.html is missing.")
    return FileResponse(INDEX_HTML)


def main() -> None:
    import uvicorn

    host = os.environ.get("SPIKENET_HOST", "127.0.0.1")
    port = int(os.environ.get("SPIKENET_PORT", 8000))
    print(f"\n  SpikeNet Lab running at  http://{host}:{port}\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
