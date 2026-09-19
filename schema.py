"""
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


def _clean_position(value: Any) -> dict[str, float]:
    """
    """
    if not isinstance(value, dict):
        return {"x": 0.0, "y": 0.0}
    out: dict[str, float] = {}
    for axis in ("x", "y"):
        try:
            number = float(value.get(axis))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            number = 0.0
        out[axis] = number if math.isfinite(number) else 0.0
    return out

# --------------------------------------------------------------------------
# Populations
# --------------------------------------------------------------------------


class NeuronNode(BaseModel):
    """A NeuronGroup: one node on the canvas."""

    id: str
    label: str = "population"
    n: int = Field(gt=0, le=100_000)

    # Brian2's signature feature: the model IS a string of equations.
    equations: str
    threshold: str | None = None
    reset: str | None = None
    refractory: str | None = None  # e.g. "2*ms"
    method: Literal["euler", "exact", "heun", "milstein", "rk2", "rk4"] = "euler"

    params: dict[str, str] = Field(default_factory=dict)

    initial: dict[str, str] = Field(default_factory=dict)

    position: dict[str, float] = Field(default_factory=lambda: {"x": 0.0, "y": 0.0})

    @field_validator("equations")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("A population needs at least one equation.")
        return v

    @field_validator("position", mode="before")
    @classmethod
    def _position(cls, v: Any) -> dict[str, float]:
        return _clean_position(v)


# --------------------------------------------------------------------------
# Connections
# --------------------------------------------------------------------------


class SynapseEdge(BaseModel):
    """A Synapses object: one edge on the canvas."""

    id: str
    source: str  # NeuronNode.id
    target: str  # NeuronNode.id
    label: str = "synapse"

    # Connectivity. Exactly one of `condition` or `probability` should be set;
    # `condition` wins if both are given.
    condition: str | None = None  # e.g. "i != j"
    probability: float | None = Field(default=None, ge=0.0, le=1.0)
    n_per_pair: int = Field(default=1, ge=1)

    # Synaptic model + pathways.
    model: str = ""  # e.g. "w : volt" for per-synapse weights
    on_pre: str | None = None  # e.g. "s_e_post += w"
    on_post: str | None = None  # for STDP etc.
    delay: str | None = None  # "1.5*ms"

    params: dict[str, str] = Field(default_factory=dict)
    initial: dict[str, str] = Field(default_factory=dict)  # {"w": "0.1*mV"}


# --------------------------------------------------------------------------
# Stimulation
# --------------------------------------------------------------------------

StimulusKind = Literal[
    "poisson",       # PoissonInput  -- the usual Brunel drive
    "poisson_group", # PoissonGroup routed through a real Synapses object
    "dc",            # constant current/voltage injection
    "step",          # on at t_start, off at t_stop
    "sinusoid",      # amplitude * sin(2*pi*f*t) -- drives the resonance probe
    "chirp",         # linear frequency sweep: one run gives the whole tuning curve
    "ou_noise",      # Ornstein-Uhlenbeck coloured noise
    "spike_train",   # SpikeGeneratorGroup, exact user-specified spikes
]


class StimulusNode(BaseModel):
    """An input source node feeding one population.
    """

    id: str
    target: str  # NeuronNode.id
    kind: StimulusKind = "poisson"
    label: str = "input"

    target_var: str = "v"

    # Shared / per-kind parameters, all as unit strings.
    #   poisson       : n_inputs, rate, weight
    #   dc / step     : amplitude, (t_start, t_stop)
    #   sinusoid      : amplitude, frequency, offset
    #   chirp         : amplitude, f_start, f_stop, duration
    #   ou_noise      : mean, sigma, tau
    #   spike_train   : indices, times (plain lists, not unit strings)
    params: dict[str, Any] = Field(default_factory=dict)
    position: dict[str, float] = Field(default_factory=lambda: {"x": 0.0, "y": 0.0})

    @field_validator("position", mode="before")
    @classmethod
    def _position(cls, v: Any) -> dict[str, float]:
        return _clean_position(v)


# --------------------------------------------------------------------------
# Recording + run control
# --------------------------------------------------------------------------


class MonitorSpec(BaseModel):
    id: str
    target: str  # NeuronNode.id
    kind: Literal["spike", "rate", "state"] = "spike"
    variables: list[str] = Field(default_factory=list)  # for kind="state"
    record: int | list[int] | bool = True  # which indices; True = all
    dt: str | None = None  # subsample state recording, e.g. "1*ms"


class RunConfig(BaseModel):
    duration: str = "1*second"
    dt: str = "0.1*ms"
    transient: str = "0*second"  # discarded before analysis
    seed: int | None = None
    codegen_target: Literal["numpy", "cython", "auto"] = "numpy"


class NetworkGraph(BaseModel):
    """The complete document the canvas serialises. This is the whole API."""

    version: Literal["1"] = "1"
    name: str = "untitled network"
    populations: list[NeuronNode] = Field(default_factory=list)
    synapses: list[SynapseEdge] = Field(default_factory=list)
    stimuli: list[StimulusNode] = Field(default_factory=list)
    monitors: list[MonitorSpec] = Field(default_factory=list)
    run: RunConfig = Field(default_factory=RunConfig)

    # Constants visible to every node unless shadowed locally. This is the
    # "control everything globally" half of the local/global split: a slider
    # bound here retunes the entire network at once.
    globals: dict[str, str] = Field(default_factory=dict)

    def population_ids(self) -> set[str]:
        return {p.id for p in self.populations}
