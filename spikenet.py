"""SpikeNet Lab -- a node-graph front end for Brian2 spiking network models."""

from analysis import (
    amplitude_at_frequency,
    classify_regime,
    isi_histogram,
    rate_distribution,
    rate_spectrum,
    resonant_frequency,
    spectral_entropy,
    spike_statistics,
    summarise,
    synchrony_index,
)
from builder import BuildError, build_network, export_script
from runner import SimulationResult, run_graph
from safe_units import UnsafeExpression, safe_eval_quantity
from schema import (
    MonitorSpec,
    NetworkGraph,
    NeuronNode,
    RunConfig,
    StimulusNode,
    SynapseEdge,
)
from sweep import METRICS, run_sweep
from validation import Diagnostic, ValidationResult, validate

__version__ = "0.1.0"

__all__ = [
    "NetworkGraph", "NeuronNode", "SynapseEdge", "StimulusNode",
    "MonitorSpec", "RunConfig",
    "validate", "ValidationResult", "Diagnostic",
    "build_network", "export_script", "BuildError",
    "run_graph", "SimulationResult",
    "safe_eval_quantity", "UnsafeExpression",
    "rate_spectrum", "spectral_entropy", "resonant_frequency",
    "spike_statistics", "synchrony_index", "classify_regime",
    "amplitude_at_frequency", "isi_histogram", "rate_distribution",
    "summarise",
    "run_sweep", "METRICS",
]
