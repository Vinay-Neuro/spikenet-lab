"""
Validation in three layers, cheapest first.

  Layer 1  structural   -- pydantic types + referential integrity of the graph
  Layer 2  static       -- identifier allowlist + Brian2's equation parser
  Layer 3  dry run      -- build the real objects and run for ZERO milliseconds

Layer 3 is the important one and it is worth explaining, because the obvious
approach is wrong. You cannot check Brian2 unit consistency by inspecting the
AST or matching regexes: `(v + I)/tau` is dimensionally valid or invalid
depending on what v, I and tau were declared as, which requires the same
sympy-backed dimensional algebra Brian2 already implements. Reimplementing it
means reimplementing it badly.

Measured against Brian2 2.10, unit errors are NOT raised by `Equations(...)` and
NOT raised by `NeuronGroup(...)`. They surface only during `Network.before_run`,
which `run()` calls first. So `net.run(0*ms)` triggers the complete check --
equations, thresholds, resets, synaptic pathways, namespace resolution -- and
then returns without integrating a single timestep. On a 2500-neuron Brunel
network that costs about 50 ms.

Layer 2 is not redundant with layer 3: it is the security boundary. Layer 3
hands user strings to Brian2's code generator, so anything reaching it must
already be known-safe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from brian2.core.namespace import DEFAULT_CONSTANTS, DEFAULT_FUNCTIONS, DEFAULT_UNITS
from brian2.equations.equations import Equations
from brian2.utils.stringtools import get_identifiers

from safe_units import UnsafeExpression, safe_eval_quantity
from schema import NetworkGraph

Severity = Literal["error", "warning"]

# Names any user string may reference without being declared: units, pi/e/inf,
# Brian2's built-in functions, and the handful of special variables.
_BUILTIN_NAMES: set[str] = (
    set(DEFAULT_UNITS)
    | set(DEFAULT_CONSTANTS)
    | set(DEFAULT_FUNCTIONS)
    | {"t", "dt", "i", "j", "N", "N_pre", "N_post", "N_incoming", "N_outgoing", "xi"}
)


@dataclass
class Diagnostic:
    """One problem, addressed to one node so the UI can highlight it."""

    severity: Severity
    message: str
    node_id: str | None = None
    field: str | None = None

    def __str__(self) -> str:
        if self.node_id and self.field:
            where = f"[{self.node_id}.{self.field}] "
        elif self.node_id:
            where = f"[{self.node_id}] "
        else:
            where = ""
        return f"{self.severity.upper()}: {where}{self.message}"


@dataclass
class ValidationResult:
    diagnostics: list[Diagnostic] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(d.severity == "error" for d in self.diagnostics)

    @property
    def errors(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity == "error"]

    def add(self, severity: Severity, message: str, node_id=None, field=None) -> None:
        self.diagnostics.append(Diagnostic(severity, message, node_id, field))

    def report(self) -> str:
        if not self.diagnostics:
            return "No problems found."
        return "\n".join(str(d) for d in self.diagnostics)


# --------------------------------------------------------------------------
# Layer 1: structure
# --------------------------------------------------------------------------


def check_structure(graph: NetworkGraph, result: ValidationResult) -> None:
    """Referential integrity: every edge points at a population that exists."""
    pop_ids = graph.population_ids()

    # Ids must be unique within their kind, and across kinds too: the builder
    # derives Brian2 object names from them, and two objects with the same name
    # fail deep inside Brian2 with a message that points nowhere useful.
    seen: dict[str, str] = {}
    for kind, items in (("population", graph.populations), ("synapse", graph.synapses),
                        ("input", graph.stimuli), ("monitor", graph.monitors)):
        for item in items:
            if item.id in seen:
                result.add(
                    "error",
                    f"Duplicate id {item.id!r}: already used by a "
                    f"{seen[item.id]}. Ids must be unique across the graph.",
                    item.id, "id",
                )
            else:
                seen[item.id] = kind

    for syn in graph.synapses:
        for end, pid in (("source", syn.source), ("target", syn.target)):
            if pid not in pop_ids:
                result.add(
                    "error",
                    f"Synapse {end} {pid!r} does not match any population.",
                    syn.id,
                    end,
                )
        if syn.condition is None and syn.probability is None:
            result.add(
                "warning",
                "No condition and no probability: this connects all-to-all.",
                syn.id,
                "condition",
            )
        if syn.on_pre is None and syn.on_post is None:
            result.add(
                "warning",
                "Synapse has no on_pre/on_post pathway, so it transmits nothing.",
                syn.id,
                "on_pre",
            )

    for stim in graph.stimuli:
        if stim.target not in pop_ids:
            result.add(
                "error",
                f"Stimulus target {stim.target!r} does not match any population.",
                stim.id,
                "target",
            )

    for mon in graph.monitors:
        if mon.target not in pop_ids:
            result.add(
                "error",
                f"Monitor target {mon.target!r} does not match any population.",
                mon.id,
                "target",
            )
        if mon.kind == "state" and not mon.variables:
            result.add(
                "error", "A state monitor must name at least one variable.",
                mon.id, "variables",
            )

    if not graph.populations:
        result.add("error", "The network has no populations.")


# --------------------------------------------------------------------------
# Layer 2: static checks + security allowlist
# --------------------------------------------------------------------------


def _check_names(
    mapping: dict, node_id: str, field_name: str, result: ValidationResult
) -> None:
    """Check only the keys. Used where the values are Brian2 expressions.

    Initial values like "rand()*20*mV" or "V_r" are evaluated by Brian2 against
    the group, not by the unit parser, so running them through
    safe_eval_quantity would reject perfectly good input.
    """
    for key in mapping:
        if not isinstance(key, str) or not key.isidentifier():
            result.add(
                "error",
                f"{key!r} is not a valid name. Use letters, digits and "
                f"underscores, not starting with a digit.",
                node_id, f"{field_name}.{key}",
            )


#: Stimulus parameters whose values are code or names, not quantities.
_STIM_TEXT_KEYS = {"on_pre", "var_name", "unit"}


def _check_param_strings(
    params: dict, node_id: str, field_name: str, result: ValidationResult
) -> None:
    for key, raw in params.items():
        # The key becomes a Brian2 namespace entry and, on export, a Python
        # variable name. A key like "tau m" is silently ignored by the namespace
        # lookup -- the equation then fails with "unknown identifier tau_m"
        # pointing at the equation rather than at the typo that caused it.
        if not isinstance(key, str) or not key.isidentifier():
            result.add(
                "error",
                f"{key!r} is not a valid name. Use letters, digits and "
                f"underscores, not starting with a digit.",
                node_id, f"{field_name}.{key}",
            )
            continue
        if not isinstance(raw, str):
            continue
        try:
            safe_eval_quantity(raw)
        except UnsafeExpression as exc:
            result.add("error", str(exc), node_id, f"{field_name}.{key}")
        except Exception as exc:
            # Anything the arithmetic itself can throw -- ZeroDivisionError from
            # "1/0", OverflowError from a huge power. Validation is user-facing
            # and must always come back as a diagnostic, never as a 500.
            result.add("error", f"{type(exc).__name__}: {exc}",
                       node_id, f"{field_name}.{key}")


#: Numeric stimulus parameters that Brian2 will not complain about but that are
#: meaningless outside these ranges. `PoissonInput(N=-5)` and `connect(p=3.0)`
#: both run without error and both mean nothing.
_STIM_RANGES = {
    "n_inputs": (1, None, "a positive whole number of inputs"),
    "probability": (0.0, 1.0, "a connection probability between 0 and 1"),
}


def _check_stimulus_params(stim, result: ValidationResult) -> None:
    for key, (lo, hi, what) in _STIM_RANGES.items():
        if key not in stim.params:
            continue
        raw = stim.params[key]
        try:
            value = float(raw)
        except (TypeError, ValueError):
            result.add("error", f"{key} must be {what}, got {raw!r}.",
                       stim.id, f"params.{key}")
            continue
        if not math.isfinite(value) or value < lo or (hi is not None and value > hi):
            result.add("error", f"{key} must be {what}, got {raw!r}.",
                       stim.id, f"params.{key}")

    if stim.kind == "spike_train":
        indices = stim.params.get("indices")
        times = stim.params.get("times_ms")
        if indices is not None and times is not None:
            try:
                if len(indices) != len(times):
                    result.add(
                        "error",
                        f"indices has {len(indices)} entries but times_ms has "
                        f"{len(times)}; they describe the same spikes.",
                        stim.id, "params.times_ms",
                    )
            except TypeError:
                result.add("error", "indices and times_ms must both be lists.",
                           stim.id, "params.indices")


def _check_identifiers(
    expr: str, declared: set[str], node_id: str, field_name: str,
    result: ValidationResult,
) -> None:
    """Allowlist: every name in a user string must be declared or built in.

    This is a positive allowlist, not a blocklist of scary words. Blocklists
    lose; something like `__import__` fails here not because it is on a list of
    bad names but because it is on no list of good ones.
    """
    for name in sorted(get_identifiers(expr)):
        if name in declared or name in _BUILTIN_NAMES:
            continue
        if name.startswith("d") and name[1:] in declared:
            continue  # dv in "dv/dt"
        if name.startswith("__"):
            result.add(
                "error",
                f"Identifier {name!r} is not allowed in model strings.",
                node_id, field_name,
            )
        else:
            result.add(
                "error",
                f"Unknown identifier {name!r}. Declare it in this node's "
                f"params, in the network globals, or as a model variable.",
                node_id, field_name,
            )


def check_static(graph: NetworkGraph, result: ValidationResult) -> None:
    from builder import stimulus_variable

    global_names = set(graph.globals)
    _check_param_strings(graph.globals, "network", "globals", result)

    # Continuous stimuli inject a variable (default I_ext) into their target's
    # equations at build time. The user's equations may reference it, so it
    # counts as declared even though it does not appear in their text.
    injected: dict[str, set[str]] = {}
    for stim in graph.stimuli:
        var = stimulus_variable(stim)
        if not var:
            continue
        # Both the injected variable name and its unit are pasted verbatim into
        # the target's equation text, so they have to look like identifiers.
        for fname, value in (("var_name", var), ("unit", stim.params.get("unit", "volt"))):
            if not isinstance(value, str) or not value.isidentifier():
                result.add(
                    "error",
                    f"Input {fname} must be a plain identifier, got {value!r}.",
                    stim.id, f"params.{fname}",
                )
        bag = injected.setdefault(stim.target, set())
        if var in bag:
            result.add(
                "error",
                f"Two continuous inputs both inject {var!r} into "
                f"{stim.target!r}. Give one of them a different var_name.",
                stim.id, "params.var_name",
            )
        bag.add(var)

    for stim in graph.stimuli:
        _check_stimulus_params(stim, result)
        _check_param_strings(
            {k: v for k, v in stim.params.items()
             if isinstance(v, str) and k not in _STIM_TEXT_KEYS},
            stim.id, "params", result,
        )

    parsed: dict[str, Equations] = {}

    for pop in graph.populations:
        _check_param_strings(pop.params, pop.id, "params", result)
        _check_names(pop.initial, pop.id, "initial", result)

        try:
            eqs = Equations(pop.equations)
            parsed[pop.id] = eqs
        except Exception as exc:  # EquationError and friends
            result.add("error", f"Equation syntax: {exc}", pop.id, "equations")
            continue

        declared = (
            set(eqs.names) | set(pop.params) | global_names | injected.get(pop.id, set())
        )
        _check_identifiers(pop.equations, declared, pop.id, "equations", result)

        for fname, expr in (
            ("threshold", pop.threshold),
            ("reset", pop.reset),
            ("refractory", pop.refractory),
        ):
            if expr:
                _check_identifiers(expr, declared, pop.id, fname, result)

        for var, expr in pop.initial.items():
            if var not in eqs.names:
                result.add(
                    "error",
                    f"Initial value set for {var!r}, which the equations do not define.",
                    pop.id, "initial",
                )
            _check_identifiers(str(expr), declared, pop.id, f"initial.{var}", result)

        if pop.reset and not pop.threshold:
            result.add(
                "warning",
                "A reset without a threshold never fires.",
                pop.id, "reset",
            )

    # Synapses see their own model variables plus _pre/_post views of both ends.
    by_id = {p.id: p for p in graph.populations}
    for syn in graph.synapses:
        _check_param_strings(syn.params, syn.id, "params", result)
        _check_names(syn.initial, syn.id, "initial", result)
        src, tgt = by_id.get(syn.source), by_id.get(syn.target)
        if src is None or tgt is None:
            continue  # already reported in layer 1

        declared = set(syn.params) | global_names
        try:
            if syn.model.strip():
                declared |= set(Equations(syn.model).names)
        except Exception as exc:
            result.add("error", f"Synapse model syntax: {exc}", syn.id, "model")
            continue

        for pop, suffix in ((src, "_pre"), (tgt, "_post")):
            if pop.id in parsed:
                names = set(parsed[pop.id].names)
                declared |= {f"{n}{suffix}" for n in names} | names

        for fname in ("on_pre", "on_post", "condition"):
            expr = getattr(syn, fname)
            if expr:
                _check_identifiers(expr, declared, syn.id, fname, result)

    # An event-driven input carries a custom on_pre just like a synapse does,
    # and it reaches the same code generator, so it earns the same allowlist.
    # Without this it still fails, but as a raw Brian2 KeyError from the dry run
    # rather than a message naming the input and the offending identifier.
    for stim in graph.stimuli:
        on_pre = stim.params.get("on_pre")
        if not isinstance(on_pre, str) or not on_pre.strip():
            continue
        target = by_id.get(stim.target)
        if target is None or target.id not in parsed:
            continue
        names = set(parsed[target.id].names)
        declared = (
            names
            | {f"{n}_post" for n in names}
            | {k for k in stim.params if isinstance(k, str) and k.isidentifier()}
            | global_names
            | {"w_ext"}
        )
        _check_identifiers(on_pre, declared, stim.id, "params.on_pre", result)


# --------------------------------------------------------------------------
# Layer 3: dry run
# --------------------------------------------------------------------------


def _root_cause(exc: BaseException) -> BaseException:
    """Brian2 wraps errors in BrianObjectException; dig out the real one."""
    seen = set()
    while (exc.__cause__ or exc.__context__) is not None:
        nxt = exc.__cause__ or exc.__context__
        if id(nxt) in seen:
            break
        seen.add(id(nxt))
        exc = nxt
    return exc


def _blame_node(exc: BaseException, graph: NetworkGraph) -> str | None:
    """Work out which node on the canvas an exception belongs to.

    Brian2's BrianObjectException carries `_brian_objname`, but it names the
    internal sub-object rather than the group: a bad threshold on population
    'exc' surfaces as 'exc_spike_thresholder'. Since the builder derives every
    Brian2 name from the node id, the longest matching prefix wins.

    Without this, unit errors arrive with node_id=None and the UI can report
    them in the status bar but cannot highlight the node that caused them.
    """
    from builder import _safe_name

    name = None
    cursor: BaseException | None = exc
    seen = set()
    while cursor is not None and id(cursor) not in seen:
        seen.add(id(cursor))
        candidate = getattr(cursor, "_brian_objname", None)
        if candidate:
            name = candidate
            break
        cursor = cursor.__cause__ or cursor.__context__

    if not name:
        return None

    best_id, best_len = None, 0
    for node in [*graph.populations, *graph.synapses, *graph.stimuli]:
        safe = _safe_name(node.id)
        if name.startswith(safe) and len(safe) > best_len:
            best_id, best_len = node.id, len(safe)
    return best_id


def check_dry_run(graph: NetworkGraph, result: ValidationResult) -> None:
    """Build the real Brian2 objects and run for zero seconds.

    Only reached when layers 1 and 2 passed, so the strings handed to Brian2's
    code generator have already cleared the allowlist.
    """
    from brian2 import ms  # local import: keeps module import cheap

    from builder import BuildError, build_network

    try:
        built = build_network(graph)
    except BuildError as exc:
        result.add("error", str(exc), exc.node_id)
        return
    except Exception as exc:
        root = _root_cause(exc)
        result.add("error", f"{type(root).__name__}: {root}", _blame_node(exc, graph))
        return

    try:
        built.network.run(0 * ms)
    except Exception as exc:
        root = _root_cause(exc)
        result.add(
            "error",
            f"{type(root).__name__}: {root}",
            _blame_node(exc, graph),
            "units",
        )


# --------------------------------------------------------------------------


def validate(graph: NetworkGraph, dry_run: bool = True) -> ValidationResult:
    """Run all layers, stopping before the expensive one if anything failed."""
    result = ValidationResult()

    check_structure(graph, result)
    if not result.ok:
        return result

    check_static(graph, result)
    if not result.ok:
        return result

    if dry_run:
        check_dry_run(graph, result)

    return result
