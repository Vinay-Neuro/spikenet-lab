"""
Graph -> Brian2 objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from brian2 import (
    Equations,
    Network,
    NeuronGroup,
    PoissonGroup,
    PoissonInput,
    PopulationRateMonitor,
    SpikeGeneratorGroup,
    SpikeMonitor,
    StateMonitor,
    Synapses,
    defaultclock,
    prefs,
    second,
)

from safe_units import safe_eval_quantity
from schema import NetworkGraph, NeuronNode, StimulusNode


class BuildError(RuntimeError):
    """Assembly failed. Carries the node id so the UI can highlight it."""

    def __init__(self, message: str, node_id: str | None = None) -> None:
        super().__init__(message)
        self.node_id = node_id


def _set_variables(obj, values: dict[str, str], node_id: str) -> None:
    """Apply initial values, preferring our safe parser, falling back to Brian2.

    A plain number or unit expression ("10*mV") is parsed by safe_eval_quantity.
    Anything else ("rand()*20*mV", "i*0.1*mV") is a Brian2 expression string and
    is handed to Brian2 to evaluate against the group, which is what makes
    heterogeneous initialisation possible.
    """
    for var, expr in values.items():
        try:
            setattr(obj, var, safe_eval_quantity(expr))
            continue
        except Exception:
            pass
        try:
            setattr(obj, var, expr)
        except Exception as exc:
            raise BuildError(
                f"initial value for {var!r}: {exc}", node_id
            ) from exc


@dataclass
class BuiltNetwork:
    """A live Brian2 Network plus the lookup tables needed to read it back."""

    network: Network
    groups: dict[str, NeuronGroup] = field(default_factory=dict)
    synapses: dict[str, Synapses] = field(default_factory=dict)
    monitors: dict[str, Any] = field(default_factory=dict)
    namespace: dict[str, Any] = field(default_factory=dict)


def _resolve(params: dict[str, str]) -> dict[str, Any]:
    """Turn {"tau_m": "20*ms"} into {"tau_m": 20*ms} without eval()."""
    return {k: safe_eval_quantity(v) for k, v in params.items()}


# --------------------------------------------------------------------------
# Stimulus equations
# --------------------------------------------------------------------------


#: Stimulus kinds that contribute an equation line rather than a separate object.
CONTINUOUS_KINDS = {"dc", "step", "sinusoid", "chirp", "ou_noise"}


def stimulus_variable(stim: StimulusNode) -> str | None:
    """The variable a continuous stimulus injects, or None for event-driven ones.

    Validation needs this: the population's equations legitimately reference
    `I_ext`, but that name is generated here, not written by the user, so the
    identifier allowlist would otherwise reject the user's own equation.
    """
    if stim.kind in CONTINUOUS_KINDS:
        return stim.params.get("var_name", "I_ext")
    return None


def _stimulus_equation(stim: StimulusNode) -> tuple[str, dict[str, Any]]:
    """Build the extra equation line a time-varying stimulus contributes.

    The generated variable (default `I_ext`) is referenced by the population's
    own equations, which is the idiomatic Brian2 pattern:

        dv/dt = (-v + I_ext)/tau_m : volt      <- written by the user
        I_ext = amplitude*sin(2*pi*freq*t) : volt   <- generated here

    Returns (equation_text, namespace_fragment). Event-driven stimuli
    (poisson, spike_train) return ("", {}): they attach as separate objects.
    """
    p = stim.params
    var = p.get("var_name", "I_ext")
    unit = p.get("unit", "volt")
    ns: dict[str, Any] = {}

    def q(key: str, default: str | None = None):
        raw = p.get(key, default)
        if raw is None:
            raise BuildError(f"Stimulus {stim.id!r} ({stim.kind}) needs {key!r}.", stim.id)
        return safe_eval_quantity(raw)

    pre = f"{stim.id}_"  # namespace prefix, so two stimuli cannot collide

    if stim.kind == "dc":
        ns[pre + "amp"] = q("amplitude")
        return f"{var} = {pre}amp : {unit}", ns

    if stim.kind == "step":
        ns[pre + "amp"] = q("amplitude")
        ns[pre + "t0"] = q("t_start", "0*second")
        ns[pre + "t1"] = q("t_stop", "1e9*second")
        eq = (
            f"{var} = {pre}amp * int(t >= {pre}t0) * int(t < {pre}t1) : {unit}"
        )
        return eq, ns

    if stim.kind == "sinusoid":
        ns[pre + "amp"] = q("amplitude")
        ns[pre + "freq"] = q("frequency")
        ns[pre + "off"] = q("offset", "0*" + unit)
        eq = (
            f"{var} = {pre}off + {pre}amp * sin(2*pi*{pre}freq*t) : {unit}"
        )
        return eq, ns

    if stim.kind == "chirp":
        # Linear sweep f(t) = f0 + k*t, so phase = 2*pi*(f0*t + k*t^2/2).
        # One run sweeps the whole band -- the cheapest way to read a
        # resonance curve straight off a single simulation.
        f0 = q("f_start")
        f1 = q("f_stop")
        dur = q("duration")
        ns[pre + "amp"] = q("amplitude")
        ns[pre + "f0"] = f0
        ns[pre + "k"] = (f1 - f0) / dur
        eq = (
            f"{var} = {pre}amp * sin(2*pi*({pre}f0*t + 0.5*{pre}k*t*t)) : {unit}"
        )
        return eq, ns

    if stim.kind == "ou_noise":
        ns[pre + "mu"] = q("mean", "0*" + unit)
        ns[pre + "sigma"] = q("sigma")
        ns[pre + "tau"] = q("tau", "5*ms")
        eq = (
            f"d{var}/dt = ({pre}mu - {var})/{pre}tau "
            f"+ {pre}sigma*sqrt(2/{pre}tau)*xi : {unit}"
        )
        return eq, ns

    return "", {}


# --------------------------------------------------------------------------
# Populations
# --------------------------------------------------------------------------


def _build_population(
    pop: NeuronNode, stimuli: list[StimulusNode], globals_ns: dict[str, Any]
) -> tuple[NeuronGroup, dict[str, Any]]:
    namespace = dict(globals_ns)
    namespace.update(_resolve(pop.params))

    eq_text = pop.equations
    for stim in stimuli:
        line, frag = _stimulus_equation(stim)
        if line:
            eq_text = eq_text.rstrip() + "\n" + line
            namespace.update(frag)

    try:
        equations = Equations(eq_text)
    except Exception as exc:
        raise BuildError(f"Population {pop.id!r}: {exc}", pop.id) from exc

    kwargs: dict[str, Any] = {
        "model": equations,
        "method": pop.method,
        "namespace": namespace,
        "name": _safe_name(pop.id),
    }
    if pop.threshold:
        kwargs["threshold"] = pop.threshold
    if pop.reset:
        kwargs["reset"] = pop.reset
    if pop.refractory:
        kwargs["refractory"] = safe_eval_quantity(pop.refractory)

    group = NeuronGroup(pop.n, **kwargs)

    _set_variables(group, pop.initial, pop.id)

    return group, namespace


def _safe_name(node_id: str) -> str:
    """Brian2 object names must be valid identifiers."""
    cleaned = "".join(c if c.isalnum() or c == "_" else "_" for c in node_id)
    return cleaned if cleaned[:1].isalpha() else f"n_{cleaned}"


# --------------------------------------------------------------------------
# Whole network
# --------------------------------------------------------------------------


def build_network(graph: NetworkGraph) -> BuiltNetwork:
    """Assemble every node and edge into a runnable Brian2 Network."""
    if graph.run.codegen_target != "auto":
        prefs.codegen.target = graph.run.codegen_target
    defaultclock.dt = safe_eval_quantity(graph.run.dt)

    globals_ns = _resolve(graph.globals)
    built = BuiltNetwork(network=Network(), namespace=globals_ns)

    stim_by_target: dict[str, list[StimulusNode]] = {}
    for stim in graph.stimuli:
        stim_by_target.setdefault(stim.target, []).append(stim)

    # --- populations -----------------------------------------------------
    for pop in graph.populations:
        group, ns = _build_population(pop, stim_by_target.get(pop.id, []), globals_ns)
        built.groups[pop.id] = group
        built.network.add(group)

    # --- synapses --------------------------------------------------------
    for syn in graph.synapses:
        try:
            src = built.groups[syn.source]
            tgt = built.groups[syn.target]
        except KeyError as exc:
            raise BuildError(f"Synapse {syn.id!r} references missing {exc}.", syn.id) from exc

        ns = dict(globals_ns)
        ns.update(_resolve(syn.params))

        kwargs: dict[str, Any] = {"namespace": ns, "name": _safe_name(syn.id)}
        if syn.model.strip():
            kwargs["model"] = syn.model
        if syn.on_pre:
            kwargs["on_pre"] = syn.on_pre
        if syn.on_post:
            kwargs["on_post"] = syn.on_post
        if syn.delay:
            kwargs["delay"] = safe_eval_quantity(syn.delay)

        obj = Synapses(src, tgt, **kwargs)

        if syn.condition:
            obj.connect(condition=syn.condition, n=syn.n_per_pair)
        elif syn.probability is not None:
            obj.connect(p=syn.probability, n=syn.n_per_pair)
        else:
            obj.connect(n=syn.n_per_pair)

        _set_variables(obj, syn.initial, syn.id)

        built.synapses[syn.id] = obj
        built.network.add(obj)

    # --- event-driven stimuli --------------------------------------------
    for stim in graph.stimuli:
        target = built.groups[stim.target]
        p = stim.params

        if stim.kind == "poisson":
            src = PoissonInput(
                target,
                target_var=stim.target_var,
                N=int(p.get("n_inputs", 1)),
                rate=safe_eval_quantity(p["rate"]),
                weight=safe_eval_quantity(p["weight"]),
            )
            built.network.add(src)

        elif stim.kind == "poisson_group":
            n = int(p.get("n_inputs", 100))
            pg = PoissonGroup(n, rates=safe_eval_quantity(p["rate"]),
                              name=_safe_name(stim.id))
            conn = Synapses(
                pg, target,
                on_pre=p.get("on_pre", f"{stim.target_var} += w_ext"),
                namespace={"w_ext": safe_eval_quantity(p["weight"])},
                name=_safe_name(stim.id) + "_syn",
            )
            conn.connect(p=float(p.get("probability", 1.0)))
            built.network.add(pg, conn)

        elif stim.kind == "spike_train":
            indices = np.asarray(p.get("indices", []), dtype=int)
            times = np.asarray(p.get("times_ms", []), dtype=float)
            n = int(p.get("n_inputs", max(int(indices.max()) + 1, 1) if len(indices) else 1))
            sg = SpikeGeneratorGroup(
                n, indices, times * 1e-3 * second, name=_safe_name(stim.id)
            )
            conn = Synapses(
                sg, target,
                on_pre=p.get("on_pre", f"{stim.target_var} += w_ext"),
                namespace={"w_ext": safe_eval_quantity(p["weight"])},
                name=_safe_name(stim.id) + "_syn",
            )
            conn.connect(p=float(p.get("probability", 1.0)))
            built.network.add(sg, conn)

    # --- monitors --------------------------------------------------------
    for mon in graph.monitors:
        target = built.groups[mon.target]
        name = _safe_name(mon.id)

        if mon.kind == "spike":
            obj = SpikeMonitor(target, record=mon.record, name=name)
        elif mon.kind == "rate":
            obj = PopulationRateMonitor(target, name=name)
        else:
            kw: dict[str, Any] = {"record": mon.record, "name": name}
            if mon.dt:
                kw["dt"] = safe_eval_quantity(mon.dt)
            obj = StateMonitor(target, mon.variables, **kw)

        built.monitors[mon.id] = obj
        built.network.add(obj)

    return built


# --------------------------------------------------------------------------
# One-way script export
# --------------------------------------------------------------------------


def _lit(text: str) -> str:
    """Quote a user string as a Python literal.
    """
    return repr(text)


def _stimulus_lines(graph: NetworkGraph) -> list[str]:
    """Render the external inputs.
    """
    lines: list[str] = []
    for stim in graph.stimuli:
        name = _safe_name(stim.id)
        target = _safe_name(stim.target)
        p = stim.params
        lines.append(f"# --- input: {stim.label} ({stim.kind}) ---")

        if stim.kind in CONTINUOUS_KINDS:
            var = p.get("var_name", "I_ext")
            eq, ns = _stimulus_equation(stim)
            lines.append(
                f"# {var} is added to {target}'s equations; the group above must "
                f"reference it."
            )
            for key, value in ns.items():
                lines.append(f"{key} = {value!r}")
            lines.append(f"# appended equation: {eq}")
            lines.append("")
            continue

        if stim.kind == "poisson":
            lines.append(
                f"{name} = PoissonInput({target}, {_lit(stim.target_var)}, "
                f"N={int(p.get('n_inputs', 1))}, rate={p.get('rate')}, "
                f"weight={p.get('weight')})"
            )
        elif stim.kind == "poisson_group":
            lines.append(f"{name} = PoissonGroup({int(p.get('n_inputs', 100))}, "
                         f"rates={p.get('rate')})")
            lines.append(f"w_ext_{name} = {p.get('weight')}")
            on_pre = p.get("on_pre", f"{stim.target_var} += w_ext_{name}")
            lines.append(f"{name}_syn = Synapses({name}, {target}, "
                         f"on_pre={_lit(on_pre)})")
            lines.append(f"{name}_syn.connect(p={float(p.get('probability', 1.0))})")
        elif stim.kind == "spike_train":
            lines.append(f"{name} = SpikeGeneratorGroup("
                         f"{int(p.get('n_inputs', 1))}, "
                         f"{list(p.get('indices', []))}, "
                         f"{list(p.get('times_ms', []))}*ms)")
            lines.append(f"w_ext_{name} = {p.get('weight')}")
            on_pre = p.get("on_pre", f"{stim.target_var} += w_ext_{name}")
            lines.append(f"{name}_syn = Synapses({name}, {target}, "
                         f"on_pre={_lit(on_pre)})")
            lines.append(f"{name}_syn.connect(p={float(p.get('probability', 1.0))})")
        lines.append("")
    return lines


def export_script(graph: NetworkGraph) -> str:
    """Render the graph as a standalone Brian2 script the user can keep.
    """
    L = [
        "from brian2 import *",
        "",
        f"# {graph.name}",
    ]
    if graph.run.codegen_target != "auto":
        # Without this the script uses Brian2's default target while the app
        # used the one set here. Different targets draw from different random
        # streams, so the same seed produces a different spike count and the
        # exported script quietly fails to reproduce what you just watched.
        L.append(f"prefs.codegen.target = {_lit(graph.run.codegen_target)}")
    L.append(f"defaultclock.dt = {graph.run.dt}")
    if graph.run.seed is not None:
        L.append(f"seed({graph.run.seed})")
    L.append("")

    if graph.globals:
        L.append("# --- global constants ---")
        L += [f"{k} = {v}" for k, v in graph.globals.items()]
        L.append("")

    # Continuous stimuli append a line to their target's equations, so the
    # population has to be rendered with that line already in place.
    extra_eqs: dict[str, list[str]] = {}
    for stim in graph.stimuli:
        if stim.kind in CONTINUOUS_KINDS:
            eq, _ = _stimulus_equation(stim)
            if eq:
                extra_eqs.setdefault(stim.target, []).append(eq)

    for pop in graph.populations:
        name = _safe_name(pop.id)
        L.append(f"# --- population: {pop.label} ---")
        L += [f"{k} = {v}" for k, v in pop.params.items()]
        body = pop.equations.strip()
        for eq in extra_eqs.get(pop.id, []):
            body += "\n" + eq
        L.append(f"{name}_eqs = {_lit(body)}")
        args = [str(pop.n), f"{name}_eqs"]
        if pop.threshold:
            args.append(f"threshold={_lit(pop.threshold)}")
        if pop.reset:
            args.append(f"reset={_lit(pop.reset.strip())}")
        if pop.refractory:
            args.append(f"refractory={pop.refractory}")
        args.append(f"method={_lit(pop.method)}")
        L.append(f"{name} = NeuronGroup({', '.join(args)})")
        for var, expr in pop.initial.items():
            L.append(f"{name}.{var} = {expr}")
        L.append("")

    for syn in graph.synapses:
        name = _safe_name(syn.id)
        L.append(f"# --- synapse: {syn.label} ---")
        L += [f"{k} = {v}" for k, v in syn.params.items()]
        args = [_safe_name(syn.source), _safe_name(syn.target)]
        if syn.model.strip():
            args.append(f"model={_lit(syn.model.strip())}")
        if syn.on_pre:
            args.append(f"on_pre={_lit(syn.on_pre)}")
        if syn.on_post:
            args.append(f"on_post={_lit(syn.on_post)}")
        if syn.delay:
            args.append(f"delay={syn.delay}")
        L.append(f"{name} = Synapses({', '.join(args)})")
        if syn.condition:
            L.append(f"{name}.connect(condition={_lit(syn.condition)}, n={syn.n_per_pair})")
        elif syn.probability is not None:
            L.append(f"{name}.connect(p={syn.probability}, n={syn.n_per_pair})")
        else:
            L.append(f"{name}.connect(n={syn.n_per_pair})")
        for var, expr in syn.initial.items():
            L.append(f"{name}.{var} = {expr}")
        L.append("")

    L += _stimulus_lines(graph)

    if graph.monitors:
        L.append("# --- recording ---")
    for mon in graph.monitors:
        name, target = _safe_name(mon.id), _safe_name(mon.target)
        if mon.kind == "spike":
            L.append(f"{name} = SpikeMonitor({target})")
        elif mon.kind == "rate":
            L.append(f"{name} = PopulationRateMonitor({target})")
        else:
            extra = f", dt={mon.dt}" if mon.dt else ""
            L.append(f"{name} = StateMonitor({target}, {mon.variables!r}, "
                     f"record={mon.record}{extra})")

    L += ["", f"run({graph.run.duration})"]
    return "\n".join(L)
