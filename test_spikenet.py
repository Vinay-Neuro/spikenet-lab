import copy
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from brian2 import BrianLogger

from spikenet import (
    NetworkGraph,
    UnsafeExpression,
    amplitude_at_frequency,
    classify_regime,
    export_script,
    rate_spectrum,
    resonant_frequency,
    safe_eval_quantity,
    spectral_entropy,
    spike_statistics,
    synchrony_index,
    validate,
)

BrianLogger.suppress_name("resolution_conflict")
BrianLogger.suppress_name("method_choice")

EXAMPLE = Path(__file__).resolve().parent / "brunel.json"


@pytest.fixture
def graph() -> NetworkGraph:
    return NetworkGraph.model_validate(json.loads(EXAMPLE.read_text()))


# --------------------------------------------------------------------------
# safe_units
# --------------------------------------------------------------------------


def test_parses_unit_expressions():
    from brian2 import Hz, mV, ms, second

    assert safe_eval_quantity("20*ms") == 20 * ms
    assert safe_eval_quantity("0.1*mV") == 0.1 * mV
    assert safe_eval_quantity("4.5") == 4.5
    assert safe_eval_quantity("1/second") == 1 / second
    assert safe_eval_quantity("2*20*Hz") == 40 * Hz


@pytest.mark.parametrize(
    "payload",
    [
        '__import__("os").system("id")',
        "open('/etc/passwd').read()",
        "(lambda: 1)()",
        "[x for x in range(10)]",
        "ms.__class__.__mro__",
        "eval('1+1')",
        "globals()",
    ],
)
def test_rejects_code_in_parameters(payload):
    with pytest.raises(UnsafeExpression):
        safe_eval_quantity(payload)


def test_rejects_unknown_unit():
    with pytest.raises(UnsafeExpression):
        safe_eval_quantity("20*parsecs")


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def test_reference_graph_is_valid(graph):
    assert validate(graph).ok


def test_catches_unit_mismatch_in_threshold(graph):
    graph.populations[0].threshold = "v > 20"
    report = validate(graph)
    assert not report.ok
    assert any("unit" in d.message.lower() for d in report.errors)


def test_catches_unit_mismatch_in_synapse(graph):
    graph.synapses[0].on_pre = "s_e_post += 1*amp"
    assert not validate(graph).ok


def test_catches_undeclared_identifier(graph):
    graph.populations[0].threshold = "v > theta_typo"
    report = validate(graph, dry_run=False)
    assert not report.ok
    assert any("theta_typo" in d.message for d in report.errors)


def test_catches_dangling_synapse(graph):
    graph.synapses[0].target = "nope"
    report = validate(graph, dry_run=False)
    assert not report.ok
    assert report.errors[0].node_id == "ee"


def test_catches_injection_in_model_string(graph):
    graph.populations[0].reset = '__import__("os").system("id")'
    assert not validate(graph, dry_run=False).ok


def test_equation_syntax_error_is_reported(graph):
    graph.populations[0].equations = "dv/dt = (-v)/tau_m volt"
    report = validate(graph, dry_run=False)
    assert not report.ok
    assert report.errors[0].field == "equations"


def test_warns_on_reset_without_threshold(graph):
    graph.populations[0].threshold = None
    report = validate(graph, dry_run=False)
    assert any(d.severity == "warning" for d in report.diagnostics)


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------


def _synthetic_rate(freq=20.0, depth=0.6, rate=10.0, n=1000, dt=1e-4, dur=2.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur / dt)) * dt
    modulated = rate * (1 + depth * np.sin(2 * np.pi * freq * t))
    counts = rng.poisson(n * modulated * dt)
    return t, counts / (n * dt)


def test_spectrum_finds_the_injected_frequency():
    t, r = _synthetic_rate(freq=20.0)
    freqs, amps = rate_spectrum(t, r)
    peak = resonant_frequency(freqs, amps, low=1.0, high=100.0)
    assert abs(peak.peak_frequency - 20.0) < 1.0


def test_entropy_is_lower_for_a_tuned_spectrum():
    t, tuned = _synthetic_rate(depth=0.9)
    t2, flat = _synthetic_rate(depth=0.0)
    f1, a1 = rate_spectrum(t, tuned)
    f2, a2 = rate_spectrum(t2, flat)
    assert spectral_entropy(f1, a1) < spectral_entropy(f2, a2)


def test_entropy_is_bounded():
    t, r = _synthetic_rate(depth=0.0)
    freqs, amps = rate_spectrum(t, r)
    assert 0.0 <= spectral_entropy(freqs, amps) <= 1.0


def test_synchrony_correction_zeroes_an_asynchronous_population():
    """The whole point of the correction: no false positive on Poisson noise."""
    t, r = _synthetic_rate(depth=0.0, rate=10.0)
    naive = synchrony_index(t, r, 1000, finite_size_correction=False)
    fixed = synchrony_index(t, r, 1000)
    assert naive > 0.3          # uncorrected measure is badly fooled
    assert fixed < 0.1          # corrected measure is not


def test_synchrony_recovers_known_modulation_depth():
    """A sinusoid of depth d has CV = d/sqrt(2)."""
    depth = 0.6
    t, r = _synthetic_rate(depth=depth, rate=20.0)
    assert abs(synchrony_index(t, r, 1000) - depth / np.sqrt(2)) < 0.05


def test_spike_statistics_on_a_poisson_process():
    rng = np.random.default_rng(1)
    n, dur, rate = 200, 5.0, 20.0
    times, indices = [], []
    for neuron in range(n):
        k = rng.poisson(rate * dur)
        times.append(np.sort(rng.uniform(0, dur, k)))
        indices.append(np.full(k, neuron))
    st, si = np.concatenate(times), np.concatenate(indices)

    stats = spike_statistics(st, si, n, dur)
    assert abs(stats.mean_rate - rate) < 1.0
    assert abs(stats.cv_isi - 1.0) < 0.15      # Poisson => CV ~ 1
    assert abs(stats.fano_factor - 1.0) < 0.4  # Poisson => Fano ~ 1


def test_regime_labels():
    assert "AI" in classify_regime(cv_isi=1.0, sync=0.05)
    assert "SI" in classify_regime(cv_isi=1.0, sync=0.9)
    assert "SR" in classify_regime(cv_isi=0.1, sync=0.9)
    assert "AR" in classify_regime(cv_isi=0.1, sync=0.05)
    assert classify_regime(np.nan, 0.5) == "insufficient activity"


# --------------------------------------------------------------------------
# round trips
# --------------------------------------------------------------------------


def test_graph_json_round_trip(graph):
    restored = NetworkGraph.model_validate(json.loads(graph.model_dump_json()))
    assert restored == graph


def test_export_script_mentions_every_node(graph):
    script = export_script(graph)
    assert "from brian2 import *" in script
    for pop in graph.populations:
        assert pop.id in script
    for syn in graph.synapses:
        assert syn.id in script


def test_positions_survive_the_round_trip(graph):
    """Canvas layout must persist -- users lose trust fast if it does not."""
    graph.populations[0].position = {"x": 42.0, "y": 7.0}
    restored = NetworkGraph.model_validate(json.loads(graph.model_dump_json()))
    assert restored.populations[0].position == {"x": 42.0, "y": 7.0}


# --------------------------------------------------------------------------
# stimuli
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind,params",
    [
        ("dc", {"amplitude": "2*mV"}),
        ("step", {"amplitude": "2*mV", "t_start": "100*ms", "t_stop": "500*ms"}),
        ("sinusoid", {"amplitude": "2*mV", "frequency": "20*Hz"}),
        ("chirp", {"amplitude": "2*mV", "f_start": "1*Hz", "f_stop": "80*Hz",
                   "duration": "1*second"}),
        ("ou_noise", {"sigma": "1*mV", "tau": "5*ms"}),
    ],
)
def test_time_varying_stimuli_build_and_validate(graph, kind, params):
    """Each stimulus kind must produce equations Brian2 accepts dimensionally."""
    g = copy.deepcopy(graph)
    g.stimuli = [s for s in g.stimuli if s.target != "exc"]
    g.populations[0].equations = g.populations[0].equations.replace(
        "dv/dt = (-v + s_e + s_i - w)/tau_m : volt",
        "dv/dt = (-v + s_e + s_i - w + I_ext)/tau_m : volt",
    )
    g.stimuli.append(
        type(graph.stimuli[0])(
            id="probe", target="exc", kind=kind,
            params={**params, "var_name": "I_ext", "unit": "volt"},
        )
    )
    report = validate(g)
    assert report.ok, report.report()


# --------------------------------------------------------------------------
# error attribution: the UI can only highlight a node if the diagnostic names it
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mutate,expected",
    [
        (lambda g: setattr(g.populations[0], "threshold", "v > 20"), "exc"),
        (lambda g: setattr(g.synapses[2], "on_pre", "s_i_post -= 1*amp"), "ie"),
        (lambda g: g.populations[0].initial.__setitem__("v", "5*amp"), "exc"),
    ],
)
def test_unit_errors_name_the_offending_node(graph, mutate, expected):
    mutate(graph)
    report = validate(graph)
    assert not report.ok
    assert report.errors[0].node_id == expected


def test_heterogeneous_initialisation_is_allowed(graph):
    """Brian2 expression strings must survive the safe-parser fallback."""
    graph.populations[0].initial["v"] = "rand()*20*mV"
    assert validate(graph).ok


# --------------------------------------------------------------------------
# position is decoration and must never block a run
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        {"x": None, "y": None},      # NaN serialised by JSON.stringify
        {"x": 10, "y": None},
        {},                          # field present but empty
        None,                        # field omitted entirely
        "nonsense",
        {"x": float("nan"), "y": 3},
        {"x": float("inf"), "y": 3},
    ],
)
def test_broken_positions_do_not_block_validation(graph, bad):
    payload = json.loads(graph.model_dump_json())
    payload["stimuli"][1]["position"] = bad
    restored = NetworkGraph.model_validate(payload)
    pos = restored.stimuli[1].position
    assert isinstance(pos["x"], float) and isinstance(pos["y"], float)
    assert validate(restored, dry_run=False).ok


def test_example_file_gives_every_node_a_position():
    """The frontend drags from these values; missing ones become NaN."""
    raw = json.loads(EXAMPLE.read_text())
    for section in ("populations", "stimuli"):
        for node in raw[section]:
            assert "position" in node, f"{section}: {node['id']} has no position"
            assert isinstance(node["position"]["x"], (int, float))
            assert isinstance(node["position"]["y"], (int, float))


# --------------------------------------------------------------------------
# sweeps
# --------------------------------------------------------------------------


def test_sweep_over_a_global(graph):
    from spikenet import run_sweep

    graph.run.duration = "0.4*second"
    graph.run.transient = "0.1*second"
    out = run_sweep(graph, {
        "scope": "globals", "key": "g", "unit": "", "target": "exc",
        "values": [2.0, 6.0], "metric": "peak_frequency",
    })
    assert out["parameter"] == "g"
    assert len(out["points"]) == 2
    assert all(p["metric"] is not None for p in out["points"]), out["points"]


def test_sweep_amplitude_at_fixed_vs_resonant(graph):
    """The two modes answer different questions and must not be conflated."""
    from spikenet import run_sweep

    graph.run.duration = "0.4*second"
    graph.run.transient = "0.1*second"
    base = {"scope": "globals", "key": "g", "unit": "", "target": "exc",
            "values": [2.0, 8.0], "metric": "amplitude_at"}
    fixed = run_sweep(graph, {**base, "options": {"at_frequency": 20}})
    peak = run_sweep(graph, {**base, "options": {"at_frequency": "resonant"}})
    for out in (fixed, peak):
        assert all(p["metric"] is not None for p in out["points"])
    # tracking the peak can never report less power than a fixed bin
    for a, b in zip(fixed["points"], peak["points"], strict=True):
        assert b["metric"] >= a["metric"] - 1e-12


def test_sweep_survives_a_failing_point(graph):
    """One bad parameter value must not destroy the whole curve."""
    from spikenet import run_sweep

    graph.run.duration = "0.4*second"
    graph.run.transient = "0.1*second"
    # Raising the threshold far out of reach silences the population. CV of ISI
    # is then genuinely undefined, which must surface as a labelled gap rather
    # than a NaN travelling onward into JSON.
    out = run_sweep(graph, {
        "scope": "globals", "key": "theta", "unit": "mV",
        "target": "exc", "values": [20.0, 5000.0], "metric": "cv_isi",
    })
    assert out["points"][0]["metric"] is not None
    bad = out["points"][1]
    assert bad["metric"] is None and bad["error"] is not None


def test_sweep_never_emits_nan(graph):
    """JSON cannot represent NaN, so it must not escape the sweep."""
    from spikenet import run_sweep
    import json as _json

    graph.run.duration = "0.3*second"
    out = run_sweep(graph, {
        "scope": "globals", "key": "theta", "unit": "mV",
        "target": "exc", "values": [20.0, 5000.0], "metric": "cv_isi",
    })
    _json.dumps(out, allow_nan=False)   # raises if any NaN survived


def test_unknown_metric_is_rejected(graph):
    from spikenet import run_sweep

    with pytest.raises(ValueError):
        run_sweep(graph, {"key": "g", "values": [1], "metric": "not_a_metric"})


def test_amplitude_at_frequency_reports_the_bin_it_used():
    freqs = np.array([0.0, 5.0, 10.0, 15.0])
    amps = np.array([0.1, 0.2, 0.9, 0.3])
    f, a = amplitude_at_frequency(freqs, amps, 11.0)
    assert f == 10.0 and a == 0.9


def test_summary_includes_the_new_distributions(graph):
    t, r = _synthetic_rate()
    rng = np.random.default_rng(3)
    n, dur = 100, 2.0
    st = np.sort(rng.uniform(0, dur, 3000))
    si = rng.integers(0, n, 3000)
    from analysis import summarise
    out = summarise(t, r, st, si, n_neurons=n, duration=dur)
    assert len(out["isi"]["counts"]) > 0
    assert sum(out["rate_distribution"]["counts"]) == n


# --------------------------------------------------------------------------
# audit regressions: every one of these was a real bug at some point
# --------------------------------------------------------------------------


def test_exported_script_is_valid_python(graph):
    """The Code button is worthless if what it emits will not parse.

    A multi-line reset used to be interpolated into a single-quoted string,
    which put a raw newline inside the literal and made the whole script a
    SyntaxError.
    """
    assert "\n" in graph.populations[0].reset, "the example must exercise this"
    compile(export_script(graph), "<export>", "exec")


def test_exported_script_survives_awkward_strings(graph):
    graph.populations[0].reset = "v = V_r\nw += 1*mV"
    graph.populations[0].threshold = "v > theta"
    graph.synapses[0].on_pre = "s_e_post += J"
    graph.name = "it's a \"test\"\\network"
    compile(export_script(graph), "<export>", "exec")


def test_exported_script_includes_the_inputs(graph):
    """Omitting the stimuli produced a script that ran but stayed silent."""
    script = export_script(graph)
    for stim in graph.stimuli:
        assert stim.id in script, f"{stim.id} missing from the export"
    assert "PoissonInput" in script


def test_exported_script_includes_continuous_stimulus_equations(graph):
    graph.stimuli = [graph.stimuli[0]]
    graph.stimuli[0].kind = "sinusoid"
    graph.stimuli[0].params = {"amplitude": "2*mV", "frequency": "20*Hz"}
    graph.populations[0].equations = graph.populations[0].equations.replace(
        "dv/dt = (-v + s_e + s_i - w)/tau_m : volt",
        "dv/dt = (-v + s_e + s_i - w + I_ext)/tau_m : volt",
    )
    script = export_script(graph)
    compile(script, "<export>", "exec")
    assert "I_ext" in script


@pytest.mark.parametrize("payload", ["9**9**9", "2**5000", "9**9**9**9"])
def test_exponent_bomb_is_rejected(payload):
    """`9**9**9` used to hang the process. Validation runs in the web
    process, so that hung the whole server with no timeout to recover."""
    with pytest.raises(UnsafeExpression):
        safe_eval_quantity(payload)


def test_small_exponents_still_work():
    assert safe_eval_quantity("2**10") == 1024
    assert safe_eval_quantity("10**-3") == 0.001


def test_division_by_zero_is_a_diagnostic_not_a_crash(graph):
    """It used to escape as ZeroDivisionError and become an HTTP 500."""
    graph.globals["J"] = "1/0"
    report = validate(graph, dry_run=False)
    assert not report.ok
    assert any("zero" in d.message.lower() for d in report.errors), report.report()


@pytest.mark.parametrize("kind", ["synapse", "stimulus", "monitor"])
def test_duplicate_ids_are_caught_for_every_node_kind(graph, kind):
    if kind == "synapse":
        graph.synapses[1].id = graph.synapses[0].id
    elif kind == "stimulus":
        graph.stimuli[1].id = graph.stimuli[0].id
    else:
        graph.monitors[1].id = graph.monitors[0].id
    report = validate(graph, dry_run=False)
    assert not report.ok
    assert any("Duplicate id" in d.message for d in report.errors), report.report()


def test_id_collision_across_kinds_is_caught(graph):
    graph.synapses[0].id = graph.populations[0].id
    report = validate(graph, dry_run=False)
    assert any("Duplicate id" in d.message for d in report.errors)


def test_two_continuous_inputs_cannot_share_a_variable(graph):
    for stim in graph.stimuli:
        stim.kind = "dc"
        stim.params = {"amplitude": "1*mV"}
        stim.target = "exc"
    report = validate(graph, dry_run=False)
    assert not report.ok
    assert any("var_name" in (d.field or "") for d in report.errors), report.report()


def test_injected_variable_name_must_be_an_identifier(graph):
    graph.stimuli[0].kind = "dc"
    graph.stimuli[0].params = {"amplitude": "1*mV", "var_name": "I : volt\nevil"}
    report = validate(graph, dry_run=False)
    assert not report.ok
    assert any("identifier" in d.message for d in report.errors)


def test_sweep_rejects_a_spec_with_no_parameter(graph):
    from spikenet import run_sweep

    with pytest.raises(ValueError):
        run_sweep(graph, {"values": [1, 2], "metric": "mean_rate"})


def test_sweep_reports_a_missing_node_per_point(graph):
    """A stale panel naming a deleted node must not fail the whole request."""
    from spikenet import run_sweep

    graph.run.duration = "0.2*second"
    out = run_sweep(graph, {
        "scope": "population", "node_id": "deleted", "key": "tau_m", "unit": "ms",
        "target": "exc", "values": [20.0], "metric": "mean_rate",
    })
    assert out["points"][0]["metric"] is None
    assert "deleted" in out["points"][0]["error"]


def test_exported_script_reproduces_the_app_run(graph, tmp_path):
    """The strongest guarantee this project can offer: the code shown in the
    Code panel, run on its own, gives the same answer as the app.

    Reproducing it needs the codegen target pinned as well as the seed --
    different targets draw from different random streams, so without that line
    the script diverges silently while looking correct.
    """
    from spikenet import run_graph

    graph.run.duration = "0.3*second"
    graph.run.transient = "0.1*second"
    graph.run.seed = 4242

    app = run_graph(graph)
    assert app.ok, app.validation.report()
    expected = app.metrics["exc"]["spikes"]["n_spikes"]

    script = tmp_path / "exported.py"
    script.write_text(export_script(graph) + textwrap.dedent("""

        import numpy as _np
        _t = _np.asarray(spikes_e.t / second)
        print("POST_TRANSIENT", int((_t >= 0.1).sum()))
        """))

    proc = subprocess.run([sys.executable, str(script)],
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-1500:]
    line = [x for x in proc.stdout.splitlines() if x.startswith("POST_TRANSIENT")]
    assert line, proc.stdout[-800:]
    assert int(line[0].split()[1]) == expected


# --------------------------------------------------------------------------
# resonance: bandwidth definitions, checked against an analytic Lorentzian
# --------------------------------------------------------------------------


def _lorentzian(f0=20.0, Q=5.0, n=40000):
    """A(f) = 1/sqrt(1 + (2Q(f-f0)/f0)^2): half-power width is exactly f0/Q."""
    f = np.linspace(0.05, 200.0, n)
    return f, 1.0 / np.sqrt(1.0 + (2 * Q * (f - f0) / f0) ** 2)


def test_half_power_bandwidth_and_q_match_theory():
    """Q must be defined on the -3 dB width, not the amplitude FWHM.

    On an amplitude spectrum, half power is A_peak/sqrt(2), not A_peak/2.
    Using the FWHM threshold gives a wider band and a Q roughly sqrt(3) too
    small, which would make these numbers incomparable with anyone else's.
    """
    f0, Q = 20.0, 5.0
    f, a = _lorentzian(f0, Q)
    r = resonant_frequency(f, a, low=1.0, high=150.0)
    assert abs(r.peak_frequency - f0) < 0.05
    assert abs(r.bandwidth_half_power - f0 / Q) < 0.05
    assert abs(r.q_factor - Q) < 0.05
    assert not r.truncated


def test_amplitude_fwhm_is_wider_than_half_power_by_root_three():
    """For a Lorentzian the two widths differ by exactly sqrt(3)."""
    f, a = _lorentzian(20.0, 5.0)
    r = resonant_frequency(f, a, low=1.0, high=150.0)
    assert r.bandwidth_fwhm > r.bandwidth_half_power
    assert abs(r.bandwidth_fwhm / r.bandwidth_half_power - np.sqrt(3)) < 0.02


def test_bandwidth_is_undefined_when_the_peak_exceeds_the_window():
    """Clipping at the band edge would report a measured-looking lower bound."""
    f = np.linspace(1.0, 30.0, 3000)
    r = resonant_frequency(f, np.ones_like(f), low=1.0, high=30.0)
    assert np.isnan(r.bandwidth_half_power)
    assert np.isnan(r.q_factor)
    assert r.truncated


# --------------------------------------------------------------------------
# input validation gaps found in review
# --------------------------------------------------------------------------


@pytest.mark.parametrize("key,value", [
    ("n_inputs", -5), ("n_inputs", 0), ("n_inputs", "lots"),
    ("probability", 3.0), ("probability", -0.1),
])
def test_out_of_range_stimulus_parameters_are_caught(graph, key, value):
    """Brian2 accepts PoissonInput(N=-5) and connect(p=3.0) without complaint."""
    graph.stimuli[0].params[key] = value
    report = validate(graph, dry_run=False)
    assert not report.ok
    assert any(key in (d.field or "") for d in report.errors), report.report()


@pytest.mark.parametrize("where", ["params", "initial"])
def test_non_identifier_keys_are_caught(graph, where):
    """A key like 'tau m' silently never matches, and breaks script export."""
    getattr(graph.populations[0], where)["tau m"] = "1*mV"
    report = validate(graph, dry_run=False)
    assert not report.ok
    assert any("not a valid name" in d.message for d in report.errors)


def test_stimulus_on_pre_gets_the_identifier_allowlist(graph):
    stim = graph.stimuli[0]
    stim.kind = "poisson_group"
    stim.params = {"n_inputs": 50, "rate": "100*Hz", "weight": "0.1*mV",
                   "on_pre": "v_post += bogus_name"}
    report = validate(graph, dry_run=False)
    assert not report.ok
    assert any("bogus_name" in d.message for d in report.errors), report.report()


def test_valid_stimulus_on_pre_is_accepted(graph):
    stim = graph.stimuli[0]
    stim.kind = "poisson_group"
    stim.params = {"n_inputs": 50, "rate": "100*Hz", "weight": "0.1*mV",
                   "probability": 0.5, "on_pre": "v_post += w_ext"}
    assert validate(graph).ok


def test_mismatched_spike_train_lengths_are_caught(graph):
    graph.stimuli[0].kind = "spike_train"
    graph.stimuli[0].params = {"indices": [0, 1, 2], "times_ms": [1.0, 2.0],
                               "weight": "0.1*mV"}
    report = validate(graph, dry_run=False)
    assert not report.ok


def test_expression_initial_values_are_not_run_through_the_unit_parser(graph):
    """Initial values are Brian2 expressions, not unit literals. Checking them
    with safe_eval_quantity rejects 'V_r' and 'rand()*20*mV', which are fine."""
    graph.populations[0].initial["v"] = "rand()*20*mV"
    assert validate(graph).ok
    graph.populations[0].initial["v"] = "V_r"
    assert validate(graph).ok


def test_server_finds_its_files_beside_itself(graph):
    """Both are resolved from __file__, not the working directory, so the
    server behaves identically however it is launched."""
    import server

    assert server.INDEX_HTML.is_file()
    assert server._example_path() is not None
    assert server._example_path().name == "brunel.json"
