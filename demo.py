"""
End-to-end demo. Run: python demo.py

Shows the validation layers rejecting bad graphs, then a full Brunel run with
spectral analysis, all driven from brunel.json.
"""

import copy
import json
from pathlib import Path

from brian2 import BrianLogger

from spikenet import NetworkGraph, export_script, run_graph, validate

BrianLogger.suppress_name("resolution_conflict")
BrianLogger.suppress_name("method_choice")

HERE = Path(__file__).parent
RULE = "=" * 68


def load() -> NetworkGraph:
    raw = json.loads((HERE / "brunel.json").read_text())
    return NetworkGraph.model_validate(raw)


def show(title: str, graph: NetworkGraph, dry_run: bool = True) -> None:
    report = validate(graph, dry_run=dry_run)
    status = "PASS" if report.ok else "FAIL"
    print(f"\n[{status}] {title}")
    for d in report.diagnostics:
        print(f"       {d}")
    if report.ok and not report.diagnostics:
        print("       no problems found")


def main() -> None:
    print(RULE)
    print("1. VALIDATION")
    print(RULE)

    base = load()
    show("the reference Brunel graph", base)

    # --- unit mismatch: the error a regex checker cannot catch --------------
    g = copy.deepcopy(base)
    g.populations[0].equations = g.populations[0].equations.replace(
        "ds_e/dt = -s_e/tau_syn_e : volt", "ds_e/dt = -s_e/tau_syn_e : amp"
    )
    show("s_e redeclared as amp while v is volt", g)

    # --- undeclared constant -----------------------------------------------
    g = copy.deepcopy(base)
    g.populations[0].threshold = "v > theta_typo"
    show("threshold references an undeclared constant", g)

    # --- dimensionless comparison ------------------------------------------
    g = copy.deepcopy(base)
    g.populations[0].threshold = "v > 20"
    show("threshold compares volts against a bare number", g)

    # --- synaptic pathway with the wrong dimension -------------------------
    g = copy.deepcopy(base)
    g.synapses[0].on_pre = "s_e_post += 1*amp"
    show("synapse adds amps to a volt variable", g)

    # --- code injection attempt --------------------------------------------
    g = copy.deepcopy(base)
    g.populations[0].threshold = '__import__("os").system("id") > 0'
    show("threshold containing a Python import", g, dry_run=False)

    g = copy.deepcopy(base)
    g.globals["J"] = '__import__("os").popen("id").read()'
    show("parameter value containing a Python import", g, dry_run=False)

    # --- structural error ---------------------------------------------------
    g = copy.deepcopy(base)
    g.synapses[0].target = "does_not_exist"
    show("synapse pointing at a missing population", g, dry_run=False)

    print()
    print(RULE)
    print("2. SIMULATION")
    print(RULE)

    result = run_graph(base)
    if not result.ok:
        print(result.validation.report())
        return

    print(f"simulated {result.duration:.2f} s of model time "
          f"in {result.wall_time:.2f} s wall time")

    m = result.metrics["exc"]
    res = m["resonance"]
    sp = m["spikes"]
    print(f"\nexcitatory population ({base.populations[0].n} neurons)")
    print(f"  spikes recorded    : {sp['n_spikes']}")
    print(f"  mean rate          : {sp['mean_rate']:.2f} Hz/neuron")
    print(f"  CV of ISI          : {sp['cv_isi']:.3f}")
    print(f"  Fano factor        : {sp['fano_factor']:.3f}")
    print(f"  active fraction    : {sp['active_fraction']:.3f}")
    print(f"  synchrony index    : {m['synchrony']:.3f}")
    print(f"  spectral entropy   : {m['spectrum']['entropy']:.4f}")
    print(f"  resonant frequency : {res['peak_frequency']:.2f} Hz")
    print(f"  FWHM (amplitude)   : {res['bandwidth_fwhm']:.2f} Hz")
    print(f"  -3 dB bandwidth    : {res['bandwidth_half_power']:.2f} Hz")
    print(f"  Q factor           : {res['q_factor']:.2f}")
    print(f"  regime             : {m['regime']}")

    print()
    print(RULE)
    print("3. PARAMETER SWEEPS")
    print(RULE)

    def sweep(label, mutate, values, header):
        print(f"\n{label}")
        print(f"{header:>8}  {'rate Hz':>8}  {'CV ISI':>7}  {'sync':>6}  "
              f"{'peak Hz':>8}  {'entropy':>8}  regime")
        for value in values:
            graph = load()
            graph.run.duration = "1.0*second"
            graph.run.transient = "0.3*second"
            mutate(graph, value)
            r = run_graph(graph)
            if not r.ok:
                print(f"{value:>8}  {r.validation.errors[0].message[:50]}")
                continue
            m = r.metrics["exc"]
            if "error" in m:
                print(f"{value:>8}  {m['error'][:50]}")
                continue
            print(f"{value:>8}  {m['spikes']['mean_rate']:>8.2f}  "
                  f"{m['spikes']['cv_isi']:>7.3f}  {m['synchrony']:>6.3f}  "
                  f"{m['resonance']['peak_frequency']:>8.2f}  "
                  f"{m['spectrum']['entropy']:>8.4f}  {m['regime']}")

    def set_g(graph, value):
        graph.globals["g"] = value

    sweep("inhibitory strength -- shifts the spectral peak",
          set_g, ("2.0", "4.5", "8.0"), "g")

    # Drive is expressed relative to the threshold rate, nu_thr =
    # theta/(J*C_ext*tau_m) = 100 Hz for this graph. Sweeping across it walks
    # the network from fluctuation-driven firing (below threshold, irregular)
    # to mean-driven firing (above threshold, regular), which is the clearest
    # demonstration that CV_ISI is measuring something real.
    def set_drive(graph, value):
        graph.globals["adaptation_strength"] = "0*mV"
        for stim in graph.stimuli:
            stim.params["rate"] = f"{float(value) * 100.0}*Hz"

    sweep("external drive relative to threshold -- fluctuation vs mean driven",
          set_drive, ("0.90", "0.95", "1.00", "1.05", "1.20"), "drive")

    print()
    print(RULE)
    print("4. SCRIPT EXPORT (first 22 lines)")
    print(RULE)
    for line in export_script(base).splitlines()[:22]:
        print(line)


if __name__ == "__main__":
    main()
