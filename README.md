# SpikeNet Lab

A population-based, graph-driven simulator and analysis framework for spiking neural networks, built on [Brian2](https://brian2.readthedocs.io/).

SpikeNet Lab represents a network as a graph of populations, synapses, inputs, and monitors. That graph is translated into Brian2 objects, simulated, and analysed.

The default example is a simple recurrent excitatory-inhibitory (Brunel-style) network.

**Status: Just a fun prototype.** Feel free to reach out to me (@dr.vinay.neuro@gmail.com) with any suggestions or ideas. This is intended for exploration and experimentation rather than as a validated reference simulator. The simulation engine is Brian2; SpikeNet Lab adds the graph representation, validation, interface, analysis, and parameter-sweep layers.

## Quick start

Requires Python 3.10+.

```bash
git clone https://github.com/Vinay-Neuro/spikenet-lab.git
cd spikenet-lab
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python run.py
```

Open **http://127.0.0.1:8000/**.

For a browser-free run:

```bash
python demo.py
```

Run the tests with:

```bash
python -m pytest test_spikenet.py -q
```


## The model

A SpikeNet Lab network is made from four object types:

- Populations: groups of neurons with their own equations, thresholds, resets, initial conditions, and parameters.
- Synapses: directed connections with probabilities or conditions, delays, pathways (`on_pre` / `on_post`), constants, and optional per-synapse state.
- Inputs: external drives such as Poisson, DC, sinusoidal, chirp, spike-train, and Ornstein-Uhlenbeck noise.
- Monitors: recordings of spikes, population activity, or state variables.

The graph is stored as JSON. It describes the network rather than Python source code or Brian2 class names; the backend performs the translation.

Synapses define the event-triggered effect of spikes, while the receiving population's equations determine how that effect evolves over time.

## Using the interface

The interface has three main areas: a graph canvas, an inspector, and analysis panels.

- Drag nodes to move them; drag the background to pan; scroll to zoom.
- Toggle "Connect" and click two nodes to create a directed connection.
- Select a node or edge to edit equations, parameters, pathways, and connection rules.
- Use "Run" to simulate the current graph.
- Use "Code" to inspect the generated Brian2 script.
- Use "+ Panel" to inspect the same run from multiple views.

After a run, the transport bar can replay the recorded activity. The mapping between model time and playback time is shown explicitly in the interface.

## Analyses

The analysis layer includes:

| Analysis | Purpose |
|---|---|
| Spike raster / neuron grid | inspect individual spikes and population activity |
| Population rate | activity over time |
| Amplitude / power spectrum | frequency-domain structure |
| ISI histogram / rate distribution | single-neuron statistics |
| Synchrony | population-level coordination with a finite-size noise correction |
| Spectral entropy | concentration of spectral power |
| Resonance / bandwidth / Q | preferred frequency and tuning |
| Parameter sweep | measure a network metric across parameter values |

For the population-rate spectrum, the transient and mean are removed before the Fourier transform. Spectral amplitude is then normalised within the analysis band.

For an amplitude spectrum, SpikeNet Lab distinguishes:

- **Amplitude FWHM:** width at half the peak amplitude, `A_peak / 2`.
- **Half-power / -3 dB bandwidth:** width at `A_peak / sqrt(2)`.

The Q factor uses the half-power bandwidth:

```text
Q = f0 / Δf_-3dB
```

Synchrony: population-level coordination, with a correction for finite-size noise.

## Parameter sweeps

A sweep turns a model parameter into an experimental axis.

```text
g = 2 → 3 → 4 → 5 → 6
          ↓
      one run each
          ↓
    metric versus g
```

Any numeric constant can be swept, with metrics such as resonant frequency, spectral entropy, ISI CV, synchrony, amplitude, or power at a selected frequency.

For amplitude and power, the frequency can either be fixed or set to `resonant`. These answer different questions: a fixed frequency measures one band, while `resonant` follows each run's own peak.

A sweep of `N` points performs `N` simulations.

## How it maps to Brian2

The graph is converted into ordinary Brian2 objects:

| SpikeNet Lab | Brian2 |
|---|---|
| population | `NeuronGroup` |
| synapse | `Synapses` |
| Poisson input | `PoissonInput` / `PoissonGroup` |
| spike-train input | `SpikeGeneratorGroup` |
| spike monitor | `SpikeMonitor` |
| population-rate monitor | `PopulationRateMonitor` |
| state monitor | `StateMonitor` |

Physical quantities are stored as strings with explicit units, for example `20*ms` or `0.1*mV`.

The generated Python is a **one-way export** for inspection, copying, and use in a notebook or script. The application does not read that generated script back into the graph.


## Safety and limitations

SpikeNet Lab checks the network and user-supplied expressions before simulation and does not use `eval()` on UI input.

Simulations run in separate processes with configurable time and resource limits.

The server is intended for local use and has no authentication.



## Extending the simulator

**New input:** add the type to the schema, implement its Brian2 translation/export, add the UI option, and add tests.

**New metric:** implement it in `analysis.py`, register it for summaries and sweeps, and test it against a signal with a known answer where possible.

**New panel:** add the panel type and its drawing function in the frontend.

## Testing

```bash
python -m pytest test_spikenet.py -q
```

For the UI smoke test:

```bash
python run.py
npm install jsdom
node smoke.js
```

## Known limitations

This is still a prototype. In particular:

- analysis panels focus on one population at a time;
- parameter sweeps run sequentially;
- long rasters are subsampled for browser display;
- AI/SI/SR/AR regime labels are threshold conventions, not fitted phase boundaries;
- the default synaptic model is simplified and is not a full conductance-based or biophysical model;
- Cython is not the primary tested code-generation path;
- the complete tool has not been validated against published benchmark results.

## References

- Goodman, D. F. M. & Brette, R. (2008). *The Brian simulator for spiking neural networks in Python.* Frontiers in Neuroinformatics.  
  https://pmc.ncbi.nlm.nih.gov/articles/PMC2605403/

- Stimberg, M., Brette, R. & Goodman, D. F. M. (2019). *Brian 2, an intuitive and efficient neural simulator.* eLife, 8:e47314.  
  https://elifesciences.org/articles/47314

### Brunel network

- Brunel, N. (2000). *Dynamics of sparsely connected networks of excitatory and inhibitory spiking neurons.* Journal of Computational Neuroscience, 8, 183–208.

### Project context

The default network and the initial set of experiments were developed from a group project at the 2026 edition of the summer-school "Advanced Tools for Data Analyses in Neuroscience"" at the University of Strasbourg *Parameter-Dependent Emergent Oscillatory Dynamics in Recurrent Excitatory–Inhibitory Networks*, by Vinay,Julie Muzzolon & Ebru E., supervised by Jyotika Bahuguna.

## Acknowledgements

SpikeNet Lab is built on [Brian2](https://brian2.readthedocs.io/). The concept and direction of the project are mine, the implementation was developed with Claude Opus.

## License

SpikeNet Lab is released under the **MIT License**.

Brian2 is a separate project and remains under its own license. See the Brian2 repository and license for its terms.
