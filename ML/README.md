# ML

Models, integer references, and the hardware handoff for the neuromorphic seizure detector.

| File | What it is |
|---|---|
| `common.py` | Spec shared by every model: `CHANNELS`, `N_IN`, `FS`, window size, input encoding, int8/int16 ranges, `spike_fn` surrogate for training |
| `lif.py` | Model A: LIF SNN N_IN -> 64 -> 2 (integer reference + float training model) |
| `alif.py` | Model B: adaptive-threshold (ALIF) SNN N_IN -> 64 -> 2. Same inputs, outputs and decision as `lif`; hidden neurons add an adaptation register. Reuses `lif.py`'s output layer and `LIFNet` weight handling |
| `scnn.py` | Model C: spiking convolutional NN. Temporal filters shared by every channel (K = 16 taps, dilations 1 and 3), depthwise spatial layer, 2 output neurons; 656 weights. The convolution is delayed synapses, so it's the same LIF math with a spike-history buffer. Every config knob (`kernel_len`, `dilations`, `group_filters`, `depth_mult`, `readout`, `share_filters`) works through `train.py --set` |
| `scnn_float.py` | Ablation: `scnn`'s architecture with ReLUs instead of spikes. Software only |
| `cnn.py` | Baseline: small 1D CNN on the same spike windows, logits instead of spike counts. Software only (never exported) |
| `train.py` | Per-patient training: float -> QAT -> int8, agreement check vs the integer reference, real-window handoff. `python train.py --patient chb01` (`--help` for options). `--set KEY=VALUE` overrides the model's CONFIG, `--float-only` skips QAT / int8 for a hardware model; both are added to the run folder name |
| `inspect_alarms.py` | Lists a finished run's validation alarms (detection vs false alarm); `--plot` saves a PNG per false alarm |
| `summarize.py` | One table of every run in `--out` (validation, test if run, cost columns), mean +- std per model / variant; `--csv`, `--md` |
| `ablate.py` | Runs the Model C ablations (plus `lif` / `alif` for comparison) through `train.py`; resumable. `--dry-run` prints the commands |
| `diversity.py` | Pairwise disagreement / error correlation / false-positive overlap between runs, and which seizures only one model catches (input to the voter) |
| `test_scnn.py` | Model C tests: causality, tap mapping, integer reference == QAT forward on every layer, export round-trip, on every config variant |
| `export.py` | Writes `handoff/<model>/` for any hardware model: `python export.py lif alif scnn` |
| `data/` | CHB-MIT pipeline: `preprocess.py` (EDF -> spike cache), `dataset.py` (windows, labels, split). See `data/CLAUDE.md` |
| `handoff/` | What the hardware team gets. Generated; don't edit by hand |

Setup: `python -m venv .venv`, then `.venv/Scripts/pip install numpy torch mne scipy matplotlib` (run from the repo root).

## Adding a model

Copy `lif.py` to `<name>.py` (e.g. `alif.py`) and change it. `export.py` doesn't need
any changes: `python export.py <name>` finds your module by name.

Your module must provide:

| Name | What |
|---|---|
| `CONFIG` | `{**COMMON_CONFIG, ...your constants}`. Don't override anything from `COMMON_CONFIG` |
| `weight_shapes(cfg=CONFIG)` | `{name: shape}` for every 2D int8 weight array. Each becomes `weights_<name>.mem` |
| `run_int_reference(spikes, weights, cfg=CONFIG)` | Integer-only (NumPy ints, no floats). `spikes` is `[T, n_inputs]` 0/1, `weights` is `{name: int8 array}`. Returns a dict: every per-timestep trace as a 2D `[T, n]` array (each written to `expected_<key>.mem`; keys starting with `spk` are written as packed bits, everything else as int16), plus `"out"` (`[T, n_out]` output spikes), `"counts"` (spike count per output neuron) and `"decision"` (0/1) |
| `REFERENCE_FUNCS` | List of functions copied into the standalone `reference_<name>.py`. Must include `weight_shapes`, `run_int_reference` and every helper they call. They may use `np`, `CONFIG`, `W_MIN/W_MAX/V_MIN/V_MAX` and nothing else |
| `test_cases(rng, cfg=CONFIG)` | List of `(folder, description, weights, spikes)`. Folder `"."` is the top of `handoff/<name>/`. Include at least one case that exercises saturation |
| `spec_md(cfg=CONFIG)` | Markdown for your model's section of `spec.md`: constants, exact math rules, weight file layouts, trace meanings |

Rules:
- Every model uses the same `N_IN = 2 * len(CHANNELS)` input lines and `T = 128`-step windows
  (1 s at 128 Hz, new window every 0.5 s), all from `common.py`, so the ensemble can share inputs.
  Never hard-code the input count.
- The integer reference IS the hardware contract. Write the math rules in `spec_md` precisely
  enough that someone could implement them in Verilog without reading your Python.
- After `python export.py <name>`, run `python handoff/<name>/reference_<name>.py`: every case must PASS.
- To train a model with `train.py --model <name>`, also provide `Net` (a `torch.nn.Module`) with
  `forward(x, qat=False) -> (output spike counts [B, n_out], mean hidden spike rate)`, where
  `qat=True` computes exactly what `run_int_reference` does; `clamp_weights()`; `quantize()` ->
  `{name: int8 array}`; and `load_int_weights(weights)`. See `LIFNet` in `lif.py`.
- Optional, for the cost columns in `results.json` / `summarize.py`: a module-level `hardware_cost(cfg)`
  (weights, membranes, history frames, worst-case adds per step) and `Net.cost_stats(x, qat)` (measured
  spike rates and adds per step, summed over the batch). See `lif.py` and `scnn.py`.
- Software-only models (e.g. `cnn.py`) don't need any of this; they never get exported. They only need
  `CONFIG` and `Net` with the same `forward(x, qat=False) -> (logits [B, n_out], activity)` signature.
  `train.py` sees there's no `run_int_reference` and runs float training only (no QAT, int8, agreement
  check or handoff), saving `weights.pt` instead of `weights_int8.npz`. Decision = logit 1 > logit 0.
