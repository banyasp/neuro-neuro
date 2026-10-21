# ML

Models, integer references, and the hardware handoff for the neuromorphic seizure detector.

| File | What it is |
|---|---|
| `common.py` | Spec shared by every model: `CHANNELS`, `N_IN`, `FS`, window size, input encoding, int8/int16 ranges, `spike_fn` surrogate for training |
| `lif.py` | Model A: LIF SNN N_IN -> 64 -> 2 (integer reference + float training model) |
| `train.py` | Per-patient training: float -> QAT -> int8, agreement check vs the integer reference, real-window handoff. `python train.py --patient chb01` (`--help` for options) |
| `export.py` | Writes `handoff/<model>/` for any model: `python export.py lif` |
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
- Software-only models (e.g. the 1D-CNN baseline) don't need any of this; they never get exported.
