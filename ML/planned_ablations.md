# Planned ablations

As of 2026-11-27.

## Overview

Every ablation changes one thing from a model's default configuration, so any change in results
comes from that one thing. Together they answer our three research questions: how the models
compare, whether an ensemble helps, and what each design choice costs in hardware.

Protocol:

1. **Screen** every ablation on 2 patients (chb01, chb03) with 1 seed:
   `python ablate.py --patients chb01 chb03 --seeds 0 --workers N` (default groups; finished runs are skipped;
   `--workers` = runs at once, about the CPU core count minus 1; see `colab.ipynb` section 10).
2. **Confirm** the defaults and the strongest variants on every patient with 2 seeds:
   `python ablate.py --groups confirm --patients chb01 chb03 chb05 chb08 chb10 --seeds 0 1 --workers N`
   (`ablate.py` `CONFIRM`: each model's default and k = 4, plus SCNN `d1` and `k1` at k = 4).
   Add variants from the new screen runs with `--only` if they clearly win.
3. **Compare on validation only.** Test is scored once, at the end, for the final configurations.
4. **Report** event-level results (false alarms per hour, seizures detected, onset delay) alongside
   cost (weights, worst-case and measured adds per timestep).

Validation has only a few seizures per patient, so event-level differences between configs are
noisy at the screening stage. Window-level F1 and false positives per hour are steadier for screening.

**Selection rule** (decided after the screen, 2026-11-23): pick configs by mean window-level F1 on
validation across patients and seeds, with FP/h as the second check. When two configs are within one
standard deviation, take the cheaper one (fewer adds per step). Event-level results (detections, FA/h,
delay) are reported on test, once, for the final configs only: with 1-2 validation seizures per patient
they can't separate configs.

**Input density:** the delta threshold stays at alpha = 0.3 for every run, so all results are
comparable. Measured input spike density at that setting is 18-27% (not the 5-10% first assumed), and
hardware cost estimates must use the measured adds per step, not the 5% estimate.

## Shared ablations (every model)

These change the inputs or the alarm logic, so they apply to A, B and C alike. The first two cost
nothing extra: every training run sweeps them automatically.

| Ablation | Settings | What changes | Why | Status |
|---|---|---|---|---|
| Alarm persistence N | 1 to 10 windows | Number of consecutive positive windows before an alarm | Trades false alarms against detection delay; N = 2 to 3 keeps us near the 2 s target | Swept in every run |
| Decision margin m | 0 and up | A window is positive if seizure spikes exceed non-seizure spikes by more than m | A cheap way to make a model more conservative without retraining | Swept in every run |
| Window length | 0.5 / 1 / 2 s | Length of each window; the step stays at half a window | 0.5 s holds only about 1.5 cycles of a 3 Hz seizure rhythm; 2 s adds context but delays decisions | In `ablate.py` for Models A and B (`lif_win05`, `lif_win2`, `alif_win05`, `alif_win2`) |
| Channel subsets | 23 / 22 / 21 channels, or other combinations | Which EEG channels feed the model | Two of the 23 are redundant; fewer channels mean fewer input lines and less hardware | Needs a preprocessed cache per channel set |
| Hybrid encoding | delta only vs delta + rate-coded | Adds rate-coded spikes (spike probability from signal amplitude) alongside the delta spikes | Prior work found the hybrid much better than either encoding alone | Needs a new encoder in the pipeline |
| Delta threshold alpha | 0.2 / 0.3 / 0.5 | How big a signal change triggers an input spike | Input spike density sets most of the hardware cost; higher alpha means fewer spikes | Needs a preprocessed cache per value |

## Model A: LIF

Default: 46 inputs, 64 hidden LIF neurons, 2 outputs, leak shift k = 5, int8 weights.

| Ablation | Settings | What changes | Why | Status |
|---|---|---|---|---|
| Hidden size | 32 / 64 / 128 / 192 | Number of hidden neurons | Weights and adds grow with size; this shows where accuracy stops improving | `--hidden`; in `ablate.py` |
| Leak shift k | 4 / 5 / 6 | How fast the membrane forgets: time constant of about 0.125 / 0.25 / 0.5 s | Sets how much recent history each neuron integrates; changes no hardware cost | `--set leak_shift=...`; in `ablate.py` |
| int8 cost | float vs int8 | Train in float and stop, instead of finishing with quantization-aware training (QAT) and int8 weights | The hardware must run int8. The gap shows how much accuracy quantization costs | `--float-only`; in `ablate.py` |

## Model B: ALIF

Default: same layout as Model A, but each hidden neuron's threshold rises each time it fires (by
`adapt_inc` = 16) and decays back with a time constant of about 1 s (`adapt_shift` = 7). The idea
is to ignore persistent artifacts that keep a neuron firing.

| Ablation | Settings | What changes | Why | Status |
|---|---|---|---|---|
| Adaptation off | `adapt_inc` = 0 | The threshold never rises, so the model becomes Model A | Shows whether adaptation helps at all | `--set`; in `ablate.py` |
| Adaptation strength | `adapt_inc` = 8 / 16 / 32 | How much the threshold rises per spike | Too weak does nothing; too strong silences neurons during a real seizure | `--set`; in `ablate.py` |
| Adaptation time constant | `adapt_shift` = 6 / 7 / 8 (about 0.5 / 1 / 2 s) | How long the raised threshold lasts | Should match how long artifacts persist compared with seizure activity | `--set`; in `ablate.py` |
| Hidden size | 32 / 64 / 128 / 192 | Number of hidden neurons | Same sizes as Model A, so A and B are compared at matched size. ALIF costs more per neuron (an extra adaptation register), so this shows whether that cost pays off | `--hidden`; in `ablate.py` |
| int8 cost | float vs int8 | Same as Model A | Same as Model A | `--float-only`; in `ablate.py` |
| Leak shift k | 4 / 5 / 6 | Same as Model A | k = 4 helped Models A and C in the screen; checks whether it helps B too | `alif_leak4`, `alif_leak6` in `ablate.py` |

## Model C: SCNN

Default: 8 temporal filters run over each channel's recent spikes. Each filter has K = 16 taps (how
many past frames it weighs); 4 filters use dilation d = 1 (taps on consecutive frames, about 125 ms
reach) and 4 use d = 3 (every third frame, about 359 ms). All 23 channels share the same filters,
giving 184 spiking neurons. A depthwise spatial stage then combines each filter's output across
channels, with D = 2 neurons per filter (16 in total), and those feed 2 outputs. 656 weights, leak
shift k = 5. All runs below are in `ablate.py`.

| Name | What changes | Why |
|---|---|---|
| baseline | Nothing | The reference every other row is compared against |
| k1 | K = 1: each filter sees only the current sample | Does temporal filtering help at all? The only memory left is the membrane leak. If k1 matches the baseline, the convolution isn't worth its cost |
| d1 | All 8 filters at d = 1 (125 ms reach) | Is the longer reach of the d = 3 filters needed? |
| d124 | 12 filters, 4 each at d = 1, 2, 4 (up to about 475 ms) | Does even longer reach help? Also adds 4 filters, so not a pure single change |
| k32 | K = 32 at d = 1 (250 ms, every frame used, twice the layer-1 weights) | Is it better to reach far with a long dense filter or with sparse dilated taps? |
| fc | Spatial stage removed: the 184 neurons connect straight to the 2 outputs (624 weights) | Is the spatial stage worth it? Weight count is about the same, so this tests the structure, not the size |
| leak shift | k = 4 / 5 / 6 (about 0.125 / 0.25 / 0.5 s) | The leak interacts with the filters' reach, so it may matter more here than in Model A. In `ablate.py` as `leak4` and `leak6` |
| float_spiking | Same spiking network, trained in float, no int8 | How much accuracy int8 costs |
| float_relu | Spikes replaced by ReLU activations; software only | How much accuracy spiking costs, with the architecture held fixed. Cleaner than comparing to the 1D-CNN, which also differs in architecture |

Width and sharing (`nice` group, now in the default screen):

| Name | What changes | Why |
|---|---|---|
| unshared | Each channel gets its own filters (layer-1 weights go from 256 to 5,888) | Do channels need different filters? With few training seizures it may overfit |
| c4 | 4 filters (2 + 2) | Is a smaller filter bank enough? Cheaper hardware |
| c16 | 16 filters (8 + 8) | Does a bigger filter bank help? Roughly doubles cost |
| d_mult1 | D = 1 spatial neuron per filter (8 instead of 16) | Is one spatial pattern per filter enough? |

The three results that matter most for the report are k1 (is the convolution worth it?), fc (is
the spatial stage worth it?) and float_relu (what does spiking cost?).

## Model C with ALIF neurons (`scnn_alif`)

Default: Model C's architecture unchanged (656 weights, 46-frame history), but the temporal and
spatial layers use Model B's ALIF neuron (`adapt_inc` = 16, `adapt_shift` = 7, time constant about
1 s). The output layer stays plain LIF. Hardware cost: Model C plus 200 int16 adaptation registers.
Module `scnn_alif.py`; in `ablate.py` as the `scnn_alif` group. It gets Model C's ablations (each
compared with the Model C run of the same name, so the only difference is the neuron) plus Model B's
adaptation knobs. `float_relu` has no ALIF twin (ReLUs have no threshold to adapt).

| Name | What changes | Compare against |
|---|---|---|
| scnn_alif | Hidden neurons LIF -> ALIF: does adaptation help the convolutional model the way it was meant to help Model B (ignoring persistent artifacts)? | `baseline` |
| scnn_alif_k1 / _d1 / _d124 / _k32 | Model C's filter-length and dilation ablations | `k1` / `d1` / `d124` / `k32` |
| scnn_alif_fc | Fully connected readout | `fc` |
| scnn_alif_float | Float only, no int8 | `float_spiking` |
| scnn_alif_leak4 / _leak6 | Leak k = 4 / 6 | `leak4` / `leak6` |
| scnn_alif_inc8 / _inc32 | Adaptation strength `adapt_inc` 8 / 32 | `scnn_alif` (Model B's `alif_inc8` / `alif_inc32`) |
| scnn_alif_ashift6 / _ashift8 | Adaptation time constant about 0.5 / 2 s | `scnn_alif` (Model B's `alif_ashift6` / `alif_ashift8`) |

## Cross-model analysis

- **Diversity (`diversity.py`):** for each pair of models, how often they disagree, how correlated
  their errors are, how many false positives they share, and which seizures only one model catches.
  An ensemble only helps if the models make different mistakes, so this is the input to the voter
  design (the voter itself is a teammate's part).
- **1D-CNN baseline:** one non-spiking reference model on the same windows, preprocessing and split
  as the SNNs. No ablations planned; it's the comparison point for research question 1.

## Stretch goal: streaming instead of windows

Today every window starts with all membranes at zero, and windows overlap by 50%, so the hardware
processes every EEG sample twice. In streaming mode the model would run continuously, keeping its
membranes from one step to the next and never resetting.

- **Why:** it halves hardware compute, and the model can output a decision at any step instead of
  every 0.5 s, which could shorten detection delay.
- **Risk:** this changes the hardware math contract (when state resets), so it needs agreement from
  the hardware team first. Training also changes, since the model must learn on long continuous
  segments instead of independent windows.
- **Applies to:** all three models.

## Other candidates (not yet decided)

These came up in discussion but aren't committed to.

| Ablation | Settings | What changes | Why | Effort |
|---|---|---|---|---|
| Spike-rate penalty | `--rate-penalty` 0 / small / large | Penalizes spiking during training | Fewer spikes mean fewer adds and less energy; gives an accuracy vs cost curve | Flag exists |
| Weight bit width | int8 / int6 / int4 | Weight precision | Smaller weights need less on-chip memory | Small code change |
| Channel dropout at test time | Zero one channel | Simulates an electrode coming loose | Tests robustness to artifacts, and whether the models fail differently | Evaluation script only, no retraining |
| 64 Hz sampling | 64 timesteps per window | Halves the time resolution | Halves the cycles per window | Needs a lower low-pass cutoff (about 30 Hz) and new caches |
| Training balance | `--neg-per-pos`, `--pos-weight` | Ratio of seizure to non-seizure windows, and seizure loss weight | Shifts the trade-off between sensitivity and false alarms | Flags exist; tuning rather than a finding |

Not planned: firing threshold theta (mostly redundant with the weight scale under QAT), reset to
zero instead of subtracting (changes the hardware math for little expected gain), and voter rules
(a teammate's part).
