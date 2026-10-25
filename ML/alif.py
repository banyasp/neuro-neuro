"""
Model B: fully connected adaptive LIF (ALIF) SNN (N_IN -> 64 -> 2) for EEG seizure detection.
N_IN = 2 * len(common.CHANNELS) (46 with all 23 CHB-MIT channels).

Same inputs, outputs, sizes and decision rule as Model A (lif.py). The only difference:
every hidden neuron has an adaptation variable `a` that rises by a fixed step each time
the neuron spikes and leaks back toward 0, raising that neuron's threshold to theta + a.
Neurons that fire a lot get harder to fire, which gives the hidden layer memory longer than
the membrane time constant (spike-frequency adaptation). The output layer is plain LIF,
exactly as in lif.py, so the output spike counts are read the same way.

This file has three parts:
  1. CONFIG: the shared input spec (common.py) plus this model's constants.
  2. Integer reference (NumPy only): bit-exact model of the hardware math.
     This generates the golden vectors. No floats anywhere.
  3. Float model (PyTorch): the same network for surrogate-gradient training.
     Same order of operations as the integer reference.

Integer math, per timestep (the hardware contract):
  Layer 1 (ALIF), membrane v, adaptation a:
  1. Leak:      v = v - (v >> k)        arithmetic shift, rounds toward -inf
  2. Adapt leak: a = a - (a >> k_a)
  3. Integrate: for each input i with s_in[i] == 1, in increasing i order:
                    v = clip(v + W[i, :], -32768, 32767)   (saturate after EVERY add)
  4. Fire:      s_out = (v >= theta + a)
  5. Reset:     v[s_out] -= theta
  6. Adapt:     a[s_out] += adapt_inc
  Layer 2 (LIF): steps 1, 3, 4, 5 with a = 0 (same as lif.py).
Layer 2 uses layer 1's spikes from the same timestep.
Membranes and adaptation start at 0 at the beginning of every window.
Decision: seizure if output neuron 1 spikes more times than neuron 0 (tie = no seizure).
"""

import numpy as np

from common import COMMON_CONFIG, N_IN, T, W_MIN, W_MAX, V_MIN, V_MAX, random_spikes, random_int8
from lif import lif_step_int, random_weights

# ---------------------------------------------------------------------------
# 1. Config (the spec)
# ---------------------------------------------------------------------------

CONFIG = {
    **COMMON_CONFIG,
    "n_hidden": 64,
    "n_outputs": 2,           # neuron 0 = non-seizure, neuron 1 = seizure
    "leak_shift": 5,          # k in v = v - (v >> k), decay 1 - 2^-5 = 0.96875, tau ~ 31.5 steps ~ 0.25 s at 128 Hz
    "theta": 64,              # base firing threshold (integer units), same for both layers
    "adapt_shift": 7,         # k_a in a = a - (a >> k_a), decay 1 - 2^-7, tau ~ 127.5 steps ~ 1 s at 128 Hz
    "adapt_inc": 16,          # a += adapt_inc per hidden spike (theta / 4)
}


def weight_shapes(cfg=CONFIG):
    """Name -> shape of every int8 weight array. Files are weights_<name>.mem.
    Uses cfg["n_inputs"] (= N_IN) because this function is copied into the standalone
    reference, which only has CONFIG."""
    return {"l1": (cfg["n_inputs"], cfg["n_hidden"]),
            "l2": (cfg["n_hidden"], cfg["n_outputs"])}


# ---------------------------------------------------------------------------
# 2. Integer reference (bit-exact with hardware)
# ---------------------------------------------------------------------------

def alif_step_int(v, a, s_in, W, k, theta, k_a, inc):
    """One timestep of one integer ALIF layer.

    v:    int32 array [n_out], current membrane (always within int16 range)
    a:    int32 array [n_out], current adaptation (always in [0, (inc + 1) * 2^k_a))
    s_in: 0/1 array [n_in], input spikes this timestep
    W:    int8 array [n_in, n_out]
    Returns (new v, new a, output spikes as uint8 [n_out]).
    """
    # 1-2. Leak membrane and adaptation. NumPy >> on signed ints is an arithmetic shift, like Verilog >>>.
    v = v - (v >> k)
    a = a - (a >> k_a)

    # 3. Integrate + saturate, one input row at a time, in increasing input index.
    for i in np.flatnonzero(s_in):
        v = np.clip(v + W[i].astype(np.int32), V_MIN, V_MAX)

    # 4. Fire against the adapted threshold
    s_out = (v >= theta + a).astype(np.uint8)

    # 5. Reset by subtracting the BASE threshold (v >= theta + a >= theta > 0, so no underflow)
    v = v - s_out.astype(np.int32) * theta

    # 6. Adapt: every spike raises this neuron's threshold
    a = a + s_out.astype(np.int32) * inc
    return v, a, s_out


def run_int_reference(spikes, weights, cfg=CONFIG):
    """Run one window through the integer model.

    spikes:  0/1 array [T, n_inputs]
    weights: {"l1": int8 [n_inputs, n_hidden], "l2": int8 [n_hidden, n_outputs]}

    Returns a dict of per-timestep traces [T, n] (membranes are after reset, adaptation is
    after the increment), each written to expected_<key>.mem, plus "out" output spikes,
    "counts", "decision". Keys starting with "spk" are written as packed bits; all others as int16.
    """
    spikes = np.asarray(spikes)
    T = spikes.shape[0]
    assert spikes.shape[1] == cfg["n_inputs"]
    assert set(np.unique(spikes)) <= {0, 1}, "spikes must be 0/1"
    for name, shape in weight_shapes(cfg).items():
        assert weights[name].dtype == np.int8 and weights[name].shape == shape, name
    W1, W2 = weights["l1"], weights["l2"]

    k, theta = cfg["leak_shift"], cfg["theta"]
    k_a, inc = cfg["adapt_shift"], cfg["adapt_inc"]
    # a never exceeds (inc + 1) * 2^k_a - 1, so a and theta + a fit in int16 without saturation
    assert theta + ((inc + 1) << k_a) <= V_MAX, "adaptation could overflow int16"
    v1 = np.zeros(cfg["n_hidden"], dtype=np.int32)
    a1 = np.zeros(cfg["n_hidden"], dtype=np.int32)
    v2 = np.zeros(cfg["n_outputs"], dtype=np.int32)

    mem1 = np.zeros((T, cfg["n_hidden"]), dtype=np.int16)
    adapt1 = np.zeros((T, cfg["n_hidden"]), dtype=np.int16)
    spk1 = np.zeros((T, cfg["n_hidden"]), dtype=np.uint8)
    mem2 = np.zeros((T, cfg["n_outputs"]), dtype=np.int16)
    spk2 = np.zeros((T, cfg["n_outputs"]), dtype=np.uint8)

    for t in range(T):
        v1, a1, s1 = alif_step_int(v1, a1, spikes[t], W1, k, theta, k_a, inc)
        v2, s2 = lif_step_int(v2, s1, W2, k, theta)
        mem1[t], adapt1[t], spk1[t] = v1, a1, s1
        mem2[t], spk2[t] = v2, s2

    counts = spk2.sum(axis=0).astype(np.int32)
    decision = int(counts[1] > counts[0])
    return {"mem_l1": mem1, "adapt_l1": adapt1, "spk_l1": spk1, "mem_l2": mem2, "out": spk2,
            "counts": counts, "decision": decision}


# Functions copied into the standalone handoff/<model>/reference_<model>.py
REFERENCE_FUNCS = [weight_shapes, alif_step_int, lif_step_int, run_int_reference]


# ---------------------------------------------------------------------------
# Handoff: test cases and spec text (used by export.py)
# ---------------------------------------------------------------------------

def test_cases(rng, cfg=CONFIG):
    """List of (folder, description, weights, spikes). Folder "." = top of handoff/alif/.
    The first four cases are the same draws as lif.py, so the two handoffs share inputs."""
    cases = [(".", "random int8 weights, 5% random input spikes",
              random_weights(rng, cfg), random_spikes(rng, 0.05))]

    # Saturation: hidden 0-15 all +127, 16-31 all -128, all inputs on for the first half.
    w = random_weights(rng, cfg)
    w["l1"][:, 0:16] = W_MAX
    w["l1"][:, 16:32] = W_MIN
    spikes = np.zeros((T, N_IN), dtype=np.uint8)
    spikes[: T // 2] = 1
    r = run_int_reference(spikes, w, cfg)
    # Recorded membranes are after reset, so the + rail shows up as V_MAX - theta
    assert r["mem_l1"].max() == V_MAX - cfg["theta"] and r["mem_l1"].min() == V_MIN, \
        "saturation case didn't saturate"
    cases.append(("saturation", "hidden 0-15 all +127, 16-31 all -128; all inputs on for "
                  f"t < {T // 2}, then off. Hits both int16 rails, then leaks. "
                  f"The + rail appears as {V_MAX - cfg['theta']} (32767 minus reset)", w, spikes))

    cases.append(("dense", "random weights, 30% input spikes (heavy load)",
                  random_weights(rng, cfg), random_spikes(rng, 0.30)))

    # Seizure: every hidden neuron excites output 1 and inhibits output 0, so decision = 1.
    w = random_weights(rng, cfg)
    w["l2"][:, 0] = W_MIN
    w["l2"][:, 1] = W_MAX
    spikes = random_spikes(rng, 0.05)
    assert run_int_reference(spikes, w, cfg)["decision"] == 1, "seizure case didn't decide 1"
    cases.append(("seizure", "random l1, l2 column 0 all -128 and column 1 all +127, "
                  "5% input spikes. Decision = 1", w, spikes))

    # Adaptation: constant drive on input 0 only. A plain LIF neuron would fire at a steady
    # rate; here every hidden neuron's threshold climbs, so it fires less and less often.
    # Appended last so the cases above keep their random draws (bit-identical to lif's inputs).
    w = random_weights(rng, cfg)
    w["l1"][:] = 0
    w["l1"][0, :] = 48
    spikes = np.zeros((T, N_IN), dtype=np.uint8)
    spikes[:, 0] = 1
    r = run_int_reference(spikes, w, cfg)
    early, late = r["spk_l1"][: T // 4].sum(), r["spk_l1"][-T // 4:].sum()
    assert late < early and r["adapt_l1"].max() > 0, "adaptation case didn't adapt"
    cases.append(("adaptation", "l1 all 0 except input 0 -> every hidden neuron = +48; input 0 on "
                  "every step, all others off. Hidden spike rate drops as the threshold adapts "
                  f"({early} hidden spikes in the first {T // 4} steps, {late} in the last {T // 4})",
                  w, spikes))
    return cases


def spec_md(cfg=CONFIG):
    """Model-specific section of spec.md."""
    nh, no, ni = cfg["n_hidden"], cfg["n_outputs"], cfg["n_inputs"]
    k, ka, inc, theta = cfg["leak_shift"], cfg["adapt_shift"], cfg["adapt_inc"], cfg["theta"]
    a_max = (inc + 1) * 2 ** ka - 1
    return f"""## Model B: ALIF SNN {ni} -> {nh} -> {no}

Same inputs, sizes, output layer and decision rule as Model A (`lif`). The hidden layer adds
one adaptation register `a` per neuron that raises its threshold after every spike.

| Name | Value | Notes |
|---|---|---|
| `N_HIDDEN` | {nh} | adaptive (ALIF) neurons |
| `N_OUT` | {no} | plain LIF. Neuron 0 = non-seizure, neuron 1 = seizure |
| `K` (leak shift) | {k} | membrane leak, both layers |
| `THETA` | {theta} | base threshold, same for both layers |
| `K_A` (adaptation shift) | {ka} | hidden layer only |
| `A_INC` (adaptation step) | {inc} | hidden layer only |
| Bias | none | |

Time scale: one timestep = one EEG sample at {cfg['fs']} Hz, so a {cfg['window_len']}-step window is
{cfg['window_len'] / cfg['fs']:g} s, with a new window every {cfg['window_step'] / cfg['fs']:g} s. The membrane leak
`K = {k}` gives decay `1 - 2^-{k}` per step, a time constant of about
{1 / -np.log(1 - 2.0 ** -k):.1f} steps = {1 / -np.log(1 - 2.0 ** -k) / cfg['fs']:.2f} s. The adaptation leak
`K_A = {ka}` gives a time constant of about {1 / -np.log(1 - 2.0 ** -ka):.1f} steps = {1 / -np.log(1 - 2.0 ** -ka) / cfg['fs']:.2f} s.

Input spikes come from per-channel delta modulation in Python: each channel is divided by a
fixed per-channel scale, then thresholded at `alpha = {cfg['delta_alpha']}` (so each channel has its
own threshold `d_c = alpha * scale_c`). None of this is visible to hardware, which just reads
the spike raster.

### Math, per timestep

**Layer 1 (ALIF)**, membrane `v` (int16), adaptation `a` (int16, never negative), input spikes
`s_in` (0/1), weights `W1` (int8, `[n_in][n_out]`):

1. **Leak:** `v = v - (v >>> K)` (arithmetic shift).
2. **Adaptation leak:** `a = a - (a >>> K_A)`.
3. **Integrate + saturate:** for each input `i` with `s_in[i] == 1`, in **increasing `i`**:
   `v = sat16(v + W1[i][:])`. Saturate after **every** add, not once at the end.
4. **Fire:** `s_out = (v >= THETA + a)` (signed compare, using `a` from step 2).
5. **Reset:** where `s_out == 1`, `v = v - THETA` (the **base** threshold, not `THETA + a`).
6. **Adapt:** where `s_out == 1`, `a = a + A_INC`.

**Layer 2 (LIF)**, exactly Model A's math: steps 1, 3, 4, 5 with `a = 0` (no adaptation register).

- Layer 2 uses layer 1's spikes **from the same timestep**.
- All membranes **and adaptation registers** reset to 0 at the start of every window.
- **Decision:** seizure (1) if neuron 1 spiked **more** times than neuron 0 over the window. Tie = 0.

Notes:
- The arithmetic shift rounds toward -inf: a negative `v` leaks all the way to 0, but a
  positive `v` in 1..{2 ** k - 1} has `v >>> K == 0` and does **not** leak. Likewise `a` in
  1..{2 ** ka - 1} does not leak.
- `a` is always in `[0, {a_max}]` (`(A_INC + 1) * 2^K_A - 1`), so `a` and `THETA + a` (at most
  {theta + a_max}) fit in int16 and need **no** saturation logic.
- The reset only runs when `v >= THETA + a >= THETA > 0`, so it can't underflow. Only step 3
  needs saturation logic.
- Order of adds matters only when saturation happens (see the `saturation/` case).

### Weights and traces

| File | Contents |
|---|---|
| `weights_l1.mem` | W1 `[{ni}][{nh}]`, {ni * nh} words: address `i*{nh} + j` = `W1[i][j]` (grouped by input, suits event-driven fetch) |
| `weights_l2.mem` | W2 `[{nh}][{no}]`, {nh * no} words: address `i*{no} + j` = `W2[i][j]` |
| `expected_mem_l1.mem` | T*{nh} words, int16: address `t*{nh} + j` = layer-1 membrane of neuron `j` at step `t`, **after reset** |
| `expected_adapt_l1.mem` | T*{nh} words, int16: address `t*{nh} + j` = layer-1 adaptation `a` of neuron `j` at step `t`, **after step 6** |
| `expected_spk_l1.mem` | T words of {nh} bits: bit `j` = layer-1 neuron `j` spiked at step `t` |
| `expected_mem_l2.mem` | T*{no} words, int16: address `t*{no} + j`, **after reset** |
"""


# ---------------------------------------------------------------------------
# 3. Trainable model (PyTorch): float mode and QAT mode
# ---------------------------------------------------------------------------
# Units: as in lif.py, float mode uses theta = 1.0 and W_int = round(W_float * cfg["theta"]).
# Adaptation uses the same units: float a = integer a / theta, so the float step is adapt_inc / theta.
#
# Float mode (stage 1), hidden layer:
#   v = v - v * 2^-k + W x;  a = a - a * 2^-k_a;  s = v >= 1 + a;  v -= s;  a += s * adapt_inc / theta
# QAT mode   (stage 2): the same steps in integer units, i.e. the run_int_reference math.
#   Both leak floors use straight-through gradients. a stays an exact integer.
#   As in lif.py, the int16 clip is applied once after the whole sum (train.py checks agreement).
# Gradients flow through a into earlier spikes (as in Bellec et al. 2018), so training can
# learn to use the longer adaptation memory. The output layer reuses LIFNet's steps unchanged.

try:
    import torch
    import torch.nn.functional as F
    from common import spike_fn
    from lif import LIFNet
except ImportError:  # the integer reference above works without PyTorch
    torch = None

if torch is not None:

    class ALIFNet(LIFNet):
        """Model B for training. Input x: [batch, T, N_IN] of 0/1.
        Weights, init, clamp_weights, quantize and load_int_weights are inherited from LIFNet."""

        def __init__(self, cfg=CONFIG):
            super().__init__(cfg)
            # Fixed (not learnable) adaptation so it matches a - (a >> k_a) and a += adapt_inc on hardware
            self.rho = 1.0 - 2.0 ** -cfg["adapt_shift"]

        def float_alif_step(self, v, a, current):
            v = self.beta * v + current                     # leak, then integrate
            a = self.rho * a                                # adaptation leak
            s = spike_fn(v - (1.0 + a))                     # fire against the adapted threshold
            v = v - s.detach()                              # reset by the base threshold
            a = a + s * (self.cfg["adapt_inc"] / self.scale)
            return v, a, s

        def qat_alif_step(self, v, a, current):
            k, theta = self.cfg["leak_shift"], self.cfg["theta"]
            leak = v / 2 ** k
            v = v - (leak + (torch.floor(leak) - leak).detach())   # v - (v >> k), straight-through
            leak_a = a / 2 ** self.cfg["adapt_shift"]
            a = a - (leak_a + (torch.floor(leak_a) - leak_a).detach())   # a - (a >> k_a)
            v = torch.clamp(v + current, V_MIN, V_MAX)             # integrate + saturate (once)
            s = spike_fn((v - (theta + a)) / theta)  # v >= theta + a; surrogate in float-mode units
            v = v - s.detach() * theta
            a = a + s * self.cfg["adapt_inc"]
            return v, a, s

        def forward(self, x, qat=False):
            """Returns (output spike counts [B, n_outputs], used as logits; mean hidden spike rate)."""
            if qat:
                w1, w2 = self.int_weight(self.fc1), self.int_weight(self.fc2)
                step1, step2 = self.qat_alif_step, self.qat_step
            else:
                w1, w2 = self.fc1.weight, self.fc2.weight
                step1, step2 = self.float_alif_step, self.float_step
            B, T, _ = x.shape
            current1 = F.linear(x, w1)            # [B, T, n_hidden], all timesteps at once
            v1 = x.new_zeros(B, self.cfg["n_hidden"])
            a1 = x.new_zeros(B, self.cfg["n_hidden"])
            v2 = x.new_zeros(B, self.cfg["n_outputs"])
            counts, hidden = 0, 0
            for t in range(T):
                v1, a1, s1 = step1(v1, a1, current1[:, t])
                v2, s2 = step2(v2, F.linear(s1, w2))   # layer 2 sees layer 1's spikes from this step
                counts = counts + s2
                hidden = hidden + s1.mean()
            return counts, hidden / T

    Net = ALIFNet   # train.py looks for <model>.Net
