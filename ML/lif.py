"""
Model A: fully connected LIF SNN (N_IN -> 64 -> 2) for EEG seizure detection.
N_IN = 2 * len(common.CHANNELS) (46 with all 23 CHB-MIT channels).

This file has three parts:
  1. CONFIG: the shared input spec (common.py) plus this model's constants.
  2. Integer reference (NumPy only): bit-exact model of the hardware math.
     This generates the golden vectors. No floats anywhere.
  3. Float model (PyTorch): the same network for surrogate-gradient training.
     Same order of operations as the integer reference.

Integer math, per timestep, per layer (the hardware contract):
  1. Leak:      v = v - (v >> k)        arithmetic shift, rounds toward -inf
  2. Integrate: for each input i with s_in[i] == 1, in increasing i order:
                    v = clip(v + W[i, :], -32768, 32767)   (saturate after EVERY add)
  3. Fire:      s_out = (v >= theta)
  4. Reset:     v[s_out] -= theta
Layer 2 uses layer 1's spikes from the same timestep.
Membranes start at 0 at the beginning of every window.
Decision: seizure if output neuron 1 spikes more times than neuron 0 (tie = no seizure).
"""

import numpy as np

from common import COMMON_CONFIG, N_IN, T, W_MIN, W_MAX, V_MIN, V_MAX, random_spikes, random_int8

# ---------------------------------------------------------------------------
# 1. Config (the spec)
# ---------------------------------------------------------------------------

CONFIG = {
    **COMMON_CONFIG,
    "n_hidden": 64,
    "n_outputs": 2,           # neuron 0 = non-seizure, neuron 1 = seizure
    "leak_shift": 5,          # k in v = v - (v >> k), decay 1 - 2^-5 = 0.96875, tau ~ 31.5 steps ~ 0.25 s at 128 Hz
    "theta": 64,              # firing threshold (integer units), same for both layers
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

def lif_step_int(v, s_in, W, k, theta):
    """One timestep of one integer LIF layer.

    v:    int32 array [n_out], current membrane (always within int16 range)
    s_in: 0/1 array [n_in], input spikes this timestep
    W:    int8 array [n_in, n_out]
    Returns (new v, output spikes as uint8 [n_out]).
    """
    # 1. Leak. NumPy >> on signed ints is an arithmetic shift, like Verilog >>>.
    v = v - (v >> k)

    # 2. Integrate + saturate, one input row at a time, in increasing input index.
    for i in np.flatnonzero(s_in):
        v = np.clip(v + W[i].astype(np.int32), V_MIN, V_MAX)

    # 3. Fire
    s_out = (v >= theta).astype(np.uint8)

    # 4. Reset by subtraction (v >= theta > 0 here, so this can't underflow)
    v = v - s_out.astype(np.int32) * theta
    return v, s_out


def run_int_reference(spikes, weights, cfg=CONFIG):
    """Run one window through the integer model.

    spikes:  0/1 array [T, n_inputs]
    weights: {"l1": int8 [n_inputs, n_hidden], "l2": int8 [n_hidden, n_outputs]}

    Returns a dict of per-timestep traces [T, n] (membranes are after reset),
    each written to expected_<key>.mem, plus "out" output spikes, "counts", "decision".
    Keys starting with "spk" are written as packed bits; all others as int16.
    """
    spikes = np.asarray(spikes)
    T = spikes.shape[0]
    assert spikes.shape[1] == cfg["n_inputs"]
    assert set(np.unique(spikes)) <= {0, 1}, "spikes must be 0/1"
    for name, shape in weight_shapes(cfg).items():
        assert weights[name].dtype == np.int8 and weights[name].shape == shape, name
    W1, W2 = weights["l1"], weights["l2"]

    k, theta = cfg["leak_shift"], cfg["theta"]
    v1 = np.zeros(cfg["n_hidden"], dtype=np.int32)
    v2 = np.zeros(cfg["n_outputs"], dtype=np.int32)

    mem1 = np.zeros((T, cfg["n_hidden"]), dtype=np.int16)
    spk1 = np.zeros((T, cfg["n_hidden"]), dtype=np.uint8)
    mem2 = np.zeros((T, cfg["n_outputs"]), dtype=np.int16)
    spk2 = np.zeros((T, cfg["n_outputs"]), dtype=np.uint8)

    for t in range(T):
        v1, s1 = lif_step_int(v1, spikes[t], W1, k, theta)
        v2, s2 = lif_step_int(v2, s1, W2, k, theta)
        mem1[t], spk1[t] = v1, s1
        mem2[t], spk2[t] = v2, s2

    counts = spk2.sum(axis=0).astype(np.int32)
    decision = int(counts[1] > counts[0])
    return {"mem_l1": mem1, "spk_l1": spk1, "mem_l2": mem2, "out": spk2,
            "counts": counts, "decision": decision}


# Functions copied into the standalone handoff/<model>/reference_<model>.py
REFERENCE_FUNCS = [weight_shapes, lif_step_int, run_int_reference]


# ---------------------------------------------------------------------------
# Handoff: test cases and spec text (used by export.py)
# ---------------------------------------------------------------------------

def random_weights(rng, cfg=CONFIG):
    return {name: random_int8(rng, shape) for name, shape in weight_shapes(cfg).items()}


def test_cases(rng, cfg=CONFIG):
    """List of (folder, description, weights, spikes). Folder "." = top of handoff/lif/."""
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
    # Appended last so the cases above keep their random draws (bit-identical files).
    w = random_weights(rng, cfg)
    w["l2"][:, 0] = W_MIN
    w["l2"][:, 1] = W_MAX
    spikes = random_spikes(rng, 0.05)
    assert run_int_reference(spikes, w, cfg)["decision"] == 1, "seizure case didn't decide 1"
    cases.append(("seizure", "random l1, l2 column 0 all -128 and column 1 all +127, "
                  "5% input spikes. Decision = 1", w, spikes))
    return cases


def spec_md(cfg=CONFIG):
    """Model-specific section of spec.md."""
    nh, no, ni = cfg["n_hidden"], cfg["n_outputs"], cfg["n_inputs"]
    return f"""## Model A: LIF SNN {ni} -> {nh} -> {no}

| Name | Value | Notes |
|---|---|---|
| `N_HIDDEN` | {nh} | |
| `N_OUT` | {no} | neuron 0 = non-seizure, neuron 1 = seizure |
| `K` (leak shift) | {cfg['leak_shift']} | |
| `THETA` | {cfg['theta']} | same threshold for both layers |
| Bias | none | |

Time scale: one timestep = one EEG sample at {cfg['fs']} Hz, so a {cfg['window_len']}-step window is
{cfg['window_len'] / cfg['fs']:g} s, with a new window every {cfg['window_step'] / cfg['fs']:g} s. The leak
`K = {cfg['leak_shift']}` gives decay `1 - 2^-{cfg['leak_shift']}` per step, a time constant of about
{1 / -np.log(1 - 2.0 ** -cfg['leak_shift']):.1f} steps = {1 / -np.log(1 - 2.0 ** -cfg['leak_shift']) / cfg['fs']:.2f} s.

Input spikes come from per-channel delta modulation in Python: each channel is divided by a
fixed per-channel scale, then thresholded at `alpha = {cfg['delta_alpha']}` (so each channel has its
own threshold `d_c = alpha * scale_c`). None of this is visible to hardware, which just reads
the spike raster.

### Math, per timestep, per layer

Layer with membrane `v` (int16), input spikes `s_in` (0/1), weights `W` (int8, `[n_in][n_out]`):

1. **Leak:** `v = v - (v >>> K)` (arithmetic shift).
2. **Integrate + saturate:** for each input `i` with `s_in[i] == 1`, in **increasing `i`**:
   `v = sat16(v + W[i][:])`. Saturate after **every** add, not once at the end.
3. **Fire:** `s_out = (v >= THETA)` (signed compare).
4. **Reset:** where `s_out == 1`, `v = v - THETA`.

- Layer 2 uses layer 1's spikes **from the same timestep**.
- All membranes reset to 0 at the start of every window.
- **Decision:** seizure (1) if neuron 1 spiked **more** times than neuron 0 over the window. Tie = 0.

Notes:
- The arithmetic shift rounds toward -inf: a negative `v` leaks all the way to 0, but a
  positive `v` in 1..{2 ** cfg['leak_shift'] - 1} has `v >>> K == 0` and does **not** leak.
- The leak can't overflow, and the reset only runs when `v >= THETA > 0`, so it can't
  underflow. Only step 2 needs saturation logic.
- Order of adds matters only when saturation happens (see the `saturation/` case).

### Weights and traces

| File | Contents |
|---|---|
| `weights_l1.mem` | W1 `[{ni}][{nh}]`, {ni * nh} words: address `i*{nh} + j` = `W1[i][j]` (grouped by input, suits event-driven fetch) |
| `weights_l2.mem` | W2 `[{nh}][{no}]`, {nh * no} words: address `i*{no} + j` = `W2[i][j]` |
| `expected_mem_l1.mem` | T*{nh} words, int16: address `t*{nh} + j` = layer-1 membrane of neuron `j` at step `t`, **after reset** |
| `expected_spk_l1.mem` | T words of {nh} bits: bit `j` = layer-1 neuron `j` spiked at step `t` |
| `expected_mem_l2.mem` | T*{no} words, int16: address `t*{no} + j`, **after reset** |
"""


# ---------------------------------------------------------------------------
# 3. Trainable model (PyTorch): float mode and QAT mode
# ---------------------------------------------------------------------------
# Units: float mode uses theta = 1.0. Quantization maps W_int = round(W_float * cfg["theta"]),
# so float theta 1.0 becomes integer theta 64 and the two modes line up.
#
# Float mode (stage 1):  v = v - v * 2^-k + W x;  s = v >= 1;  v -= s
# QAT mode   (stage 2):  the same steps in integer units, i.e. the run_int_reference math:
#   weights round(64 w) clipped to int8, leak v - floor(v / 2^k), int16 clip, s = v >= 64, v -= 64.
#   Rounding and the leak floor use straight-through gradients.
#   One difference: the spec saturates after EVERY add (in input order); here we clip once
#   after the whole sum. That gives identical results unless a partial sum leaves int16, which
#   has ~512 thresholds of headroom. train.py checks agreement with run_int_reference to be sure.
# We write the LIF loop by hand instead of using snntorch.Leaky so the order of operations
# matches the integer reference exactly.

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from common import spike_fn
except ImportError:  # the integer reference above works without PyTorch
    torch = None

if torch is not None:

    class LIFNet(nn.Module):
        """Model A for training. Input x: [batch, T, N_IN] of 0/1."""

        def __init__(self, cfg=CONFIG):
            super().__init__()
            self.cfg = cfg
            self.fc1 = nn.Linear(N_IN, cfg["n_hidden"], bias=False)
            self.fc2 = nn.Linear(cfg["n_hidden"], cfg["n_outputs"], bias=False)
            # PyTorch's default init is too small here: at ~6% input density the hidden layer
            # barely fires (<1%) and the outputs never do, so surrogate gradients vanish.
            # std 0.5 gives ~10% hidden activity and output spikes in most windows.
            nn.init.normal_(self.fc1.weight, 0.0, 0.5)
            nn.init.normal_(self.fc2.weight, 0.0, 0.5)
            # Fixed (not learnable) leak so it matches v - (v >> k) on hardware
            self.beta = 1.0 - 2.0 ** -cfg["leak_shift"]
            self.scale = cfg["theta"]   # float weight * scale = integer weight

        def float_step(self, v, current):
            v = self.beta * v + current           # leak, then integrate
            s = spike_fn(v - 1.0)                 # fire (theta = 1.0)
            v = v - s.detach()                    # reset by subtraction
            return v, s

        def qat_step(self, v, current):
            k, theta = self.cfg["leak_shift"], self.cfg["theta"]
            leak = v / 2 ** k
            v = v - (leak + (torch.floor(leak) - leak).detach())   # v - (v >> k), straight-through
            v = torch.clamp(v + current, V_MIN, V_MAX)             # integrate + saturate (once)
            s = spike_fn((v - theta) / theta)     # v >= theta; surrogate in the same units as float mode
            v = v - s.detach() * theta
            return v, s

        def int_weight(self, fc):
            """round(scale * w) clipped to int8, straight-through gradient. Same rule as quantize()."""
            w = fc.weight * self.scale
            return w + (torch.clamp(torch.round(w), W_MIN, W_MAX) - w).detach()

        def forward(self, x, qat=False):
            """Returns (output spike counts [B, n_outputs], used as logits; mean hidden spike rate)."""
            if qat:
                w1, w2, step = self.int_weight(self.fc1), self.int_weight(self.fc2), self.qat_step
            else:
                w1, w2, step = self.fc1.weight, self.fc2.weight, self.float_step
            B, T, _ = x.shape
            current1 = F.linear(x, w1)            # [B, T, n_hidden], all timesteps at once
            v1 = x.new_zeros(B, self.cfg["n_hidden"])
            v2 = x.new_zeros(B, self.cfg["n_outputs"])
            counts, hidden = 0, 0
            for t in range(T):
                v1, s1 = step(v1, current1[:, t])
                v2, s2 = step(v2, F.linear(s1, w2))   # layer 2 sees layer 1's spikes from this step
                counts = counts + s2
                hidden = hidden + s1.mean()
            return counts, hidden / T

        def clamp_weights(self):
            """Keep float weights in [-2, 127/64] so round(64 w) fits int8. Call after every step."""
            with torch.no_grad():
                for fc in (self.fc1, self.fc2):
                    fc.weight.clamp_(W_MIN / self.scale, W_MAX / self.scale)

        def quantize(self):
            """Return {"l1": W1, "l2": W2} as int8 NumPy arrays in [n_in, n_out] layout."""
            out = {}
            for name, fc in (("l1", self.fc1), ("l2", self.fc2)):
                w = fc.weight.detach().cpu().numpy().T     # nn.Linear stores [out, in]
                out[name] = np.clip(np.round(w * self.scale), W_MIN, W_MAX).astype(np.int8)
            return out

        def load_int_weights(self, weights):
            """Inverse of quantize(): float weights = W_int / scale (exact, so quantize() gives W_int back)."""
            with torch.no_grad():
                for name, fc in (("l1", self.fc1), ("l2", self.fc2)):
                    fc.weight.copy_(torch.from_numpy(weights[name].T.astype(np.float32)) / self.scale)

    Net = LIFNet   # train.py looks for <model>.Net
