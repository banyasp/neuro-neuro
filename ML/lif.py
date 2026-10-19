"""
Model A: fully connected LIF SNN (46 -> 64 -> 2) for EEG seizure detection.

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

from common import COMMON_CONFIG, W_MIN, W_MAX, V_MIN, V_MAX, random_spikes, random_int8

# ---------------------------------------------------------------------------
# 1. Config (the spec)
# ---------------------------------------------------------------------------

CONFIG = {
    **COMMON_CONFIG,
    "n_hidden": 64,
    "n_outputs": 2,           # neuron 0 = non-seizure, neuron 1 = seizure
    "leak_shift": 5,          # k in v = v - (v >> k), decay factor 1 - 2^-5 = 0.96875
    "theta": 64,              # firing threshold (integer units), same for both layers
}


def weight_shapes(cfg=CONFIG):
    """Name -> shape of every int8 weight array. Files are weights_<name>.txt."""
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
    each written to expected_<key>.txt, plus "out" output spikes, "counts", "decision".
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
              random_weights(rng, cfg), random_spikes(rng, 0.05, cfg))]

    # Saturation: hidden 0-15 all +127, 16-31 all -128, all inputs on for the first half.
    w = random_weights(rng, cfg)
    w["l1"][:, 0:16] = W_MAX
    w["l1"][:, 16:32] = W_MIN
    spikes = np.zeros((cfg["window_len"], cfg["n_inputs"]), dtype=np.uint8)
    spikes[: cfg["window_len"] // 2] = 1
    r = run_int_reference(spikes, w, cfg)
    # Recorded membranes are after reset, so the + rail shows up as V_MAX - theta
    assert r["mem_l1"].max() == V_MAX - cfg["theta"] and r["mem_l1"].min() == V_MIN, \
        "saturation case didn't saturate"
    cases.append(("saturation", "hidden 0-15 all +127, 16-31 all -128; all inputs on for "
                  f"t < {cfg['window_len'] // 2}, then off. Hits both int16 rails, then leaks. "
                  f"The + rail appears as {V_MAX - cfg['theta']} (32767 minus reset)", w, spikes))

    cases.append(("dense", "random weights, 30% input spikes (heavy load)",
                  random_weights(rng, cfg), random_spikes(rng, 0.30, cfg)))
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
| `weights_l1.txt` | W1 `[{ni}][{nh}]`, {ni * nh} lines: line `i*{nh} + j` = `W1[i][j]` |
| `weights_l2.txt` | W2 `[{nh}][{no}]`, {nh * no} lines: line `i*{no} + j` = `W2[i][j]` |
| `expected_mem_l1.txt` | T rows, {nh} ints: layer-1 membrane **after reset** |
| `expected_spk_l1.txt` | T rows, {nh} 0/1: layer-1 spikes |
| `expected_mem_l2.txt` | T rows, {no} ints: layer-2 membrane **after reset** |
"""


# ---------------------------------------------------------------------------
# 3. Float model for training (PyTorch)
# ---------------------------------------------------------------------------
# Units: the float model uses theta = 1.0. Quantization maps float weights to
# integers with W_int = round(W_float * cfg["theta"]), so float theta 1.0
# becomes integer theta 64 and the two models line up.
# We write the LIF loop by hand instead of using snntorch.Leaky so the order
# of operations matches the integer reference exactly.

try:
    import torch
    import torch.nn as nn
    from common import spike_fn
except ImportError:  # the integer reference above works without PyTorch
    torch = None

if torch is not None:

    class LIFNet(nn.Module):
        """Float version of Model A. Input x: [batch, T, n_inputs] of 0/1."""

        def __init__(self, cfg=CONFIG):
            super().__init__()
            self.cfg = cfg
            self.fc1 = nn.Linear(cfg["n_inputs"], cfg["n_hidden"], bias=False)
            self.fc2 = nn.Linear(cfg["n_hidden"], cfg["n_outputs"], bias=False)
            self.beta = 1.0 - 2.0 ** -cfg["leak_shift"]   # fixed leak, matches v - (v >> k)
            self.theta = 1.0

        def lif_step(self, v, current):
            v = self.beta * v + current           # leak, then integrate
            s = spike_fn(v - self.theta)          # fire
            v = v - s.detach() * self.theta       # reset by subtraction
            return v, s

        def forward(self, x):
            B, T, _ = x.shape
            v1 = x.new_zeros(B, self.cfg["n_hidden"])
            v2 = x.new_zeros(B, self.cfg["n_outputs"])
            out_spikes = []
            for t in range(T):
                v1, s1 = self.lif_step(v1, self.fc1(x[:, t]))
                v2, s2 = self.lif_step(v2, self.fc2(s1))
                out_spikes.append(s2)
            out_spikes = torch.stack(out_spikes, dim=1)   # [B, T, n_outputs]
            counts = out_spikes.sum(dim=1)                # [B, n_outputs], use as logits
            return counts, out_spikes

        def quantize(self):
            """Return {"l1": W1, "l2": W2} as int8 NumPy arrays in [n_in, n_out] layout."""
            scale = self.cfg["theta"]
            out = {}
            for name, fc in (("l1", self.fc1), ("l2", self.fc2)):
                w = fc.weight.detach().cpu().numpy().T     # nn.Linear stores [out, in]
                out[name] = np.clip(np.round(w * scale), W_MIN, W_MAX).astype(np.int8)
            return out
