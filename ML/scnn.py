"""
Model C: spiking convolutional NN (SCNN) for EEG seizure detection.

A spiking, int8 version of the first stage of EEGNet (Lawhern et al. 2018): temporal filters
shared across channels, then per-filter (depthwise) spatial filters, then a small output layer.
Models A and B (lif.py, alif.py) are fully connected and only remember the past through one
fixed leak; this model learns the *shape* of each channel's signal over the last few hundred ms.

    input s[ch, p](t)        23 channels x p in {UP, DOWN} (the same 46 lines as lif / alif)
      -> layer 1  temporal   C filters with K taps, SHARED by every channel   -> LIF, 23 x C neurons
                             filter group g uses dilation d_g (default: 4 filters at d = 1
                             spanning 125 ms, 4 at d = 3 spanning 375 ms)
      -> layer 2  spatial    depthwise: for each filter f, 23 channels -> D units -> LIF, C x D neurons
      -> layer 3  output     fully connected C x D -> 2                        -> LIF, 2 neurons
    decision: seizure if output neuron 1 spikes more than neuron 0 (tie = no seizure), as in lif

With readout = "fc", layers 2 and 3 are replaced by one fully connected layer 23 x C -> 2.

The temporal convolution is just delayed synapses: at step t, layer-1 neuron (ch, f) adds
W1[2k + p, f] for every tap k and polarity p where input s[ch, p] spiked at step t - k * d_f.
Tap k = 0 is the current step. Steps before the window start count as no spike.

Every neuron uses Model A's math unchanged (lif.lif_step_int; conv_step_int is the same thing
for all channels of a filter group at once): leak v -= v >> k, integrate in increasing input
order with int16 saturation after every add, fire at v >= theta, reset by subtracting theta. Each layer uses the previous layer's spikes from the same timestep. Every
layer passes binary spikes, so the hardware never multiplies.

With neuron = "alif" (module scnn_alif.py), the hidden layers (layer 1, and layer 2 if depthwise)
use Model B's ALIF neuron instead (alif.alif_step_int; conv_alif_step_int for layer 1): an
adaptation a per neuron that leaks as a -= a >> k_a, raises the threshold to theta + a and grows by
adapt_inc per spike. The output layer stays plain LIF, as in alif.py.

This file has three parts, like lif.py:
  1. CONFIG: the shared input spec (common.py) plus this model's constants.
  2. Integer reference (NumPy only): bit-exact model of the hardware math.
  3. Float model (PyTorch) for surrogate-gradient training, float and QAT modes.
"""

import numpy as np

from common import COMMON_CONFIG, N_IN, T, W_MIN, W_MAX, V_MIN, V_MAX, random_spikes, random_int8
from alif import alif_step_int
from lif import lif_step_int

# ---------------------------------------------------------------------------
# 1. Config (the spec)
# ---------------------------------------------------------------------------

CONFIG = {
    **COMMON_CONFIG,
    "kernel_len": 16,         # K taps per temporal filter
    "dilations": [1, 3],      # one dilation per filter group: taps at t, t - d, ..., t - (K-1) d
    "group_filters": [4, 4],  # filters in each group, so C = 8 filters in total
    "depth_mult": 2,          # D spatial units per filter (depthwise readout only)
    "readout": "depthwise",   # "depthwise" (layers 2 + 3) or "fc" (one layer 23 x C -> 2)
    "share_filters": True,    # False: every channel gets its own filter bank (ablation)
    "n_outputs": 2,           # neuron 0 = non-seizure, neuron 1 = seizure
    "leak_shift": 5,          # k in v = v - (v >> k), same as lif
    "theta": 64,              # firing threshold, same for every layer
    "neuron": "lif",          # "alif": hidden layers (1, and 2 if depthwise) use alif.py's neuron
    "adapt_shift": 7,         # ALIF only, as in alif.py: a = a - (a >> k_a), tau ~ 1 s
    "adapt_inc": 16,          # ALIF only, as in alif.py: a += adapt_inc per spike
}


def filter_groups(cfg=CONFIG):
    """[(dilation, first filter, end filter), ...] for each filter group."""
    groups, first = [], 0
    for d, n in zip(cfg["dilations"], cfg["group_filters"]):
        groups.append((d, first, first + n))
        first += n
    return groups


def weight_shapes(cfg=CONFIG):
    """Name -> shape of every int8 weight array, all [n_in][n_out]. Files are weights_<name>.mem.

    l1: [2K][C]        row 2k + p = tap k, polarity p (0 = UP, 1 = DOWN); column f = filter
        [23 * 2K][C]   if share_filters is False: row ch * 2K + 2k + p
    l2: [23][C * D]    depthwise: W2[ch][f * D + j], unit j of filter f, from channel ch
        [23 * C][2]    readout "fc": row ch * C + f = layer-1 neuron (ch, f)
    l3: [C * D][2]     depthwise only: row f * D + j
    """
    assert len(cfg["dilations"]) == len(cfg["group_filters"]), "one dilation per filter group"
    assert cfg["readout"] in ("depthwise", "fc"), cfg["readout"]
    n_ch, K, n_out = cfg["n_channels"], cfg["kernel_len"], cfg["n_outputs"]
    C = sum(cfg["group_filters"])
    l1 = (2 * K, C) if cfg["share_filters"] else (n_ch * 2 * K, C)
    if cfg["readout"] == "depthwise":
        D = cfg["depth_mult"]
        return {"l1": l1, "l2": (n_ch, C * D), "l3": (C * D, n_out)}
    return {"l1": l1, "l2": (n_ch * C, n_out)}


def history_depth(cfg=CONFIG):
    """Frames of input-spike history layer 1 reads (current frame included): (K-1) * max(d) + 1."""
    return (cfg["kernel_len"] - 1) * max(cfg["dilations"]) + 1


# ---------------------------------------------------------------------------
# 2. Integer reference (bit-exact with hardware)
# ---------------------------------------------------------------------------

def tap_inputs(spikes, t, d, K):
    """Layer-1 inputs of every channel at step t for dilation d: 0/1 array [n_ch, 2K],
    entry [ch, 2k + p] = spikes[t - k*d, 2*ch + p] (0 before the window start).
    The column order (tap k ascending, UP before DOWN) is the hardware add order."""
    x = np.zeros((K, spikes.shape[1]), dtype=np.int64)
    for k in range(K):
        if t - k * d >= 0:
            x[k] = spikes[t - k * d]
    return x.reshape(K, -1, 2).transpose(1, 0, 2).reshape(spikes.shape[1] // 2, 2 * K)


def conv_integrate(v, x, W):
    """Integrate step of a layer-1 filter group, every channel at once: for each input i = 2k + p
    in ascending order, add W[i] to the channels whose input i spiked, saturating after every add.

    v: int32 [n_ch, n_filters] (updated in place and returned);  x: 0/1 [n_ch, 2K] from tap_inputs
    W: int8 [2K, n_filters] (shared by every channel) or [n_ch, 2K, n_filters] (one bank per channel)
    """
    for i in range(x.shape[1]):
        on = x[:, i] == 1                                  # channels whose input i spiked
        if on.any():
            w = W[i] if W.ndim == 2 else W[on, i]
            v[on] = np.clip(v[on] + w.astype(np.int32), V_MIN, V_MAX)
    return v


def conv_step_int(v, x, W, k, theta):
    """One timestep of one layer-1 filter group, every channel at once. The same neuron math as
    lif.lif_step_int; each row is one channel, which has its own active inputs.
    Returns (new v, spikes uint8 [n_ch, n_filters])."""
    v = v - (v >> k)                                       # 1. leak
    v = conv_integrate(v, x, W)                            # 2. integrate + saturate
    s = (v >= theta).astype(np.uint8)                      # 3. fire
    v = v - s.astype(np.int32) * theta                     # 4. reset by subtraction
    return v, s


def conv_alif_step_int(v, a, x, W, k, theta, k_a, inc):
    """ALIF version of conv_step_int (neuron = "alif"): the same math as alif.alif_step_int.
    a: int32 [n_ch, n_filters] adaptation. Returns (new v, new a, spikes uint8 [n_ch, n_filters])."""
    v = v - (v >> k)                                       # 1. leak membrane
    a = a - (a >> k_a)                                     # 2. leak adaptation
    v = conv_integrate(v, x, W)                            # 3. integrate + saturate
    s = (v >= theta + a).astype(np.uint8)                  # 4. fire against the adapted threshold
    v = v - s.astype(np.int32) * theta                     # 5. reset by the base threshold
    a = a + s.astype(np.int32) * inc                       # 6. adapt
    return v, a, s


def run_int_reference(spikes, weights, cfg=CONFIG):
    """Run one window through the integer model.

    spikes:  0/1 array [T, n_inputs]
    weights: {name: int8 array}, shapes from weight_shapes(cfg)

    Returns per-timestep traces [T, n] (membranes after reset), each written to
    expected_<key>.mem, plus "out" output spikes, "counts", "decision".
    Layer-1 neuron (ch, f) is column ch * C + f; layer-2 unit (f, j) is column f * D + j.
    """
    spikes = np.asarray(spikes)
    T = spikes.shape[0]
    assert spikes.shape[1] == cfg["n_inputs"] == 2 * cfg["n_channels"]
    assert set(np.unique(spikes)) <= {0, 1}, "spikes must be 0/1"
    for name, shape in weight_shapes(cfg).items():
        assert weights[name].dtype == np.int8 and weights[name].shape == shape, name

    n_ch, K, n_out = cfg["n_channels"], cfg["kernel_len"], cfg["n_outputs"]
    k, theta = cfg["leak_shift"], cfg["theta"]
    k_a, inc = cfg["adapt_shift"], cfg["adapt_inc"]
    alif = cfg["neuron"] == "alif"       # hidden layers (1, and 2 if depthwise) are ALIF; output stays LIF
    assert cfg["neuron"] in ("lif", "alif"), cfg["neuron"]
    # a never exceeds (inc + 1) * 2^k_a - 1, so a and theta + a fit in int16 without saturation
    assert not alif or theta + ((inc + 1) << k_a) <= V_MAX, "adaptation could overflow int16"
    C = sum(cfg["group_filters"])
    depthwise = cfg["readout"] == "depthwise"
    n2 = C * cfg["depth_mult"] if depthwise else n_out    # layer-2 size
    W1 = weights["l1"]

    v1 = np.zeros((n_ch, C), dtype=np.int32)
    a1 = np.zeros((n_ch, C), dtype=np.int32)
    v2 = np.zeros(n2, dtype=np.int32)
    a2 = np.zeros(n2, dtype=np.int32)
    v3 = np.zeros(n_out, dtype=np.int32)
    mem1 = np.zeros((T, n_ch * C), dtype=np.int16)
    adapt1 = np.zeros((T, n_ch * C), dtype=np.int16)
    spk1 = np.zeros((T, n_ch * C), dtype=np.uint8)
    mem2 = np.zeros((T, n2), dtype=np.int16)
    adapt2 = np.zeros((T, n2), dtype=np.int16)
    spk2 = np.zeros((T, n2), dtype=np.uint8)
    mem3 = np.zeros((T, n_out), dtype=np.int16)
    spk3 = np.zeros((T, n_out), dtype=np.uint8)

    for t in range(T):
        # Layer 1: in each filter group, neuron (ch, f) is an LIF (or ALIF) neuron whose 2K inputs
        # are channel ch's delayed spikes, with weights W1[:, f].
        s1 = np.zeros((n_ch, C), dtype=np.uint8)
        W = W1 if cfg["share_filters"] else W1.reshape(n_ch, 2 * K, C)
        for d, f0, f1 in filter_groups(cfg):
            x = tap_inputs(spikes, t, d, K)
            if alif:
                v1[:, f0:f1], a1[:, f0:f1], s1[:, f0:f1] = conv_alif_step_int(
                    v1[:, f0:f1], a1[:, f0:f1], x, W[..., f0:f1], k, theta, k_a, inc)
            else:
                v1[:, f0:f1], s1[:, f0:f1] = conv_step_int(v1[:, f0:f1], x, W[..., f0:f1], k, theta)
        mem1[t], adapt1[t], spk1[t] = v1.reshape(-1), a1.reshape(-1), s1.reshape(-1)

        if depthwise:
            # Layer 2: filter f's D units see only filter f's 23 layer-1 neurons, ch ascending
            D = cfg["depth_mult"]
            s2 = np.zeros(n2, dtype=np.uint8)
            for f in range(C):
                u = slice(f * D, (f + 1) * D)
                if alif:
                    v2[u], a2[u], s2[u] = alif_step_int(v2[u], a2[u], s1[:, f], weights["l2"][:, u],
                                                        k, theta, k_a, inc)
                else:
                    v2[u], s2[u] = lif_step_int(v2[u], s1[:, f], weights["l2"][:, u], k, theta)
            # Layer 3 (output, always LIF): fully connected, input f * D + j ascending
            v3, s3 = lif_step_int(v3, s2, weights["l3"], k, theta)
            mem2[t], adapt2[t], spk2[t], mem3[t], spk3[t] = v2, a2, s2, v3, s3
        else:
            # Readout "fc": layer 2 is the output layer (LIF), input ch * C + f ascending
            v2, s2 = lif_step_int(v2, s1.reshape(-1), weights["l2"], k, theta)
            mem2[t], spk3[t] = v2, s2

    counts = spk3.sum(axis=0).astype(np.int32)
    r = {"mem_l1": mem1, "spk_l1": spk1, "mem_l2": mem2}
    if alif:
        r["adapt_l1"] = adapt1
    if depthwise:
        r.update({"spk_l2": spk2, "mem_l3": mem3})
        if alif:
            r["adapt_l2"] = adapt2
    return {**r, "out": spk3, "counts": counts, "decision": int(counts[1] > counts[0])}


# Functions copied into the standalone handoff/<model>/reference_<model>.py
REFERENCE_FUNCS = [filter_groups, weight_shapes, history_depth, tap_inputs, conv_integrate, conv_step_int,
                   conv_alif_step_int, lif_step_int, alif_step_int, run_int_reference]


def hardware_cost(cfg=CONFIG):
    """Static hardware cost of a config: weights, membranes, history buffer, worst-case adds per step.
    Worst case assumes the delta encoder's at most one spike (UP or DOWN) per channel per step."""
    shapes = weight_shapes(cfg)
    n_ch, K, n_out = cfg["n_channels"], cfg["kernel_len"], cfg["n_outputs"]
    C = sum(cfg["group_filters"])
    if cfg["readout"] == "depthwise":
        D = cfg["depth_mult"]
        membranes, adds = n_ch * C + C * D + n_out, n_ch * C * K + n_ch * C * D + C * D * n_out
    else:
        membranes, adds = n_ch * C + n_out, n_ch * C * K + n_ch * C * n_out
    cost = {"weights": int(sum(np.prod(s) for s in shapes.values())), "membranes": membranes,
            "history_frames": history_depth(cfg), "worst_adds_per_step": adds}
    if cfg["neuron"] == "alif":          # one adaptation register per hidden (non-output) neuron
        cost["adapt_registers"] = membranes - n_out
    return cost


# ---------------------------------------------------------------------------
# Handoff: test cases and spec text (used by export.py)
# ---------------------------------------------------------------------------

def random_weights(rng, cfg=CONFIG):
    return {name: random_int8(rng, shape) for name, shape in weight_shapes(cfg).items()}


def test_cases(rng, cfg=CONFIG):
    """List of (folder, description, weights, spikes). Folder "." = top of handoff/scnn/."""
    n_ch, K = cfg["n_channels"], cfg["kernel_len"]
    groups = filter_groups(cfg)
    last = list(weight_shapes(cfg))[-1]          # output layer's weights
    cases = [(".", "random int8 weights, 5% random input spikes",
              random_weights(rng, cfg), random_spikes(rng, 0.05))]

    # Saturation: group-0 filters all +127, the other groups all -128, all inputs on for the first half.
    w = random_weights(rng, cfg)
    a0, b0 = groups[0][1], groups[0][2]
    w["l1"][:] = W_MIN
    w["l1"][:, a0:b0] = W_MAX
    spikes = np.zeros((T, N_IN), dtype=np.uint8)
    spikes[: T // 2] = 1
    r = run_int_reference(spikes, w, cfg)
    assert r["mem_l1"].max() == V_MAX - cfg["theta"] and r["mem_l1"].min() == V_MIN, \
        "saturation case didn't saturate"
    cases.append(("saturation", f"l1 filters {a0}-{b0 - 1} all +127, the rest all -128; all inputs on "
                  f"for t < {T // 2}, then off. Hits both int16 rails, then leaks. "
                  f"The + rail appears as {V_MAX - cfg['theta']} (32767 minus reset)", w, spikes))

    w, spikes = random_weights(rng, cfg), random_spikes(rng, 0.30)
    if cfg["neuron"] == "alif":
        r = run_int_reference(spikes, w, cfg)
        assert r["adapt_l1"].max() > cfg["adapt_inc"], "dense case: layer-1 adaptation never built up"
    cases.append(("dense", "random weights, 30% input spikes (heavy load"
                  + ("; adaptation builds up over several spikes)" if cfg["neuron"] == "alif" else ")"),
                  w, spikes))

    # Seizure: the output layer's column 0 all -128 and column 1 all +127, so decision = 1.
    w = random_weights(rng, cfg)
    w[last][:, 0] = W_MIN
    w[last][:, 1] = W_MAX
    spikes = random_spikes(rng, 0.05)
    assert run_int_reference(spikes, w, cfg)["decision"] == 1, "seizure case didn't decide 1"
    cases.append(("seizure", f"random weights except {last} column 0 all -128 and column 1 all +127, "
                  "5% input spikes. Decision = 1", w, spikes))

    # Taps: one UP spike on channel 0 at t0, every UP tap weight = +1 (DOWN = 0). Layer-1 neuron
    # (0, f) then gains +1 at exactly t0 + k * d_f for k = 0..K-1. Its membrane stays below 32,
    # so it neither leaks nor fires: expected_mem_l1 column f is the number of taps reached so far.
    w = random_weights(rng, cfg)
    w["l1"][:] = 0
    w["l1"][0::2] = 1
    t0 = 10
    spikes = np.zeros((T, N_IN), dtype=np.uint8)
    spikes[t0, 0] = 1
    r = run_int_reference(spikes, w, cfg)
    for d, a, b in groups:
        want = np.array([sum(t0 + kk * d <= t for kk in range(K)) for t in range(T)])
        for f in range(a, b):
            assert np.array_equal(r["mem_l1"][:, f], want), "taps case: tap mapping is off"
    assert not r["mem_l1"][:, sum(cfg["group_filters"]):].any(), "taps case: other channels moved"
    spans = ", ".join(f"filters {a}-{b - 1} (d = {d}) at t = {t0}, {t0 + d}, ..., {t0 + (K - 1) * d}"
                      for d, a, b in groups)
    cases.append(("taps", f"tap / dilation mapping: one UP spike on channel 0 at t = {t0}; l1 UP-tap "
                  f"weights +1, DOWN-tap weights 0. Layer-1 neurons (0, f) step up by 1 at: {spans}. "
                  "Every other layer-1 neuron stays 0", w, spikes))
    return cases


def spec_md(cfg=CONFIG):
    """Model-specific section of spec.md."""
    n_ch, K, n_out = cfg["n_channels"], cfg["kernel_len"], cfg["n_outputs"]
    C = sum(cfg["group_filters"])
    depthwise = cfg["readout"] == "depthwise"
    D = cfg["depth_mult"]
    shapes = weight_shapes(cfg)
    cost = hardware_cost(cfg)
    dt_ms = 1000 / cfg["fs"]
    group_rows = "\n".join(
        f"| {a}-{b - 1} | {d} | t, t-{d}, ..., t-{(K - 1) * d} | {(K - 1) * d * dt_ms + dt_ms:.0f} ms |"
        for d, a, b in filter_groups(cfg))
    l1_layout = (f"`[{2 * K}][{C}]`, address `(2k + p)*{C} + f`. **Shared by every channel**"
                 if cfg["share_filters"] else
                 f"`[{n_ch * 2 * K}][{C}]`, address `(ch*{2 * K} + 2k + p)*{C} + f` (one bank per channel)")
    l1_row = "W1[2k + p][f]" if cfg["share_filters"] else f"W1[ch*{2 * K} + 2k + p][f]"
    alif = cfg["neuron"] == "alif"
    hid = "ALIF" if alif else "LIF"      # hidden-layer neuron; the output layer is always LIF

    if depthwise:
        arch = (f"{cfg['n_inputs']} inputs -> layer 1 temporal ({n_ch} x {C} = {n_ch * C} {hid}) -> "
                f"layer 2 depthwise spatial ({C} x {D} = {C * D} {hid}) -> layer 3 output ({n_out} LIF)")
        later = f"""**Layer 2 (depthwise spatial)**, unit `u = f*{D} + j` (filter `f`, unit `j < {D}`): its inputs are
the {n_ch} layer-1 neurons of filter `f` only. For each `ch` with `s1[ch][f] == 1`, in **increasing `ch`**:
`v = sat16(v + W2[ch][u])`.

**Layer 3 (output)**, neuron `o`: inputs are the {C * D} layer-2 units. For each `u` with `s2[u] == 1`, in
**increasing `u`**: `v = sat16(v + W3[u][o])`."""
        weight_rows = f"""| `weights_l2.mem` | W2 `[{n_ch}][{C * D}]`, address `ch*{C * D} + u`, `u = f*{D} + j` |
| `weights_l3.mem` | W3 `[{C * D}][{n_out}]`, address `u*{n_out} + o` |"""
        trace_rows = f"""| `expected_mem_l2.mem` | T*{C * D} words, int16: address `t*{C * D} + u`, after reset |
| `expected_spk_l2.mem` | T words of {C * D} bits: bit `u` = layer-2 unit `u` |
| `expected_mem_l3.mem` | T*{n_out} words, int16: output membranes, after reset |"""
        if alif:
            trace_rows += f"\n| `expected_adapt_l2.mem` | T*{C * D} words, int16: layer-2 adaptation `a`, after the increment |"
    else:
        arch = (f"{cfg['n_inputs']} inputs -> layer 1 temporal ({n_ch} x {C} = {n_ch * C} {hid}) -> "
                f"layer 2 output, fully connected ({n_out} LIF)")
        later = f"""**Layer 2 (output, fully connected)**, neuron `o`: inputs are the {n_ch * C} layer-1 neurons,
index `n = ch*{C} + f`. For each `n` with `s1[n] == 1`, in **increasing `n`**: `v = sat16(v + W2[n][o])`."""
        weight_rows = f"| `weights_l2.mem` | W2 `[{n_ch * C}][{n_out}]`, address `n*{n_out} + o`, `n = ch*{C} + f` |"
        trace_rows = f"| `expected_mem_l2.mem` | T*{n_out} words, int16: output membranes, after reset |"

    if alif:
        adapt_rows = (f"\n| `K_ADAPT` (adaptation shift) | {cfg['adapt_shift']} | hidden layers only |"
                      f"\n| `ADAPT_INC` | {cfg['adapt_inc']} | hidden layers only |")
        neuron_math = f"""**Hidden layers (layer 1{', layer 2' if depthwise else ''}) are ALIF neurons**, exactly Model B's (`alif`):
each has an adaptation register `a` (int16, starts at 0 every window). Per step: **leak** `v = v - (v >>> K_LEAK)`,
**adaptation leak** `a = a - (a >>> K_ADAPT)`, **integrate** its active inputs in the order below with
`v = sat16(v + w)` after **every** add, **fire** `s = (v >= THETA + a)`, **reset** `v = v - THETA` where `s == 1`
(the base threshold), **adapt** `a = a + ADAPT_INC` where `s == 1`. `a` stays below
`(ADAPT_INC + 1) * 2^K_ADAPT = {(cfg['adapt_inc'] + 1) << cfg['adapt_shift']}`, so `THETA + a` never overflows.
**The output layer is plain LIF** (Model A's neuron): the same steps without `a`."""
        trace_rows += (f"\n| `expected_adapt_l1.mem` | T*{n_ch * C} words, int16: layer-1 adaptation `a`, "
                       f"address `t*{n_ch * C} + n`, after the increment |")
        adapt_cost = f" Plus {cost['adapt_registers']} int16 adaptation registers (one per hidden neuron)."
    else:
        adapt_rows, adapt_cost = "", ""
        neuron_math = """Every neuron is Model A's LIF neuron, unchanged: **leak** `v = v - (v >>> K_LEAK)`, **integrate** its active
inputs in the order below with `v = sat16(v + w)` after **every** add, **fire** `s = (v >= THETA)`,
**reset** `v = v - THETA` where `s == 1`."""

    return f"""## Model C: spiking convolutional NN (SCNN){', ALIF hidden neurons' if alif else ''}

{arch}. {cost['weights']} int8 weights, {cost['membranes']} int16 membranes.{adapt_cost}

| Name | Value | Notes |
|---|---|---|
| `N_CH` | {n_ch} | EEG channels; channel `ch` owns input lines `2*ch` (UP) and `2*ch + 1` (DOWN) |
| `K` | {K} | taps per temporal filter |
| `C` | {C} | temporal filters (layer-1 neurons per channel) |
| `D` | {D if depthwise else '-'} | spatial units per filter (layer 2) |
| Readout | {cfg['readout']} | |
| `K_LEAK` (leak shift) | {cfg['leak_shift']} | every layer |
| `THETA` | {cfg['theta']} | every layer |{adapt_rows}
| Bias | none | |

Filter groups (tap `k` of filter `f` reads the input frame from `k * d_f` steps ago; tap 0 = now):

| Filters | Dilation `d` | Frames read | Span |
|---|---|---|---|
{group_rows}

**Spike history:** layer 1 needs the last **{cost['history_frames']} input frames** (current one included), i.e. a
{cost['history_frames']}-deep x {cfg['n_inputs']}-bit shift register. Frames before the window start are all-zero
(clear the history at the start of every window, like the membranes).

Time scale: one timestep = one EEG sample at {cfg['fs']} Hz ({dt_ms:.2f} ms).

### Math, per timestep

{neuron_math} Each layer uses the previous layer's spikes **from the same
timestep** (layer 1, then layer 2, then layer 3, within one step).

**Layer 1 (temporal, convolution as delayed synapses)**, neuron `(ch, f)`, filter `f` in a group with
dilation `d`: its {2 * K} inputs are `x[2k + p] = s_in[t - k*d][2*ch + p]` for tap `k = 0..{K - 1}` and polarity
`p` (0 = UP, 1 = DOWN). For each `i = 2k + p` with `x[i] == 1`, in **increasing `i`** (tap ascending, UP
before DOWN): `v = sat16(v + {l1_row})`.

{later}

- All membranes{', adaptation registers' if alif else ''} and the spike history reset to 0 at the start of every window.
- **Decision:** seizure (1) if output neuron 1 spiked **more** times than neuron 0. Tie = 0.
- Layer-1 neuron `(ch, f)` is index `n = ch*{C} + f` in every trace.

### Cost

Worst case **{cost['worst_adds_per_step']} adds per step**, assuming at most one of UP / DOWN fires per
channel per step (true for the delta encoder; the random test vectors break this, so don't rely on it).
Layer 1 accounts for {n_ch * C * K} of them ({n_ch} channels x {C} filters x {K} taps).

### Weights and traces

| File | Contents |
|---|---|
| `weights_l1.mem` | W1 {l1_layout} |
{weight_rows}
| `expected_mem_l1.mem` | T*{n_ch * C} words, int16: address `t*{n_ch * C} + n`, `n = ch*{C} + f`, after reset |
| `expected_spk_l1.mem` | T words of {n_ch * C} bits: bit `n` = layer-1 neuron `n` |
{trace_rows}

The `taps/` case checks the tap and dilation mapping on its own (see its description below).
"""


# ---------------------------------------------------------------------------
# 3. Trainable model (PyTorch): float mode and QAT mode
# ---------------------------------------------------------------------------
# Same units and steps as lif.py (LIFNet.float_step / qat_step are reused as is): float theta 1.0,
# W_int = round(W_float * theta). In QAT mode the int16 clip is applied once after the whole sum
# instead of after every add; train.py checks agreement with run_int_reference.
# Layer 1's currents for the whole window are computed up front (the input spikes are known), then
# each layer's LIF runs over all T steps before the next layer starts. For a feedforward net with
# same-step timing that's the same as interleaving the layers step by step.

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from alif import ALIFNet
    from lif import LIFNet
except ImportError:  # the integer reference above works without PyTorch
    torch = None

if torch is not None:

    def tap_tensor(x, d, K):
        """[B, T, N_IN] 0/1 -> [B, T, n_ch, 2K]: entry 2k + p = x[t - k*d, 2*ch + p] (0 before t = 0).
        The torch version of tap_inputs, for every step and channel at once."""
        B, T, _ = x.shape
        xc = x.view(B, T, -1, 2)
        taps = [F.pad(xc, (0, 0, 0, 0, k * d, 0))[:, :T] for k in range(K)]   # shift k*d steps later
        return torch.stack(taps, dim=3).flatten(3)

    def layer1_currents(x, w1, cfg):
        """Layer-1 input currents [B, T, n_ch, C] for every step: sum over taps and polarities of
        W1[2k + p, f] * x[t - k*d_f, 2*ch + p]. w1 is [2K, C] (shared) or [n_ch * 2K, C]."""
        K, n_ch = cfg["kernel_len"], cfg["n_channels"]
        out = []
        for d, a, b in filter_groups(cfg):
            h = tap_tensor(x, d, K)                                   # [B, T, n_ch, 2K]
            if cfg["share_filters"]:
                out.append(h @ w1[:, a:b])
            else:
                out.append(torch.einsum("btci,cif->btcf", h, w1.view(n_ch, 2 * K, -1)[:, :, a:b]))
        return torch.cat(out, dim=-1)

    class SCNNNet(nn.Module):
        """Model C for training. Input x: [batch, T, N_IN] of 0/1.
        Weights are stored in the int8 file layout ([n_in, n_out], see weight_shapes)."""

        float_step = LIFNet.float_step      # same neuron as lif.py, float and QAT
        qat_step = LIFNet.qat_step
        float_alif_step = ALIFNet.float_alif_step   # neuron = "alif": same neuron as alif.py
        qat_alif_step = ALIFNet.qat_alif_step
        INIT_STD = 0.5                      # as in lif.py (see the note there), at the default K = 16

        def __init__(self, cfg=CONFIG):
            super().__init__()
            self.cfg = cfg
            # Scale the init by fan-in so every variant starts with about the default's drive per
            # neuron: layer 1 sees 2K inputs (K = 1 barely fires otherwise; capped inside the int8
            # grid, +-2), layer 2 sees 23 (depthwise) or 23 x C (fc, which otherwise pins one output).
            shapes = weight_shapes(cfg)
            std = {"l1": min(1.5, self.INIT_STD * (16 / cfg["kernel_len"]) ** 0.5),
                   "l2": self.INIT_STD * (cfg["n_channels"] / shapes["l2"][0]) ** 0.5}
            self.w = nn.ParameterDict({name: nn.Parameter(torch.randn(shape) * std.get(name, self.INIT_STD))
                                       for name, shape in shapes.items()})
            self.beta = 1.0 - 2.0 ** -cfg["leak_shift"]
            self.rho = 1.0 - 2.0 ** -cfg["adapt_shift"]   # ALIF adaptation decay (used only if neuron = "alif")
            self.scale = cfg["theta"]

        def int_weight(self, w):
            """round(scale * w) clipped to int8, straight-through gradient. Same rule as quantize()."""
            w = w * self.scale
            return w + (torch.clamp(torch.round(w), W_MIN, W_MAX) - w).detach()

        def run_lif(self, current, step):
            """LIF over all steps: current [B, T, ...] -> spikes [B, T, ...]."""
            v = current.new_zeros(current[:, 0].shape)
            spikes = []
            for t in range(current.shape[1]):
                v, s = step(v, current[:, t])
                spikes.append(s)
            return torch.stack(spikes, dim=1)

        def run_alif(self, current, step):
            """ALIF over all steps (membrane v and adaptation a, both start at 0): -> spikes [B, T, ...]."""
            v = current.new_zeros(current[:, 0].shape)
            a = torch.zeros_like(v)
            spikes = []
            for t in range(current.shape[1]):
                v, a, s = step(v, a, current[:, t])
                spikes.append(s)
            return torch.stack(spikes, dim=1)

        def layers(self, x, qat=False):
            """Every layer's spikes: (s1 [B, T, n_ch, C], s2 [B, T, C*D] or None, out [B, T, n_out])."""
            w = {n: self.int_weight(p) if qat else p for n, p in self.w.items()}
            step = self.qat_step if qat else self.float_step           # output layer: always LIF
            if self.cfg["neuron"] == "alif":                           # hidden layers
                alif_step = self.qat_alif_step if qat else self.float_alif_step
                hidden = lambda current: self.run_alif(current, alif_step)
            else:
                hidden = lambda current: self.run_lif(current, step)
            s1 = hidden(layer1_currents(x, w["l1"], self.cfg))
            if self.cfg["readout"] == "fc":
                return s1, None, self.run_lif(s1.flatten(2) @ w["l2"], step)   # input ch*C + f
            D = self.cfg["depth_mult"]
            # depthwise: unit f*D + j sums filter f over channels
            s2 = hidden((s1.repeat_interleave(D, dim=-1) * w["l2"]).sum(dim=2))
            return s1, s2, self.run_lif(s2 @ w["l3"], step)

        def forward(self, x, qat=False):
            """Returns (output spike counts [B, n_outputs], used as logits; mean hidden spike rate)."""
            s1, s2, out = self.layers(x, qat)
            n1 = s1[0, 0].numel()
            n2 = 0 if s2 is None else s2.shape[-1]
            hidden = s1.mean() if s2 is None else (s1.mean() * n1 + s2.mean() * n2) / (n1 + n2)
            return out.sum(dim=1), hidden

        @torch.no_grad()
        def cost_stats(self, x, qat=False):
            """Measured activity on a batch: spike rate per layer and mean adds per step (what an
            event-driven implementation actually does). Sums over the batch, for averaging."""
            cfg = self.cfg
            s1, s2, out = self.layers(x, qat)
            B, T = x.shape[:2]
            adds1 = sum(tap_tensor(x, d, cfg["kernel_len"]).sum() * (b - a) for d, a, b in filter_groups(cfg))
            if s2 is None:
                adds2, adds3 = s1.sum() * cfg["n_outputs"], 0.0
            else:
                adds2, adds3 = s1.sum() * cfg["depth_mult"], s2.sum() * cfg["n_outputs"]
            stats = {"rate_l1": s1.mean(), "rate_l2": s2.mean() if s2 is not None else None,
                     "rate_out": out.mean(), "adds_l1": adds1 / (B * T), "adds_l2": adds2 / (B * T),
                     "adds_l3": adds3 / (B * T)}
            stats = {k: (None if v is None else float(v) * B) for k, v in stats.items()}
            stats["adds_per_step"] = sum(stats[k] for k in ("adds_l1", "adds_l2", "adds_l3"))
            return stats

        def clamp_weights(self):
            """Keep float weights in [-2, 127/64] so round(64 w) fits int8. Call after every step."""
            with torch.no_grad():
                for p in self.w.values():
                    p.clamp_(W_MIN / self.scale, W_MAX / self.scale)

        def quantize(self):
            """{name: int8 NumPy array} in the weight_shapes layout."""
            return {n: np.clip(np.round(p.detach().cpu().numpy() * self.scale), W_MIN, W_MAX).astype(np.int8)
                    for n, p in self.w.items()}

        def load_int_weights(self, weights):
            """Inverse of quantize(): float weights = W_int / scale."""
            with torch.no_grad():
                for n, p in self.w.items():
                    p.copy_(torch.from_numpy(weights[n].astype(np.float32)) / self.scale)

    Net = SCNNNet   # train.py looks for <model>.Net
