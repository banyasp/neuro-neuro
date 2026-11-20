"""
Ensemble accumulator ("hardware council" voter): combines the 2-bit output streams of the
member models (lif, alif, the spiking CNN, ...) into one seizure decision per window.

Every member sees the same input window and emits 2 output spikes per timestep:
neuron 0 = non-seizure, neuron 1 = seizure. The accumulator reads those 2 bits from every
member at every timestep, so its input is a [T, 2 * n_members] spike raster:
line 2*m = member m's neuron 0, line 2*m + 1 = member m's neuron 1 (same layout as the
UP/DOWN pairs of the EEG input).

This file has the same parts as lif.py (no float model: nothing here is trained by gradient):
  1. CONFIG: the shared spec (common.py) plus this block's constants.
  2. Integer reference (NumPy only): bit-exact model of the hardware math, plus a vectorised
     version that works from per-window spike counts (for fitting on saved val outputs).
  3. Handoff test cases and spec text, so `python export.py accumulator` works unchanged.
fit_accumulator.py chooses the parameters (shifts, margin, mode) from the members' saved
validation outputs.

Integer math, per window (the hardware contract):
  Per member m: two unsigned counters c0[m], c1[m], reset to 0 at the start of every window.
  Each timestep t:
    1. Count:   c0[m] += s[t, 2m];  c1[m] += s[t, 2m + 1]          for every member m
    2. Contribute (mode register):
         mode 0 "count": (p0[m], p1[m]) = (c0[m], c1[m])           pooled spike counts
         mode 1 "vote":  d = (c1[m] > c0[m]);  (p0[m], p1[m]) = (1 - d, d)   one vote per member
    3. Tally:   A0 = sum_m p0[m] << shift[m];  A1 = sum_m p1[m] << shift[m]   (no multipliers)
    4. Output:  out[t] = one-hot (bit 1 = (A1 - A0 > margin), bit 0 = its complement)
  Decision = out[T-1] bit 1, i.e. seizure if A1 - A0 > margin at the end of the window.

With all shifts 0 and margin 0 this is the members' own rule (count1 > count0, tie = no
seizure) applied to the pooled counts (mode 0), or a strict majority vote (mode 1; with 3
members, seizure if >= 2 members decide seizure). Shifts weight members by powers of two
(the proposal's confidence-weighted voter). The persistence rule (alarm after N consecutive
positive windows) runs after this block, exactly as for a single model in train.py.

Fixed rules first. The same registers also express the classic fixed (untrained) rules, named in
fixed_rules(): pooled counts, and "k of M" votes (any, majority, unanimous). Literature basis:
  - Ghawaly et al., "Exploring Spiking Neural Networks for Binary Classification in Multivariate
    Time Series at the Edge", 2025 International Joint Conference on Neural Networks (IJCNN),
    23 Oct 2025 (doi:10.1109/IJCNN64981.2025.11228041, arXiv:2510.20997):
    SNN ensembles on CHB-MIT with an FPGA deployment:
    a 3-member majority vote clearly beat single networks on EEG, while any-vote degraded it.
    They used fixed votes for interpretability, low cost and serial/parallel member execution.
  - R.P.W. Duin, "The Combining Classifier: to Train or Not to Train?", ICPR 2002:
    Duin argues fixed rules are suboptimal when members differ in reliability; with little data,
    plain majority voting is often recommended over trained combiners. Each patient here has ~2
    validation seizures, so the fixed rules are the default and weights must earn their place.
  - Learned weights, when used, come from a logistic regression on the members' outputs, rounded
    to powers of two by params_from_linear() so they run on the same block. This follows the
    best-performing aggregation in Borovac et al., "Ensemble Learning Using Individual Neonatal
    Data for Seizure Detection", IEEE J. Transl. Eng. Health Med., 18 Aug 2022
    (doi:10.1109/JTEHM.2022.3201167, arXiv:2204.07043): a weighted mean whose weights come from a
    logistic-regression stacking classifier, which beat the plain mean and majority vote.
"""

import itertools

import numpy as np

from common import COMMON_CONFIG, T, V_MAX

# ---------------------------------------------------------------------------
# 1. Config (the spec)
# ---------------------------------------------------------------------------

MEMBERS = ["lif", "alif", "scnn"]   # member module names: models A, B, C (lif.py, alif.py, scnn.py)
MODES = {"count": 0, "vote": 1}     # value of the mode register


def make_config(members=MEMBERS, max_shift=3):
    """Config for an accumulator over the given members.

    Args:
        members: Member model names, in input-line order (member m drives lines 2m, 2m + 1).
        max_shift: Largest allowed per-member shift (weights 1, 2, ..., 2**max_shift).

    Returns:
        The config dict. "n_inputs" is overridden: the accumulator's inputs are the members'
        output spikes (2 per member), not the EEG lines, which it never sees.
    """
    cfg = {
        **COMMON_CONFIG,
        "n_inputs": 2 * len(members),
        "members": list(members),
        "n_members": len(members),
        "n_outputs": 2,                       # one-hot: neuron 0 = non-seizure, neuron 1 = seizure
        "count_bits": int(np.ceil(np.log2(T + 1))),   # per-member counters hold 0..T
        "max_shift": max_shift,
    }
    # Largest tally: every member fires both neurons every step, at the largest shift.
    assert len(members) * (T << max_shift) <= V_MAX, "tallies would not fit int16: lower max_shift"
    return cfg


CONFIG = make_config()


def weight_shapes(cfg=CONFIG):
    """Name -> shape of every int8 parameter array. Files are weights_<name>.mem.

    These are run-time registers, not synaptic weights: shift[m] in [0, max_shift],
    margin (signed), mode (0 = count, 1 = vote).
    """
    return {"shift": (cfg["n_members"], 1), "margin": (1, 1), "mode": (1, 1)}


# ---------------------------------------------------------------------------
# 2. Integer reference (bit-exact with hardware)
# ---------------------------------------------------------------------------

def check_params(weights, cfg=CONFIG):
    """Assert that the parameter arrays have the spec's shapes, types and ranges.

    Args:
        weights: {"shift": int8 [n_members, 1], "margin": int8 [1, 1], "mode": int8 [1, 1]}.
        cfg: Config dict.
    """
    for name, shape in weight_shapes(cfg).items():
        assert weights[name].dtype == np.int8 and weights[name].shape == shape, name
    assert ((weights["shift"] >= 0) & (weights["shift"] <= cfg["max_shift"])).all(), "shift out of range"
    assert int(weights["mode"][0, 0]) in (0, 1), "mode must be 0 (count) or 1 (vote)"


def tally(c0, c1, weights):
    """Steps 2 and 3: the two weighted tallies from the per-member counters.

    Args:
        c0: Non-seizure spike counts, int array [..., n_members].
        c1: Seizure spike counts, int array [..., n_members].
        weights: Parameter dict (see check_params).

    Returns:
        (A0, A1) as int64 arrays of shape [...].
    """
    shift = weights["shift"][:, 0].astype(np.int64)
    if int(weights["mode"][0, 0]) == 1:          # vote: each member contributes one vote
        p1 = (c1 > c0).astype(np.int64)
        p0 = 1 - p1
    else:                                        # count: pooled spike counts
        p0, p1 = c0.astype(np.int64), c1.astype(np.int64)
    return (p0 << shift).sum(axis=-1), (p1 << shift).sum(axis=-1)


def run_int_reference(spikes, weights, cfg=CONFIG):
    """Run one window of member outputs through the integer accumulator.

    Args:
        spikes: 0/1 array [T, 2 * n_members]; line 2m / 2m + 1 = member m's neuron 0 / 1.
        weights: Parameter dict (see check_params).
        cfg: Config dict.

    Returns:
        Per-timestep traces [T, n] written to expected_<key>.mem as int16: "cnt" (counters
        after step t: c0[m] at column 2m, c1[m] at 2m + 1) and "tally" (A0, A1 after step t).
        Plus "out" (one-hot [T, 2]), "counts" ([A0, A1] at the end of the window) and
        "decision" (0/1).
    """
    spikes = np.asarray(spikes)
    T = spikes.shape[0]
    assert spikes.shape[1] == cfg["n_inputs"]
    assert set(np.unique(spikes)) <= {0, 1}, "spikes must be 0/1"
    check_params(weights, cfg)
    margin = int(weights["margin"][0, 0])
    M = cfg["n_members"]

    c0 = np.zeros(M, dtype=np.int64)
    c1 = np.zeros(M, dtype=np.int64)
    cnt = np.zeros((T, 2 * M), dtype=np.int16)
    tallies = np.zeros((T, 2), dtype=np.int16)
    out = np.zeros((T, 2), dtype=np.uint8)

    for t in range(T):
        c0 = c0 + spikes[t, 0::2]                       # 1. count
        c1 = c1 + spikes[t, 1::2]
        a0, a1 = tally(c0, c1, weights)                 # 2-3. contribute + tally
        seizure = int(a1 - a0 > margin)                 # 4. output (signed compare)
        cnt[t, 0::2], cnt[t, 1::2] = c0, c1
        tallies[t] = a0, a1
        out[t] = 1 - seizure, seizure

    return {"cnt": cnt, "tally": tallies, "out": out,
            "counts": tallies[-1].astype(np.int32), "decision": int(out[-1, 1])}


# Functions copied into the standalone handoff/accumulator/reference_accumulator.py
REFERENCE_FUNCS = [weight_shapes, check_params, tally, run_int_reference]


def decide_from_counts(member_counts, weights):
    """Vectorised window decisions from per-window member spike counts (no timestep loop).

    The decision only depends on the end-of-window counters, so this equals
    run_int_reference(...)["decision"] for every window (test_cases asserts it).

    Args:
        member_counts: Int array [n_windows, n_members, 2] of (count0, count1) per member,
            e.g. stacked from each member's val_outputs.npz.
        weights: Parameter dict (see check_params).

    Returns:
        (tallies int64 [n_windows, 2], decisions int64 [n_windows]).
    """
    c = np.asarray(member_counts, dtype=np.int64)
    a0, a1 = tally(c[..., 0], c[..., 1], weights)
    return np.stack([a0, a1], axis=-1), (a1 - a0 > int(weights["margin"][0, 0])).astype(np.int64)


def make_params(shifts, margin=0, mode="count"):
    """Parameter dict from plain Python values.

    Args:
        shifts: Per-member shift, in member order.
        margin: Seizure if A1 - A0 > margin (int8).
        mode: "count" or "vote".

    Returns:
        {"shift": int8 [n_members, 1], "margin": int8 [1, 1], "mode": int8 [1, 1]}.
    """
    return {"shift": np.array(shifts, dtype=np.int8).reshape(-1, 1),
            "margin": np.array([[margin]], dtype=np.int8),
            "mode": np.array([[MODES[mode]]], dtype=np.int8)}


def vote_margin(k, n_members):
    """Margin that turns vote mode (all shifts 0) into "seizure if at least k members say so".

    With equal shifts, A1 = votes for seizure and A0 = n_members - A1, so
    A1 - A0 > margin  <=>  2 * A1 - n_members > margin. Choosing margin = 2k - n_members - 1
    makes that true exactly when A1 >= k.

    Args:
        k: Votes needed, 1 <= k <= n_members.
        n_members: Number of members.

    Returns:
        The int margin.
    """
    assert 1 <= k <= n_members
    return 2 * k - n_members - 1


def fixed_rules(n_members):
    """The untrained combination rules, as parameter dicts (all shifts 0).

    Args:
        n_members: Number of members.

    Returns:
        Ordered {name: params}: "pooled" (members' rule on summed spike counts), then the k-of-M
        votes "any" (k = 1), "majority" (k = n_members // 2 + 1) and "unanimous" (k = n_members).
        Names that would repeat a rule (e.g. majority = unanimous for 2 members) are skipped.
    """
    zeros = [0] * n_members
    rules, ks = {"pooled": make_params(zeros, 0, "count")}, set()
    for name, k in (("any", 1), ("majority", n_members // 2 + 1), ("unanimous", n_members)):
        if k not in ks:
            ks.add(k)
            rules[name] = make_params(zeros, vote_margin(k, n_members), "vote")
    return rules


def features(member_counts, mode):
    """Per-member signed evidence for seizure, as the accumulator sums it (before the shifts).

    With all shifts 0, A1 - A0 = sum_m features[m] in both modes, so a linear model on these
    features is exactly what the accumulator can compute once its weights are powers of two.

    Args:
        member_counts: Int array [n_windows, n_members, 2] of (count0, count1) per member.
        mode: "count" (feature = count1 - count0) or "vote" (feature = +1 if the member decides
            seizure, i.e. count1 > count0, else -1).

    Returns:
        Int array [n_windows, n_members].
    """
    c = np.asarray(member_counts, dtype=np.int64)
    diff = c[..., 1] - c[..., 0]
    return diff if mode == "count" else np.where(diff > 0, 1, -1)


def params_from_linear(weights, bias, mode, max_shift):
    """Round a linear rule "seizure if bias + sum_m weights[m] * features[m] > 0" to accumulator
    parameters (power-of-two weights and an integer margin).

    The smallest representable weight is w_ref = max(smallest positive weight,
    largest weight / 2**max_shift), so the largest weight fits in max_shift. Then
    shift[m] = round(log2(weights[m] / w_ref)) clipped to [0, max_shift] (weights <= 0, which the
    block cannot express, get shift 0), and margin = round(-bias / w_ref) clipped to int8.

    Args:
        weights: Real per-member weights, length n_members.
        bias: Real intercept.
        mode: "count" or "vote" (the feature definition the weights were fitted on).
        max_shift: Largest allowed shift.

    Returns:
        Parameter dict (see make_params), or None if no weight is positive.
    """
    w = np.asarray(weights, dtype=float)
    if not (w > 0).any():
        return None
    w_ref = max(w[w > 0].min(), w.max() / 2 ** max_shift)
    shifts = [int(np.clip(np.round(np.log2(x / w_ref)), 0, max_shift)) if x > 0 else 0 for x in w]
    margin = int(np.clip(np.round(-bias / w_ref), -128, 127))
    return make_params(shifts, margin, mode)


def shift_grid(n_members, max_shift):
    """Every shift vector with min 0. Adding 1 to every shift only doubles both tallies, so
    vectors with min > 0 repeat a min-0 vector at a doubled margin."""
    return [s for s in itertools.product(range(max_shift + 1), repeat=n_members) if min(s) == 0]


# ---------------------------------------------------------------------------
# Handoff: test cases and spec text (used by export.py)
# ---------------------------------------------------------------------------

def member_raster(rng, p0, p1):
    """Synthetic member outputs [T, 2 * n_members]: member m's neuron 0 / 1 fire with p0[m] / p1[m]."""
    p = np.stack([p0, p1], axis=1).reshape(-1)          # line order 2m, 2m + 1
    return (rng.random((T, len(p))) < p).astype(np.uint8)


def test_cases(rng, cfg=CONFIG):
    """List of (folder, description, weights, spikes). Folder "." = top of handoff/accumulator/.

    Assumes 3 members (the default CONFIG). Every case also checks decide_from_counts
    against run_int_reference.
    """
    assert cfg["n_members"] == 3, "test cases are written for 3 members"
    cases = []

    # The k-of-M rules, checked against plain vote counting on every pattern of member decisions
    # (member says seizure: counts (0, 1); says non-seizure: counts (1, 0)).
    for n in (2, 3, 4):
        for k in range(1, n + 1):
            p = make_params([0] * n, vote_margin(k, n), "vote")
            for says in itertools.product((0, 1), repeat=n):
                counts = np.array([[[1 - s, s] for s in says]])
                assert decide_from_counts(counts, p)[1][0] == int(sum(says) >= k), f"{k}-of-{n} on {says}"
    assert list(fixed_rules(3)) == ["pooled", "any", "majority", "unanimous"]
    assert list(fixed_rules(2)) == ["pooled", "any", "majority"]   # 2-of-2 majority = unanimous

    # Rounding linear rules to shifts, against values worked out by hand.
    def rounded(w, b, mode="count", ms=3):
        p = params_from_linear(w, b, mode, ms)
        return p["shift"][:, 0].tolist(), int(p["margin"][0, 0]), int(p["mode"][0, 0])
    assert rounded([1, 2, 4], -3) == ([0, 1, 2], 3, 0)           # exact powers of two, w_ref = 1
    assert rounded([0.5, 0.5, 8], 0) == ([0, 0, 3], 0, 0)       # w_ref = 8 / 2**3 = 1: 0.5 -> 2^-1 -> clip 0
    assert rounded([3, 6, 12], 1.4, "vote") == ([0, 1, 2], 0, 1)  # w_ref = 3; margin round(-1.4/3) = 0
    assert rounded([2, -1, 2.9], -5) == ([0, 0, 1], 2, 0)       # negative -> 0; log2(1.45) = 0.54 -> 1; 5/2 -> 2
    assert params_from_linear([-1, 0, -2], 0, "count", 3) is None
    assert features(np.array([[[3, 5], [5, 3], [4, 4]]]), "count").tolist() == [[2, -2, 0]]
    assert features(np.array([[[3, 5], [5, 3], [4, 4]]]), "vote").tolist() == [[1, -1, -1]]  # tie -> -1
    # Exact powers of two + integer bias: the block must reproduce the linear rule on any counts.
    counts = np.random.default_rng(662).integers(0, T + 1, size=(2000, 3, 2))   # own rng: cases below unchanged
    for mode in MODES:
        w, b = np.array([1, 4, 2]), -3
        _, dec = decide_from_counts(counts, params_from_linear(w, b, mode, cfg["max_shift"]))
        assert (dec == (b + features(counts, mode) @ w > 0)).all(), f"linear rule != block ({mode})"

    def add(folder, desc, weights, spikes, expect=None):
        r = run_int_reference(spikes, weights, cfg)
        counts = np.stack([spikes[:, 0::2].sum(0), spikes[:, 1::2].sum(0)], axis=-1)[None]
        tallies, dec = decide_from_counts(counts, weights)
        assert dec[0] == r["decision"] and (tallies[0] == r["counts"]).all(), f"{folder}: vectorised != reference"
        if expect is not None:
            assert r["decision"] == expect, f"{folder}: expected decision {expect}"
        cases.append((folder, desc, weights, spikes))
        return r

    # Count mode, shifts [0, 1, 2]: realistic-looking streams, members mildly disagree.
    add(".", "count mode, shifts [0, 1, 2], margin 0; random member outputs "
        "(member 0 leans non-seizure, members 1-2 lean seizure)",
        make_params([0, 1, 2]), member_raster(rng, [0.30, 0.10, 0.10], [0.10, 0.25, 0.20]))

    # The shifts flip the decision: unweighted pooled counts say no seizure, weighted say seizure.
    spikes = member_raster(rng, [0.60, 0.05, 0.05], [0.0, 0.20, 0.20])
    assert decide_from_counts(np.stack([spikes[:, 0::2].sum(0), spikes[:, 1::2].sum(0)], -1)[None],
                              make_params([0, 0, 0]))[1][0] == 0
    add("weighted", "count mode, shifts [0, 2, 2]: member 0 says non-seizure loudly, members 1-2 "
        "say seizure quietly. Unweighted would decide 0; the shifts make it 1",
        make_params([0, 2, 2]), spikes, expect=1)

    # Vote mode: 2 of 3 members decide seizure -> majority says seizure.
    add("vote", "vote mode, shifts [0, 0, 0], margin 0: members 1-2 decide seizure, member 0 "
        "doesn't. Strict majority (>= 2 of 3) -> 1",
        make_params([0, 0, 0], 0, "vote"), member_raster(rng, [0.30, 0.05, 0.05], [0.05, 0.30, 0.30]), expect=1)

    # Tie: A1 == A0 with margin 0 -> no seizure (same tie rule as the members).
    spikes = np.zeros((T, cfg["n_inputs"]), dtype=np.uint8)
    spikes[::4, 0] = 1                                  # member 0: 32 non-seizure spikes
    spikes[::8, 3] = 1                                  # member 1: 16 seizure spikes, x2 by shift 1
    r = add("tie", "count mode, shifts [0, 1, 0], margin 0: member 0 fires neuron 0 every 4th step, "
            "member 1 fires neuron 1 every 8th step. Final A0 = 32, A1 = 16 << 1 = 32: tie -> 0 "
            "(out is 1 at some earlier steps, where A1 is briefly ahead)", make_params([0, 1, 0]), spikes, expect=0)
    assert r["counts"].tolist() == [32, 32], "tie case: hand-computed tallies are 32, 32"
    assert r["out"][0].tolist() == [0, 1], "tie case: at t = 0, A0 = 1 and A1 = 1 << 1 = 2, so out = seizure"

    # Largest tallies: every line on every step, max shifts, most negative margin.
    ms = cfg["max_shift"]
    spikes = np.ones((T, cfg["n_inputs"]), dtype=np.uint8)
    r = add("max", f"count mode, shifts all {ms}, margin -128, every line on every step: "
            f"counters reach {T}, tallies reach {cfg['n_members'] * (T << ms)} (widest values). "
            "A1 - A0 = 0 > -128 -> 1", make_params([ms] * 3, -128), spikes, expect=1)
    assert r["counts"].max() == cfg["n_members"] * (T << ms)
    return cases


def spec_md(cfg=CONFIG):
    """Accumulator section of spec.md."""
    M, ms, cb = cfg["n_members"], cfg["max_shift"], cfg["count_bits"]
    members = ", ".join(f"{m} = `{name}`" for m, name in enumerate(cfg["members"]))
    max_tally = M * (T << ms)
    return f"""## Accumulator: {M}-member voter

**Inputs are not EEG.** This block reads the members' output spikes, so its `N_IN` is
`2 * N_MEMBERS = {cfg['n_inputs']}` (the EEG `N_IN` in the table above does not apply here).
Line `2*m` = member `m`'s output neuron 0 (non-seizure), line `2*m + 1` = its neuron 1
(seizure), from the **same timestep** of the same window. Members: {members}.

| Name | Value | Notes |
|---|---|---|
| `N_MEMBERS` | {M} | |
| `N_IN` | {cfg['n_inputs']} | 2 output lines per member |
| `COUNT_BITS` | {cb} | unsigned per-member counters, 0..{T} |
| `MAX_SHIFT` | {ms} | member weights 1, 2, ..., {2 ** ms} |
| Tally width | int16 | max {max_tally}, fits without saturation |

Run-time parameters (loaded from `weights_*.mem`, int8):

| Register | Shape | Meaning |
|---|---|---|
| `shift[m]` | [{M}][1] | member `m`'s weight is `2^shift[m]`, 0 <= shift <= {ms} |
| `margin` | [1][1] | signed; seizure if `A1 - A0 > margin` |
| `mode` | [1][1] | 0 = count (pooled spike counts), 1 = vote (one vote per member) |

### Math, per window

All counters reset to 0 at the start of every window. Each timestep `t`:

1. **Count:** for every member `m`: `c0[m] += s[2m]`, `c1[m] += s[2m+1]`.
2. **Contribute:** mode 0: `(p0[m], p1[m]) = (c0[m], c1[m])`.
   Mode 1: `d = (c1[m] > c0[m])`, `(p0[m], p1[m]) = (1 - d, d)` (tie = non-seizure vote).
3. **Tally:** `A0 = sum_m (p0[m] << shift[m])`, `A1 = sum_m (p1[m] << shift[m])`.
4. **Output:** `out[t][1] = (A1 - A0 > margin)` (signed compare), `out[t][0] = !out[t][1]`.

- **Decision** = `out[T-1][1]`. The traces before `T-1` are the running decision (for early exit).
- Mode 0 tallies only grow by `(s << shift)` each step, so hardware can update `A0`/`A1`
  incrementally instead of recomputing the sum. Mode 1 depends on the current counters.
- No multipliers: the shifts are wiring offsets (or a small barrel shifter if `shift` stays a register).
- The persistence rule (alarm after `N` consecutive positive windows) is downstream of this block.

### Parameters and traces

| File | Contents |
|---|---|
| `weights_shift.mem` | {M} words, int8: address `m` = `shift[m]` |
| `weights_margin.mem` | 1 word, int8 |
| `weights_mode.mem` | 1 word, int8: 0 = count, 1 = vote |
| `expected_cnt.mem` | T*{2 * M} words, int16: address `t*{2 * M} + 2m` = `c0[m]`, `+ 2m+1` = `c1[m]`, after step `t` |
| `expected_tally.mem` | T*2 words, int16: address `t*2` = `A0`, `t*2 + 1` = `A1`, after step `t` |
| `expected_out.mem` | T words of 2 bits: one-hot running decision |
| `expected_counts.mem` | 2 words: final `A0`, `A1` (not spike counts for this block) |
"""
