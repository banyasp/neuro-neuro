"""
Write the hardware handoff folder for one or more models.

    python export.py            # exports lif
    python export.py lif alif   # exports each named model to handoff/<name>/

Each model is a module (lif.py, alif.py, ...) that provides the interface described
in ML/README.md: CONFIG, weight_shapes, run_int_reference, REFERENCE_FUNCS,
test_cases, spec_md. This file never needs to change when a model is added.

Files are .mem (hex, for Verilog $readmemh):
  - signed values: two's complement, fixed width (int8 = 2 hex digits, int16 = 4)
  - spike rasters: one line per timestep, all lines packed into one word, bit i = line i (LSB first)
  - multi-value traces (membranes): one value per line, address t*N + j

handoff/<name>/ is deleted and regenerated on every run.
Right now test cases use random int8 weights and synthetic spikes (fixed seed).
Later, train.py / preprocess.py call export() with trained weights and real EEG spikes.
"""

import importlib
import inspect
import pprint
import shutil
import sys
from pathlib import Path

import numpy as np

from common import COMMON_CONFIG

HANDOFF_DIR = Path(__file__).parent / "handoff"

# Keys in a run_int_reference result that are not per-timestep traces
NOT_TRACES = ("out", "counts", "decision")
COUNT_BITS = 16   # width of each output spike count in expected_counts.mem


# ---------------------------------------------------------------------------
# All handoff file writing lives here
# ---------------------------------------------------------------------------

def write_mem_ints(path, values, bits, comment):
    """One value per line, hex two's complement, ceil(bits/4) digits. 2D input is row-major."""
    digits = (bits + 3) // 4
    mask = (1 << bits) - 1
    with open(path, "w") as f:
        f.write(f"// {comment}\n")
        for v in np.asarray(values).reshape(-1):
            f.write(f"{int(v) & mask:0{digits}x}\n")


def write_mem_bits(path, rows, comment):
    """One line per row: the row's 0/1 values packed into one word, bit i = column i."""
    rows = np.asarray(rows)
    digits = (rows.shape[1] + 3) // 4
    with open(path, "w") as f:
        f.write(f"// {comment}\n")
        for row in rows:
            word = sum(int(b) << i for i, b in enumerate(row))
            f.write(f"{word:0{digits}x}\n")


def export(model, weights, spikes, out_dir):
    """Run model's integer reference on one window and write its test-vector files.

    model:   the model module (e.g. lif)
    weights: {name: int8 array}, shapes from model.weight_shapes()
    spikes:  0/1 array [T, n_inputs]
    Returns the reference result dict.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = model.CONFIG
    r = model.run_int_reference(spikes, weights, cfg)
    T = len(spikes)

    for name, W in weights.items():
        n_in, n_out = W.shape
        write_mem_ints(out_dir / f"weights_{name}.mem", W, cfg["weight_bits"],
                       f"weights_{name}: int{cfg['weight_bits']} [{n_in}][{n_out}], "
                       f"{W.size} words, addr = i*{n_out} + j")
    write_mem_bits(out_dir / "input_spikes.mem", spikes,
                   f"input_spikes: {T} words of {spikes.shape[1]} bits, addr = t, bit i = input line i")

    for key, trace in r.items():
        if key in NOT_TRACES:
            continue
        n = trace.shape[1]
        if key.startswith("spk"):
            write_mem_bits(out_dir / f"expected_{key}.mem", trace,
                           f"expected_{key}: {T} words of {n} bits, addr = t, bit j = neuron j")
        else:
            write_mem_ints(out_dir / f"expected_{key}.mem", trace, cfg["mem_bits"],
                           f"expected_{key}: int{cfg['mem_bits']}, {T * n} words, addr = t*{n} + j")

    n_out = r["out"].shape[1]
    write_mem_bits(out_dir / "expected_out.mem", r["out"],
                   f"expected_out: {T} words of {n_out} bits, addr = t, bit j = output neuron j")
    write_mem_ints(out_dir / "expected_counts.mem", r["counts"], COUNT_BITS,
                   f"expected_counts: {n_out} words, unsigned {COUNT_BITS}-bit spike count per output neuron")
    write_mem_ints(out_dir / "expected_decision.mem", [r["decision"]], 1,
                   "expected_decision: 1 word, 1 = seizure, 0 = no seizure")
    return r


def case_stats(spikes, r):
    parts = [f"input density {spikes.mean():.3f}"]
    for key, trace in r.items():
        if key.startswith("spk"):
            parts.append(f"{key} density {trace.mean():.3f}")
        elif key.startswith("mem"):
            parts.append(f"{key} range [{trace.min()}, {trace.max()}]")
    parts.append(f"output counts {r['counts'].tolist()}, decision {r['decision']}")
    return ", ".join(parts)


def write_spec(out_dir, model, name, cases):
    """spec.md = shared input spec + the model's own section + file list + test cases."""
    c = COMMON_CONFIG
    case_lines = "\n".join(f"| `{folder}` | {desc} | {stats} |" for folder, desc, stats in cases)
    spec = f"""# Hardware spec: `{name}`

Generated by `ML/export.py` from `ML/common.py` and `ML/{name}.py`. Do not edit by hand.

Treat sizes and constants as Verilog **parameters** (they may be tuned).
The **math rules** must match exactly; ML will tell you before changing them.

## Inputs (shared by every model)

| Name | Value | Notes |
|---|---|---|
| `N_IN` | {c['n_inputs']} | {c['n_channels']} EEG channels x 2 (UP, DOWN) |
| `T` | {c['window_len']} | timesteps per window ({c['window_len'] / c['fs']:g} s at {c['fs']} Hz) |
| `WINDOW_STEP` | {c['window_step']} | new window every {c['window_step']} samples ({c['window_step'] / c['fs']:g} s) |
| Weight type | signed int{c['weight_bits']} | |
| Membrane type | signed int{c['mem_bits']}, saturating | [{-2 ** (c['mem_bits'] - 1)}, {2 ** (c['mem_bits'] - 1) - 1}] |

Input line order: line `2*ch` = UP spike of channel `ch`, line `2*ch + 1` = DOWN spike.

{model.spec_md(model.CONFIG)}
## File format (`.mem`, for `$readmemh`)

- Hex, no `0x` prefix. First line of every file is a `//` comment describing it.
- **Signed values** (weights, membranes): two's complement, fixed width. int8 = 2 hex digits
  (`-1` = `ff`), int16 = 4 hex digits (`-1` = `ffff`). Sign-extend when you load them.
- **Spike rasters** (`input_spikes`, `expected_spk_*`, `expected_out`): one word per timestep
  (address = `t`), all lines packed into the word, **bit `i` = line/neuron `i`** (LSB = 0).
  E.g. `reg [N_IN-1:0] spikes [0:T-1]; $readmemh("input_spikes.mem", spikes);`
- **Multi-value traces** (`expected_mem_*`): one value per word, address = `t*N + j`.

| File | Contents |
|---|---|
| `weights_<name>.mem` | int8, one per word, row-major (layouts above) |
| `input_spikes.mem` | T words of N_IN bits |
| `expected_spk_<layer>.mem` | T words, one bit per neuron |
| `expected_mem_<layer>.mem` | T*N words, int16 |
| `expected_out.mem` | T words, one bit per output neuron |
| `expected_counts.mem` | one unsigned {COUNT_BITS}-bit word per output neuron: spikes over the window |
| `expected_decision.mem` | one word: 1 = seizure, 0 = no seizure |

`reference_{name}.py` (top folder only) is the standalone integer reference (NumPy only).
`python reference_{name}.py` re-checks every case.

## Test cases

| Folder | What it tests | Stats |
|---|---|---|
{case_lines}

Weights are random for now. Trained weights will come in the same format.
"""
    (Path(out_dir) / "spec.md").write_text(spec)


# Appended to reference_<name>.py so hardware can re-check (or regenerate) the vectors.
CHECK_CODE = f'''

# ---------------------------------------------------------------------------
# Re-check the handoff .mem files: python reference_<name>.py
# ---------------------------------------------------------------------------

NOT_TRACES = ("out", "counts", "decision")
COUNT_BITS = {COUNT_BITS}
'''

CHECK_CODE += '''

def read_mem(path):
    """Hex words from a $readmemh file, skipping // comments and blank lines."""
    words = []
    for line in Path(path).read_text().splitlines():
        line = line.split("//")[0].strip()
        if line:
            words.append(int(line, 16))
    return words


def to_signed(words, bits):
    return np.array([w - (1 << bits) if w >> (bits - 1) else w for w in words], dtype=np.int64)


def unpack_bits(words, n):
    """Packed words -> [len(words), n] 0/1 array, bit i = column i."""
    return np.array([[(w >> i) & 1 for i in range(n)] for w in words], dtype=np.int64)


def load_case(folder):
    folder = Path(folder)
    weights = {}
    for name, shape in weight_shapes().items():
        w = to_signed(read_mem(folder / f"weights_{name}.mem"), CONFIG["weight_bits"])
        weights[name] = w.reshape(shape).astype(np.int8)
    spikes = unpack_bits(read_mem(folder / "input_spikes.mem"), CONFIG["n_inputs"])
    return weights, spikes


def check_case(folder):
    folder = Path(folder)
    weights, spikes = load_case(folder)
    r = run_int_reference(spikes, weights)
    expected = {}
    for key, trace in r.items():
        if key in NOT_TRACES:
            continue
        words = read_mem(folder / f"expected_{key}.mem")
        if key.startswith("spk"):
            expected[key] = unpack_bits(words, trace.shape[1])
        else:
            expected[key] = to_signed(words, CONFIG["mem_bits"]).reshape(trace.shape)
    expected["out"] = unpack_bits(read_mem(folder / "expected_out.mem"), r["out"].shape[1])
    expected["counts"] = np.array(read_mem(folder / "expected_counts.mem"))
    expected["decision"] = read_mem(folder / "expected_decision.mem")[0]

    ok = True
    for key, value in expected.items():
        if not np.array_equal(value, r[key]):
            print(f"  MISMATCH in {key}")
            ok = False
    print(f"{folder.name}: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    here = Path(__file__).parent
    folders = [here] + sorted(p for p in here.iterdir() if (p / "input_spikes.mem").exists())
    results = [check_case(f) for f in folders]
    raise SystemExit(0 if all(results) else 1)
'''


def write_reference_copy(out_dir, model, name):
    """Write a standalone reference_<name>.py (NumPy only) next to the vectors."""
    src = (
        f'"""Standalone integer reference for `{name}`. Generated by ML/export.py.\n'
        f'Same code as ML/{name}.py. Run it to re-check every test case in this folder."""\n\n'
        "from pathlib import Path\n\nimport numpy as np\n\n"
        f"CONFIG = {pprint.pformat(model.CONFIG, sort_dicts=False)}\n\n"
        'W_MIN = -(2 ** (CONFIG["weight_bits"] - 1))\n'
        'W_MAX = 2 ** (CONFIG["weight_bits"] - 1) - 1\n'
        'V_MIN = -(2 ** (CONFIG["mem_bits"] - 1))\n'
        'V_MAX = 2 ** (CONFIG["mem_bits"] - 1) - 1\n\n\n'
        + "\n\n".join(inspect.getsource(f) for f in model.REFERENCE_FUNCS)
        + CHECK_CODE
    )
    (Path(out_dir) / f"reference_{name}.py").write_text(src)


def export_model(name):
    """Regenerate handoff/<name>/ from the model's test cases."""
    model = importlib.import_module(name)
    out = HANDOFF_DIR / name
    if out.exists():
        shutil.rmtree(out)

    rng = np.random.default_rng(model.CONFIG["seed"])
    cases = []
    for folder, desc, weights, spikes in model.test_cases(rng):
        r = export(model, weights, spikes, out / folder)
        cases.append((folder, desc, case_stats(spikes, r)))

    write_spec(out, model, name, cases)
    write_reference_copy(out, model, name)

    print(f"wrote {out}")
    for folder, _, stats in cases:
        print(f"  {folder:12s} {stats}")


if __name__ == "__main__":
    for name in sys.argv[1:] or ["lif"]:
        export_model(name)
