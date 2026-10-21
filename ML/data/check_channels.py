"""Settle the CHB-MIT channel count (42 vs 44 input lines) on real data.

Usage (from ML/):
    ../.venv/Scripts/python data/check_channels.py data/raw/chb01/chb01_01.edf [more .edf files ...]

For each file it checks two things:
  1. Are the two T8-P8 channels identical?          -> if yes, drop one (46 -> 44)
  2. Is P7-T7 exactly the negative of T7-P7?        -> if yes, drop it  (44 -> 42)
"""
import sys

import mne
import numpy as np

mne.set_log_level("ERROR")


def find(raw, label):
    """Return indices of every channel whose name starts with `label` (MNE renames duplicates to T8-P8-0, T8-P8-1)."""
    return [i for i, ch in enumerate(raw.ch_names) if ch.upper().startswith(label)]


def report(name, diff, ref):
    # EDF stores 16-bit integers, so an exact copy/negative can still differ by ~1 rounding step.
    # Count it as identical if the worst-case difference is under 0.1% of the signal's std.
    max_diff_uv = np.max(np.abs(diff)) * 1e6
    ref_uv = np.std(ref) * 1e6
    same = max_diff_uv < 1e-3 * ref_uv
    verdict = "IDENTICAL (within EDF rounding)" if same else "NOT identical"
    print(f"  {name}: max |difference| = {max_diff_uv:.4f} uV  (signal std {ref_uv:.1f} uV)  -> {verdict}")
    return same


all_t8 = []
all_p7 = []
for path in sys.argv[1:]:
    raw = mne.io.read_raw_edf(path, preload=True)
    data = raw.get_data()  # volts, shape (n_channels, n_samples)
    print(f"\n{path}  ({len(raw.ch_names)} channels)")

    t8 = find(raw, "T8-P8")
    if len(t8) >= 2:
        all_t8.append(report("T8-P8 copy 1 vs copy 2", data[t8[0]] - data[t8[1]], data[t8[0]]))
    else:
        print(f"  T8-P8: found {len(t8)} copies, can't compare")

    a, b = find(raw, "T7-P7"), find(raw, "P7-T7")
    if a and b:
        corr = np.corrcoef(data[a[0]], data[b[0]])[0, 1]
        print(f"  correlation(T7-P7, P7-T7) = {corr:.6f}")
        all_p7.append(report("P7-T7 + T7-P7 (should be 0 if exact negative)", data[a[0]] + data[b[0]], data[a[0]]))
    else:
        print("  T7-P7 / P7-T7: one is missing, can't compare")

print("\n=== RESULT ===")
if not all_t8 or not all_p7:
    print("Not enough channels found to decide.")
else:
    drop_t8 = all(all_t8)
    drop_p7 = all(all_p7)
    n_channels = 23 - int(drop_t8) - int(drop_p7)
    print(f"Duplicate T8-P8 identical in every file: {drop_t8}")
    print(f"P7-T7 exact negative in every file:      {drop_p7}")
    print(f"=> {n_channels} channels, {2 * n_channels} input lines")