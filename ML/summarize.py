"""
Collect every finished run into one table (validation; test too once --test has been run).

    # from ML/
    python summarize.py --out <runs>                  # one row per run, then mean +- std per model
    python summarize.py --out <runs> --csv table.csv  # also write every row to a CSV

Reads <out>/*/results.json (and test_results.json if present). Runs made before the margin
sweep need train.py --rescore first; they're listed and skipped.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import FS

COLUMNS = [   # (key, header, format)
    ("patient", "patient", "s"), ("model", "model", "s"), ("seed", "seed", "d"), ("window_s", "win s", ".1f"),
    ("hidden", "hid n", "d"),
    ("val_f1", "F1", ".3f"), ("val_sens", "sens", ".3f"), ("val_prec", "prec", ".3f"),
    ("val_fp_h", "FP/h", ".1f"),
    ("spec_n", "N", "d"), ("spec_detected", "det", "s"), ("spec_fa_h", "FA/h", ".2f"),
    ("spec_delay_s", "delay", ".1f"),
    ("best_m", "m*", "d"), ("best_n", "N*", "d"), ("best_detected", "det*", "s"), ("best_fa_h", "FA/h*", ".2f"),
    ("best_delay_s", "delay*", ".1f"),
    ("agree", "agree", ".3f"), ("input_rate", "in", ".3f"), ("hidden_rate", "hid", ".3f"),
    ("test_detected", "test det", "s"), ("test_fa_h", "test FA/h", ".2f"), ("test_delay_s", "test delay", ".1f"),
    ("test_best_detected", "test det*", "s"), ("test_best_fa_h", "test FA/h*", ".2f"),
]
MEAN_KEYS = ["val_f1", "val_sens", "val_prec", "val_fp_h", "spec_fa_h", "spec_delay_s", "best_fa_h",
             "best_delay_s", "hidden_rate", "test_fa_h", "test_delay_s", "test_best_fa_h"]


def find(sweep, n, m=0):
    return next(a for a in sweep if a["n_consec"] == n and a.get("margin", 0) == m)


def hidden_size(run, r):
    """n_hidden from results.json, or (runs made before it was saved) from the int8 weights."""
    n = (r.get("config") or {}).get("n_hidden")
    if n is None and (run / "weights_int8.npz").exists():
        n = int(np.load(run / "weights_int8.npz")["l1"].shape[1])
    return n


def row(run):
    r = json.loads((run / "results.json").read_text())
    if "best_alarm" not in r:
        return None
    spec = find(r["alarm_sweep"], r["alarm_n"])
    m, n = r["best_alarm"]["margin"], r["best_alarm"]["n_consec"]
    best = find(next(x["alarms"] for x in r["margin_sweep"] if x["margin"] == m), n, m)
    v = r["val"]
    out = {"run": run.name, "patient": r["patient"], "model": r["model"], "seed": r["seed"],
           "window_s": r["window"] / FS, "hidden": hidden_size(run, r),
           "val_f1": v["f1"], "val_sens": v["sensitivity"], "val_prec": v["precision"], "val_fp_h": v["fp_per_hour"],
           "spec_n": r["alarm_n"], "spec_detected": f"{spec['detected']}/{spec['seizures']}",
           "spec_fa_h": spec["fa_per_hour"], "spec_delay_s": spec["mean_delay_s"],
           "best_m": m, "best_n": n, "best_detected": f"{best['detected']}/{best['seizures']}",
           "best_fa_h": best["fa_per_hour"], "best_delay_s": best["mean_delay_s"],
           "agree": r["agreement"]["rate"] if r.get("agreement") else None,
           "input_rate": r["density"]["input"], "hidden_rate": r["density"]["hidden"]}
    t_path = run / "test_results.json"
    if t_path.exists():
        t = json.loads(t_path.read_text())
        out.update({"test_detected": f"{t['alarm']['detected']}/{t['alarm']['seizures']}",
                    "test_fa_h": t["alarm"]["fa_per_hour"], "test_delay_s": t["alarm"]["mean_delay_s"]})
        if "best_margin" in t:
            b = t["best_margin"]["alarm"]
            out.update({"test_best_detected": f"{b['detected']}/{b['seizures']}", "test_best_fa_h": b["fa_per_hour"]})
    return out


def cell(v, fmt):
    if v is None:
        return "-"
    return format(v, fmt) if fmt != "s" else str(v)


def print_table(rows, cols):
    head = [h for _, h, _ in cols]
    body = [[cell(r.get(k), f) for k, _, f in cols] for r in rows]
    widths = [max(len(x) for x in col) for col in zip(head, *body)]
    for line in [head, ["-" * w for w in widths]] + body:
        print("  ".join(x.rjust(w) for x, w in zip(line, widths)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "runs")
    p.add_argument("--csv", type=Path, help="also write all rows to this CSV file")
    args = p.parse_args()

    rows, stale = [], []
    for run in sorted(d for d in args.out.iterdir() if (d / "results.json").exists()):
        r = row(run)
        (rows.append(r) if r else stale.append(run.name))
    if stale:
        print(f"skipped (run train.py --rescore on these): {', '.join(stale)}\n")
    if not rows:
        raise SystemExit(f"no finished runs in {args.out}")

    # hide test columns until some run has them
    cols = [c for c in COLUMNS if not c[0].startswith("test") or any(r.get(c[0]) is not None for r in rows)]
    print("Per run (validation unless marked test). N / det / FA/h / delay = alarm at m = 0 (spec);")
    print("starred = best (m, N) from the margin ablation. det = SzCORE events detected / total.\n")
    print_table(rows, cols)

    print("\nMean +- std across runs, per model, window and hidden size:")
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["window_s"], r["hidden"])].append(r)
    for (model, win, hidden), rs in groups.items():
        parts = []
        for k in MEAN_KEYS:
            vals = [r[k] for r in rs if r.get(k) is not None]
            if vals:
                parts.append(f"{k} {np.mean(vals):.3g}+-{np.std(vals):.2g}")
        patients = sorted({r["patient"] for r in rs})
        size = f", {hidden} hidden" if hidden is not None else ""
        print(f"  {model} ({win:.1f} s windows{size}; {len(rs)} runs, {len(patients)} patients: {', '.join(patients)})")
        for i in range(0, len(parts), 4):
            print("    " + "   ".join(parts[i:i + 4]))

    if args.csv:
        keys = ["run"] + [k for k, _, _ in COLUMNS]
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
