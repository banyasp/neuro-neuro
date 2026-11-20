"""
How differently do the models fail? Pairwise diversity on validation, from saved val_outputs.npz.

    # from ML/, after train.py has run each model on the patient
    python diversity.py --patient chb01 --cache <cache> --out <runs>                  # lif, alif, scnn, seed 0
    python diversity.py --patient chb01 ... --runs chb01_lif_s0 chb01_scnn_s0_kernel_len1
    python diversity.py --patient chb01 ... --best   # each run's best (margin, N) instead of the spec (m = 0)

A voter can only cut false alarms if its members raise them on different windows, so this
reports, for every pair of runs (validation windows, natural prevalence):
  disagree    fraction of windows where the two decisions differ
  err corr    correlation of the two models' error indicators (1 = always wrong together)
  FP overlap  of the windows where both are tested, |FP_a and FP_b| / min(|FP_a|, |FP_b|):
              near 1 means a 2-of-2 vote would remove few false positives
  miss overlap the same for missed seizure windows
and, at event level (persistence alarm + SzCORE tolerance, as in train.py), which annotated
seizures each run detects and which seizures only one run catches.
The voter itself (majority / weighted / cascade) is simulated separately.
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

import train
from common import FS
from data.dataset import CACHE_DIR, load_recordings


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--patient", required=True)
    p.add_argument("--cache", type=Path, default=CACHE_DIR)
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "runs")
    p.add_argument("--runs", nargs="+", help="run folder names in --out (default: <patient>_{lif,alif,scnn}_s<seed>)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--best", action="store_true", help="use each run's best (margin, N), not the spec m = 0")
    return p.parse_args()


def load_run(run_dir, best):
    r = json.loads((run_dir / "results.json").read_text())
    z = np.load(run_dir / "val_outputs.npz")
    m, n = (r["best_alarm"]["margin"], r["best_alarm"]["n_consec"]) if best else (0, r["alarm_n"])
    return {"name": run_dir.name, "dec": train.decide(z["counts"], m), "labels": z["labels"],
            "recording": z["recording"], "start": z["start"], "window": r["window"], "margin": m, "n": n}


def overlap(a, b):
    both, small = int((a & b).sum()), min(int(a.sum()), int(b.sum()))
    return both / small if small else None


def corr(a, b):
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def detected_seizures(run, recs):
    """{(recording, onset): detected?} for every annotated seizure, alarm rule as in train.alarm_metrics."""
    out = {}
    for name in np.unique(run["recording"]):
        sel = np.flatnonzero(run["recording"] == name)
        events = train.alarm_events(run["dec"][sel], run["start"][sel], run["window"], run["n"])
        for on, off in train.seizure_intervals(recs[name]["seizure_mask"]):
            lo, hi = on - train.SZ_BEFORE_S * FS, off + train.SZ_AFTER_S * FS
            out[(str(name), int(on))] = any(a < hi and b > lo for _, a, b in events)
    return out


def f(v, spec=".3f"):
    return "-" if v is None else format(v, spec)


def main():
    args = parse_args()
    names = args.runs or [f"{args.patient}_{m}_s{args.seed}" for m in ("lif", "alif", "scnn")]
    runs = [load_run(args.out / n, args.best) for n in names]
    for r in runs[1:]:
        if not (np.array_equal(r["recording"], runs[0]["recording"]) and np.array_equal(r["start"], runs[0]["start"])):
            raise SystemExit(f"{r['name']} and {runs[0]['name']} were scored on different windows "
                             "(different --window or patient?)")
    y = runs[0]["labels"]
    recs = {r["recording"]: r for r in load_recordings([args.patient], args.cache)}
    hours = sum(len(recs[n]["seizure_mask"]) for n in np.unique(runs[0]["recording"])) / FS / 3600

    print(f"{args.patient} validation: {len(y)} windows ({int(y.sum())} seizure), {hours:.1f} h; "
          f"{'best (m, N) per run' if args.best else 'spec decision (m = 0) and spec N'}\n")
    print(f"{'run':40s} {'m':>3s} {'N':>3s} {'FP win':>7s} {'missed':>7s}")
    for r in runs:
        print(f"{r['name']:40s} {r['margin']:3d} {r['n']:3d} {int(((r['dec'] == 1) & (y == 0)).sum()):7d} "
              f"{int(((r['dec'] == 0) & (y == 1)).sum()):7d}")

    print(f"\n{'pair':60s} {'disagree':>8s} {'err corr':>8s} {'FP overlap':>10s} {'miss overlap':>12s}")
    for a, b in combinations(runs, 2):
        ea, eb = a["dec"] != y, b["dec"] != y
        fpa, fpb = (a["dec"] == 1) & (y == 0), (b["dec"] == 1) & (y == 0)
        mia, mib = (a["dec"] == 0) & (y == 1), (b["dec"] == 0) & (y == 1)
        print(f"{a['name'] + ' vs ' + b['name']:60s} {f(float((a['dec'] != b['dec']).mean()), '.4f'):>8s} "
              f"{f(corr(ea.astype(float), eb.astype(float))):>8s} {f(overlap(fpa, fpb)):>10s} "
              f"{f(overlap(mia, mib)):>12s}")

    det = {r["name"]: detected_seizures(r, recs) for r in runs}
    seizures = sorted(det[runs[0]["name"]])
    print(f"\nEvent level: annotated seizures detected (alarm within -{train.SZ_BEFORE_S} s / "
          f"+{train.SZ_AFTER_S} s of the seizure)")
    print(f"{'seizure':24s} " + " ".join(f"{r['name'][-14:]:>14s}" for r in runs))
    for s in seizures:
        print(f"{s[0] + ' @ ' + f'{s[1] // FS}s':24s} " + " ".join(f"{'yes' if det[r['name']][s] else '.':>14s}" for r in runs))
    for r in runs:
        others = [o for o in runs if o is not r]
        only = [s for s in seizures if det[r["name"]][s] and not any(det[o["name"]][s] for o in others)]
        print(f"  caught only by {r['name']}: {len(only)}" + (f" ({', '.join(f'{a} @ {b // FS}s' for a, b in only)})" if only else ""))


if __name__ == "__main__":
    main()
