"""
List every alarm a finished run raises on validation, and mark which are false alarms.

    # from ML/, after train.py (or train.py --rescore) has written val_outputs.npz
    python inspect_alarms.py --patient chb01 --cache <cache> --out <runs>            # best (m, N)
    python inspect_alarms.py --patient chb01 --cache <cache> --out <runs> --spec     # m = 0 choice
    python inspect_alarms.py ... --margin 8 --n 2 --plot                            # any setting, + plots

Alarms are built and scored exactly as in train.py (persistence rule + SzCORE event rules), so
the false-alarm count here equals the one in results.json. With --plot, each false alarm gets a
PNG (decision margin over time + input spike rate per channel, 5 min either side), to judge
whether it looks like an artifact or a seizure the annotations missed.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import train
from common import CHANNELS, FS
from data.dataset import CACHE_DIR, load_recordings

CONTEXT_S = 300   # plot this much either side of an alarm


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--patient", required=True)
    p.add_argument("--model", default="lif")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache", type=Path, default=CACHE_DIR)
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "runs")
    p.add_argument("--run", type=Path, help="run folder (default: <out>/<patient>_<model>_s<seed>)")
    p.add_argument("--spec", action="store_true", help="use the m = 0 (spec) choice instead of the best")
    p.add_argument("--margin", type=int, help="override the margin")
    p.add_argument("--n", type=int, help="override N (consecutive positive windows)")
    p.add_argument("--plot", action="store_true", help="save a PNG per false alarm in <run>/alarms/")
    return p.parse_args()


def hms(samples):
    s = int(samples // FS)
    return f"{s // 3600:d}:{s // 60 % 60:02d}:{s % 60:02d}"


def score_events(seizures, alarms, n_samples):
    """SzCORE per-event labels for one recording: (merged alarm events, is_false_alarm list).
    Same rules as train.szcore_score, which is checked against it."""
    ref = train.split_events(train.merge_events(seizures, train.SZ_MERGE_S * FS), train.SZ_MAX_EVENT_S * FS)
    hyp = train.split_events(train.merge_events(alarms, train.SZ_MERGE_S * FS), train.SZ_MAX_EVENT_S * FS)
    zones = [(max(0, on - train.SZ_BEFORE_S * FS), min(n_samples, off + train.SZ_AFTER_S * FS)) for on, off in ref]
    false = [not any(on < hi and off > lo for lo, hi in zones) for on, off in hyp]
    assert sum(false) == train.szcore_score(seizures, alarms, n_samples)[2]
    return hyp, false


def plot_alarm(path, rec, sel_starts, margins, window, on, off, seizures):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(rec["seizure_mask"])
    lo, hi = max(0, on - CONTEXT_S * FS), min(n, off + CONTEXT_S * FS)
    spikes = np.unpackbits(rec["spikes"][lo:hi], axis=1, count=2 * len(CHANNELS)).astype(np.float32)
    per_ch = spikes[:, 0::2] + spikes[:, 1::2]                    # UP + DOWN per channel
    bins = per_ch[: len(per_ch) // FS * FS].reshape(-1, FS, len(CHANNELS)).mean(axis=1)   # 1 s bins

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, height_ratios=[1, 2], layout="constrained")
    keep = (sel_starts >= lo) & (sel_starts < hi)
    t = (sel_starts[keep] + window / 2 - lo) / FS
    a1.plot(t, margins[keep], lw=0.8)
    a1.axhline(0, color="k", lw=0.5)
    a1.set_ylabel("count1 - count0")
    im = a2.imshow(bins.T, aspect="auto", origin="lower", extent=[0, len(bins), 0, len(CHANNELS)],
                   cmap="viridis", interpolation="nearest")
    a2.set_yticks(np.arange(len(CHANNELS)) + 0.5, CHANNELS, fontsize=6)
    a2.set_xlabel(f"seconds from {hms(lo)} in the recording")
    fig.colorbar(im, ax=(a1, a2), label="spikes / sample", shrink=0.6, anchor=(0, 0))  # both panels keep one width
    for ax in (a1, a2):
        ax.axvspan((on - lo) / FS, (off - lo) / FS, color="red", alpha=0.15, label="false alarm")
        for s_on, s_off in seizures:
            if s_off > lo and s_on < hi:
                ax.axvspan((s_on - lo) / FS, (s_off - lo) / FS, color="green", alpha=0.15, label="seizure")
    a1.legend(loc="upper right", fontsize=7)
    a1.set_title(f"{rec['recording']}: false alarm at {hms(on)}-{hms(off)}")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    run = args.run or args.out / f"{args.patient}_{args.model}_s{args.seed}"
    results = json.loads((run / "results.json").read_text())
    if "best_alarm" not in results:
        raise SystemExit("results.json predates the margin sweep: run train.py --rescore first")
    if args.spec:
        m, n = 0, results["alarm_n"]
    else:
        m, n = results["best_alarm"]["margin"], results["best_alarm"]["n_consec"]
    m = args.margin if args.margin is not None else m
    n = args.n if args.n is not None else n
    window = results["window"]

    z = np.load(run / "val_outputs.npz")
    counts, names, starts = z["counts"], z["recording"], z["start"]
    margins = counts[:, 1].astype(np.int64) - counts[:, 0]
    decisions = (margins > m).astype(np.int64)
    recs = {r["recording"]: r for r in load_recordings([args.patient], args.cache)}
    hours = sum(len(recs[r]["seizure_mask"]) for r in np.unique(names)) / FS / 3600

    print(f"{run.name}, validation ({hours:.1f} h), seizure if count1 - count0 > {m}, alarm after N = {n}")
    print(f"{'recording':12s} {'start':>8s} {'end':>8s} {'dur s':>6s} {'peak m':>6s}  verdict")
    total_fa, plots = 0, []
    for name in sorted(np.unique(names), key=lambda s: (len(s), s)):
        sel = np.flatnonzero(names == name)
        rec = recs[name]
        seizures = [[int(a), int(b)] for a, b in train.seizure_intervals(rec["seizure_mask"])]
        raw = train.alarm_events(decisions[sel], starts[sel], window, n)
        events, false = score_events(seizures, [[int(lo), int(hi)] for _, lo, hi in raw], len(rec["seizure_mask"]))
        for s_on, s_off in seizures:
            print(f"{name:12s} {hms(s_on):>8s} {hms(s_off):>8s} {(s_off - s_on) / FS:6.0f} {'':>6s}  (annotated seizure)")
        for (on, off), is_fa in zip(events, false):
            inside = (starts[sel] >= on) & (starts[sel] < off)
            peak = int(margins[sel][inside].max()) if inside.any() else 0
            print(f"{name:12s} {hms(on):>8s} {hms(off):>8s} {(off - on) / FS:6.0f} {peak:6d}  "
                  f"{'FALSE ALARM' if is_fa else 'detection'}")
            if is_fa:
                total_fa += 1
                if args.plot:
                    out = run / "alarms"
                    out.mkdir(exist_ok=True)
                    path = out / f"fa_m{m}_n{n}_{name}_{int(on // FS)}s.png"
                    plot_alarm(path, rec, starts[sel], margins[sel], window, on, off, seizures)
                    plots.append(path)
    print(f"\n{total_fa} false alarms in {hours:.1f} h = {total_fa / hours:.2f} / h")
    for p in plots:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
