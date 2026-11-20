"""
Choose the accumulator's parameters (mode, shifts, margin) and the alarm persistence N from the
members' saved validation outputs. No member is re-run: their int8 forward passes are already
in <run>/val_outputs.npz (written by train.py or train.py --rescore).

    # from ML/, after every member has a finished (or rescored) run for this patient
    python fit_accumulator.py --patient chb01 --cache <cache> --out <runs> --members lif alif
    python fit_accumulator.py ... --test     # ONCE, at the very end, after every member's --test

Reads   <out>/<patient>_<member>_s<seed>/val_outputs.npz and results.json, for each member
Writes  <out>/<patient>_accumulator_<member+member+...>_s<seed>/
          results.json      baselines (each member alone, plain pooled vote, majority),
                            the chosen setting, and the top settings
          params_int8.npz   {"shift", "margin", "mode"} int8 arrays, as accumulator.weight_shapes()
          test_results.json only after --test

Selection uses train.py's alarm scoring (persistence rule + SzCORE event rules) and
train.choose_alarm, over every mode x shift vector x margin x N. That is a lot of choices for a
validation split with a handful of seizures, so compare the chosen setting with the plain
baselines before trusting it, and prefer the baselines when they are close.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import accumulator as acc
import train
from common import STEP, WINDOW
from data.dataset import CACHE_DIR, WindowDataset, load_recordings

MARGINS = (-4, -2, -1, 0, 1, 2, 4, 8, 16, 32)   # accumulator margins tried (int8)
TOP_K = 10                                       # settings listed in results.json


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--patient", required=True, help="e.g. chb01")
    p.add_argument("--members", nargs="+", default=acc.MEMBERS,
                   help="member model names, in accumulator input order (default: accumulator.MEMBERS)")
    p.add_argument("--member-runs", nargs="+", type=Path,
                   help="member run folders, same order as --members (default: <out>/<patient>_<member>_s<seed>)")
    p.add_argument("--seed", type=int, default=0, help="seed of the member runs (and the output folder tag)")
    p.add_argument("--cache", type=Path, default=CACHE_DIR, help="preprocess.py output folder")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "runs")
    p.add_argument("--modes", nargs="+", default=list(acc.MODES), choices=list(acc.MODES))
    p.add_argument("--max-shift", type=int, default=acc.CONFIG["max_shift"])
    p.add_argument("--margins", nargs="+", type=int, default=MARGINS)
    p.add_argument("--test", action="store_true", help="apply the saved parameters to the test split (once)")
    p.add_argument("--force", action="store_true", help="let --test run again")
    return p.parse_args()


def member_dirs(args):
    """Run folder of every member, in --members order."""
    if args.member_runs:
        if len(args.member_runs) != len(args.members):
            raise SystemExit("--member-runs needs one folder per --members entry")
        return list(args.member_runs)
    return [args.out / f"{args.patient}_{m}_s{args.seed}" for m in args.members]


def load_member_outputs(dirs, split):
    """Stack the members' per-window outputs for one split.

    Args:
        dirs: Member run folders, in accumulator input order.
        split: "val" or "test" (reads val_outputs.npz or test_outputs.npz).

    Returns:
        (member_counts int64 [n_windows, n_members, 2], labels, recording, start), after
        checking that every member scored exactly the same windows in the same order.
    """
    counts, ref = [], None
    for d in dirs:
        path = d / f"{split}_outputs.npz"
        if not path.exists():
            hint = "train.py --rescore" if split == "val" else "train.py --test"
            raise SystemExit(f"{path} not found: run {hint} for this member first")
        results = json.loads((d / "results.json").read_text())
        if (results["window"], results["step"]) != (WINDOW, STEP):
            raise SystemExit(f"{d.name}: window/step {results['window']}/{results['step']} is not the hardware spec")
        z = np.load(path)
        if ref is None:
            ref = z
        elif not (np.array_equal(z["recording"], ref["recording"]) and np.array_equal(z["start"], ref["start"])
                  and np.array_equal(z["labels"], ref["labels"])):
            raise SystemExit(f"{d.name} scored different {split} windows than {dirs[0].name}")
        counts.append(z["counts"].astype(np.int64))
    return np.stack(counts, axis=1), ref["labels"], ref["recording"], ref["start"]


def dataset_for(recordings, split, recording, start):
    """The split's WindowDataset, checked against the windows the members scored (the alarm
    scoring needs the recordings' seizure masks, which only the cache has)."""
    ds = WindowDataset(recordings, split, WINDOW, STEP)
    if not (np.array_equal(ds.recording, recording) and np.array_equal(ds.start, start)):
        raise SystemExit(f"cache {split} windows differ from the members' outputs: same --cache as training?")
    return ds


def describe(params, members):
    """Short text for one parameter setting."""
    mode = {v: k for k, v in acc.MODES.items()}[int(params["mode"][0, 0])]
    shifts = ", ".join(f"{m}<<{int(s)}" for m, s in zip(members, params["shift"][:, 0]))
    return f"{mode:5s} [{shifts}] margin {int(params['margin'][0, 0]):3d}"


def fmt(a):
    delay = a["mean_delay_s"] if a["mean_delay_s"] is not None else float("nan")
    return (f"N = {a['n_consec']:2d}: detected {a['detected']}/{a['seizures']}  "
            f"FA/h {a['fa_per_hour'] or 0:.2f}  mean delay {delay:.1f} s")


def rank_key(a):
    """Sort key with the same priorities as train.choose_alarm, for listing the top settings:
    detecting under the FA/h target first (by sensitivity), then detecting (by false alarms),
    then the rest; ties by shorter delay, smaller N, smaller margin."""
    delay = a["mean_delay_s"] if a["mean_delay_s"] is not None else float("inf")
    under = a["detected"] > 0 and a["fa_per_hour"] is not None and a["fa_per_hour"] < train.TARGET_FA_PER_HOUR
    if under:
        return (0, -a["sensitivity"], 0, delay, a["n_consec"], a["margin"])
    if a["detected"] > 0:
        return (1, a["false_alarms"], -a["sensitivity"], delay, a["n_consec"], a["margin"])
    return (2, a["false_alarms"], 0, 0, a["n_consec"], a["margin"])


def sweep_setting(ds, decisions, hours, cache):
    """Alarm metrics for every N, cached by decision vector (many settings decide identically)."""
    key = decisions.tobytes()
    if key not in cache:
        cache[key] = [train.alarm_metrics(ds, decisions, n, hours) for n in train.ALARM_NS]
    return cache[key]


def fit(args, dirs):
    member_counts, labels, recording, start = load_member_outputs(dirs, "val")
    recordings = load_recordings([args.patient], args.cache)
    ds = dataset_for(recordings, "val", recording, start)
    hours = train.split_hours(recordings, "val")
    cfg = acc.make_config(args.members, args.max_shift)
    M = cfg["n_members"]
    print(f"{args.patient} validation: {len(labels)} windows ({int(labels.sum())} seizure), {hours:.1f} h, "
          f"members {', '.join(args.members)}")

    cache = {}

    def evaluate(params):
        """Window metrics plus the best N for one parameter setting."""
        acc.check_params(params, cfg)
        _, dec = acc.decide_from_counts(member_counts, params)
        alarms = [{**a, "margin": int(params["margin"][0, 0])} for a in sweep_setting(ds, dec, hours, cache)]
        best = train.choose_alarm(alarms)
        return {"setting": describe(params, args.members),
                "mode": int(params["mode"][0, 0]), "shift": params["shift"][:, 0].tolist(),
                "margin": int(params["margin"][0, 0]),
                "window": train.window_metrics(dec, labels, hours), "alarm": best, "alarm_sweep": alarms}

    # Baselines: every member alone (its own spec rule), the plain pooled vote, the plain majority
    print("\nBaselines (each member alone at margin 0, then the unweighted accumulator):")
    baselines = {}
    for m, name in enumerate(args.members):
        dec = train.decide(member_counts[:, m])
        alarms = sweep_setting(ds, dec, hours, cache)
        baselines[name] = {"window": train.window_metrics(dec, labels, hours), "alarm": train.choose_alarm(alarms)}
        print(f"  {name:28s} {fmt(baselines[name]['alarm'])}")
    for label, mode in (("pooled counts", "count"), ("majority vote", "vote")):
        baselines[label] = evaluate(acc.make_params([0] * M, 0, mode))
        print(f"  {label:28s} {fmt(baselines[label]['alarm'])}")

    # Full sweep: mode x shift vector x margin, each with its best N
    settings = [acc.make_params(s, m, mode) for mode in args.modes
                for s in acc.shift_grid(M, args.max_shift) for m in args.margins]
    print(f"\nSweeping {len(settings)} settings x {len(train.ALARM_NS)} values of N "
          f"(target FA/h < {train.TARGET_FA_PER_HOUR})...")
    rows = [evaluate(p) for p in settings]
    best_alarm = train.choose_alarm([r["alarm"] for r in rows])
    best = next(r for r in rows if r["alarm"] is best_alarm)
    ranked = sorted(rows, key=lambda r: rank_key(r["alarm"]))
    print(f"  chosen: {best['setting']}  {fmt(best['alarm'])}")
    print(f"  window level: sens {best['window']['sensitivity'] or 0:.3f}  "
          f"prec {best['window']['precision'] or 0:.3f}  FP/h {best['window']['fp_per_hour'] or 0:.1f}")
    print(f"  ({len(cache)} distinct decision vectors among {len(settings)} settings)")

    run_dir = args.out / f"{args.patient}_accumulator_{'+'.join(args.members)}_s{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    params = acc.make_params(best["shift"], best["margin"], {v: k for k, v in acc.MODES.items()}[best["mode"]])
    np.savez(run_dir / "params_int8.npz", **params)
    strip = lambda r: {k: v for k, v in r.items() if k != "alarm_sweep"}
    results = {"patient": args.patient, "members": args.members, "member_runs": [str(d) for d in dirs],
               "seed": args.seed, "config": cfg, "args": vars(args), "val_hours": hours,
               "chosen": {**strip(best), "alarm_n": best["alarm"]["n_consec"]},
               "baselines": {k: strip(v) for k, v in baselines.items()},
               "top": [strip(r) for r in ranked[:TOP_K]], "n_settings": len(settings)}
    (run_dir / "results.json").write_text(train.to_json(results))
    print(f"wrote {run_dir / 'results.json'} and params_int8.npz")


def test(args, dirs):
    """Apply the saved parameters and N to the members' test outputs. Meant to run once."""
    run_dir = args.out / f"{args.patient}_accumulator_{'+'.join(args.members)}_s{args.seed}"
    out = run_dir / "test_results.json"
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists: the test set is evaluated once. Use --force to override.")
    results = json.loads((run_dir / "results.json").read_text())
    params = {k: v for k, v in np.load(run_dir / "params_int8.npz").items()}
    member_counts, labels, recording, start = load_member_outputs(dirs, "test")
    recordings = load_recordings([args.patient], args.cache)
    ds = dataset_for(recordings, "test", recording, start)
    hours = train.split_hours(recordings, "test")
    _, dec = acc.decide_from_counts(member_counts, params)
    n = results["chosen"]["alarm_n"]
    test = {"setting": describe(params, args.members), "hours": hours,
            "window": train.window_metrics(dec, labels, hours),
            "alarm": train.alarm_metrics(ds, dec, n, hours, int(params["margin"][0, 0]))}
    out.write_text(train.to_json(test))
    print(f"TEST ({hours:.1f} h), {test['setting']}: {fmt(test['alarm'])}")
    print(f"wrote {out}")


def main():
    args = parse_args()
    dirs = member_dirs(args)
    if args.test:
        test(args, dirs)
    else:
        fit(args, dirs)


if __name__ == "__main__":
    main()
