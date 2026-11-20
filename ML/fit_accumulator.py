"""
Choose the accumulator's parameters (mode, shifts, margin) and the alarm persistence N from the
members' saved validation outputs. No member is re-run: their int8 forward passes are already
in <run>/val_outputs.npz (written by train.py or train.py --rescore).

    # from ML/, after every member has a finished (or rescored) run for this patient
    python fit_accumulator.py --patient chb01 --cache <cache> --out <runs> --members lif alif
    python fit_accumulator.py ... --test     # ONCE, at the very end, after every member's --test

Reads   <out>/<patient>_<member>_s<seed>/val_outputs.npz and results.json, for each member
Writes  <out>/<patient>_accumulator_<member+member+...>_s<seed>/
          results.json      baselines (each member alone, every fixed rule),
                            the chosen setting, and the top settings
          params_int8.npz   {"shift", "margin", "mode"} int8 arrays, as accumulator.weight_shapes()
          test_results.json only after --test

Selection uses train.py's alarm scoring (persistence rule + SzCORE event rules) and
train.choose_alarm. By default (--search fixed) it only chooses among the untrained rules in
accumulator.fixed_rules() (pooled counts; any / majority / unanimous vote), each with its best N:
with ~2 validation seizures per patient, fixed rules are the literature's safer choice (see the
accumulator.py docstring). Two opt-in searches for weighted settings:
  --search grid      every mode x shift vector x margin (many choices for a handful of seizures)
  --search logistic  one logistic regression per mode on the members' outputs (count differences
                     or +-1 votes), rounded to shifts and a margin by accumulator.params_from_linear
Either way the weighted result only replaces the best fixed rule if it is clearly better
(clearly_better). Both are fitted on the same validation windows the members were early-stopped
on, which flatters them; the gate is there for that reason.
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
MIN_FA_GAIN = 2                                  # false alarms a weighted setting must save (see clearly_better)
L2 = 1e-3                                        # logistic regression ridge, on features scaled to [-1, 1]


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
    p.add_argument("--search", choices=["fixed", "grid", "logistic"], default="fixed",
                   help="fixed: choose among the untrained rules only (default); grid / logistic: also try "
                        "weighted settings (see above)")
    p.add_argument("--modes", nargs="+", default=list(acc.MODES), choices=list(acc.MODES),
                   help="--search grid / logistic")
    p.add_argument("--max-shift", type=int, default=acc.CONFIG["max_shift"], help="--search grid / logistic")
    p.add_argument("--margins", nargs="+", type=int, default=MARGINS, help="--search grid only")
    p.add_argument("--l2", type=float, default=L2, help="--search logistic: ridge penalty")
    p.add_argument("--test", action="store_true", help="apply the saved parameters to the test split (once)")
    p.add_argument("--min-fa-gain", type=int, default=MIN_FA_GAIN,
                   help="--search grid: weights must save at least this many false alarms (same detections) "
                        "to replace the best fixed rule")
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


def fit_logistic(X, y, l2=L2, iters=100):
    """Class-balanced, L2-regularised logistic regression by Newton's method (IRLS), NumPy only.

    Seizure windows are rare, so each class gets half the total sample weight; otherwise the
    fit would mostly learn "no seizure". Features are scaled to [-1, 1] for the fit (the penalty
    and conditioning depend on scale) and the weights are returned in the original units.

    Args:
        X: Features [n_windows, n_members] (accumulator.features).
        y: 0/1 labels [n_windows]; both classes must be present.
        l2: Ridge penalty on the scaled weights (not on the intercept).
        iters: Maximum Newton steps.

    Returns:
        (weights float [n_members], bias float): seizure if bias + X @ weights > 0.
    """
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    pos = y.mean()
    if not 0 < pos < 1:
        raise SystemExit("logistic fit needs both seizure and non-seizure windows")
    scale = np.abs(X).max(axis=0)
    scale[scale == 0] = 1
    Z = np.hstack([X / scale, np.ones((len(X), 1))])
    sw = np.where(y == 1, 0.5 / pos, 0.5 / (1 - pos)) / len(y)     # balanced, sums to 1
    R = l2 * np.eye(Z.shape[1])
    R[-1, -1] = 0
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
        grad = Z.T @ (sw * (p - y)) + R @ w
        hess = (Z * (sw * p * (1 - p))[:, None]).T @ Z + R
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-9:
            break
    return w[:-1] / scale, float(w[-1])


def clearly_better(a, b, min_fa_gain=MIN_FA_GAIN):
    """Whether alarm result `a` (a weighted setting) beats `b` (the best fixed rule) by enough
    to justify its extra parameters, judged on validation counts rather than rates or delay.

    Args:
        a: Alarm metrics dict (train.alarm_metrics) of the candidate.
        b: Alarm metrics dict of the incumbent.
        min_fa_gain: False alarms `a` must save when both detect the same seizures.

    Returns:
        (bool, reason): True only if `a` detects more seizures; or detects as many with at least
        `min_fa_gain` fewer false alarms; or meets the FA/h target when `b` does not, with no fewer
        detections. A shorter delay alone never counts: with ~2 seizures it is mostly noise.
    """
    def under_target(x):
        return x["fa_per_hour"] is not None and x["fa_per_hour"] < train.TARGET_FA_PER_HOUR

    if a["detected"] > b["detected"]:
        return True, f"detects {a['detected']} vs {b['detected']} seizures"
    if a["detected"] < b["detected"]:
        return False, f"detects fewer seizures ({a['detected']} vs {b['detected']})"
    if under_target(a) and not under_target(b):
        return True, f"meets the FA/h target ({a['fa_per_hour']:.2f} vs {b['fa_per_hour']:.2f})"
    saved = b["false_alarms"] - a["false_alarms"]
    if saved >= min_fa_gain:
        return True, f"saves {saved} false alarms (>= {min_fa_gain})"
    return False, f"saves {saved} false alarms (< {min_fa_gain}), same detections: not worth the weights"


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

    # Baselines: every member alone (its own spec rule), then every fixed (untrained) rule
    print("\nBaselines (each member alone at margin 0, then the fixed rules):")
    baselines = {}
    for m, name in enumerate(args.members):
        dec = train.decide(member_counts[:, m])
        alarms = sweep_setting(ds, dec, hours, cache)
        baselines[name] = {"window": train.window_metrics(dec, labels, hours), "alarm": train.choose_alarm(alarms)}
        print(f"  {name:28s} {fmt(baselines[name]['alarm'])}")
    fixed = acc.fixed_rules(M)
    for label, params in fixed.items():
        baselines[label] = {**evaluate(params), "rule": label}
        print(f"  {label:28s} {fmt(baselines[label]['alarm'])}")

    # Candidates: the fixed rules only (default), the full mode x shift vector x margin grid,
    # or one rounded logistic regression per mode
    logistic = {}
    if args.search == "fixed":
        rows = [baselines[label] for label in fixed]
        print(f"\nChoosing among the {len(rows)} fixed rules (target FA/h < {train.TARGET_FA_PER_HOUR})...")
    elif args.search == "grid":
        settings = [acc.make_params(s, m, mode) for mode in args.modes
                    for s in acc.shift_grid(M, args.max_shift) for m in args.margins]
        print(f"\nSweeping {len(settings)} settings x {len(train.ALARM_NS)} values of N "
              f"(target FA/h < {train.TARGET_FA_PER_HOUR})...")
        rows = [evaluate(p) for p in settings]
    else:
        print(f"\nLogistic regression per mode (l2 {args.l2}), rounded to shifts <= {args.max_shift}:")
        rows = []
        for mode in args.modes:
            X = acc.features(member_counts, mode)
            w, b = fit_logistic(X, labels, args.l2)
            float_dec = (b + X @ w > 0).astype(np.int64)            # before rounding, for comparison
            float_alarm = train.choose_alarm(sweep_setting(ds, float_dec, hours, cache))
            params = acc.params_from_linear(w, b, mode, args.max_shift)
            logistic[mode] = {"weights": w.tolist(), "bias": b, "float_alarm": float_alarm,
                              "float_window": train.window_metrics(float_dec, labels, hours),
                              "rounded": describe(params, args.members) if params is not None else None}
            weights = ", ".join(f"{m} {x:+.3g}" for m, x in zip(args.members, w))
            print(f"  {mode:5s} weights [{weights}], bias {b:+.3g}")
            print(f"        float:   {fmt(float_alarm)}")
            if params is None:
                print("        no positive weight: the block cannot express this fit, skipped")
                continue
            rows.append(evaluate(params))
            print(f"        rounded: {rows[-1]['setting']}  {fmt(rows[-1]['alarm'])}")

    def pick(candidates):
        """The row whose best alarm train.choose_alarm prefers."""
        chosen_alarm = train.choose_alarm([r["alarm"] for r in candidates])
        return next(r for r in candidates if r["alarm"] is chosen_alarm)

    best = best_fixed = pick([baselines[label] for label in fixed])
    selection = {"best_fixed": best_fixed["rule"], "weights_accepted": False, "reason": f"--search {args.search}"}
    if args.search != "fixed" and not rows:
        selection["reason"] = "no weighted setting could be expressed: keeping the best fixed rule"
        print(f"  {selection['reason']}")
    elif args.search != "fixed":
        best_weighted = pick(rows)
        accepted, reason = clearly_better(best_weighted["alarm"], best_fixed["alarm"], args.min_fa_gain)
        print(f"  best fixed rule: {best_fixed['rule']} = {best_fixed['setting']}  {fmt(best_fixed['alarm'])}")
        print(f"  best weighted:   {best_weighted['setting']}  {fmt(best_weighted['alarm'])}")
        print(f"  {'use' if accepted else 'keep the fixed rule, not'} the weighted setting: {reason}")
        selection = {**selection, "best_weighted": best_weighted["setting"], "weights_accepted": accepted,
                     "reason": reason}
        if accepted:
            best = best_weighted
    ranked = sorted(rows, key=lambda r: rank_key(r["alarm"]))
    name = f"{best['rule']} = " if "rule" in best else ""
    print(f"  chosen: {name}{best['setting']}  {fmt(best['alarm'])}")
    print(f"  window level: sens {best['window']['sensitivity'] or 0:.3f}  "
          f"prec {best['window']['precision'] or 0:.3f}  FP/h {best['window']['fp_per_hour'] or 0:.1f}")
    print(f"  ({len(cache)} distinct decision vectors among {len(rows)} settings and the member baselines)")

    run_dir = args.out / f"{args.patient}_accumulator_{'+'.join(args.members)}_s{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    params = acc.make_params(best["shift"], best["margin"], {v: k for k, v in acc.MODES.items()}[best["mode"]])
    np.savez(run_dir / "params_int8.npz", **params)
    strip = lambda r: {k: v for k, v in r.items() if k != "alarm_sweep"}
    results = {"patient": args.patient, "members": args.members, "member_runs": [str(d) for d in dirs],
               "seed": args.seed, "config": cfg, "args": vars(args), "val_hours": hours,
               "chosen": {**strip(best), "alarm_n": best["alarm"]["n_consec"]}, "selection": selection,
               "baselines": {k: strip(v) for k, v in baselines.items()},
               "search": args.search, "logistic": logistic,
               "top": [strip(r) for r in ranked[:TOP_K]], "n_settings": len(rows)}
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
