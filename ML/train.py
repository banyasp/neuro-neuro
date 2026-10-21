"""
Train one model for one patient: float training, QAT fine-tuning, int8 weights, agreement check.

    # from ML/ (on Colab, point --cache and --out at Google Drive)
    python train.py --patient chb01 --cache /content/drive/MyDrive/neuro/cache --out /content/drive/MyDrive/neuro/runs
    python train.py --patient chb01 ... --resume   # continue after a disconnect (same arguments)
    python train.py --patient chb01 ... --test     # ONCE, at the very end: score the saved weights on test

Stages
  1. Float mode until validation stops improving (early stopping keeps the best epoch).
  2. QAT mode from the best stage-1 weights at a lower learning rate. Its starting point
     (plain rounding, no fine-tuning) is evaluated first, so QAT can only improve on it.
  Then quantize to int8 and run validation windows through run_int_reference: output counts
  must agree 100% with the QAT forward pass, or the script fails (op-order or saturation bug).
  Software-only models (no run_int_reference, e.g. --model cnn) run stage 1 only and skip the
  int8 / agreement / handoff steps; their outputs are logits instead of spike counts.

Metrics use the hardware decision rule (count1 > count0, tie = no seizure). There is no
probability threshold to tune, because hardware can't apply one. The knobs that move the
sensitivity / false-alarm trade-off are training ones: --neg-per-pos (the 1:N undersampling
sets the prior the model learns) and --pos-weight. Class weights are deliberately mild (1.2x,
as in [Mehrabi 2026]) rather than inverse-frequency: on top of 1:4 sampling, full balancing
would train for a 50/50 prior and inflate false alarms at the real ~1:300.
Early stopping uses window-level F1 on validation (accuracy is meaningless at ~1:300).

Outputs in <out>/<patient>_<model>_s<seed>/:
  last.pt            checkpoint, rewritten every epoch (for --resume)
  weights_int8.npz   int8 weights, {"l1": [N_IN, H], "l2": [H, 2]}  (software-only models: weights.pt)
  results.json       patient, seed, split, val metrics, alarm sweep, agreement, spike densities
  handoff/           golden vectors from real val windows in the export.py format:
                     "." decides seizure, background/ doesn't. Check with handoff/reference_<model>.py
                     (not for software-only models)
  test_results.json  only after --test
"""

import argparse
import importlib
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import export
from common import FS, STEP, WINDOW
from data.dataset import CACHE_DIR, WindowDataset, balanced_indices, load_recordings

ALARM_NS = range(1, 11)      # persistence rule sweep: alarm after N consecutive positive windows
TARGET_FA_PER_HOUR = 0.2     # [Proposal]
MARGINS = (0, 1, 2, 4, 8)    # count1 - count0 > margin. Only 0 is the current spec


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--patient", required=True, help="e.g. chb01")
    p.add_argument("--model", default="lif", help="module name: lif, alif, or cnn (software-only baseline)")
    p.add_argument("--window", type=float, default=WINDOW / FS, help="window length in seconds (step = half)")
    p.add_argument("--epochs", type=int, default=200, help="max stage-1 epochs (early stopping usually ends sooner)")
    p.add_argument("--qat-epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--qat-lr", type=float, default=2e-4)
    p.add_argument("--patience", type=int, default=15, help="epochs without val improvement before stopping")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--min-steps", type=int, default=50,
                   help="min batches per epoch; with few seizures, several balanced draws are stacked")
    p.add_argument("--neg-per-pos", type=int, default=4, help="training undersampling, non-seizure : seizure")
    p.add_argument("--pos-weight", type=float, default=1.2, help="seizure class weight in the loss")
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--rate-penalty", type=float, default=0.0,
                   help="loss += this * mean hidden spike rate (sparser = fewer hardware cycles)")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache", type=Path, default=CACHE_DIR, help="preprocess.py output folder")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "runs")
    p.add_argument("--verify-n", type=int, default=2000, help="windows checked against run_int_reference (0 = all)")
    p.add_argument("--resume", action="store_true", help="continue from <run>/last.pt")
    p.add_argument("--test", action="store_true", help="evaluate saved weights on the test split (once)")
    p.add_argument("--force", action="store_true", help="let --test run again")
    return p.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)                       # also seeds CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False  # QAT sums must stay exact integers


def is_hardware(model_mod):
    """Hardware models have an integer reference; software-only ones (cnn) don't."""
    return hasattr(model_mod, "run_int_reference")


def clone_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def decide(counts, margin=0):
    """Hardware rule: seizure if count1 > count0. A nonzero margin would be a spec change."""
    return (counts[:, 1] - counts[:, 0] > margin).astype(np.int64)


def ratio(a, b):
    return a / b if b else None


def window_metrics(decisions, labels, hours):
    tp = int(((decisions == 1) & (labels == 1)).sum())
    fp = int(((decisions == 1) & (labels == 0)).sum())
    fn = int(((decisions == 0) & (labels == 1)).sum())
    tn = int(((decisions == 0) & (labels == 0)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "sensitivity": ratio(tp, tp + fn), "specificity": ratio(tn, tn + fp),
            "precision": ratio(tp, tp + fp), "f1": ratio(2 * tp, 2 * tp + fp + fn) or 0.0,
            "fp_per_hour": ratio(fp, hours)}


def seizure_intervals(mask):
    """[(onset, offset), ...] in samples from a 0/1 mask."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int64), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def alarm_metrics(ds, decisions, n_consec, hours):
    """Event-level scoring, replaying each recording in time order.

    An alarm fires at the end of the N-th consecutive positive window and lasts until the run of
    positives ends. It's a true alarm if it overlaps an annotated seizure, else a false alarm.
    Delay = alarm time - seizure onset in seconds (negative = before the annotated onset).
    """
    n_seizures, false_alarms, delays = 0, 0, []
    for r in np.unique(ds.rec):
        sel = np.flatnonzero(ds.rec == r)          # this recording's windows, in time order
        d, starts = decisions[sel], ds.start[sel]
        seizures = seizure_intervals(ds.recordings[r]["seizure_mask"])
        n_seizures += len(seizures)
        events, run = [], 0                        # [alarm time, span start, span end] in samples
        for i in range(len(d)):
            run = run + 1 if d[i] else 0
            if run == n_consec:
                events.append([starts[i] + ds.window, starts[i - n_consec + 1], starts[i] + ds.window])
            elif run > n_consec:
                events[-1][2] = starts[i] + ds.window
        first_alarm = [None] * len(seizures)
        for t_alarm, lo, hi in events:
            hits = [j for j, (on, off) in enumerate(seizures) if lo < off and hi > on]
            false_alarms += not hits
            for j in hits:
                if first_alarm[j] is None:
                    first_alarm[j] = t_alarm
        delays += [(t - on) / FS for (on, _), t in zip(seizures, first_alarm) if t is not None]
    return {"n_consec": n_consec, "seizures": n_seizures, "detected": len(delays),
            "sensitivity": ratio(len(delays), n_seizures), "false_alarms": false_alarms,
            "fa_per_hour": ratio(false_alarms, hours),
            "mean_delay_s": float(np.mean(delays)) if delays else None,
            "max_delay_s": float(np.max(delays)) if delays else None}


def choose_n(sweep):
    """Best persistence N on validation: highest event sensitivity with FA/h under target
    (smallest N on ties, since each extra window adds 0.5 s of delay); else fewest false alarms."""
    ok = [a for a in sweep if a["fa_per_hour"] is not None and a["fa_per_hour"] < TARGET_FA_PER_HOUR]
    if ok and any(a["sensitivity"] is not None for a in ok):
        return max(ok, key=lambda a: (a["sensitivity"] or 0, -a["n_consec"]))["n_consec"]
    return min(sweep, key=lambda a: (a["false_alarms"], a["n_consec"]))["n_consec"]


def split_hours(recordings, split):
    return sum(len(r["seizure_mask"]) for r in recordings if r["split"] == split) / FS / 3600


# ---------------------------------------------------------------------------
# Running the model
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(model, ds, qat, device, batch=1024):
    """Output spike counts [n, n_outputs] (logits for software-only models) for every window of ds
    in order, plus mean hidden and input rates."""
    model.eval()
    counts, hidden, inputs = [], 0.0, 0.0
    for x, _ in DataLoader(ds, batch_size=batch):
        x = x.to(device)
        c, h = model(x, qat=qat)
        counts.append(c.round().long().cpu() if qat else c.cpu())   # float-mode counts are already whole
        hidden += float(h) * len(x)
        inputs += float(x.mean()) * len(x)
    n = max(1, len(ds))
    return torch.cat(counts).numpy(), hidden / n, inputs / n


def evaluate(model, ds, hours, qat, device):
    counts, hidden, inputs = predict(model, ds, qat, device)
    m = window_metrics(decide(counts), ds.labels, hours)
    # Early-stopping score: F1, or (if this split has no seizure windows) fewest false positives
    score = m["f1"] if ds.labels.any() else -m["fp"]
    return {**m, "score": score, "hidden_rate": hidden, "input_rate": inputs, "counts": counts}


def train_epoch(model, ds, draw, qat, opt, class_w, args, device):
    """One pass over fresh 1:neg_per_pos balanced draws of the training windows.

    A patient with few seizures gives tiny draws (e.g. 100 seizure windows -> 4 batches), so
    draws are stacked until the epoch has at least args.min_steps batches.
    """
    model.train()
    idx, j = [], 0
    while len(idx) < args.min_steps * args.batch:
        idx = np.concatenate([idx, balanced_indices(ds.labels, draw * 1000 + j, args.neg_per_pos, args.seed)]).astype(np.int64)
        j += 1
    total = 0.0
    for x, y in DataLoader(Subset(ds, idx), batch_size=args.batch):
        x, y = x.to(device), y.to(device)
        counts, hidden = model(x, qat=qat)      # spike counts are the logits
        loss = F.cross_entropy(counts, y, weight=class_w, label_smoothing=args.label_smoothing)
        loss = loss + args.rate_penalty * hidden
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if hasattr(model, "clamp_weights"):
            model.clamp_weights()               # keep weights inside the int8 grid
        total += loss.item() * len(x)
    return total / len(idx)


def fmt(m):
    def f(v, spec=".3f"):
        return "-" if v is None else format(v, spec)
    return (f"F1 {f(m['f1'])}  sens {f(m['sensitivity'])}  prec {f(m['precision'])}  "
            f"FP/h {f(m['fp_per_hour'], '.1f')}  hidden {m['hidden_rate']:.3f}")


def train(model, ds_train, ds_val, val_hours, args, device, run_dir, hw=True):
    """Stage 1 (float) then stage 2 (QAT, hardware models only), with early stopping and a
    checkpoint every epoch. Leaves the best weights of the last stage in model.
    Returns (history, summaries)."""
    ckpt_path = run_dir / "last.pt"
    class_w = torch.tensor([1.0, args.pos_weight], device=device)
    ck = None
    if args.resume:
        if ckpt_path.exists():
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        else:
            print(f"no checkpoint at {ckpt_path}, starting fresh")
    history = ck["history"] if ck else []
    stage1_best = ck["stage1_best"] if ck else None
    summaries = ck["summaries"] if ck else {}

    stages = [(1, False, args.lr, args.epochs), (2, True, args.qat_lr, args.qat_epochs)][: 2 if hw else 1]
    for stage, qat, lr, max_epochs in stages:
        if ck and ck["stage"] > stage:
            continue
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
        if ck and ck["stage"] == stage and ck["model"] is not None:
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            start, best, best_state, bad = ck["epoch"], ck["best_score"], ck["best_state"], ck["bad"]
            print(f"resumed stage {stage} at epoch {start}")
        else:
            start, best, best_state, bad = 0, -float("inf"), None, 0
            if stage == 2:
                model.load_state_dict(stage1_best)
                ev = evaluate(model, ds_val, val_hours, True, device)   # plain rounding, before QAT
                best, best_state = ev["score"], clone_state(model)
                summaries["ptq"] = {k: v for k, v in ev.items() if k != "counts"}
                print(f"stage 2 start (rounded stage-1 weights): {fmt(ev)}")
        ck = None   # only resume once

        name = "float" if stage == 1 else "QAT"
        for epoch in range(start, max_epochs):
            if bad >= args.patience:
                break
            draw = epoch if stage == 1 else args.epochs + epoch   # different balanced draw every epoch
            loss = train_epoch(model, ds_train, draw, qat, opt, class_w, args, device)
            ev = evaluate(model, ds_val, val_hours, qat, device)
            improved = ev["score"] > best
            if improved:
                best, best_state, bad = ev["score"], clone_state(model), 0
                summaries[f"stage{stage}_best"] = {k: v for k, v in ev.items() if k != "counts"}
            else:
                bad += 1
            history.append({"stage": stage, "epoch": epoch, "loss": loss,
                            **{k: ev[k] for k in ("f1", "sensitivity", "precision", "fp_per_hour", "hidden_rate")}})
            print(f"[{name} {epoch:3d}] loss {loss:.4f}  val {fmt(ev)}{'  *' if improved else ''}")
            torch.save({"stage": stage, "epoch": epoch + 1, "model": model.state_dict(), "opt": opt.state_dict(),
                        "best_score": best, "best_state": best_state, "bad": bad, "history": history,
                        "stage1_best": stage1_best, "summaries": summaries}, ckpt_path)

        model.load_state_dict(best_state)
        if stage == 1:
            stage1_best = best_state
            # Mark stage 1 done, so --resume after this point goes straight to stage 2
            torch.save({"stage": 2, "epoch": 0, "history": history, "stage1_best": stage1_best,
                        "summaries": summaries, "model": None}, ckpt_path)
    if not hw:
        model.load_state_dict(stage1_best)   # also covers --resume after stage 1 finished
    return history, summaries


# ---------------------------------------------------------------------------
# Quantize, verify, export
# ---------------------------------------------------------------------------

def verify(model_mod, weights, ds, counts, n, seed):
    """Run windows through the integer reference; counts must match the QAT forward exactly.

    Checks every seizure-labelled or seizure-deciding window (up to n/2) plus random others.
    Returns (number checked, count mismatches, decision mismatches).
    """
    idx = np.arange(len(ds))
    if n and n < len(ds):
        rng = np.random.default_rng(seed)
        important = np.flatnonzero((ds.labels == 1) | (decide(counts) == 1))
        important = rng.permutation(important)[: n // 2]
        others = rng.choice(np.setdiff1d(idx, important), n - len(important), replace=False)
        idx = np.sort(np.concatenate([important, others]))
    count_bad = dec_bad = 0
    for i in idx:
        r = model_mod.run_int_reference(ds.window_spikes(i), weights)
        count_bad += not np.array_equal(r["counts"], counts[i])
        dec_bad += r["decision"] != decide(counts[i:i + 1])[0]
    return len(idx), count_bad, dec_bad


def export_handoff(model_mod, weights, ds, counts, out_dir, note):
    """Golden vectors from real windows, same format as handoff/<model>/ (export.py)."""
    dec = decide(counts)
    seizure = np.flatnonzero((dec == 1) & (ds.labels == 1))
    if not len(seizure):
        seizure = np.flatnonzero(dec == 1)
    background = np.flatnonzero((dec == 0) & (ds.labels == 0))
    picks = []
    if len(seizure):
        picks.append((".", "real EEG window, decides seizure", seizure[len(seizure) // 2]))
    else:
        print("WARNING: no validation window decides seizure; handoff has no decision-1 case")
    if len(background):
        picks.append(("background" if picks else ".", "real EEG window, non-seizure", background[len(background) // 2]))

    if out_dir.exists():
        shutil.rmtree(out_dir)
    cases = []
    for folder, desc, i in picks:
        spikes = ds.window_spikes(i)
        r = export.export(model_mod, weights, spikes, out_dir / folder)
        desc += f" ({ds.recording[i]}, start sample {ds.start[i]}, label {ds.labels[i]})"
        cases.append((folder, desc, export.case_stats(spikes, r)))
    export.write_spec(out_dir, model_mod, model_mod.__name__, cases)
    export.write_reference_copy(out_dir, model_mod, model_mod.__name__)
    spec = out_dir / "spec.md"
    spec.write_text(spec.read_text().replace(
        "Weights are random for now. Trained weights will come in the same format.", note))


def to_json(obj):
    return json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def split_info(recordings, window, step):
    info = {}
    for split in ("train", "val", "test"):
        recs = [r for r in recordings if r["split"] == split]
        ds = WindowDataset(recordings, split, window, step)
        info[split] = {"recordings": [r["recording"] for r in recs],
                       "hours": split_hours(recordings, split),
                       "seizures": sum(len(seizure_intervals(r["seizure_mask"])) for r in recs),
                       "windows": len(ds), "seizure_windows": int(ds.labels.sum())}
    return info


def report_val(ds, counts, hours):
    """Window metrics at each margin, and the alarm-level sweep over N, on validation."""
    margins = {m: window_metrics(decide(counts, m), ds.labels, hours) for m in MARGINS}
    print("\nValidation, window level (margin 0 = current spec; others would need a spec change):")
    for m, w in margins.items():
        print(f"  count1 - count0 > {m}:  sens {w['sensitivity'] or 0:.3f}  prec {w['precision'] or 0:.3f}  "
              f"FP/h {w['fp_per_hour'] or 0:.1f}")
    sweep = [alarm_metrics(ds, decide(counts), n, hours) for n in ALARM_NS]
    print(f"\nValidation, alarm level (alarm after N consecutive positive windows, target FA/h < {TARGET_FA_PER_HOUR}):")
    for a in sweep:
        print(f"  N = {a['n_consec']:2d}: detected {a['detected']}/{a['seizures']}  FA/h {a['fa_per_hour'] or 0:.2f}  "
              f"mean delay {a['mean_delay_s'] if a['mean_delay_s'] is not None else float('nan'):.1f} s")
    n = choose_n(sweep)
    print(f"  chosen N = {n}")
    return margins, sweep, n


def run_test(args, model_mod, run_dir, recordings, window, step, device):
    """Score the saved int8 weights on the test split. Meant to run once, at the end."""
    out = run_dir / "test_results.json"
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists: the test set is evaluated once. Use --force to override.")
    results = json.loads((run_dir / "results.json").read_text())
    hw = is_hardware(model_mod)
    model = model_mod.Net().to(device)
    if hw:
        weights = dict(np.load(run_dir / "weights_int8.npz"))
        model.load_int_weights(weights)
    else:
        model.load_state_dict(torch.load(run_dir / "weights.pt", map_location=device))
    ds = WindowDataset(recordings, "test", window, step)
    hours = split_hours(recordings, "test")
    counts, hidden, inputs = predict(model, ds, hw, device)
    checked, count_bad, dec_bad = verify(model_mod, weights, ds, counts, args.verify_n, args.seed) if hw else (0, 0, 0)
    n = results["alarm_n"]
    test = {"window": window_metrics(decide(counts), ds.labels, hours),
            "alarm": alarm_metrics(ds, decide(counts), n, hours),
            "alarm_sweep": [alarm_metrics(ds, decide(counts), k, hours) for k in ALARM_NS],
            "hours": hours, "hidden_rate": hidden, "input_rate": inputs,
            "agreement": {"checked": checked, "count_mismatches": count_bad, "decision_mismatches": dec_bad} if hw else None}
    out.write_text(to_json(test))
    w, a = test["window"], test["alarm"]
    print(f"TEST ({hours:.1f} h): window sens {w['sensitivity']}, spec {w['specificity']}, FP/h {w['fp_per_hour']}")
    print(f"  alarms (N = {n}): detected {a['detected']}/{a['seizures']}, FA/h {a['fa_per_hour']}, "
          f"mean delay {a['mean_delay_s']} s")
    if hw:
        print(f"  agreement with run_int_reference: {checked - count_bad}/{checked}")
    print(f"wrote {out}")
    if count_bad:
        raise SystemExit("FAIL: QAT forward and run_int_reference disagree on test windows")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_mod = importlib.import_module(args.model)
    hw = is_hardware(model_mod)

    window = round(args.window * FS)
    step = window // 2
    if (window, step) != (WINDOW, STEP):
        print(f"NOTE: window {window} / step {step} differs from the hardware spec ({WINDOW} / {STEP})")
    tag = f"{args.patient}_{args.model}_s{args.seed}" + ("" if window == WINDOW else f"_w{window}")
    run_dir = args.out / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"device {device}, run dir {run_dir}")

    recordings = load_recordings([args.patient], args.cache)
    if args.test:
        run_test(args, model_mod, run_dir, recordings, window, step, device)
        return

    splits = split_info(recordings, window, step)
    for split, s in splits.items():
        print(f"  {split:5s}: {len(s['recordings']):3d} files, {s['hours']:6.1f} h, {s['seizures']} seizures, "
              f"{s['windows']} windows ({s['seizure_windows']} seizure)")
    ds_train = WindowDataset(recordings, "train", window, step)
    ds_val = WindowDataset(recordings, "val", window, step)
    val_hours = split_hours(recordings, "val")
    if not ds_val.labels.any():
        print("WARNING: no seizure windows in validation; early stopping will minimise false positives only")

    model = model_mod.Net().to(device)
    history, summaries = train(model, ds_train, ds_val, val_hours, args, device, run_dir, hw)

    # Hardware models: quantize, then score validation with the integer-exact QAT forward.
    # Software-only models: score the float model as is.
    final = evaluate(model, ds_val, val_hours, hw, device)
    counts = final.pop("counts")
    print(f"\nfinal {'int8' if hw else 'float'} model, val: {fmt(final)}")
    margins, sweep, alarm_n = report_val(ds_val, counts, val_hours)

    if hw:
        weights = model.quantize()
        checked, count_bad, dec_bad = verify(model_mod, weights, ds_val, counts, args.verify_n, args.seed)
        agreement = {"checked": checked, "count_mismatches": count_bad, "decision_mismatches": dec_bad,
                     "rate": (checked - count_bad) / max(1, checked)}
        print(f"\nagreement with run_int_reference on {checked} val windows: "
              f"{count_bad} count mismatches, {dec_bad} decision mismatches")
        weights_path = run_dir / "weights_int8.npz"
        np.savez(weights_path, **weights)
    else:
        count_bad, agreement = 0, None
        weights_path = run_dir / "weights.pt"
        torch.save(model.state_dict(), weights_path)
    results = {"patient": args.patient, "model": args.model, "seed": args.seed, "window": window, "step": step,
               "args": vars(args), "split": splits, "val": final, "val_by_margin": margins,
               "alarm_sweep": sweep, "alarm_n": alarm_n, "agreement": agreement,
               "density": {"input": final["input_rate"], "hidden": final["hidden_rate"]},
               "summaries": summaries, "history": history}
    (run_dir / "results.json").write_text(to_json(results))
    print(f"wrote {weights_path} and results.json")

    if count_bad:
        raise SystemExit(f"FAIL: {count_bad}/{checked} windows disagree between the QAT forward and "
                         "run_int_reference (op-order bug, or int16 saturation inside a sum). Not exporting.")

    if hw and window == WINDOW:
        note = (f"Weights are TRAINED ({args.patient}, seed {args.seed}); input spikes are real EEG "
                "windows from the validation set.")
        export_handoff(model_mod, weights, ds_val, counts, run_dir / "handoff", note)
        print(f"wrote {run_dir / 'handoff'}")


if __name__ == "__main__":
    main()
