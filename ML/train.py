"""
Train one model for one patient: float training, QAT fine-tuning, int8 weights, agreement check.

    # from ML/ (on Colab, point --cache and --out at Google Drive)
    python train.py --patient chb01 --cache /content/drive/MyDrive/neuro/cache --out /content/drive/MyDrive/neuro/runs
    python train.py --patient chb01 ... --resume   # continue after a disconnect (same arguments)
    python train.py --patient chb01 ... --rescore  # redo the val alarm / margin report from saved weights
    python train.py --patient chb01 ... --test     # ONCE, at the very end: score the saved weights on test
    python train.py --patient chb01 --model scnn --set kernel_len=32 dilations=1 group_filters=8   # ablation
    python train.py --patient chb01 --model scnn --float-only   # spiking but float: skip QAT / int8

Stages
  1. Float mode until validation stops improving (early stopping keeps the best epoch).
  2. QAT mode from the best stage-1 weights at a lower learning rate. Its starting point
     (plain rounding, no fine-tuning) is evaluated first, so QAT can only improve on it.
  Then quantize to int8 and run validation windows through run_int_reference: output counts
  must agree 100% with the QAT forward pass, or the script fails (op-order or saturation bug).
  Software-only models (no run_int_reference, e.g. --model cnn) run stage 1 only and skip the
  int8 / agreement / handoff steps; their outputs are logits instead of spike counts.
  --float-only does the same for a hardware model (ablation: what does int8 cost?).
  --set KEY=VALUE changes any of the model's own CONFIG entries (not the shared input spec).
  Like --hidden, every non-default value is added to the run folder name, so pass the same
  --set / --hidden / --float-only again for --resume / --rescore / --test.

Metrics use the hardware decision rule (count1 > count0, tie = no seizure). There is no
probability threshold to tune, because hardware can't apply one. The one exception is the
margin ablation: validation is also scored with count1 - count0 > m for m in MARGINS, jointly
with the alarm N, and the best (m, N) is saved next to the m = 0 spec choice. The knobs that move the
sensitivity / false-alarm trade-off are training ones: --neg-per-pos (the 1:N undersampling
sets the prior the model learns) and --pos-weight. Class weights are deliberately mild (1.2x,
as in [Mehrabi 2026]) rather than inverse-frequency: on top of 1:4 sampling, full balancing
would train for a 50/50 prior and inflate false alarms at the real ~1:300.
Early stopping uses window-level F1 on validation (accuracy is meaningless at ~1:300).

Outputs in <out>/<patient>_<model>_s<seed>[_w<window>][_h<hidden>][_<key><value>...][_float]/:
  last.pt            checkpoint, rewritten every epoch (for --resume)
  weights_int8.npz   int8 weights, {"l1": [N_IN, H], "l2": [H, 2]}  (software-only models: weights.pt)
  results.json       patient, seed, split, config, val metrics, alarm sweep, margin sweep, agreement,
                     densities, cost (weights, worst-case and measured adds per step, if the model has them)
  val_outputs.npz    per-window val outputs (counts, labels, recording, start), for the voter
  handoff/           golden vectors from real val windows in the export.py format:
                     "." decides seizure, background/ doesn't. Check with handoff/reference_<model>.py
                     (not for software-only models)
  test_results.json  only after --test (spec result and best-margin result), plus test_outputs.npz
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
from common import COMMON_CONFIG, FS, STEP, WINDOW
from data.dataset import CACHE_DIR, WindowDataset, balanced_indices, load_recordings

ALARM_NS = range(1, 11)      # persistence rule sweep: alarm after N consecutive positive windows
TARGET_FA_PER_HOUR = 0.2     # [Proposal]
# Decision margin ablation: seizure if count1 - count0 > margin. Only 0 is the current hardware
# spec; any other value is a (cheap) spec change, so it's reported next to the spec result.
MARGINS = (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
# SzCORE event-scoring defaults (their timescoring package), in seconds [SzCORE]
SZ_BEFORE_S, SZ_AFTER_S = 30, 60   # tolerance around each annotated seizure
SZ_MERGE_S = 90                    # merge events closer than this
SZ_MAX_EVENT_S = 300               # split events longer than this


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--patient", required=True, help="e.g. chb01")
    p.add_argument("--model", default="lif",
                   help="module name: lif, alif, scnn, or a software-only model (cnn, scnn_float)")
    p.add_argument("--window", type=float, default=WINDOW / FS, help="window length in seconds (step = half)")
    p.add_argument("--hidden", type=int, help="hidden neurons for lif / alif (default: the model's CONFIG, 64). "
                                              "Pass the same value again for --resume / --rescore / --test")
    p.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE",
                   help="override the model's CONFIG, e.g. kernel_len=32 dilations=1,3 share_filters=false")
    p.add_argument("--float-only", action="store_true",
                   help="hardware model trained and scored in float mode only (no QAT, int8 or handoff)")
    p.add_argument("--epochs", type=int, default=200, help="max stage-1 epochs (early stopping usually ends sooner)")
    p.add_argument("--qat-epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--qat-lr", type=float, default=2e-4)
    p.add_argument("--patience", type=int, default=30,
                   help="epochs without val improvement before stopping (chb01 sat at F1 ~0.01 for 14 epochs before learning)")
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
    p.add_argument("--skip-done", action="store_true", help="do nothing if this run already has results.json")
    p.add_argument("--rescore", action="store_true",
                   help="redo the validation alarm / margin report from saved weights (no training)")
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


def parse_value(text, current):
    """A --set value, in the type of the CONFIG entry it replaces ("1,3" for a list)."""
    if isinstance(current, bool):
        if text.lower() not in ("true", "false", "1", "0"):
            raise SystemExit(f"--set: expected true / false, got '{text}'")
        return text.lower() in ("true", "1")
    if isinstance(current, (list, tuple)):
        return [parse_value(t, current[0]) for t in text.split(",")]
    return type(current)(text)


def apply_overrides(config, items, model_name):
    """Apply --set KEY=VALUE items to the model's CONFIG in place. Returns the run-folder tag."""
    tag = ""
    for item in sorted(items):
        key, eq, text = item.partition("=")
        if not eq or key not in config:
            options = ", ".join(k for k in config if k not in COMMON_CONFIG)
            raise SystemExit(f"--set {item}: not a {model_name} CONFIG entry (options: {options})")
        if key in COMMON_CONFIG:
            raise SystemExit(f"--set {key}: part of the shared input spec; change common.py instead")
        value = parse_value(text, config[key])
        if value != config[key]:
            tag += f"_{key}" + ("-".join(map(str, value)) if isinstance(value, list) else str(value))
        config[key] = value
    return tag


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


def alarm_events(decisions, starts, window, n_consec):
    """Persistence rule on one recording's windows (in time order).

    An alarm fires at the end of the N-th consecutive positive window. Its event spans the whole
    run of positive windows. Returns [[alarm time, span start, span end], ...] in samples."""
    events, run = [], 0
    for i in range(len(decisions)):
        run = run + 1 if decisions[i] else 0
        if run == n_consec:
            events.append([starts[i] + window, starts[i - n_consec + 1], starts[i] + window])
        elif run > n_consec:
            events[-1][2] = starts[i] + window
    return events


def merge_events(events, gap):
    """Merge events separated by less than gap samples (SzCORE). events sorted by start."""
    out = []
    for on, off in events:
        if out and on - out[-1][1] < gap:
            out[-1][1] = max(out[-1][1], off)
        else:
            out.append([on, off])
    return out


def split_events(events, max_len):
    """Split events longer than max_len samples into max_len pieces (SzCORE)."""
    out = []
    for on, off in events:
        while off - on > max_len:
            out.append([on, on + max_len])
            on += max_len
        out.append([on, off])
    return out


def szcore_score(seizures, alarms, n_samples):
    """SzCORE event scoring of one recording, same rules as their `timescoring` package [SzCORE].

    1. Merge events less than 90 s apart, then split events longer than 5 min, in both the
       annotations and the alarms (so a 7 min seizure counts as 2 reference events).
    2. A reference event is detected (TP) if any alarm overlaps it, extended by 30 s before
       onset and 60 s after offset.
    3. An alarm is a false positive if it overlaps no detected (extended) reference event.
    seizures, alarms: [[start, end], ...] in samples. Returns (tp, n_reference_events, fp).
    (timescoring works at 10 Hz; we use the 128 Hz samples directly, which only changes rounding.)
    """
    ref = split_events(merge_events(seizures, SZ_MERGE_S * FS), SZ_MAX_EVENT_S * FS)
    hyp = split_events(merge_events(alarms, SZ_MERGE_S * FS), SZ_MAX_EVENT_S * FS)
    hyp_mask = np.zeros(n_samples, bool)
    for on, off in hyp:
        hyp_mask[on:off] = True
    tp, tp_mask = 0, np.zeros(n_samples, bool)
    for on, off in ref:
        lo, hi = max(0, on - SZ_BEFORE_S * FS), min(n_samples, off + SZ_AFTER_S * FS)
        if hyp_mask[lo:hi].any():
            tp += 1
            tp_mask[lo:hi] = True
    fp = sum(not tp_mask[on:off].any() for on, off in hyp)
    return tp, len(ref), fp


def alarm_metrics(ds, decisions, n_consec, hours, margin=0):
    """Event-level scoring with SzCORE rules, replaying each recording in time order.

    Alarms come from the persistence rule (alarm_events) and are scored with szcore_score.
    Delay, per annotated seizure: first alarm time that overlaps the seizure's tolerance
    interval minus the annotated onset, in seconds (negative = before the annotated onset).
    margin is only recorded in the result (the decisions were already made with it).
    """
    tp = n_ref = fp = n_annotated = 0
    delays = []
    for r in np.unique(ds.rec):
        sel = np.flatnonzero(ds.rec == r)          # this recording's windows, in time order
        mask = ds.recordings[r]["seizure_mask"]
        seizures = [[int(on), int(off)] for on, off in seizure_intervals(mask)]
        events = alarm_events(decisions[sel], ds.start[sel], ds.window, n_consec)
        t, n, f = szcore_score(seizures, [[int(lo), int(hi)] for _, lo, hi in events], len(mask))
        tp, n_ref, fp, n_annotated = tp + t, n_ref + n, fp + f, n_annotated + len(seizures)
        for on, off in seizures:
            lo, hi = on - SZ_BEFORE_S * FS, off + SZ_AFTER_S * FS
            hits = [t_alarm for t_alarm, a, b in events if a < hi and b > lo]
            if hits:
                delays.append((min(hits) - on) / FS)
    return {"margin": margin, "n_consec": n_consec, "seizures": n_ref, "annotated_seizures": n_annotated,
            "detected": tp, "sensitivity": ratio(tp, n_ref), "false_alarms": fp,
            "precision": ratio(tp, tp + fp), "f1": ratio(2 * tp, 2 * tp + fp + (n_ref - tp)),
            "fa_per_hour": ratio(fp, hours),
            "mean_delay_s": float(np.mean(delays)) if delays else None,
            "median_delay_s": float(np.median(delays)) if delays else None,
            "max_delay_s": float(np.max(delays)) if delays else None}


def choose_alarm(sweep):
    """Best alarm setting from a list of alarm_metrics results (any mix of margins and N).

    Only settings that detect at least one seizure count (silence trivially has zero false alarms).
    1. Under the FA/h target: highest event sensitivity, then shortest mean delay.
    2. Target not reachable: fewest false alarms, then highest sensitivity, then shortest delay.
    3. Nothing detects anything (or no seizures in the split): fewest false alarms.
    Remaining ties go to the smallest N, then the smallest margin."""
    def delay(a):
        return a["mean_delay_s"] if a["mean_delay_s"] is not None else float("inf")

    detects = [a for a in sweep if a["detected"] > 0]
    under = [a for a in detects if a["fa_per_hour"] is not None and a["fa_per_hour"] < TARGET_FA_PER_HOUR]
    if under:
        return min(under, key=lambda a: (-a["sensitivity"], delay(a), a["n_consec"], a["margin"]))
    if detects:
        return min(detects, key=lambda a: (a["false_alarms"], -a["sensitivity"], delay(a),
                                           a["n_consec"], a["margin"]))
    return min(sweep, key=lambda a: (a["false_alarms"], a["n_consec"], a["margin"]))


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


def fmt_alarm(a):
    delay = a["mean_delay_s"] if a["mean_delay_s"] is not None else float("nan")
    return (f"m = {a['margin']:2d}, N = {a['n_consec']:2d}: detected {a['detected']}/{a['seizures']}  "
            f"FA/h {a['fa_per_hour'] or 0:.2f}  mean delay {delay:.1f} s")


def alarm_report(ds, counts, hours):
    """Margin x N sweep on validation. Picks the alarm setting twice: at margin 0 (the current
    hardware spec) and over all margins (the margin ablation). Returns the results.json entries."""
    margin_sweep = []
    for m in MARGINS:
        dec = decide(counts, m)
        margin_sweep.append({"margin": m, "window": window_metrics(dec, ds.labels, hours),
                             "alarms": [alarm_metrics(ds, dec, n, hours, m) for n in ALARM_NS]})
    spec_sweep = margin_sweep[0]["alarms"]
    spec = choose_alarm(spec_sweep)
    best = choose_alarm([a for row in margin_sweep for a in row["alarms"]])

    print(f"\nValidation, alarm level (alarm after N consecutive positive windows, margin 0 = spec):")
    for a in spec_sweep:
        print(f"  {fmt_alarm(a)}")
    print(f"\nMargin ablation (seizure if count1 - count0 > m; m > 0 needs a spec change). "
          f"Best N per margin, target FA/h < {TARGET_FA_PER_HOUR}:")
    for row in margin_sweep:
        w = row["window"]
        print(f"  window sens {w['sensitivity'] or 0:.3f}  prec {w['precision'] or 0:.3f}  "
              f"FP/h {w['fp_per_hour'] or 0:5.1f}  |  {fmt_alarm(choose_alarm(row['alarms']))}")
    print(f"  spec choice: {fmt_alarm(spec)}")
    print(f"  best choice: {fmt_alarm(best)}")
    return {"alarm_sweep": spec_sweep, "alarm_n": spec["n_consec"], "margin_sweep": margin_sweep,
            "best_alarm": {"margin": best["margin"], "n_consec": best["n_consec"]}}


@torch.no_grad()
def measure_cost(model_mod, model, ds, qat, device, n=4096, batch=512):
    """Hardware cost: the module's static numbers (weights, worst-case adds per step) plus the
    model's measured spike rates and mean adds per step on up to n random validation windows
    (natural prevalence, so mostly background, like the hardware sees). None if the model has neither."""
    cost = dict(model_mod.hardware_cost(model_mod.CONFIG)) if hasattr(model_mod, "hardware_cost") else {}
    if hasattr(model, "cost_stats") and len(ds):
        model.eval()
        idx = np.random.default_rng(0).permutation(len(ds))[:n]
        totals, seen = {}, 0
        for x, _ in DataLoader(Subset(ds, idx), batch_size=batch):
            for k, v in model.cost_stats(x.to(device), qat).items():
                totals[k] = None if v is None else totals.get(k, 0.0) + v
            seen += len(x)
        cost.update({k: None if v is None else v / seen for k, v in totals.items()})
    return cost or None


def save_val_outputs(run_dir, ds, counts):
    """Per-window validation outputs (for the voter simulation and rescoring without the model)."""
    np.savez(run_dir / "val_outputs.npz", counts=counts, labels=ds.labels,
             recording=ds.recording, start=ds.start)


def load_trained(model_mod, run_dir, device, hw):
    """The model from a finished run's saved weights (int8 for hardware models)."""
    model = model_mod.Net().to(device)
    if hw:
        weights = dict(np.load(run_dir / "weights_int8.npz"))
        expected = model_mod.weight_shapes()
        saved = {k: w.shape for k, w in weights.items()}
        if saved != expected:
            raise SystemExit(f"saved weights {saved} don't match the model {expected}: "
                             "pass the same --hidden / --set as when this run was trained")
        model.load_int_weights(weights)
    else:
        weights = None
        model.load_state_dict(torch.load(run_dir / "weights.pt", map_location=device))
    return model, weights


def rescore(model_mod, run_dir, recordings, window, step, device, hw):
    """Redo the validation alarm/margin report from saved weights, without retraining
    (e.g. for runs made before the margin sweep existed). Updates results.json in place."""
    path = run_dir / "results.json"
    results = json.loads(path.read_text())
    model, _ = load_trained(model_mod, run_dir, device, hw)
    ds = WindowDataset(recordings, "val", window, step)
    counts, _, _ = predict(model, ds, hw, device)
    results.pop("val_by_margin", None)          # replaced by margin_sweep
    results["config"] = getattr(model_mod, "CONFIG", None)
    results.update(alarm_report(ds, counts, split_hours(recordings, "val")))
    save_val_outputs(run_dir, ds, counts)
    path.write_text(to_json(results))
    print(f"updated {path} and wrote val_outputs.npz")


def run_test(args, model_mod, run_dir, recordings, window, step, device, hw):
    """Score the saved int8 weights on the test split. Meant to run once, at the end."""
    out = run_dir / "test_results.json"
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists: the test set is evaluated once. Use --force to override.")
    results = json.loads((run_dir / "results.json").read_text())
    if "best_alarm" not in results:
        raise SystemExit("results.json predates the margin sweep: run with --rescore first")
    model, weights = load_trained(model_mod, run_dir, device, hw)
    ds = WindowDataset(recordings, "test", window, step)
    hours = split_hours(recordings, "test")
    counts, hidden, inputs = predict(model, ds, hw, device)
    checked, count_bad, dec_bad = verify(model_mod, weights, ds, counts, args.verify_n, args.seed) if hw else (0, 0, 0)
    n = results["alarm_n"]
    m, n_best = results["best_alarm"]["margin"], results["best_alarm"]["n_consec"]
    test = {"window": window_metrics(decide(counts), ds.labels, hours),
            "alarm": alarm_metrics(ds, decide(counts), n, hours),
            "alarm_sweep": [alarm_metrics(ds, decide(counts), k, hours) for k in ALARM_NS],
            # margin ablation: the (margin, N) picked on validation, applied once to test
            "best_margin": {"window": window_metrics(decide(counts, m), ds.labels, hours),
                            "alarm": alarm_metrics(ds, decide(counts, m), n_best, hours, m)},
            "hours": hours, "hidden_rate": hidden, "input_rate": inputs,
            "agreement": {"checked": checked, "count_mismatches": count_bad, "decision_mismatches": dec_bad} if hw else None}
    out.write_text(to_json(test))
    np.savez(run_dir / "test_outputs.npz", counts=counts, labels=ds.labels, recording=ds.recording, start=ds.start)
    w = test["window"]
    print(f"TEST ({hours:.1f} h): window sens {w['sensitivity']}, spec {w['specificity']}, FP/h {w['fp_per_hour']}")
    print(f"  spec alarms:        {fmt_alarm(test['alarm'])}")
    print(f"  best-margin alarms: {fmt_alarm(test['best_margin']['alarm'])}")
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
    if args.float_only and not is_hardware(model_mod):
        raise SystemExit(f"--float-only: '{args.model}' is already software-only")
    hw = is_hardware(model_mod) and not args.float_only
    config = getattr(model_mod, "CONFIG", {})
    set_tag = apply_overrides(config, args.set, args.model)   # edits the module's CONFIG, like --hidden
    hidden_tag = ""
    if args.hidden is not None:
        if "n_hidden" not in config:
            raise SystemExit(f"--hidden: model '{args.model}' has no n_hidden in its CONFIG")
        if args.hidden != config["n_hidden"]:
            hidden_tag = f"_h{args.hidden}"
        # Set it in the module's CONFIG itself: Net, run_int_reference, weight_shapes, spec_md and
        # the handoff's reference copy all default to cfg=CONFIG (the same dict), so they all see it.
        config["n_hidden"] = args.hidden

    window = round(args.window * FS)
    step = window // 2
    if (window, step) != (WINDOW, STEP):
        print(f"NOTE: window {window} / step {step} differs from the hardware spec ({WINDOW} / {STEP})")
    tag = (f"{args.patient}_{args.model}_s{args.seed}" + ("" if window == WINDOW else f"_w{window}") + hidden_tag
           + set_tag + ("_float" if args.float_only else ""))
    run_dir = args.out / tag
    if (args.rescore or args.test) and not (run_dir / "results.json").exists():
        raise SystemExit(f"no finished run at {run_dir} (same --seed / --window / --hidden / --set as training?)")
    if args.skip_done and not (args.rescore or args.test) and (run_dir / "results.json").exists():
        print(f"{run_dir} is already done, skipping")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"device {device}, run dir {run_dir}" + (f", n_hidden {config['n_hidden']}" if "n_hidden" in config else ""))

    recordings = load_recordings([args.patient], args.cache)
    if args.test:
        run_test(args, model_mod, run_dir, recordings, window, step, device, hw)
        return
    if args.rescore:
        rescore(model_mod, run_dir, recordings, window, step, device, hw)
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
    report = alarm_report(ds_val, counts, val_hours)
    save_val_outputs(run_dir, ds_val, counts)
    cost = measure_cost(model_mod, model, ds_val, hw, device)
    if cost and cost.get("adds_per_step") is not None:
        print(f"cost: {cost.get('weights')} weights, {cost.get('worst_adds_per_step')} worst-case adds/step, "
              f"{cost['adds_per_step']:.0f} measured adds/step on validation")

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
               "config": getattr(model_mod, "CONFIG", None), "args": vars(args), "split": splits, "val": final, **report, "agreement": agreement,
               "density": {"input": final["input_rate"], "hidden": final["hidden_rate"]}, "cost": cost,
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
