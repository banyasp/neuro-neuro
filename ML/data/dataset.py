"""
Load the preprocess.py cache and cut windows + labels on the fly.

Window length and step are arguments (defaults from common.py: 128 samples = 1 s, step 64 = 0.5 s),
so changing them never needs re-running preprocess.py.

    from data.dataset import load_recordings, WindowDataset, balanced_indices
    recs = load_recordings(["chb01"])
    train = WindowDataset(recs, "train")          # drops ambiguous 1-50% seizure windows
    val = WindowDataset(recs, "val")              # keeps every window, natural prevalence
    idx = balanced_indices(train.labels, epoch)   # ~1 seizure : 4 non-seizure, redrawn per epoch

Sanity check (from ML/):  ../.venv/Scripts/python data/dataset.py chb01 chb04
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # so `import common` works from data/
from common import CHANNELS, N_IN, STEP, WINDOW

CACHE_DIR = Path(__file__).resolve().parent / "cache"


def load_recordings(patients, cache_dir=CACHE_DIR):
    """All cached recordings of the given patients, in time order per patient.

    Each is a dict: "spikes" (bit-packed [n_samples, ceil(N_IN/8)] uint8 to save memory),
    "seizure_mask" [n_samples], "patient", "recording", "split".
    """
    recordings = []
    for patient in patients:
        meta = np.load(Path(cache_dir) / patient / "scales.npz")
        for name in meta["recordings"]:
            z = np.load(Path(cache_dir) / patient / f"{name}.npz")
            if list(z["channels"]) != CHANNELS:
                raise ValueError(f"{patient}/{name}.npz was made with {len(z['channels'])} channels, "
                                 f"common.CHANNELS has {len(CHANNELS)}. Re-run data/preprocess.py {patient}")
            recordings.append({"spikes": np.packbits(z["spikes"], axis=1),
                               "seizure_mask": z["seizure_mask"],
                               "patient": patient, "recording": str(z["recording"]),
                               "split": str(z["split"])})
    return recordings


def cut_windows(recordings, split, window=WINDOW, step=STEP):
    """Every window of one split. Windows never cross file boundaries.

    Returns arrays rec (index into recordings), start (sample), frac (seizure overlap).
    """
    rec, start, frac = [], [], []
    for r, recording in enumerate(recordings):
        if recording["split"] != split:
            continue
        mask = recording["seizure_mask"]
        starts = np.arange(0, len(mask) - window + 1, step)
        cum = np.concatenate([[0], np.cumsum(mask, dtype=np.int64)])
        rec.append(np.full(len(starts), r))
        start.append(starts)
        frac.append((cum[starts + window] - cum[starts]) / window)
    if not rec:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    return np.concatenate(rec), np.concatenate(start), np.concatenate(frac)


class WindowDataset(torch.utils.data.Dataset):
    """Windows of one split. Item i = (spikes float32 [window, N_IN], label 0/1).

    Label = 1 if > 50% of the window is inside a seizure. Training drops windows with
    1-50% seizure overlap (ambiguous edges); val/test keep every window.
    Provenance per window: self.patient, self.recording, self.start.
    """

    def __init__(self, recordings, split, window=WINDOW, step=STEP):
        self.recordings, self.window = recordings, window
        rec, start, frac = cut_windows(recordings, split, window, step)
        if split == "train":
            keep = (frac == 0) | (frac > 0.5)
            rec, start, frac = rec[keep], start[keep], frac[keep]
        self.rec, self.start = rec, start
        self.labels = (frac > 0.5).astype(np.int64)
        self.patient = np.array([recordings[r]["patient"] for r in rec])
        self.recording = np.array([recordings[r]["recording"] for r in rec])

    def __len__(self):
        return len(self.labels)

    def window_spikes(self, i):
        packed = self.recordings[self.rec[i]]["spikes"][self.start[i]:self.start[i] + self.window]
        return np.unpackbits(packed, axis=1, count=N_IN)   # uint8 [window, N_IN]

    def __getitem__(self, i):
        return torch.from_numpy(self.window_spikes(i).astype(np.float32)), int(self.labels[i])


def balanced_indices(labels, epoch, neg_per_pos=4, seed=0):
    """Training only: every seizure window plus neg_per_pos x as many random non-seizure windows.

    Redrawn each epoch (deterministic given seed + epoch). Returns shuffled indices.
    """
    rng = np.random.default_rng([seed, epoch])
    pos = np.flatnonzero(labels == 1)
    neg = np.flatnonzero(labels == 0)
    neg = rng.choice(neg, size=min(len(neg), neg_per_pos * len(pos)), replace=False)
    return rng.permutation(np.concatenate([pos, neg]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print window counts and densities per split.")
    parser.add_argument("patients", nargs="+")
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--step", type=int, default=STEP)
    parser.add_argument("--cache", type=Path, default=CACHE_DIR)
    args = parser.parse_args()

    recs = load_recordings(args.patients, args.cache)
    print(f"{len(recs)} recordings, N_IN = {N_IN}, window {args.window}, step {args.step}")
    for split in ("train", "val", "test"):
        ds = WindowDataset(recs, split, args.window, args.step)
        n_pos = int(ds.labels.sum())
        line = f"{split:5s}: {len(ds):7d} windows, {n_pos:5d} seizure ({n_pos / max(1, len(ds)):.2%})"
        if n_pos:
            # Density on a sample of windows, to keep this quick
            rng = np.random.default_rng(0)
            pos = rng.choice(np.flatnonzero(ds.labels == 1), size=min(n_pos, 500), replace=False)
            neg = rng.choice(np.flatnonzero(ds.labels == 0), size=min(len(ds) - n_pos, 500), replace=False)
            sz = np.mean([ds.window_spikes(i).mean() for i in pos])
            bg = np.mean([ds.window_spikes(i).mean() for i in neg])
            line += f", input density seizure {sz:.3f} / non-seizure {bg:.3f}"
        print(line)
    if "train" in {r["split"] for r in recs}:
        train = WindowDataset(recs, "train", args.window, args.step)
        idx = balanced_indices(train.labels, epoch=0)
        print(f"balanced epoch 0: {len(idx)} windows, {train.labels[idx].mean():.2%} seizure")
