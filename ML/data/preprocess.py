"""
CHB-MIT EDF -> filtered, decimated, scaled, delta-encoded spike cache.

Usage (from ML/):
    ../.venv/Scripts/python data/preprocess.py chb01 chb04 [--alpha 0.3] [--keep-signal] [--plot]

Per patient (details and reasons in data/CLAUDE.md):
  1. Parse chbXX-summary.txt for seizure start/end seconds.
  2. Load every EDF in data/raw/chbXX/ in time order. Keep exactly common.CHANNELS;
     skip files missing a channel or with a flat channel.
  3. Causal filter (0.5 Hz high-pass, 8th-order 40 Hz low-pass, 60 Hz notch), decimate to 128 Hz.
  4. Chronological split by file: ~60% of hours train, ~20% val, ~20% test.
  5. scale_c = std of channel c over the TRAINING files, non-seizure samples only.
  6. Delta-encode signal / scale_c with threshold alpha over each whole recording.
  7. Write data/cache/chbXX/<recording>.npz plus data/cache/chbXX/scales.npz.

Note: chb01 and chb21 are the same person. Harmless for this per-patient split, but never put
them on opposite sides of a cross-patient split.
"""

import argparse
import re
import sys
from pathlib import Path

import mne
import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # so `import common` works from data/
from common import CHANNELS, FS, N_IN

mne.set_log_level("ERROR")

DATA_DIR = Path(__file__).resolve().parent
FLAT_UV = 0.1   # a channel with std below this (microvolts) counts as flat/disconnected


# ---------------------------------------------------------------------------
# Reading the raw data
# ---------------------------------------------------------------------------

def parse_summary(path):
    """{edf file name: [(start_s, end_s), ...]} for every file listed in chbXX-summary.txt."""
    seizures = {}
    current, start = None, None
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith("File Name:"):
            current = line.split(":", 1)[1].strip()
            seizures[current] = []
        elif line.startswith("Channels changed"):
            print(f"  summary: channel change listed after {current} (channels are matched by name)")
        elif current and re.match(r"Seizure\s*\d*\s*Start Time", line):
            start = int(re.search(r"(\d+)\s*seconds?", line).group(1))
        elif current and re.match(r"Seizure\s*\d*\s*End Time", line):
            end = int(re.search(r"(\d+)\s*seconds?", line).group(1))
            seizures[current].append((start, end))
    return seizures


def file_order_key(path):
    """chb01_03.edf -> ("chb01", 3); chb17b_57.edf -> ("chb17b", 57); chb02_16+.edf -> ("chb02", 16, "16+")."""
    stem, num = Path(path).stem.rsplit("_", 1)
    return stem, int(re.match(r"\d+", num).group()), num


def load_eeg(path):
    """Load one EDF. Returns (data [len(CHANNELS), n_samples] in uV, fs), or (None, reason) to skip it."""
    raw = mne.io.read_raw_edf(path, preload=True)
    names = [ch.upper() for ch in raw.ch_names]
    missing = [ch for ch in CHANNELS if ch.upper() not in names]
    if missing:
        return None, f"missing channels {missing}"
    data = raw.get_data(picks=[names.index(ch.upper()) for ch in CHANNELS]) * 1e6
    flat = [ch for ch, s in zip(CHANNELS, data.std(axis=1)) if s < FLAT_UV]
    if flat:
        return None, f"flat channels {flat}"
    return data, raw.info["sfreq"]


# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------

def filter_and_decimate(data, fs_raw):
    """Causal filter of one continuous recording, then keep every Nth sample to get FS Hz."""
    factor = int(round(fs_raw / FS))
    assert factor * FS == fs_raw, f"can't decimate {fs_raw} Hz to {FS} Hz"
    b, a = iirnotch(60.0, 30.0, fs=fs_raw)
    sos = np.vstack([butter(2, 0.5, "highpass", fs=fs_raw, output="sos"),
                     butter(8, 40.0, "lowpass", fs=fs_raw, output="sos"),
                     tf2sos(b, a)])
    # Start the filter in steady state for the first sample (still causal) to avoid a big
    # start-up transient from the DC offset.
    zi = sosfilt_zi(sos)[:, None, :] * data[None, :, :1]
    y, _ = sosfilt(sos, data, axis=1, zi=zi)
    return y[:, ::factor].astype(np.float32)


def seizure_mask(n_samples, intervals):
    mask = np.zeros(n_samples, dtype=np.uint8)
    for start, end in intervals:
        mask[round(start * FS):round(end * FS)] = 1
    return mask


def delta_encode(x, d):
    """Delta modulation (rule in the root CLAUDE.md) of the whole recording at once.

    x: [n_channels, n_samples] scaled signal, d: threshold.
    Returns spikes uint8 [n_samples, 2 * n_channels]: line 2*ch = UP, 2*ch + 1 = DOWN.
    """
    n_ch, n = x.shape
    xt = np.ascontiguousarray(x.T, dtype=np.float64)
    spikes = np.zeros((n, 2 * n_ch), dtype=np.uint8)
    ref = xt[0].copy()
    for t in range(1, n):   # t = 0 can't spike: ref == x[0]
        diff = xt[t] - ref
        up = diff >= d
        down = ~up & (-diff >= d)
        ref += d * up - d * down
        spikes[t, 0::2] = up
        spikes[t, 1::2] = down
    return spikes


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def chronological_split(hours, n_seizures, target=(0.6, 0.2, 0.2)):
    """Files in time order -> list of "train"/"val"/"test" (earliest files train, latest test).

    Tries every pair of cut points at file boundaries and keeps the one closest to 60/20/20 in
    both recording hours and seizure count, with at least one seizure in every split (val needs
    seizures for early stopping, test for the final score). With seizures in only 2 files,
    val is allowed to be seizure-free.
    """
    hours, n_seizures = np.asarray(hours, dtype=float), np.asarray(n_seizures)
    n_files_with_seizures = int((n_seizures > 0).sum())
    if n_files_with_seizures < 2:
        raise ValueError("need seizures in at least 2 files (one for train, one for test)")
    need_val_seizure = n_files_with_seizures >= 3
    cum_h = np.concatenate([[0], np.cumsum(hours)])
    cum_s = np.concatenate([[0], np.cumsum(n_seizures)])
    target = np.array(target)
    n = len(hours)
    best = None
    for n1 in range(1, n - 1):            # files [0, n1) -> train
        for n2 in range(n1 + 1, n):       # files [n1, n2) -> val, [n2, n) -> test
            h = np.array([cum_h[n1], cum_h[n2] - cum_h[n1], cum_h[n] - cum_h[n2]]) / cum_h[n]
            sz = np.array([cum_s[n1], cum_s[n2] - cum_s[n1], cum_s[n] - cum_s[n2]])
            if sz[0] == 0 or sz[2] == 0 or (need_val_seizure and sz[1] == 0):
                continue
            cost = np.abs(h - target).sum() + np.abs(sz / cum_s[n] - target).sum()
            if best is None or cost < best[0]:
                best = (cost, n1, n2)
    if best is None:
        raise ValueError("no chronological split gives every split a seizure")
    _, n1, n2 = best
    return ["train"] * n1 + ["val"] * (n2 - n1) + ["test"] * (n - n2)


# ---------------------------------------------------------------------------
# One patient
# ---------------------------------------------------------------------------

def preprocess_patient(patient, alpha, raw_dir, cache_dir, keep_signal=False, plot=False):
    print(f"\n=== {patient} ===")
    seizures = parse_summary(raw_dir / patient / f"{patient}-summary.txt")
    paths = sorted((raw_dir / patient).glob("*.edf"), key=file_order_key)

    # Pass 1: check every file, filter it, and keep per-channel non-seizure sums for the scales.
    files = []
    for path in paths:
        if path.name not in seizures:
            print(f"  skip {path.name}: not in summary")
            continue
        data, fs_raw = load_eeg(path)
        if data is None:
            print(f"  skip {path.name}: {fs_raw}")
            continue
        x = filter_and_decimate(data, fs_raw).astype(np.float64)
        mask = seizure_mask(x.shape[1], seizures[path.name])
        bg = x[:, mask == 0]
        files.append({"path": path, "recording": path.stem, "hours": x.shape[1] / FS / 3600,
                      "intervals": seizures[path.name], "n_seizures": len(seizures[path.name]),
                      "sum": bg.sum(axis=1), "sumsq": (bg ** 2).sum(axis=1), "count": bg.shape[1]})
    print(f"  {len(files)} files used, {len(paths) - len(files)} skipped")

    splits = chronological_split([f["hours"] for f in files], [f["n_seizures"] for f in files])
    for name in ("train", "val", "test"):
        part = [f for f, s in zip(files, splits) if s == name]
        print(f"  {name:5s}: {len(part):3d} files, {sum(f['hours'] for f in part):6.1f} h, "
              f"{sum(f['n_seizures'] for f in part):2d} seizures")
        if not part:
            print(f"  WARNING: {name} split is empty")

    # Fixed per-channel scale from training files, non-seizure samples only
    train = [f for f, s in zip(files, splits) if s == "train"]
    n = sum(f["count"] for f in train)
    mean = sum(f["sum"] for f in train) / n
    scales = np.sqrt(sum(f["sumsq"] for f in train) / n - mean ** 2)
    print(f"  channels ({len(CHANNELS)}, N_IN = {N_IN}): {', '.join(CHANNELS)}")
    print("  scales (uV): " + ", ".join(f"{s:.1f}" for s in scales))

    out_dir = cache_dir / patient
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "scales.npz", scales=scales, channels=np.array(CHANNELS), alpha=alpha,
             recordings=np.array([f["recording"] for f in files]), splits=np.array(splits),
             hours=np.array([f["hours"] for f in files]),
             n_seizures=np.array([f["n_seizures"] for f in files]))

    # Pass 2: reload, scale, encode, cache. Density stats are over training files only,
    # since that's what alpha is tuned on.
    line_spikes = np.zeros(N_IN)
    train_samples = 0
    seizure_density = []   # (spikes, samples) inside seizures, training files
    background_density = []
    for order, (f, split) in enumerate(zip(files, splits)):
        data, fs_raw = load_eeg(f["path"])
        x = filter_and_decimate(data, fs_raw) / scales[:, None].astype(np.float32)
        spikes = delta_encode(x, alpha)
        mask = seizure_mask(spikes.shape[0], f["intervals"])
        extra = {"signal": x.T.astype(np.float16)} if keep_signal else {}
        np.savez_compressed(out_dir / f"{f['recording']}.npz", spikes=spikes,
                            channels=np.array(CHANNELS), seizure_mask=mask, patient=patient,
                            recording=f["recording"], file_order=order, fs=FS, split=split,
                            alpha=alpha, **extra)
        if split == "train":
            line_spikes += spikes.sum(axis=0)
            train_samples += spikes.shape[0]
            seizure_density.append((spikes[mask == 1].sum(), (mask == 1).sum() * N_IN))
            background_density.append((spikes[mask == 0].sum(), (mask == 0).sum() * N_IN))
        print(f"  wrote {f['recording']}.npz ({split}, density {spikes.mean():.3f})")

    # Density report (training files)
    density = line_spikes / train_samples
    per_channel = density[0::2] + density[1::2]   # UP + DOWN
    print(f"  input density, training files: overall {density.mean():.3f} (target 0.05-0.10)")
    print("  per channel (UP + DOWN): " + ", ".join(f"{ch} {d:.3f}" for ch, d in zip(CHANNELS, per_channel)))
    sz = sum(s for s, _ in seizure_density) / max(1, sum(n for _, n in seizure_density))
    bg = sum(s for s, _ in background_density) / max(1, sum(n for _, n in background_density))
    print(f"  seizure density {sz:.3f}, non-seizure density {bg:.3f}")
    if not 0.05 <= density.mean() <= 0.10:
        print(f"  NOTE: density outside 5-10%; try a {'larger' if density.mean() > 0.10 else 'smaller'} --alpha")

    if plot:
        plot_checks(patient, train, scales, alpha, out_dir / "checks.png")


def plot_checks(patient, train_files, scales, alpha, out_path):
    """Raw vs filtered spectrum of channel 0, and spike rasters of a seizure vs non-seizure window."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import welch

    f = next(f for f in train_files if f["n_seizures"] > 0)
    data, fs_raw = load_eeg(f["path"])
    filt = filter_and_decimate(data, fs_raw)
    spikes = delta_encode(filt / scales[:, None].astype(np.float32), alpha)
    win = 10 * FS   # 10 s windows so the rasters are readable
    sz_start = round(f["intervals"][0][0] * FS)
    bg_start = 0 if sz_start > 2 * win else spikes.shape[0] - win

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for sig, fs, label in ((data[0], fs_raw, "raw"), (filt[0], FS, "filtered + decimated")):
        freqs, psd = welch(sig, fs=fs, nperseg=4 * int(fs))
        axes[0].semilogy(freqs, psd, label=label)
    axes[0].set(title=f"{f['recording']} {CHANNELS[0]} spectrum", xlabel="Hz", ylabel="uV^2/Hz")
    axes[0].legend()
    for ax, start, title in ((axes[1], sz_start, "seizure"), (axes[2], bg_start, "non-seizure")):
        t, line = np.nonzero(spikes[start:start + win])
        ax.scatter(t / FS, line, s=1, marker="|")
        ax.set(title=f"{title} spikes (10 s)", xlabel="s", ylabel="input line")
    fig.suptitle(f"{patient}, alpha = {alpha}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("patients", nargs="+", help="e.g. chb01 chb04")
    parser.add_argument("--alpha", type=float, default=0.3, help="delta threshold in units of channel scale")
    parser.add_argument("--raw", type=Path, default=DATA_DIR / "raw")
    parser.add_argument("--cache", type=Path, default=DATA_DIR / "cache")
    parser.add_argument("--keep-signal", action="store_true", help="also cache the scaled float signal (CNN baseline)")
    parser.add_argument("--plot", action="store_true", help="save a spectrum + raster check plot per patient")
    args = parser.parse_args()
    for p in args.patients:
        preprocess_patient(p, args.alpha, args.raw, args.cache, args.keep_signal, args.plot)
