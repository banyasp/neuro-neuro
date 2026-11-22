"""
Run the planned ablations (planned_ablations.md) through train.py: one factor at a time from
each model's default config. Model C (scnn), Model A (lif) and Model B (alif) sweeps.

    # from ML/ (on Colab, point --cache / --out at Drive). Resumable: finished runs are skipped,
    # an interrupted one continues from its checkpoint.
    python ablate.py --patients chb01 chb03 --seeds 0 --cache <cache> --out <runs>        # screen all
    python ablate.py ... --groups lif alif   # only the Model A / B sweeps (scnn = Model C must-do)
    python ablate.py --patients chb01 chb03 chb05 chb08 chb10 --seeds 0 1 2 --only baseline fc ...
    python ablate.py ... --nice              # also the Model C nice-to-have ablations
    python ablate.py ... --dry-run           # just print the train.py commands
    python ablate.py ... --train-args "--epochs 100 --verify-n 200"

Then:  python summarize.py --out <runs> --csv ablations.csv --md ablations.md
       python diversity.py --patient chb01 --cache <cache> --out <runs>      (A vs B vs C)

Suggested protocol: screen every ablation on 2 patients with 1 seed (validation seizures are few,
so event-level differences between configs are mostly noise; window-level F1 and FP/h are
steadier), then run the shortlist with 3 seeds on every patient. Never look at test for this.
"""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

# name: (question, train.py arguments). "baseline" = scnn's CONFIG: 8 filters (4 at d = 1,
# 4 at d = 3), K = 16, depthwise D = 2, spiking, int8. lif / alif are included so that
# diversity.py has A and B on the same patients and seeds.
ABLATIONS = {
    "baseline": ("Model C as specified", ["--model", "scnn"]),
    "lif": ("Model A, for comparison and diversity", ["--model", "lif"]),
    "alif": ("Model B, for comparison and diversity", ["--model", "alif"]),
    "k1": ("1. does temporal filtering help? (K = 1: per-channel LIFs only)",
           ["--model", "scnn", "--set", "kernel_len=1"]),
    "d1": ("2. temporal reach: dilation 1 only (125 ms)",
           ["--model", "scnn", "--set", "dilations=1", "group_filters=8"]),
    "d124": ("2. temporal reach: dilations 1, 2, 4 (12 filters, up to 470 ms, 61-frame history)",
             ["--model", "scnn", "--set", "dilations=1,2,4", "group_filters=4,4,4"]),
    "k32": ("2. temporal reach: K = 32 at dilation 1 (250 ms; longer kernel instead of dilation)",
            ["--model", "scnn", "--set", "kernel_len=32", "dilations=1", "group_filters=8"]),
    "fc": ("3. is the spatial stage worth it? fully connected 184 -> 2 readout",
           ["--model", "scnn", "--set", "readout=fc"]),
    "float_spiking": ("4b. cost of int8: float, spiking", ["--model", "scnn", "--float-only"]),
    "float_relu": ("4a. cost of spiking: float, non-spiking twin", ["--model", "scnn_float"]),
    "leak4": ("leak: k = 4 (tau ~ 0.125 s)", ["--model", "scnn", "--set", "leak_shift=4"]),
    "leak6": ("leak: k = 6 (tau ~ 0.5 s)", ["--model", "scnn", "--set", "leak_shift=6"]),
}
# Model A (lif) and Model B (alif): one factor at a time from their default CONFIG
# (64 hidden, k = 5; alif adapt_inc = 16, adapt_shift = 7). See planned_ablations.md.
LIF_ABLATIONS = {
    "lif_h32": ("A: hidden size 32", ["--model", "lif", "--hidden", "32"]),
    "lif_h128": ("A: hidden size 128", ["--model", "lif", "--hidden", "128"]),
    "lif_h192": ("A: hidden size 192", ["--model", "lif", "--hidden", "192"]),
    "lif_leak4": ("A: leak k = 4 (tau ~ 0.125 s)", ["--model", "lif", "--set", "leak_shift=4"]),
    "lif_leak6": ("A: leak k = 6 (tau ~ 0.5 s)", ["--model", "lif", "--set", "leak_shift=6"]),
    "lif_float": ("A: cost of int8 (float only)", ["--model", "lif", "--float-only"]),
}
ALIF_ABLATIONS = {
    "alif_noadapt": ("B: adaptation off (adapt_inc = 0, i.e. LIF)", ["--model", "alif", "--set", "adapt_inc=0"]),
    "alif_inc8": ("B: weaker adaptation (adapt_inc = 8)", ["--model", "alif", "--set", "adapt_inc=8"]),
    "alif_inc32": ("B: stronger adaptation (adapt_inc = 32)", ["--model", "alif", "--set", "adapt_inc=32"]),
    "alif_ashift6": ("B: shorter adaptation (adapt_shift = 6, tau ~ 0.5 s)", ["--model", "alif", "--set", "adapt_shift=6"]),
    "alif_ashift8": ("B: longer adaptation (adapt_shift = 8, tau ~ 2 s)", ["--model", "alif", "--set", "adapt_shift=8"]),
    "alif_h32": ("B: hidden size 32", ["--model", "alif", "--hidden", "32"]),
    "alif_h128": ("B: hidden size 128", ["--model", "alif", "--hidden", "128"]),
    "alif_h192": ("B: hidden size 192", ["--model", "alif", "--hidden", "192"]),
    "alif_float": ("B: cost of int8 (float only)", ["--model", "alif", "--float-only"]),
}
GROUPS = {"scnn": "Model C must-do (plus lif / alif defaults)", "lif": "Model A sweeps",
          "alif": "Model B sweeps", "nice": "Model C nice-to-have"}
NICE = {
    "unshared": ("weight sharing: one filter bank per channel (23x layer-1 weights)",
                 ["--model", "scnn", "--set", "share_filters=false"]),
    "c4": ("width: C = 4", ["--model", "scnn", "--set", "group_filters=2,2"]),
    "c16": ("width: C = 16", ["--model", "scnn", "--set", "group_filters=8,8"]),
    "d_mult1": ("width: D = 1 spatial unit per filter", ["--model", "scnn", "--set", "depth_mult=1"]),
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--patients", nargs="+", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--cache", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--groups", nargs="+", choices=list(GROUPS), default=["scnn", "lif", "alif"],
                   help="which sets to run (default: %(default)s). "
                        + "; ".join(f"{g} = {d}" for g, d in GROUPS.items()))
    p.add_argument("--only", nargs="+", help="ablation names instead of --groups. Options: "
                                             + ", ".join({**ABLATIONS, **LIF_ABLATIONS, **ALIF_ABLATIONS, **NICE}))
    p.add_argument("--nice", action="store_true", help="same as adding 'nice' to --groups")
    p.add_argument("--train-args", default="--verify-n 500",
                   help="extra train.py arguments for every run (default: %(default)s)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    by_group = {"scnn": ABLATIONS, "lif": LIF_ABLATIONS, "alif": ALIF_ABLATIONS, "nice": NICE}
    table = {k: v for g in by_group.values() for k, v in g.items()}
    groups = args.groups + (["nice"] if args.nice and "nice" not in args.groups else [])
    names = args.only or [n for g in groups for n in by_group[g]]
    unknown = [n for n in names if n not in table]
    if unknown:
        raise SystemExit(f"unknown ablation(s): {', '.join(unknown)}")

    common = shlex.split(args.train_args) + ["--resume", "--skip-done"]
    if args.cache:
        common += ["--cache", str(args.cache)]
    if args.out:
        common += ["--out", str(args.out)]
    jobs = [(n, patient, seed) for patient in args.patients for seed in args.seeds for n in names]
    failed = []
    for i, (name, patient, seed) in enumerate(jobs, 1):
        cmd = [sys.executable, "train.py", "--patient", patient, "--seed", str(seed), *table[name][1], *common]
        print(f"\n[{i}/{len(jobs)}] {name}: {table[name][0]}\n  {' '.join(shlex.quote(c) for c in cmd[1:])}", flush=True)
        if args.dry_run:
            continue
        if subprocess.run(cmd, cwd=Path(__file__).resolve().parent).returncode:
            failed.append(f"{name} {patient} s{seed}")
    if failed:
        print(f"\nFAILED ({len(failed)}): {'; '.join(failed)}")
        raise SystemExit(1)
    print("\nall runs done. Next: python summarize.py --out <runs> --csv ablations.csv --md ablations.md")


if __name__ == "__main__":
    main()
