"""
Run the Model C (scnn) ablations through train.py: one factor at a time from the baseline.

    # from ML/ (on Colab, point --cache / --out at Drive). Resumable: finished runs are skipped,
    # an interrupted one continues from its checkpoint.
    python ablate.py --patients chb01 chb03 --seeds 0 --cache <cache> --out <runs>        # screen
    python ablate.py --patients chb01 chb03 chb05 chb08 chb10 --seeds 0 1 2 --only baseline fc ...
    python ablate.py ... --nice              # also the nice-to-have ablations
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
}
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
    p.add_argument("--only", nargs="+", help=f"ablation names (default: all must-do). "
                                             f"Options: {', '.join(list(ABLATIONS) + list(NICE))}")
    p.add_argument("--nice", action="store_true", help="add the nice-to-have ablations")
    p.add_argument("--train-args", default="--verify-n 500",
                   help="extra train.py arguments for every run (default: %(default)s)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    table = {**ABLATIONS, **NICE}
    names = args.only or list(ABLATIONS) + (list(NICE) if args.nice else [])
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
