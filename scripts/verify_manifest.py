#!/usr/bin/env python3
"""verify_manifest.py -- FAIL-CLOSED campaign integrity gate (review 2.0 fix #7).

The confirmatory pipeline must never silently reduce n. This verifier hard-fails (exit 1) if:
  (a) any arm x seed emitted in sweep_out/jobs.tsv lacks its completion sentinel
      weights_signal/.done_<arm>_s<seed>;
  (b) any REGISTERED confirmatory cell below is missing seed{S}.json for ANY registered seed;
  (c) any registered seed overlaps the prior-campaign (10-24) or dev/pilot (>=50) ranges,
      or lies below 30.
Run by auto_campaign2.sh after S10 (dump+probe); also runnable standalone:
  python scripts/verify_manifest.py --seeds "30 31 ... 44" [--root sweep_out] [--jobs sweep_out/jobs.tsv]
"""
import argparse
import os
import sys

# Registered confirmatory dump cells (primaries + their controls + QMIX concordance set).
PRIMARY_CELLS = [
    "v13/dp_nocomm", "v13/dp_dhat", "v13/dp_raw",
    "v13/ar1r9_nocomm", "v13/ar1r9_dhat", "v13/ar1r9_raw",
    "v13/clip12_nocomm", "v13/clip12_raw", "v13/clip20_nocomm", "v13/clip20_raw",
    "v13/qmix_dp_nocomm", "v13/qmix_dp_dhat", "v13/qmix_dp_raw",
    "v13/qmix_ar1_nocomm", "v13/qmix_ar1_dhat", "v13/qmix_ar1_raw",
    "v13/qmix_clip12_nocomm", "v13/qmix_clip12_raw",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True, help='e.g. "30 31 ... 44"')
    ap.add_argument("--root", default="sweep_out")
    ap.add_argument("--jobs", default="sweep_out/jobs.tsv")
    ap.add_argument("--weights", default="weights_signal")
    a = ap.parse_args()
    seeds = sorted(int(s) for s in a.seeds.split())
    problems = []

    # (c) seed-space hygiene
    for s in seeds:
        if s < 30 or 10 <= s <= 24 or s >= 50:
            problems.append(f"SEED-SPACE: confirmatory seed {s} overlaps prior/pilot ranges")

    # (a) training completeness against the emitted manifest
    if os.path.exists(a.jobs):
        with open(a.jobs) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                arm, seed = parts[0], parts[1]
                if not os.path.exists(os.path.join(a.weights, f".done_{arm}_s{seed}")):
                    problems.append(f"TRAIN: {arm} seed {seed} has no completion sentinel")
    else:
        problems.append(f"TRAIN: jobs manifest {a.jobs} missing")

    # (b) registered confirmatory cells: the EXACT seed vector, no intersection shrinkage
    for cell in PRIMARY_CELLS:
        d = os.path.join(a.root, cell)
        if not os.path.isdir(d):
            problems.append(f"DUMP: registered cell {cell} missing entirely")
            continue
        for s in seeds:
            if not os.path.exists(os.path.join(d, f"seed{s}.json")):
                problems.append(f"DUMP: {cell} missing seed{s}.json")

    if problems:
        print(f"MANIFEST VERIFICATION FAILED ({len(problems)} problems) -- the campaign is INCOMPLETE.")
        for p in problems[:60]:
            print("  -", p)
        if len(problems) > 60:
            print(f"  ... and {len(problems)-60} more")
        print("No confirmatory analysis may run on this state (fail-closed; review 2.0 fix #7).")
        sys.exit(1)
    print(f"MANIFEST OK: all trained arms sentinel-complete; all {len(PRIMARY_CELLS)} registered cells "
          f"hold the exact {len(seeds)}-seed vector {seeds[0]}..{seeds[-1]}.")


if __name__ == "__main__":
    main()