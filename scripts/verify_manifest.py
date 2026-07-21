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
PROBE_DIRS = ["probes/iv_ar1r9_upstream", "probes/iv_ar1r9_rbroadcast",
              "probes/iv_ar1r9_rbroadcast_raw", "probes/iv_ar1r9_rbroadcast_learned",
              "probes/iv_ar1r9_rbroadcast_eps", "probes/iv_ar1r9_rbroadcast_condmean",
              "probes/iv_ar1r9_upstream_raw", "probes/iv_ar1r9_downstream_raw",
              "probes/iv_ar1r9_beta0_upstream", "probes/iv_ar1r9_beta05_upstream"]
EXTRA_V13 = ["v13/dp_dhat_up", "v13/dp_true_lambda", "v13/ar1r9_eps", "v13/ar1r9_condmean",
             "v13/ar1r9_learned", "v13/top_up_raw", "v13/top_down_raw", "v13/lag1", "v13/lag2",
             "v13/rho0_nocomm", "v13/rho0_dhat", "v13/rho0_raw", "v13/rho3_nocomm", "v13/rho3_dhat",
             "v13/rho3_raw", "v13/rho6_nocomm", "v13/rho6_dhat", "v13/rho6_raw"]
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
        # Amendment v2.1 (pre-unblinding): confirmatory seeds are 25..49. 10-24 = Study-1 prior
        # campaign; >=50 = dev/pilot (incl. the seed-50 QMIX certification runs). The original
        # 30..54 window self-contradicted the >=50 pilot exclusion -- caught fail-closed at S10
        # BEFORE any confirmatory analysis ran; see the registry's amendment_log.
        if s < 25 or s >= 50:
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
    for cell in PRIMARY_CELLS + EXTRA_V13:
        d = os.path.join(a.root, cell)
        if not os.path.isdir(d):
            problems.append(f"DUMP: registered cell {cell} missing entirely")
            continue
        for s in seeds:
            if not os.path.exists(os.path.join(d, f"seed{s}.json")):
                problems.append(f"DUMP: {cell} missing seed{s}.json")

    for d0 in PROBE_DIRS:
        dd = os.path.join(a.root, d0)
        if not os.path.isdir(dd):
            problems.append(f"PROBE: {d0} missing entirely")
        else:
            for s in seeds:
                if not os.path.exists(os.path.join(dd, f"seed{s}_iv.json")):
                    problems.append(f"PROBE: {d0} missing seed{s}_iv.json")

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