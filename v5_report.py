#!/usr/bin/env python3
"""v5_report.py -- verdict on the nocomm calibration: gate series with new-best markers,
rho=0.9 evaluation vs references, and the pass/fail line per seed. Run: python v5_report.py"""
import argparse, csv, glob, json, os, subprocess, sys
import numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--episodes", type=int, default=200)
ap.add_argument("--seeds", default="60,61")
ap.add_argument("--prefix", default="v5_nocomm"); a = ap.parse_args()
REF = {60: 4462.4, 61: 4304.7}
verdicts = []
for seed in [int(s) for s in a.seeds.split(",")]:
    tag = f"{a.prefix}_s{seed}"
    ds = [d for d in glob.glob(os.path.join(ROOT, "weights_signal", f"run_signal_*_{tag}"))
          if os.path.isdir(d)]
    if not ds:
        print(f"\n== {tag}: NO RUN DIR (training never started) =="); continue
    d = max(ds, key=os.path.getmtime)
    print(f"\n== {tag}  ({os.path.basename(d)}) ==")
    h = list(csv.DictReader(open(os.path.join(d, "metrics_heldout.csv"))))
    best, best_ep = 1e18, None
    for r in h:
        v = float(r["heldout_mean_cost"]); m = ""
        if v < best - 1e-9:
            best, best_ep, m = v, int(r["episode"]), "  <-- new best"
        print(f"   ep {r['episode']:>5}  gate {v:8.1f}{m}")
    # v5 pass criterion (cold start trivially beats its first gates): the run must still be
    # improving DEEP into training AND its deterministic policy must beat the frozen warm-start
    # instrument it replaces.
    late = best_ep is not None and best_ep >= 1000
    ck = os.path.join(d, "signal_checkpoint_best.pt")
    out = os.path.join(ROOT, "repair_out", "v5", tag)
    if not os.path.exists(os.path.join(out, f"seed{seed}.json")):
        subprocess.run([sys.executable, "agents/eval_signal.py", "--ckpt", ck,
                        "--dump-comm", out, "--dump-ar1", "0.9",
                        "--dump-episodes", str(a.episodes)], stdout=subprocess.DEVNULL)
    p = os.path.join(out, f"seed{seed}.json")
    c = (np.mean([float(x) for x in json.load(open(p)).values()])
         if os.path.exists(p) else float("nan"))
    print(f"   rho=0.9 eval: {c:.1f}   (registered nocomm {REF.get(seed,'?')}, "
          f"frontier 3747.6, static 4988.6)")
    deep = late and c == c and c < REF.get(seed, 1e18)
    v = ("PASS -- still improving at ep %s and beats the frozen v3 instrument (%.1f < %.1f)"
         % (best_ep, c, REF.get(seed, float("nan"))) if deep else
         "FAIL -- best @ ep %s, eval %.1f vs v3 frozen %.1f" % (best_ep, c, REF.get(seed, float("nan"))))
    late = deep
    print(f"   gate best @ ep {best_ep}: {v}")
    verdicts.append((tag, late, c))
if verdicts:
    print("\n== SUMMARY ==")
    for tag, late, c in verdicts:
        print(f"   {tag}: {'PASS' if late else 'FAIL'}   rho0.9={c:.1f}")
    print("   both PASS -> the categorical learner is calibrated; comm pairs next."
          "\n   FAIL with best_ep deep in training -> read fig14 first: if the train curve was"
          "\n   still descending at the stop, the stop rule is binding (rerun with --patience"
          " 4000 --min-ep 4000 --suffix f), not the head.")
