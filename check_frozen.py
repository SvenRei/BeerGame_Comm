#!/usr/bin/env python3
"""check_frozen.py -- was the agent FROZEN, or did it CONVERGE early?

Reads the scientific CSV logs every run writes (metrics_update.csv,
metrics_heldout.csv, run_meta.json) and classifies each run. Works on your
Windows machine against a downloaded archive or a fresh local run; needs only
the standard library (torch is optional, for the milestone-vs-best byte check).

USAGE
  python check_frozen.py                      # scans ./weights_signal
  python check_frozen.py path\\to\\weights_signal
  python check_frozen.py --arm ar1r9          # only arms whose name contains this

THE DISTINCTION THIS SETTLES
  A FROZEN actor (the historical DRACO failure) never moves: the base-stock
  level S_mean stays pinned at its initialization, exploration noise never
  adapts, and the PPO update produces ~zero KL because the policy after the
  update equals the policy before it. An EARLY-CONVERGED run moved, improved,
  then plateaued -- under AR(1) with FIXED mean demand (mu=12) that is the
  EXPECTED outcome: there is no regime to keep inferring, so the best gate
  lands early and patience stops the run. Identical budget-milestone files are
  then byte-copies of that early best -- correct semantics, not corruption.

VERDICT RULES (validated against a real lr=0 run vs a real healthy run: the
KL gap between them is FIVE orders of magnitude, while S_mean varies ~1-2 units
from batch stochasticity even with pinned weights -- so KL decides, S is evidence)
  FROZEN            max |approx_kl| < 1e-6  AND  action_std range < 0.02
                    (the update provably changes nothing; eval costs byte-repeat)
  CONVERGED-EARLY   KL shows the policy DID move; best gate in the first third;
                    no later improvement     (expected for AR(1) fixed-mu)
  HEALTHY           policy moved and held-out cost still improving later on
  Anything unreadable is reported, never guessed.
"""
import argparse, csv, glob, json, os, sys

def _f(row, key, default=float("nan")):
    try: return float(row.get(key, default))
    except (TypeError, ValueError): return default

def read_csv(path):
    try:
        with open(path, newline="") as fh:
            return list(csv.DictReader(fh))
    except OSError:
        return []

def analyze_run(rundir):
    name = os.path.basename(rundir)
    arm = name.split("run_signal_", 1)[-1]
    arm = arm.split("_", 1)[1] if "_" in arm else arm          # drop the wandb id
    upd = read_csv(os.path.join(rundir, "metrics_update.csv"))
    held = read_csv(os.path.join(rundir, "metrics_heldout.csv"))
    out = {"arm": arm, "dir": name, "verdict": "UNREADABLE", "why": "", "extra": ""}
    if len(upd) < 3 or len(held) < 2:
        out["why"] = f"too few rows (update={len(upd)}, heldout={len(held)}) -- was SIGNAL_CSVLOG=1?"
        return out

    S    = [_f(r, "S_mean") for r in upd]
    std  = [_f(r, "action_std") for r in upd]
    kl   = [abs(_f(r, "approx_kl", 0.0)) for r in upd]
    gn   = [_f(r, "grad_norm") for r in upd]
    cost = [_f(r, "heldout_mean_cost") for r in held]
    eps  = [int(_f(r, "episode", 0)) for r in held]

    S_rng   = max(S) - min(S)
    std_rng = (max(std) - min(std)) if all(x == x for x in std) else float("nan")
    kl_max  = max(kl) if kl else float("nan")
    best_i  = cost.index(min(cost))
    c_rng   = (max(cost) - min(cost)) / max(1e-9, abs(sum(cost) / len(cost)))
    late_improve = best_i > len(cost) // 3

    ev = (f"S_mean {min(S):.1f}->{max(S):.1f} (range {S_rng:.2f}) | action_std range "
          f"{std_rng:.3f} | max|KL| {kl_max:.2e} | grad_norm~{sum(gn)/len(gn):.1f} | "
          f"heldout {cost[0]:.0f}->{min(cost):.0f} (best@ep {eps[best_i]}, gate {best_i+1}/{len(cost)}) "
          f"| cost rel.range {100*c_rng:.2f}%")

    moved = kl_max == kl_max and kl_max >= 1e-6          # KL is the ground truth of movement
    if not moved and (std_rng != std_rng or std_rng < 0.02):
        out.update(verdict="FROZEN",
                   why=f"updates change NOTHING: max|KL|={kl_max:.1e} (healthy ~1e-2), "
                       f"exploration static; heldout repeats to {100*c_rng:.2f}%")
    elif moved and not late_improve:
        out.update(verdict="CONVERGED-EARLY",
                   why="policy moved, best found early, no later improvement "
                       "(EXPECTED for AR(1) fixed-mu; patience then stops the run)")
    else:
        out.update(verdict="HEALTHY", why="policy moved; held-out cost improving through training")
    out["extra"] = ev

    # optional: are the budget milestones byte-copies of best? (expected under early stop)
    try:
        import torch
        best_p = os.path.join(rundir, "signal_checkpoint_best.pt")
        buds = sorted(glob.glob(os.path.join(rundir, "signal_checkpoint_budget*.pt")))
        if buds and os.path.exists(best_p):
            b = torch.load(best_p, map_location="cpu", weights_only=False)
            same = 0
            for bp in buds:
                m = torch.load(bp, map_location="cpu", weights_only=False)
                d = max(float((b["actors"][i][k] - m["actors"][i][k]).abs().max())
                        for i in range(len(b["actors"])) for k in b["actors"][0])
                same += (d == 0.0)
            out["extra"] += (f" | milestones==best: {same}/{len(buds)}"
                             " (all-equal = early best copied forward: correct, not corruption)")
    except Exception:
        pass
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="weights_signal")
    ap.add_argument("--arm", default="", help="only run dirs whose name contains this substring")
    a = ap.parse_args()
    runs = sorted(d for d in glob.glob(os.path.join(a.root, "run_signal_*")) if os.path.isdir(d))
    if a.arm:
        runs = [d for d in runs if a.arm in os.path.basename(d)]
    if not runs:
        print(f"no run_signal_* directories under '{a.root}'"); sys.exit(1)

    counts = {}
    print(f"scanning {len(runs)} runs under {a.root}\n" + "=" * 100)
    for d in runs:
        r = analyze_run(d)
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
        print(f"[{r['verdict']:<15}] {r['arm']}")
        print(f"    {r['why']}")
        if r["extra"]:
            print(f"    {r['extra']}")
    print("=" * 100)
    print("SUMMARY: " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if counts.get("FROZEN"):
        print("\n!! FROZEN runs found. Do NOT rent a pod yet -- this reproduces locally and must be")
        print("   fixed first (historical causes: reward_scale, TD target, loss-target scaling).")
    elif counts.get("CONVERGED-EARLY") and not counts.get("FROZEN"):
        print("\nNo frozen runs. CONVERGED-EARLY under AR(1) fixed-mu is the expected, correct")
        print("outcome -- the flat V(budget) curve there is a finding (early redundancy), not a bug.")
    sys.exit(2 if counts.get("FROZEN") else 0)

if __name__ == "__main__":
    main()