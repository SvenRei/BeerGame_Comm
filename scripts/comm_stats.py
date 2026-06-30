"""
comm_stats.py -- cross-seed VALUE-OF-SHARING aggregator for the SIGNAL study.

Turns the H1 communication result from a point estimate into a defended claim. The unit of
analysis is the INDEPENDENT TRAINING SEED: one comm arm and one no-comm arm trained at the same
seed, each scored deterministically per-lambda on the EVAL seeds (CRN), then CRN-PAIRED by seed.

INPUT (produced by `python agents/eval_signal.py --dump-comm DIR`, one DIR per arm):
  DIR/seed{S}.json        {lambda(float): mean_cost(float)}                 # the arm's cost vector
  DIR/seed{S}_ferr.json   {lambda(float): {echelon: upstream_forecast_err}} # the Lee mechanism

WHAT IT REPORTS:
  * V_cost(seed) = mean_lambda(no_comm[seed]) - mean_lambda(comm[seed])   (+ => communication is
    cheaper => comm has value). Mean V_cost + bootstrap CI, paired Wilcoxon/sign test, and a TOST
    EQUIVALENCE test against a +/-band (the H1 NULL: is comm value within a negligible band?).
  * per-echelon FORECAST-ERROR DELTA = ferr(no_comm) - ferr(comm), upstream echelons (the
    mechanism: comm should cut UPSTREAM forecast error -- Lee-So-Tang see-through-bullwhip).

Reuses the inference primitives in scripts.c1_stats (bootstrap_ci / paired / tost), so the math is
shared with the C1 report.

Run:
  python scripts/comm_stats.py                                           # self-test (numpy only)
  python scripts/comm_stats.py report --comm-dir results/comm_on --nocomm-dir results/comm_off
"""
import os
import sys
import json
import glob
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.c1_stats import bootstrap_ci, paired, tost            # shared inference primitives

AGENTS = ["retailer", "wholesaler", "distributor", "manufacturer"]
UPSTREAM = AGENTS[1:]                                               # the Lee channel is upstream-only
TOST_BAND_FRAC = 0.02                                              # +/-2% of no-comm cost (equivalence band)


# ==============================================================================
# Loaders (mirror the eval_signal --dump-comm layout)
# ==============================================================================
def _seed_of(path):
    base = os.path.basename(path)
    try:
        return int("".join(ch for ch in base.split("seed")[1].split("_")[0].split(".")[0] if ch.isdigit()))
    except (IndexError, ValueError):
        return None


def load_cost_dir(d):
    """DIR/seed{S}.json -> {seed: {lambda: cost}} (skips the _ferr / _bw siblings)."""
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "seed*.json"))):
        if p.endswith("_ferr.json") or p.endswith("_bw.json"):
            continue
        s = _seed_of(p)
        if s is None:
            continue
        with open(p) as f:
            out[s] = {float(k): float(v) for k, v in json.load(f).items()}
    return out


def load_ferr_dir(d):
    """DIR/seed{S}_ferr.json -> {seed: {lambda: {echelon: forecast_error}}}."""
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "seed*_ferr.json"))):
        s = _seed_of(p)
        if s is None:
            continue
        with open(p) as f:
            out[s] = {float(k): {a: float(b) for a, b in v.items()} for k, v in json.load(f).items()}
    return out


def _common(comm, nocomm, lambdas=None):
    seeds = sorted(set(comm) & set(nocomm))
    if not seeds:
        raise ValueError(f"no shared seeds (comm={sorted(comm)}, nocomm={sorted(nocomm)})")
    lam_sets = [set(comm[s]) for s in seeds] + [set(nocomm[s]) for s in seeds]
    lams = set.intersection(*lam_sets)
    if lambdas is not None:
        lams &= {float(l) for l in lambdas}
    return seeds, sorted(lams)


# ==============================================================================
# Value of sharing (CRN-paired by seed)
# ==============================================================================
def value_of_sharing(comm, nocomm, lambdas=None, band_frac=TOST_BAND_FRAC, n_boot=10000, ci=0.95):
    seeds, lams = _common(comm, nocomm, lambdas)
    if not lams:
        raise ValueError("no shared lambdas between the arms")
    comm_seed = np.array([float(np.mean([comm[s][l] for l in lams])) for s in seeds])
    nocomm_seed = np.array([float(np.mean([nocomm[s][l] for l in lams])) for s in seeds])
    diffs = nocomm_seed - comm_seed                               # + => comm cheaper => comm has value
    band = band_frac * float(np.mean(nocomm_seed))
    pr = paired(diffs, np.zeros_like(diffs))                      # H0: no comm value
    eq = tost(diffs, -band, band)                                 # equivalence within +/- band
    v_mean = float(np.mean(diffs))
    return {
        "n_seeds": len(seeds), "seeds": seeds, "lambdas": lams,
        "v_cost_mean": v_mean,
        "v_cost_ci": bootstrap_ci(diffs, n_boot=n_boot, ci=ci),
        "v_cost_ci_excludes_0": bool(bootstrap_ci(diffs, n_boot=n_boot, ci=ci)[0] > 0.0),
        "v_cost_pct": float(100.0 * v_mean / max(1e-9, float(np.mean(nocomm_seed)))),
        "comm_cost_mean": float(np.mean(comm_seed)),
        "nocomm_cost_mean": float(np.mean(nocomm_seed)),
        "wilcoxon_p": pr["wilcoxon_p"], "sign_p": pr["sign_p"],
        "n_pos": pr["n_pos"], "n_nonzero": pr["n_nonzero"],
        "band": band, "tost_p": eq["p_tost"], "equivalent": eq["equivalent"],
    }


def forecast_delta(comm_ferr, nocomm_ferr, lambdas=None, n_boot=10000, ci=0.95):
    """Per-echelon mean (ferr_no_comm - ferr_comm) across seeds x lambdas. >0 => comm cuts the
    echelon's forecast error (the Lee mechanism), concentrated upstream."""
    seeds, lams = _common(comm_ferr, nocomm_ferr, lambdas)
    out = {}
    for a in AGENTS:
        deltas = []
        for s in seeds:
            for l in lams:
                co = comm_ferr[s][l].get(a)
                no = nocomm_ferr[s][l].get(a)
                if co is not None and no is not None and np.isfinite(co) and np.isfinite(no):
                    deltas.append(no - co)
        deltas = np.asarray(deltas, float)
        out[a] = {"delta_mean": float(np.mean(deltas)) if deltas.size else float("nan"),
                  "delta_ci": bootstrap_ci(deltas, n_boot=n_boot, ci=ci) if deltas.size else (float("nan"),) * 2,
                  "n": int(deltas.size)}
    return {"seeds": seeds, "lambdas": lams, "per_echelon": out}


# ==============================================================================
# Report
# ==============================================================================
def print_report(vs, fd=None):
    def _ci1(t): return f"[{t[0]:.1f}, {t[1]:.1f}]"
    print("=" * 78)
    print(f"VALUE-OF-SHARING REPORT  |  {vs['n_seeds']} seed-pairs  |  lambdas "
          f"{[f'{l:g}' for l in vs['lambdas']]}")
    print("=" * 78)
    print(f"  comm cost mean      : {vs['comm_cost_mean']:.1f}")
    print(f"  no-comm cost mean   : {vs['nocomm_cost_mean']:.1f}")
    flag = "PASS (CI excludes 0)" if vs["v_cost_ci_excludes_0"] else "NOT SIGNIFICANT (CI includes 0)"
    print(f"  V_cost (no_comm - comm) : {vs['v_cost_mean']:+.1f}  ({vs['v_cost_pct']:+.1f}%)  "
          f"95% CI {_ci1(vs['v_cost_ci'])}  -> {flag}")
    print(f"  paired Wilcoxon p = {vs['wilcoxon_p']:.4g}   sign-test p = {vs['sign_p']:.4g}   "
          f"({vs['n_pos']}/{vs['n_nonzero']} seeds comm-cheaper)")
    eqv = "EQUIVALENT to no-comm" if vs["equivalent"] else "not equivalent (effect outside band)"
    print(f"  TOST equivalence band +/-{vs['band']:.1f}  p_tost = {vs['tost_p']:.4g}  -> {eqv}")
    if vs["n_seeds"] <= 6:
        print(f"  [note] n={vs['n_seeds']}: nonparametric p is floored ~0.06; read the bootstrap CI as primary.")
    if fd is not None:
        print("-" * 78)
        print("  FORECAST-ERROR DELTA (ferr_no_comm - ferr_comm; >0 = comm cuts the echelon's forecast error)")
        print(f"    {'echelon':<13}{'delta':>9}{'95% CI':>18}{'n':>6}")
        for a in AGENTS:
            r = fd["per_echelon"][a]
            tag = "  <- upstream (Lee channel)" if a in UPSTREAM else ""
            print(f"    {a:<13}{r['delta_mean']:>+9.3f}{_ci1(r['delta_ci']):>18}{r['n']:>6}{tag}")
        print("    (the H3 mechanism: the value of sharing should show as a positive delta concentrated upstream.)")
    print("=" * 78)


def _report_cli(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="comm_stats.py report")
    ap.add_argument("--comm-dir", required=True, help="dir of seed{S}.json for the COMM arm")
    ap.add_argument("--nocomm-dir", required=True, help="dir of seed{S}.json for the NO-COMM arm")
    ap.add_argument("--lambdas", nargs="+", type=float, default=None)
    ap.add_argument("--band-frac", type=float, default=TOST_BAND_FRAC)
    args = ap.parse_args(argv)
    comm, nocomm = load_cost_dir(args.comm_dir), load_cost_dir(args.nocomm_dir)
    if not comm or not nocomm:
        print(f"no seed*.json found (comm={len(comm)}, nocomm={len(nocomm)}). "
              f"Generate them with `eval_signal.py --dump-comm`.")
        return
    vs = value_of_sharing(comm, nocomm, lambdas=args.lambdas, band_frac=args.band_frac)
    cf, nf = load_ferr_dir(args.comm_dir), load_ferr_dir(args.nocomm_dir)
    fd = forecast_delta(cf, nf, lambdas=args.lambdas) if (cf and nf) else None
    print_report(vs, fd)


# ==============================================================================
# Self-test (numpy only; scipy optional for the p-values)
# ==============================================================================
def _selftest():
    print("=" * 70)
    print("SELF-TEST: comm_stats value-of-sharing math (numpy only; scipy optional)")
    print("=" * 70)
    lams = [8.0, 16.0]
    rng = np.random.default_rng(0)

    # Build two arms over 12 seeds: comm is consistently ~200 cheaper than no-comm.
    comm = {s: {l: 3000.0 + 20 * (l - 12) + rng.normal(0, 30) for l in lams} for s in range(12)}
    nocomm = {s: {l: comm[s][l] + 200.0 + rng.normal(0, 15) for l in lams} for s in range(12)}
    vs = value_of_sharing(comm, nocomm, n_boot=3000)
    assert vs["n_seeds"] == 12 and vs["lambdas"] == lams
    assert vs["v_cost_mean"] > 150.0 and vs["v_cost_ci_excludes_0"] is True       # real, detected effect
    assert vs["equivalent"] is False                                              # 200 >> +/-2% band
    print(f"  A) comm 200 cheaper -> V_cost={vs['v_cost_mean']:+.1f} ({vs['v_cost_pct']:+.1f}%), "
          f"CI excludes 0, not equivalent  PASS")

    # Null arm: comm == no-comm up to tiny noise -> V ~ 0, equivalent within band.
    comm2 = {s: {l: 3000.0 + rng.normal(0, 8) for l in lams} for s in range(20)}
    nocomm2 = {s: {l: 3000.0 + rng.normal(0, 8) for l in lams} for s in range(20)}
    vs2 = value_of_sharing(comm2, nocomm2, n_boot=3000)
    assert vs2["v_cost_ci_excludes_0"] is False and vs2["equivalent"] is True
    print(f"  B) comm == no-comm  -> V_cost={vs2['v_cost_mean']:+.1f}, CI includes 0, EQUIVALENT  PASS")

    # Forecast delta: comm cuts UPSTREAM error (positive upstream delta), retailer ~0.
    cf = {s: {l: {"retailer": 0.0, "wholesaler": 2.0, "distributor": 3.0, "manufacturer": 4.0}
              for l in lams} for s in range(6)}
    nf = {s: {l: {"retailer": 0.0, "wholesaler": 4.0, "distributor": 6.0, "manufacturer": 8.0}
              for l in lams} for s in range(6)}
    fd = forecast_delta(cf, nf, n_boot=2000)
    assert abs(fd["per_echelon"]["retailer"]["delta_mean"]) < 1e-9
    assert fd["per_echelon"]["manufacturer"]["delta_mean"] > 3.0                  # comm cut 8->4 upstream
    print(f"  C) forecast delta: retailer~0, manufacturer="
          f"{fd['per_echelon']['manufacturer']['delta_mean']:+.2f} (upstream comm gain)  PASS")

    print_report(vs, fd)
    print("\ncomm_stats self-test PASS")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        _report_cli(sys.argv[2:])
    else:
        _selftest()
