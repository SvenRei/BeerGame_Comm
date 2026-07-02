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


_SIBLING_SUFFIXES = ("_ferr.json", "_bw.json", "_censor.json", "_iv.json")


def load_cost_dir(d):
    """DIR/seed{S}.json -> {seed: {lambda: cost}} (skips the _ferr/_bw/_censor/_iv siblings)."""
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "seed*.json"))):
        if p.endswith(_SIBLING_SUFFIXES):
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


def _paired_diffs(comm, nocomm, lambdas=None):
    """Seed-paired V vector: (seeds, V_s = mean_key(nocomm_s) - mean_key(comm_s), nocomm base)."""
    seeds, lams = _common(comm, nocomm, lambdas)
    if not lams:
        raise ValueError("no shared keys between the arms")
    d, base = [], []
    for s in seeds:
        no = float(np.mean([nocomm[s][l] for l in lams]))
        co = float(np.mean([comm[s][l] for l in lams]))
        d.append(no - co)
        base.append(no)
    return seeds, np.asarray(d, float), float(np.mean(base))


# ==============================================================================
# Substitution curve: V(training budget)  (Tier-1 attachment; exploratory-registered)
# ==============================================================================
def substitution_curve(budget_dirs, lambdas=None, n_boot=10000):
    """V as a function of TRAINING BUDGET, from the budget-milestone checkpoints train_signal now
    writes (signal_checkpoint_budget{N}.pt) dumped per budget into paired dirs.

    budget_dirs : {budget_episodes(int) -> (comm_dir, nocomm_dir)}   (same seed set per budget)

    THE CLAIM THIS FEEDS: inference capacity and information sharing are SUBSTITUTES -- the
    classical redundancy null (Raghunathan 2001) is the infinite-inference limit. Prediction
    under substitution: V(budget) DECLINES as the no-comm agent's learned inference improves.
    The opposite sign (V rises with budget: exploiting a channel itself must be learned) is a
    complementarity finding -- equally reportable; the curve decides.

    Statistic (exploratory-registered): per-seed OLS slope of V_s over log2(budget) across seeds
    with ALL budgets present + tie-naive per-seed Spearman; bootstrap-t CI over seeds."""
    budgets = sorted(int(b) for b in budget_dirs)
    per_seed, rows = {}, []
    for b in budgets:
        comm, nocomm = load_cost_dir(budget_dirs[b][0]), load_cost_dir(budget_dirs[b][1])
        seeds, d, base = _paired_diffs(comm, nocomm, lambdas)
        for s, v in zip(seeds, d):
            per_seed.setdefault(s, {})[b] = float(v)
        rows.append({"budget": b, "n_seeds": len(seeds), "v_mean": float(d.mean()),
                     "v_ci": bootstrap_ci(d, n_boot=n_boot),
                     "v_pct": float(100.0 * d.mean() / max(1e-9, base))})
    full = {s: bv for s, bv in per_seed.items() if len(bv) == len(budgets)}
    x = np.log2(np.asarray(budgets, float))

    def _spear(y):                                   # tie-naive Spearman (budgets are distinct)
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(y)).astype(float)
        rx -= rx.mean(); ry -= ry.mean()
        den = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
        return float((rx * ry).sum() / den) if den > 0 else 0.0

    ys = [np.array([full[s][b] for b in budgets]) for s in sorted(full)]
    slopes = np.array([np.polyfit(x, y, 1)[0] for y in ys]) if ys else np.array([])
    spears = np.array([_spear(y) for y in ys]) if ys else np.array([])
    return {"rows": rows, "budgets": budgets, "n_trend_seeds": len(full),
            "slope_per_log2_mean": float(slopes.mean()) if slopes.size else float("nan"),
            "slope_ci": bootstrap_ci(slopes, n_boot=n_boot) if slopes.size else (float("nan"),) * 2,
            "spearman_mean": float(spears.mean()) if spears.size else float("nan")}


def print_curve(sc):
    print("=" * 78)
    print(f"SUBSTITUTION CURVE  V(training budget)  |  trend over {sc['n_trend_seeds']} complete seed-series")
    print("=" * 78)
    print(f"    {'budget(eps)':>12}{'V_cost':>10}{'V%':>8}{'95% CI':>22}{'n':>5}")
    for r in sc["rows"]:
        print(f"    {r['budget']:>12}{r['v_mean']:>+10.1f}{r['v_pct']:>+7.1f}%"
              f"   [{r['v_ci'][0]:>7.1f}, {r['v_ci'][1]:>7.1f}]{r['n_seeds']:>5}")
    lo, hi = sc["slope_ci"]
    verdict = ("SUBSTITUTES (V falls with budget)" if hi < 0 else
               "COMPLEMENTS (V rises with budget)" if lo > 0 else "trend CI spans 0")
    print(f"    per-seed slope of V per log2(budget): {sc['slope_per_log2_mean']:+.2f}  "
          f"CI [{lo:.2f}, {hi:.2f}]  mean Spearman {sc['spearman_mean']:+.2f}  -> {verdict}")
    print("    (substitution reading: the Raghunathan null is the infinite-inference limit; the curve")
    print("     locates where learned inference makes sharing redundant. Exploratory-registered.)")
    print("=" * 78)


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
# Message-intervention aggregation (the registered content-attribution gate)
# ==============================================================================
def load_iv_dir(d):
    """DIR/seed{S}_iv.json -> {seed: payload} (written by eval_signal.py --dump-iv)."""
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "seed*_iv.json"))):
        s = _seed_of(p)
        if s is None:
            continue
        with open(p) as f:
            out[s] = json.load(f)
    return out


def interventions_summary(iv, n_boot=10000, ci=0.95):
    """Cross-SEED aggregation of the per-checkpoint message-intervention probe. The per-seed
    dumps carry episode-mean costs under do(m): honest / shuffled / cross / zeroed. Here the SEED
    is the unit (episodes inside a checkpoint are the instrument's Monte Carlo, not replication):
    per seed s, d_x(s) = cost_x(s) - cost_honest(s); report mean d_x, bootstrap CI (auto ->
    studentized bootstrap-t for the seed mean), Wilcoxon; content_share = ratio of MEAN deltas
    d_shuffled/d_zeroed (pooled, robust to near-zero per-seed denominators).
    REGISTERED GATE (prereg content_attribution_rule): a positive V is attributed to message
    CONTENT only if the cross-seed d_shuffled CI excludes 0 from below (shuffling hurts)."""
    seeds = sorted(iv)
    if not seeds:
        raise ValueError("no seed*_iv.json found")
    hon = np.array([float(iv[s]["honest"]) for s in seeds])
    out = {"n_seeds": len(seeds), "seeds": seeds, "honest_mean": float(hon.mean()), "per": {}}
    for nm in ("shuffled", "cross", "zeroed"):
        c = np.array([float(iv[s][nm]["mean"]) for s in seeds])
        d = c - hon                                   # + => intervention HURTS => content mattered
        pr = paired(d, np.zeros_like(d))
        out["per"][nm] = {"mean": float(c.mean()), "delta_mean": float(d.mean()),
                          "delta_ci": bootstrap_ci(d, n_boot=n_boot, ci=ci),
                          "wilcoxon_p": pr["wilcoxon_p"], "n_pos": pr["n_pos"],
                          "n_nonzero": pr["n_nonzero"]}
    dz = out["per"]["zeroed"]["delta_mean"]
    out["content_share"] = (float(out["per"]["shuffled"]["delta_mean"] / dz)
                            if abs(dz) > 1e-9 else float("nan"))
    out["content_gate_pass"] = bool(out["per"]["shuffled"]["delta_ci"][0] > 0.0)
    return out


def print_interventions(sm):
    def _ci1(t): return f"[{t[0]:.1f}, {t[1]:.1f}]"
    print("=" * 78)
    print(f"MESSAGE-INTERVENTION SUMMARY (cross-seed)  |  {sm['n_seeds']} seeds  |  "
          f"honest mean cost {sm['honest_mean']:.1f}")
    print("=" * 78)
    print(f"  {'do(m)':<12}{'mean cost':>11}{'delta vs honest':>17}{'95% CI':>20}{'p(Wilcoxon)':>13}")
    for nm in ("shuffled", "cross", "zeroed"):
        r = sm["per"][nm]
        print(f"  {nm:<12}{r['mean']:>11.1f}{r['delta_mean']:>+17.1f}{_ci1(r['delta_ci']):>20}"
              f"{r['wilcoxon_p']:>13.3g}   ({r['n_pos']}/{r['n_nonzero']} seeds hurt)")
    print(f"  content share (pooled) = delta(shuffled)/delta(zeroed) = {sm['content_share']:+.2f}")
    gate = "PASS: V attributable to CONTENT" if sm["content_gate_pass"] else \
           "FAIL: shuffled CI includes 0 -> V (if any) is presence/scale, NOT content"
    print(f"  registered content-attribution gate -> {gate}")
    print("=" * 78)


def _interventions_cli(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="comm_stats.py interventions")
    ap.add_argument("--dir", required=True, help="dir of seed{S}_iv.json (eval_signal --dump-iv)")
    args = ap.parse_args(argv)
    iv = load_iv_dir(args.dir)
    if not iv:
        print(f"no seed*_iv.json in {args.dir}. Generate with `eval_signal.py --ar1 --dump-iv DIR`.")
        return
    print_interventions(interventions_summary(iv))


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

    # Substitution curve: fabricate per-budget dumps where V SHRINKS with budget (substitutes).
    import tempfile, os as _os, json as _json
    tmp = tempfile.mkdtemp(prefix="curve_")
    pairs = {}
    for b, v_true in [(1000, 400.0), (2000, 250.0), (4000, 120.0), (8000, 30.0)]:
        cd, nd = _os.path.join(tmp, f"c{b}"), _os.path.join(tmp, f"n{b}")
        _os.makedirs(cd); _os.makedirs(nd)
        for s in range(12):
            no = {l: 3000.0 + rng.normal(0, 20) for l in lams}
            co = {l: no[l] - v_true + rng.normal(0, 10) for l in lams}
            _json.dump(co, open(_os.path.join(cd, f"seed{s}.json"), "w"))
            _json.dump(no, open(_os.path.join(nd, f"seed{s}.json"), "w"))
        pairs[b] = (cd, nd)
    sc = substitution_curve(pairs, n_boot=2000)
    assert sc["n_trend_seeds"] == 12 and sc["slope_ci"][1] < 0, sc["slope_ci"]     # falling V detected
    assert abs(sc["rows"][0]["v_mean"] - 400.0) < 30 and abs(sc["rows"][-1]["v_mean"] - 30.0) < 30
    print(f"  D) substitution curve: V 400->30 over budgets, per-log2 slope CI "
          f"[{sc['slope_ci'][0]:.1f},{sc['slope_ci'][1]:.1f}] < 0 (substitutes detected)  PASS")
    print_curve(sc)

    # Interventions: 12 seeds where shuffling hurts (+300) and zeroing hurts (+400) -> gate PASSES
    # with content_share ~0.75; and a null channel (shuffled == honest) -> gate FAILS.
    iv = {s: {"honest": 3000.0 + rng.normal(0, 20),
              "shuffled": {"mean": 3300.0 + rng.normal(0, 25)},
              "cross": {"mean": 3320.0 + rng.normal(0, 25)},
              "zeroed": {"mean": 3400.0 + rng.normal(0, 25)}} for s in range(12)}
    sm = interventions_summary(iv, n_boot=2000)
    assert sm["content_gate_pass"] and 0.55 < sm["content_share"] < 0.95, sm
    iv0 = {s: {"honest": 3000.0 + rng.normal(0, 20),
               "shuffled": {"mean": 3000.0 + rng.normal(0, 20)},
               "cross": {"mean": 3005.0 + rng.normal(0, 20)},
               "zeroed": {"mean": 3200.0 + rng.normal(0, 25)}} for s in range(12)}
    sm0 = interventions_summary(iv0, n_boot=2000)
    assert not sm0["content_gate_pass"], sm0
    print(f"  E) interventions: content channel gate PASS (share={sm['content_share']:+.2f}); "
          f"presence-only channel gate FAIL (share={sm0['content_share']:+.2f})  PASS")
    print_interventions(sm)

    print_report(vs, fd)
    print("\ncomm_stats self-test PASS")


def _curve_cli(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="comm_stats.py curve")
    ap.add_argument("--budget", nargs=3, action="append", required=True,
                    metavar=("EPISODES", "COMM_DIR", "NOCOMM_DIR"),
                    help="repeat per milestone, e.g. --budget 2000 results/h1b2000_comm results/h1b2000_nocomm")
    ap.add_argument("--lambdas", nargs="+", type=float, default=None)
    args = ap.parse_args(argv)
    pairs = {int(b): (c, n) for b, c, n in args.budget}
    print_curve(substitution_curve(pairs, lambdas=args.lambdas))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        _report_cli(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "curve":
        _curve_cli(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "interventions":
        _interventions_cli(sys.argv[2:])
    else:
        _selftest()