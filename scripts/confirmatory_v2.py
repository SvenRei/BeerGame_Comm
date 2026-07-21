#!/usr/bin/env python3
"""confirmatory_v2.py -- the EXACT v2.0 primary tests, frozen before hashing.

ALL INFERENCE DELEGATES TO COMMUNITY-VALIDATED LIBRARIES (v2.0 stats hardening):
  * one-sided decision  -> scipy.stats.ttest_1samp(alternative=...)   (paired-difference t-test)
  * equivalence (TOST)  -> scripts.c1_stats.tost  == Schuirmann via scipy.stats.ttest_1samp
  * effect-size CIs      -> scripts.c1_stats.bootstrap_ci == scipy.stats.bootstrap (BCa)
  * multiplicity (Holm) -> scripts.c1_stats.compare_many == statsmodels multipletests
  * nonparametric check -> scipy.stats.wilcoxon
No hand-rolled estimators remain. The paired t-test is exact under normality and CLT-robust at
n>=25; the power analysis (reports/power_v13.txt) already computes the one-sided t-test rejection
rate, so the confirmatory decision and the registered power are the SAME test.

Two REGISTERED interpretation elements accompany every primary (review hardening):
  * V-distribution: the seed-level distribution of every V (P(V>0), deciles, range). V is a
    difference of ACHIEVED costs; a positive mean with wide dispersion is reported as such, so the
    supported claim is "sharing lowers EXPECTED cost with substantial run-to-run dispersion", not
    "information always helps".
  * Optimality-gap diagnostic: VOI-via-RL identifies information value only where both policies are
    near-optimal. Each key arm's gap to the best available benchmark is reported, and each primary
    contrast is classified INFORMATION-VALUE (both arms within tau of the near-optimal reference) or
    LEARNING-GAP lower-bound. This does NOT alter the t-test decision; it bounds its interpretation.

Structure (unchanged, library-backed):
  P1 (crossover, intersection-union over FOUR one-sided components):
     Delta_DP = V_DP(dhat)-V_DP(raw) > 0  AND  Delta_AR = V_AR.9(raw)-V_AR.9(dhat) > 0  AND
     V_DP(dhat) > 0  AND  V_AR.9(raw) > 0.   p_P1 = max(component p)   (IU: all must reject).
  P2 (garbling): Gamma = V_AR.9^raw(clip12) - V_AR.9^raw(inf) > 0.
  Joint Holm over {P1, P2} at familywise alpha = .05 (statsmodels).
  Companion C-NULL: TOST |V_AR.9(dhat)| within +/-2% band.
Every cell is loaded with the EXACT registered seed vector; any missing seed aborts (fail-closed).
V is CRN seed-paired; contrasts of V share the per-seed nocomm baseline (covariance carried).

Usage:
  python scripts/confirmatory_v2.py --root sweep_out --seeds "30 31 ... 54"
  python scripts/confirmatory_v2.py --selftest          # seven registered fixture scenarios
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import ttest_1samp, wilcoxon

# --- shared, library-based inference primitives (works packaged or flat) ---
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
for _p in (_root, _here):
    if _p not in sys.path:
        sys.path.insert(0, _p)
try:
    from scripts.c1_stats import bootstrap_ci, tost, compare_many
except Exception:  # noqa: BLE001
    from c1_stats import bootstrap_ci, tost, compare_many

ALPHA_FAMILY = 0.05
BOOT = 10000
CI = 0.95


def load_cell(root, cell, seeds):
    out = {}
    for s in seeds:
        p = os.path.join(root, cell, f"seed{s}.json")
        if not os.path.exists(p):
            sys.exit(f"FAIL-CLOSED: {cell} missing seed{s}.json (run scripts/verify_manifest.py)")
        with open(p) as f:
            d = json.load(f)
        out[s] = float(np.mean(list(map(float, d.values()))))     # mean over the cell's keys
    return np.array([out[s] for s in seeds])                       # aligned to the registered vector


def p_greater(x):
    """One-sided p for H1: mean(x) > 0 via the paired-difference t-test (scipy.stats.ttest_1samp,
    alternative='greater'). Deterministic; exact under normality; CLT-robust at n>=25."""
    x = np.asarray(x, float)
    if x.size < 2:
        return 1.0
    if x.std(ddof=1) == 0.0:
        return 0.0 if x.mean() > 0 else 1.0
    return float(ttest_1samp(x, 0.0, alternative="greater").pvalue)


def p_wilcoxon_greater(x):
    """Registered nonparametric robustness for a '>0' claim: Wilcoxon signed-rank (scipy)."""
    x = np.asarray(x, float)
    nz = x[x != 0.0]
    if nz.size < 1:
        return float("nan")
    try:
        return float(wilcoxon(nz, alternative="greater").pvalue)
    except Exception:  # noqa: BLE001
        return float("nan")


def _ci(x):
    return bootstrap_ci(np.asarray(x, float), n_boot=BOOT, ci=CI)     # scipy BCa


def _vdist(v):
    """Seed-level distribution of a V vector: P(V>0), deciles, and range (all descriptive; the
    percentiles are numpy's own routine). Surfaces run-to-run dispersion the mean+CI hides."""
    v = np.asarray(v, float)
    return {"n": int(v.size), "mean": float(v.mean()), "p_pos": float(np.mean(v > 0.0)),
            "p10": float(np.percentile(v, 10)), "p50": float(np.percentile(v, 50)),
            "p90": float(np.percentile(v, 90)), "min": float(v.min()), "max": float(v.max())}


def optimality_gap(root, seeds, refs_path=None, taus=(0.10, 0.20, 0.30), ar_refs_path=None):
    """REGISTERED interpretation diagnostic (not a gate; never changes the t-test decision).
    Reports each key arm's relative gap to the best available benchmark and classifies each
    primary contrast as INFORMATION-VALUE (both arms within tau of the near-optimal reference) or
    a LEARNING-GAP lower bound. Poisson uses the per-lambda Oracle rung (a strong benchmark). The
    AR(1) static base-stock is a weak reference in an autocorrelated regime (disclosed): AR gaps
    are reported only where an AR benchmark exists, else flagged unavailable and V_AR is read as a
    lower bound on information value. Degrades gracefully (prints a disclosure) if refs absent."""
    if refs_path is None:
        refs_path = os.path.join("results", "baselines_regime_v2.json")
    if ar_refs_path is None:
        ar_refs_path = os.path.join("results", "baselines_ar_v2.json")
    if not os.path.exists(refs_path):
        return {"_note": f"benchmark refs not found at {refs_path}; interpretation rule NOT "
                         f"evaluated. V is reported as an achieved-cost difference (a lower bound "
                         f"on information value if the learned policies are sub-optimal)."}
    try:
        from scripts.c1_stats import load_rungs, mean_refs
    except Exception:  # noqa: BLE001
        from c1_stats import load_rungs, mean_refs
    rungs = load_rungs(refs_path)
    mr = mean_refs(rungs)                                      # {rung: mean cost over its lambdas}
    # near-optimal Poisson benchmark: Oracle (best rung); tolerate alt names
    oracle = mr.get("Oracle", mr.get("oracle", mr.get("Oracle_poisson")))
    out = {"taus": list(taus), "oracle_poisson": oracle, "arms": {}, "verdicts": {}}

    def armcost(cell):
        try:
            return float(load_cell(root, cell, seeds).mean())
        except SystemExit:
            return None

    def gap(c):
        return None if (c is None or not oracle) else (c - oracle) / oracle

    for k, cell in {"dp_dhat": "v13/dp_dhat", "dp_raw": "v13/dp_raw"}.items():
        c = armcost(cell)
        out["arms"][k] = {"cost": c, "gap_to_oracle": gap(c)}
    def _tau_verdicts(label, g1name, g1, g2name, g2):
        if g1 is None or g2 is None:
            return
        out["verdicts"][label] = {}
        for t in taus:
            both = (g1 <= t) and (g2 <= t)
            out["verdicts"][label][f"tau={t:.0%}"] = (
                "INFORMATION-VALUE" if both else
                f"LEARNING-GAP lower bound ({g1name} {g1:+.1%}, {g2name} {g2:+.1%})")

    gd, gr = out["arms"]["dp_dhat"]["gap_to_oracle"], out["arms"]["dp_raw"]["gap_to_oracle"]
    _tau_verdicts("V_DP(dhat vs raw) [P1 DP conjunct] vs Oracle", "dp_dhat", gd, "dp_raw", gr)

    # AR side (symmetric rule): reference = AR_BestBS rung from scripts/baselines.py ar
    if os.path.exists(ar_refs_path):
        with open(ar_refs_path) as f:
            ar = json.load(f).get("rungs", {})
        best = ar.get("AR_BestBS", {}).get("0.9")
        out["ar_bestbs_rho09"] = best
        if best:
            def argap(cell):
                c = armcost(cell)
                return c, (None if c is None else (c - best) / best)
            c_raw, g_raw = argap("v13/ar1r9_raw")
            c_dhat, g_dhat = argap("v13/ar1r9_dhat")
            out["arms"]["ar1r9_raw"] = {"cost": c_raw, "gap_to_bestbs": g_raw}
            out["arms"]["ar1r9_dhat"] = {"cost": c_dhat, "gap_to_bestbs": g_dhat}
            _tau_verdicts("V_AR(raw vs dhat) [P1 AR conjunct] vs AR_BestBS",
                          "ar_raw", g_raw, "ar_dhat", g_dhat)
            out["ar_disclosure"] = ("AR reference = AR_BestBS (privileged conditional base-stock, "
                                    "Kahn/Lee lineage; a heuristic near-optimum, not the unknown "
                                    "exact optimum -- gaps may be slightly overstated).")
    if "ar_disclosure" not in out:
        out["ar_disclosure"] = ("AR(1) reference refs not found (run scripts/baselines.py ar): "
                                "V_AR is reported as an achieved-cost difference and interpreted "
                                "as a LOWER BOUND on information value.")
    return out


def primaries(root, seeds):
    dpn = load_cell(root, "v13/dp_nocomm", seeds)
    v_dp_dhat = dpn - load_cell(root, "v13/dp_dhat", seeds)        # V = C_nocomm - C_comm, seed-paired
    v_dp_raw = dpn - load_cell(root, "v13/dp_raw", seeds)
    arn = load_cell(root, "v13/ar1r9_nocomm", seeds)
    v_ar_raw = arn - load_cell(root, "v13/ar1r9_raw", seeds)
    v_ar_dhat = arn - load_cell(root, "v13/ar1r9_dhat", seeds)
    band = 0.02 * float(arn.mean())                                # +/-2% equivalence band

    d_dp = v_dp_dhat - v_dp_raw                                    # conjunct A == C_raw - C_dhat (dpn cancels)
    d_ar = v_ar_raw - v_ar_dhat                                    # conjunct B
    # P1 intersection-union over FOUR one-sided paired t-tests (VALUE, not mere ranking):
    p_a = p_greater(d_dp)                                          # Delta_DP > 0
    p_b = p_greater(d_ar)                                          # Delta_AR > 0
    p_abs_dp = p_greater(v_dp_dhat)                               # V_DP(dhat) > 0
    p_abs_ar = p_greater(v_ar_raw)                               # V_AR(raw)  > 0
    p_p1 = max(p_a, p_b, p_abs_dp, p_abs_ar)
    # nonparametric robustness (reported, not decisive)
    w_p1 = max(p_wilcoxon_greater(d_dp), p_wilcoxon_greater(d_ar),
               p_wilcoxon_greater(v_dp_dhat), p_wilcoxon_greater(v_ar_raw))
    # Companion C-NULL: dhat redundant at rho=.9 -- TOST |V_AR(dhat)| within band (scipy Schuirmann)
    p_tost = tost(v_ar_dhat, -band, band, alpha=ALPHA_FAMILY)["p_tost"]

    g_noc = load_cell(root, "v13/clip12_nocomm", seeds)
    v_clip = g_noc - load_cell(root, "v13/clip12_raw", seeds)      # V_raw under c=12
    gamma = v_clip - v_ar_raw                                      # vs V_raw at c=inf (same seeds)
    p_p2 = p_greater(gamma)
    w_p2 = p_wilcoxon_greater(gamma)

    # A4 (registered sensitivity): two-sided p's alongside every one-sided decision
    def _p2s(x):
        x = np.asarray(x, float)
        return 1.0 if x.size < 2 or x.std(ddof=1) == 0.0 else float(ttest_1samp(x, 0.0).pvalue)
    two_sided = {"Delta_DP": _p2s(d_dp), "Delta_AR": _p2s(d_ar), "V_DP(dhat)": _p2s(v_dp_dhat),
                 "V_AR(raw)": _p2s(v_ar_raw), "Gamma": _p2s(gamma)}

    # A2: C-NULL folded into the primary family -> joint Holm over THREE members
    # (statsmodels multipletests via compare_many). C-NULL is load-bearing for the headline
    # ("dhat is redundant at rho=.9") and now carries the same FWER control as P1/P2.
    cm = compare_many({"P1": p_p1, "P2": p_p2, "C-NULL": p_tost}, method="holm")
    holm = {k: {"raw": cm[k]["raw"], "adj": cm[k]["adjusted"], "reject": cm[k]["reject"]}
            for k in ("P1", "P2", "C-NULL")}

    # frozen v13 secondaries (all library-based: t-tests + Schuirmann TOST)
    sec = {}
    try:
        v_eps = arn - load_cell(root, "v13/ar1r9_eps", seeds)
        v_cm = arn - load_cell(root, "v13/ar1r9_condmean", seeds)
        sec["H-REP raw~eps TOST"] = tost(v_ar_raw - v_eps, -band, band, alpha=ALPHA_FAMILY)["p_tost"]
        sec["H-REP raw~linpred TOST"] = tost(v_ar_raw - v_cm, -band, band, alpha=ALPHA_FAMILY)["p_tost"]
        l1 = arn - load_cell(root, "v13/lag1", seeds)
        l2 = arn - load_cell(root, "v13/lag2", seeds)
        sec["H-TIME raw>lag1"] = p_greater(v_ar_raw - l1)
        sec["H-TIME lag1>lag2"] = p_greater(l1 - l2)
        up = arn - load_cell(root, "v13/top_up_raw", seeds)
        dn = arn - load_cell(root, "v13/top_down_raw", seeds)
        sec["H-SOURCE up>down"] = p_greater(up - dn)
        g20n = load_cell(root, "v13/clip20_nocomm", seeds)
        gamma20 = (g20n - load_cell(root, "v13/clip20_raw", seeds)) - v_ar_raw
        sec["P2-dose G12>=G20"] = p_greater(gamma - gamma20)
    except SystemExit:
        sec["_note"] = "secondary cells incomplete (verify_manifest will fail-closed on the campaign)"
    # A1: FWER control WITHIN the six-member secondary family (statsmodels Holm). Raw and
    # adjusted p's are both reported; rejections are read off the adjusted values.
    numeric = {k: v for k, v in sec.items() if isinstance(v, float) and v == v}
    sec_holm = compare_many(numeric, method="holm") if len(numeric) >= 2 else {}

    return dict(d_dp=d_dp, d_ar=d_ar, gamma=gamma, v_dp_dhat=v_dp_dhat, v_ar_raw=v_ar_raw,
                v_ar_dhat=v_ar_dhat, p_a=p_a, p_b=p_b, p_abs_dp=p_abs_dp, p_abs_ar=p_abs_ar,
                p_tost=p_tost, band=band, p_p1=p_p1, p_p2=p_p2, w_p1=w_p1, w_p2=w_p2,
                ci_dp=_ci(v_dp_dhat), ci_ar=_ci(v_ar_raw), ci_gamma=_ci(gamma), holm=holm, sec=sec,
                sec_holm=sec_holm, two_sided=two_sided,
                vdist={"V_DP(dhat)": _vdist(v_dp_dhat), "V_DP(raw)": _vdist(v_dp_raw),
                       "V_AR(raw)": _vdist(v_ar_raw), "V_AR(dhat)": _vdist(v_ar_dhat),
                       "Gamma": _vdist(gamma)})


def report(r):
    print("== SIGNAL v2.0 CONFIRMATORY PRIMARIES (all inference via scipy/statsmodels) ==")
    print(f"P1 conjunct A  Delta_DP(dhat-raw)  mean={r['d_dp'].mean():+8.1f}  t-p={r['p_a']:.4f}")
    print(f"P1 conjunct B  Delta_AR(raw-dhat)  mean={r['d_ar'].mean():+8.1f}  t-p={r['p_b']:.4f}")
    print(f"P1 anchor      V_DP(dhat)>0  mean={r['v_dp_dhat'].mean():+8.1f}  t-p={r['p_abs_dp']:.4f}  "
          f"BCa95=[{r['ci_dp'][0]:+.1f},{r['ci_dp'][1]:+.1f}]")
    print(f"P1 anchor      V_AR(raw)>0   mean={r['v_ar_raw'].mean():+8.1f}  t-p={r['p_abs_ar']:.4f}  "
          f"BCa95=[{r['ci_ar'][0]:+.1f},{r['ci_ar'][1]:+.1f}]")
    print(f"P1 (IU over 4 components, max-p)   t-p={r['p_p1']:.4f}   [Wilcoxon robustness max-p={r['w_p1']:.4f}]")
    print(f"C-NULL dhat redundant @rho.9 (TOST +/-{r['band']:.0f})  p={r['p_tost']:.4f} -> "
          f"{'EQUIVALENT' if r['p_tost'] <= ALPHA_FAMILY else 'not shown'}")
    print(f"P2 Gamma = V_raw(clip12)-V_raw(inf)  mean={r['gamma'].mean():+8.1f}  t-p={r['p_p2']:.4f}  "
          f"BCa95=[{r['ci_gamma'][0]:+.1f},{r['ci_gamma'][1]:+.1f}]  [Wilcoxon p={r['w_p2']:.4f}]")
    sh = r.get("sec_holm", {})
    for k, v in r.get("sec", {}).items():
        if isinstance(v, str):
            print(f"  [secondary] {k}: {v}")
        else:
            h = sh.get(k)
            extra = (f"  Holm-adj={h['adjusted']:.4f} -> "
                     f"{'REJECT' if h['reject'] else 'n.r.'}") if h else ""
            print(f"  [secondary] {k}: raw={v:.4f}{extra}")
    ts = r.get("two_sided", {})
    if ts:
        print("  [sensitivity] two-sided p: " + "  ".join(f"{k}={v:.4f}" for k, v in ts.items()))
    for name in ("P1", "P2", "C-NULL"):
        h = r["holm"][name]
        print(f"  Holm {name}: raw={h['raw']:.4f}  adj={h['adj']:.4f} -> "
              f"{'REJECT H0 (claim supported)' if h['reject'] else 'not rejected'}")
    print("\n-- V-distribution (seed-level; V is an ACHIEVED-cost difference) --")
    print(f"  {'V':<12}{'mean':>9}{'P(V>0)':>9}{'p10':>9}{'p50':>9}{'p90':>9}{'min':>9}{'max':>9}")
    for k, d in r.get("vdist", {}).items():
        print(f"  {k:<12}{d['mean']:>+9.1f}{d['p_pos']:>9.2f}{d['p10']:>+9.1f}{d['p50']:>+9.1f}"
              f"{d['p90']:>+9.1f}{d['min']:>+9.1f}{d['max']:>+9.1f}")
    print("  (a positive mean with low P(V>0) => the EXPECTED-cost reduction coexists with runs "
          "where sharing does not help; the confirmatory claim is about the expectation.)")


def report_optgap(g):
    print("\n-- Optimality-gap diagnostic (interpretation bound; NOT a test) --")
    if "_note" in g:
        print(f"  {g['_note']}")
        return
    orc = g.get("oracle_poisson")
    taus = g.get("taus", [])
    print((f"  Poisson Oracle = {orc:.1f}" if orc else "  Poisson Oracle unavailable")
          + (f"   AR_BestBS(rho=.9) = {g['ar_bestbs_rho09']:.1f}" if g.get("ar_bestbs_rho09") else "")
          + f"   (tau grid: {', '.join(f'{t:.0%}' for t in taus)})")
    for k, d in g.get("arms", {}).items():
        cost_s = f"{d['cost']:.1f}" if d.get("cost") is not None else "NA"
        gp = d.get("gap_to_oracle", d.get("gap_to_bestbs"))
        gap_s = f"{gp:+.1%}" if gp is not None else "NA"
        print(f"    {k:<12} cost={cost_s:>9}   gap={gap_s}")
    for c, per_tau in g.get("verdicts", {}).items():
        print(f"    {c}:")
        for t, v in per_tau.items():
            print(f"        {t}: {v}")
    if g.get("ar_disclosure"):
        print(f"    {g['ar_disclosure']}")


def _mkcell(root, cell, seeds, vals):
    os.makedirs(os.path.join(root, cell), exist_ok=True)
    for s, v in zip(seeds, vals):
        json.dump({"k": float(v)}, open(os.path.join(root, cell, f"seed{s}.json"), "w"))


def selftest():
    import tempfile
    rng = np.random.default_rng(20260719)
    seeds = list(range(30, 45))
    n = len(seeds)
    scen = [
        ("both pass",        +150, +200, +180, True,  True),
        ("only DP conjunct", +150, -80,  +180, False, True),
        ("only AR conjunct", -100, +200, +180, False, True),
        ("P2 zero",          +150, +200, 0,    True,  False),
        ("P2 negative",      +150, +200, -160, True,  False),
        ("shared-baseline covariance", +150, +200, +180, True, True),
        ("ranking-only (abs anchor fails)", +150, +200, +180, False, True, "absfail"),
    ]
    ok_all = True
    for row in scen:
        name, eff_dp, eff_ar, eff_g, expA, expB = row[:6]
        absfail = len(row) > 6
        with tempfile.TemporaryDirectory() as root:
            noise = lambda sc=60: rng.normal(0, sc, n)
            base_shift = rng.normal(0, 120, n) if "covariance" in name else 0.0
            dpn = 4000 + noise() + base_shift
            _mkcell(root, "v13/dp_nocomm", seeds, dpn)
            _mkcell(root, "v13/dp_raw", seeds, dpn + (260 if absfail else -60) - noise(40))
            _mkcell(root, "v13/dp_dhat", seeds, dpn + (260 if absfail else -60) - eff_dp - noise(40))
            arn = 4100 + noise() + base_shift
            _mkcell(root, "v13/ar1r9_nocomm", seeds, arn)
            _mkcell(root, "v13/ar1r9_dhat", seeds, arn - 5 - noise(25))
            _mkcell(root, "v13/ar1r9_raw", seeds,
                    arn - 5 - eff_ar - noise(40) if eff_ar > 0 else arn - 5 + (-eff_ar) - noise(40))
            _mkcell(root, "v13/clip12_nocomm", seeds, arn + 30 + noise(50))
            gcell = (arn + 30) - (max(eff_ar, 0) + eff_g) - noise(50)
            _mkcell(root, "v13/clip12_raw", seeds, gcell)
            r = primaries(root, seeds)
            gotA, gotB = r["holm"]["P1"]["reject"], r["holm"]["P2"]["reject"]
            gotC = r["holm"]["C-NULL"]["reject"]                 # fixtures keep dhat null -> True
            verdict = (gotA == expA) and (gotB == expB) and gotC
            ok_all &= verdict
            print(f"  [{'PASS' if verdict else 'FAIL'}] {name}: P1={gotA} (exp {expA})  "
                  f"P2={gotB} (exp {expB})  C-NULL={gotC} (exp True)")
    print("SELFTEST:", "ALL SCENARIOS PASS" if ok_all else "FAILURES ABOVE")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="sweep_out")
    ap.add_argument("--seeds", default=" ".join(str(s) for s in range(25, 50)))  # n*=25; amendment v2.1
    ap.add_argument("--refs", default=None,
                    help="benchmark refs JSON (default results/baselines_regime_v2.json) for the "
                         "optimality-gap diagnostic")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        seeds = [int(s) for s in a.seeds.split()]
        report(primaries(a.root, seeds))
        report_optgap(optimality_gap(a.root, seeds, refs_path=a.refs))