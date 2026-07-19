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

    # joint Holm over the two primaries (statsmodels multipletests, via compare_many)
    cm = compare_many({"P1": p_p1, "P2": p_p2}, method="holm")
    holm = {k: {"raw": cm[k]["raw"], "adj": cm[k]["adjusted"], "reject": cm[k]["reject"]}
            for k in ("P1", "P2")}

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

    return dict(d_dp=d_dp, d_ar=d_ar, gamma=gamma, v_dp_dhat=v_dp_dhat, v_ar_raw=v_ar_raw,
                v_ar_dhat=v_ar_dhat, p_a=p_a, p_b=p_b, p_abs_dp=p_abs_dp, p_abs_ar=p_abs_ar,
                p_tost=p_tost, band=band, p_p1=p_p1, p_p2=p_p2, w_p1=w_p1, w_p2=w_p2,
                ci_dp=_ci(v_dp_dhat), ci_ar=_ci(v_ar_raw), ci_gamma=_ci(gamma), holm=holm, sec=sec)


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
    for k, v in r.get("sec", {}).items():
        print(f"  [secondary] {k}: {v if isinstance(v, str) else format(v, '.4f')}")
    for name in ("P1", "P2"):
        h = r["holm"][name]
        print(f"  Holm {name}: raw={h['raw']:.4f}  adj={h['adj']:.4f} -> "
              f"{'REJECT H0 (claim supported)' if h['reject'] else 'not rejected'}")


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
            verdict = (gotA == expA) and (gotB == expB)
            ok_all &= verdict
            print(f"  [{'PASS' if verdict else 'FAIL'}] {name}: P1={gotA} (exp {expA})  P2={gotB} (exp {expB})")
    print("SELFTEST:", "ALL SCENARIOS PASS" if ok_all else "FAILURES ABOVE")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="sweep_out")
    ap.add_argument("--seeds", default=" ".join(str(s) for s in range(30, 55)))  # n*=25 (fallback)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        report(primaries(a.root, [int(s) for s in a.seeds.split()]))