#!/usr/bin/env python3
"""confirmatory_v2.py -- the EXACT v2.0 primary tests (review 2.0 fix #8), frozen before hashing.

Implements, in one auditable file:
  P1 (conjunctive crossover, intersection-union): Delta_DP = V_DP(dhat) - V_DP(raw) > 0  AND
     Delta_AR = V_AR.9(raw) - V_AR.9(dhat) > 0. One-sided studentized bootstrap-t per conjunct;
     p_P1 = max(p_A, p_B) (IU: both must reject; no multiplicity inflation inside P1).
  P2 (garbling): Gamma = V_AR.9^raw(clip12) - V_AR.9^raw(no clip) > 0, same test.
  Joint Holm over the two primaries at familywise alpha = .05.
Strictness: every cell is loaded with the EXACT registered seed vector; any missing seed aborts
(fail-closed; no silent intersection). V is always computed CRN seed-paired; contrasts of V share
the per-seed nocomm baseline, so the seed-level covariance is carried automatically.

Usage:
  python scripts/confirmatory_v2.py --root sweep_out --seeds "30 31 ... 44"
  python scripts/confirmatory_v2.py --selftest          # six registered fixture scenarios
"""
import argparse
import json
import os
import sys

import numpy as np

B = 10000
ALPHA_FAMILY = 0.05
RNG = np.random.default_rng(20260719)


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


def boot_t_p_onesided(x):
    """One-sided p for H1: mean(x) > 0, studentized bootstrap-t (percentile-of-t under recentring)."""
    n = len(x)
    m, sd = x.mean(), x.std(ddof=1)
    if sd == 0:
        return 0.0 if m > 0 else 1.0
    t0 = m / (sd / np.sqrt(n))
    xc = x - m                                                     # impose H0
    idx = RNG.integers(0, n, size=(B, n))
    bs = xc[idx]
    bt = bs.mean(1) / (bs.std(1, ddof=1) / np.sqrt(n) + 1e-12)
    return float((np.sum(bt >= t0) + 1) / (B + 1))


def primaries(root, seeds):
    dpn = load_cell(root, "v13/dp_nocomm", seeds)
    v_dp_dhat = dpn - load_cell(root, "v13/dp_dhat", seeds)        # V = C_nocomm - C_comm, seed-paired
    v_dp_raw = dpn - load_cell(root, "v13/dp_raw", seeds)
    arn = load_cell(root, "v13/ar1r9_nocomm", seeds)
    v_ar_raw = arn - load_cell(root, "v13/ar1r9_raw", seeds)
    v_ar_dhat = arn - load_cell(root, "v13/ar1r9_dhat", seeds)
    band = 0.02 * float(arn.mean())                                # +/-2% equivalence band (Cachon-Fisher)
    d_dp = v_dp_dhat - v_dp_raw                                    # conjunct A (shared dpn cancels? no:
    #                                                                dpn cancels algebraically -- the
    #                                                                contrast is C_raw - C_dhat, still
    #                                                                seed-paired; kept in V-form for audit)
    d_ar = v_ar_raw - v_ar_dhat                                    # conjunct B
    # Review 3.0 problem 1: the headline is about VALUE, not ranking. P1's intersection-union now
    # requires BOTH ranking conjuncts AND both absolute anchors (the winning signal carries positive
    # value in its regime): p_P1 = max over the four one-sided components.
    p_a, p_b = boot_t_p_onesided(d_dp), boot_t_p_onesided(d_ar)
    p_abs_dp = boot_t_p_onesided(v_dp_dhat)                        # V_DP(dhat) > 0
    p_abs_ar = boot_t_p_onesided(v_ar_raw)                         # V_AR(raw)  > 0
    p_p1 = max(p_a, p_b, p_abs_dp, p_abs_ar)                       # intersection-union (4 components)
    # Registered companion C-NULL: dhat is REDUNDANT at rho=.9 -- TOST |V_AR(dhat)| < band.
    p_tost = max(boot_t_p_onesided(v_ar_dhat + band), boot_t_p_onesided(band - v_ar_dhat))

    g_noc = load_cell(root, "v13/clip12_nocomm", seeds)
    v_clip = g_noc - load_cell(root, "v13/clip12_raw", seeds)      # V_raw under c=12
    gamma = v_clip - v_ar_raw                                      # vs V_raw at c=inf (same seeds)
    p_p2 = boot_t_p_onesided(gamma)

    # joint Holm over the two primaries
    order = sorted([("P1", p_p1), ("P2", p_p2)], key=lambda kv: kv[1])
    holm, reject = {}, True
    for rank, (name, p) in enumerate(order):
        lvl = ALPHA_FAMILY / (2 - rank)
        reject = reject and (p <= lvl)
        holm[name] = (p, lvl, reject)
    # --- Code change 3: frozen v13 secondaries (H-REP / H-TIME / H-SOURCE / clip20 direction) ---
    sec = {}
    try:
        v_eps = arn - load_cell(root, "v13/ar1r9_eps", seeds)
        v_cm = arn - load_cell(root, "v13/ar1r9_condmean", seeds)
        sec["H-REP raw~eps TOST"] = max(boot_t_p_onesided((v_ar_raw - v_eps) + band),
                                        boot_t_p_onesided(band - (v_ar_raw - v_eps)))
        sec["H-REP raw~linpred TOST"] = max(boot_t_p_onesided((v_ar_raw - v_cm) + band),
                                            boot_t_p_onesided(band - (v_ar_raw - v_cm)))
        l1 = arn - load_cell(root, "v13/lag1", seeds); l2 = arn - load_cell(root, "v13/lag2", seeds)
        sec["H-TIME raw>lag1"] = boot_t_p_onesided(v_ar_raw - l1)
        sec["H-TIME lag1>lag2"] = boot_t_p_onesided(l1 - l2)
        up = arn - load_cell(root, "v13/top_up_raw", seeds); dn = arn - load_cell(root, "v13/top_down_raw", seeds)
        sec["H-SOURCE up>down"] = boot_t_p_onesided(up - dn)
        g20n = load_cell(root, "v13/clip20_nocomm", seeds)
        sec["P2-dose G12>=G20"] = boot_t_p_onesided(gamma - ((g20n - load_cell(root, "v13/clip20_raw", seeds)) - v_ar_raw))
    except SystemExit:
        sec["_note"] = "secondary cells incomplete (verify_manifest will fail-closed on the campaign)"
    return dict(d_dp=d_dp, d_ar=d_ar, gamma=gamma, p_a=p_a, p_b=p_b, p_abs_dp=p_abs_dp,
                p_abs_ar=p_abs_ar, p_tost=p_tost, band=band, p_p1=p_p1, p_p2=p_p2, holm=holm, sec=sec)


def report(r):
    print("== SIGNAL v2.0 CONFIRMATORY PRIMARIES (frozen; review 2.0 fix #8) ==")
    print(f"P1 conjunct A  Delta_DP(dhat-raw)  mean={r['d_dp'].mean():+8.1f}  p={r['p_a']:.4f}")
    print(f"P1 conjunct B  Delta_AR(raw-dhat)  mean={r['d_ar'].mean():+8.1f}  p={r['p_b']:.4f}")
    print(f"P1 abs anchors  V_DP(dhat)>0 p={r['p_abs_dp']:.4f}   V_AR(raw)>0 p={r['p_abs_ar']:.4f}")
    print(f"P1 (IU over 4 components, max-p)              p={r['p_p1']:.4f}")
    print(f"C-NULL dhat redundant @rho.9 (TOST +/-{r['band']:.0f}) p={r['p_tost']:.4f} -> "
          f"{'EQUIVALENT' if r['p_tost'] <= 0.05 else 'not shown'}")
    print(f"P2 Gamma = V_raw(clip12)-V_raw(inf) mean={r['gamma'].mean():+8.1f}  p={r['p_p2']:.4f}")
    for k, v in r.get("sec", {}).items():
        print(f"  [secondary] {k}: {v if isinstance(v, str) else format(v, '.4f')}")
    for name, (p, lvl, rej) in r["holm"].items():
        print(f"  Holm {name}: p={p:.4f} vs {lvl:.4f} -> {'REJECT H0 (claim supported)' if rej else 'not rejected'}")


def _mkcell(root, cell, seeds, vals):
    os.makedirs(os.path.join(root, cell), exist_ok=True)
    for s, v in zip(seeds, vals):
        json.dump({"k": float(v)}, open(os.path.join(root, cell, f"seed{s}.json"), "w"))


def selftest():
    import tempfile
    seeds = list(range(30, 45))
    n = len(seeds)
    scen = [
        ("both pass",        +150, +200, +180, ("P1", True),  ("P2", True)),
        ("only DP conjunct", +150, -80,  +180, ("P1", False), ("P2", True)),
        ("only AR conjunct", -100, +200, +180, ("P1", False), ("P2", True)),
        ("P2 zero",          +150, +200, 0,    ("P1", True),  ("P2", False)),
        ("P2 negative",      +150, +200, -160, ("P1", True),  ("P2", False)),
        ("shared-baseline covariance", +150, +200, +180, ("P1", True), ("P2", True)),
        ("ranking-only (abs anchor fails)", +150, +200, +180, ("P1", False), ("P2", True), "absfail"),
    ]
    ok_all = True
    for row in scen:
        name, eff_dp, eff_ar, eff_g, expA, expB = row[:6]
        absfail = len(row) > 6
        with tempfile.TemporaryDirectory() as root:
            noise = lambda sc=60: RNG.normal(0, sc, n)
            base_shift = RNG.normal(0, 120, n) if "covariance" in name else 0.0  # seed-quality common shock
            dpn = 4000 + noise() + base_shift
            _mkcell(root, "v13/dp_nocomm", seeds, dpn)
            _mkcell(root, "v13/dp_raw", seeds, dpn + (260 if absfail else -60) - noise(40))
            _mkcell(root, "v13/dp_dhat", seeds, dpn + (260 if absfail else -60) - eff_dp - noise(40))
            arn = 4100 + noise() + base_shift
            _mkcell(root, "v13/ar1r9_nocomm", seeds, arn)
            _mkcell(root, "v13/ar1r9_dhat", seeds, arn - 5 - noise(25))
            v_raw = 280
            _mkcell(root, "v13/ar1r9_raw", seeds, arn - 5 - eff_ar - noise(40) if eff_ar > 0 else arn - 5 + (-eff_ar) - noise(40))
            _mkcell(root, "v13/clip12_nocomm", seeds, arn + 30 + noise(50))
            gcell = (arn + 30) - (max(eff_ar, 0) + eff_g) - noise(50)
            _mkcell(root, "v13/clip12_raw", seeds, gcell)
            r = primaries(root, seeds)
            gotA = r["holm"]["P1"][2]; gotB = r["holm"]["P2"][2]
            verdict = (gotA == expA[1]) and (gotB == expB[1])
            ok_all &= verdict
            print(f"  [{ 'PASS' if verdict else 'FAIL'}] {name}: P1={gotA} (exp {expA[1]})  P2={gotB} (exp {expB[1]})")
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