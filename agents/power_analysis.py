#!/usr/bin/env python3
"""power_analysis.py -- pre-registration power analysis for SIGNAL v1.3 (determines n ONLY).

WHAT THIS DECIDES (and what it must not)
  Output: the smallest seed count n in {15, 20, 25} meeting the registered power targets for the two
  primaries and the equivalence family. It does NOT set margins: the +/-2% equivalence band is a
  SUBSTANTIVE choice (Cachon-Fisher-scale economic materiality), justified in the registration text
  (editor ruling, 2026-07-18).

METHOD (disclosed approximations)
  Seed-level paired differences V_s = C_nocomm(s) - C_comm(s) from Study-1 dumps give the observed
  (mean, sd). Power is estimated by parametric simulation: draw n seed-differences ~ Normal(effect,
  sd), apply the test, repeat n_sim times. Approximations, stated: (i) a one-sided t-test stands in
  for the registered bootstrap-t CI (negligible at these sd/n; both are location tests on the same
  statistic); (ii) the P1 conjunction multiplies the two one-sided powers -- exact under
  independence, CONSERVATIVE under the positive seed-quality correlation CRN induces (P(A and B) >=
  P(A)P(B) for positively dependent rejections). (iii) Unobserved cells (clip contrast Gamma; the
  eps/condmean equivalence contrasts) use sd proxies from the closest observed contrast, swept over
  {1.0, 1.5, 2.0}x -- the editor's "conservative covariance assumptions + sensitivity curves".

USAGE (run from the Study-1 repo root; dirs are the v1.2 sweep_out layout -- verify with `ls sweep_out`)
  python scripts/power_analysis.py \
    --pair DP_dhat    <dp_comm_dir>      <dp_nocomm_dir>    all \
    --pair AR9_dhat   sweep_out/h2/comm  sweep_out/h2/nocomm 0.9 \
    --pair AR9_raw    sweep_out/fam/ar1r9_raw sweep_out/h2/nocomm 0.9 \
    --tost-proxy AR9_raw_vs_dhat sweep_out/fam/ar1r9_raw sweep_out/h2/comm 0.9 \
    --out reports/power_v13.txt
  Each --pair NAME COMM_DIR NOCOMM_DIR KEYS loads seed{N}.json files ({key: cost}); KEYS is a comma
  list or 'all' (per-seed V = mean over keys). --tost-proxy gives the arm-vs-arm contrast whose sd
  proxies the unobserved equivalence contrasts.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

try:
    from scipy import stats
except ImportError:  # pragma: no cover
    sys.exit("scipy required (already a repo dependency)")

SEED_RE = re.compile(r"seed(\d+)\.json\Z")
N_GRID = (10, 15, 20, 25)
N_SIM = 20000
ALPHA = 0.05            # per-test one-sided level
ALPHA_HOLM = 0.025      # stricter member under Holm across the two primaries
EFFECT_FRACS = (1.00, 0.50, 0.25)
SD_INFLATE = (1.0, 1.5, 2.0)
TARGET_PRIMARY = 0.90
TARGET_TOST = 0.80
RNG = np.random.default_rng(20260718)


def load_dir(d):
    out = {}
    for p in glob.glob(os.path.join(d, "seed*.json")):
        m = SEED_RE.search(os.path.basename(p))
        if not m:
            continue
        with open(p) as f:
            out[int(m.group(1))] = {str(k): float(v) for k, v in json.load(f).items()}
    if not out:
        sys.exit(f"no seed*.json in {d}")
    return out


def paired_v(comm_dir, nocomm_dir, keys):
    c, n = load_dir(comm_dir), load_dir(nocomm_dir)
    seeds = sorted(set(c) & set(n))
    if len(seeds) < 5:
        sys.exit(f"only {len(seeds)} shared seeds between {comm_dir} and {nocomm_dir}")
    vs, base = [], []
    for s in seeds:
        ks = sorted(set(c[s]) & set(n[s])) if keys == "all" else [k for k in keys if k in c[s] and k in n[s]]
        if not ks:
            sys.exit(f"no shared keys for seed {s} ({comm_dir})")
        vs.append(float(np.mean([n[s][k] - c[s][k] for k in ks])))
        base.append(float(np.mean([n[s][k] for k in ks])))
    return np.array(vs), float(np.mean(base)), seeds


def power_onesided(effect, sd, n, alpha):
    """P(one-sided t rejects H0: mu<=0) under mu=effect, via simulation."""
    x = RNG.normal(effect, sd, size=(N_SIM, n))
    t = x.mean(1) / (x.std(1, ddof=1) / np.sqrt(n))
    return float(np.mean(t > stats.t.ppf(1 - alpha, n - 1)))


def power_tost(sd, n, band, alpha=0.05):
    """P(TOST rejects both one-sided tests) under TRUE diff = 0 (the equivalence-holds case)."""
    x = RNG.normal(0.0, sd, size=(N_SIM, n))
    se = x.std(1, ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(1 - alpha, n - 1)
    lo = (x.mean(1) + band) / se > tcrit
    hi = (band - x.mean(1)) / se > tcrit
    return float(np.mean(lo & hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=4, action="append", metavar=("NAME", "COMM", "NOCOMM", "KEYS"),
                    required=True)
    ap.add_argument("--tost-proxy", nargs=4, action="append", default=[],
                    metavar=("NAME", "ARM_A", "ARM_B", "KEYS"))
    ap.add_argument("--band-frac", type=float, default=0.02)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("=" * 96)
    emit(f"SIGNAL v1.3 POWER ANALYSIS (n only; margins substantive) | n_sim={N_SIM} | "
         f"alpha={ALPHA} (Holm member {ALPHA_HOLM})")
    emit("=" * 96)

    obs = {}
    for name, cd, nd, keys in a.pair:
        ks = keys if keys == "all" else [k.strip() for k in keys.split(",")]
        v, base, seeds = paired_v(cd, nd, ks)
        obs[name] = dict(mean=v.mean(), sd=v.std(ddof=1), base=base, n=len(seeds))
        emit(f"\n[{name}] observed: n={len(seeds)} seeds  V mean={v.mean():+.1f}  sd={v.std(ddof=1):.1f}"
             f"  nocomm base={base:.1f}  (band {a.band_frac*100:.0f}% = {a.band_frac*base:.1f})")
        emit(f"  {'n':>4} | " + " | ".join(f"pow@{int(f*100):3d}%eff (a={ALPHA}/{ALPHA_HOLM})" for f in EFFECT_FRACS))
        for n in N_GRID:
            cells = []
            for f in EFFECT_FRACS:
                p1 = power_onesided(f * v.mean(), v.std(ddof=1), n, ALPHA)
                p2 = power_onesided(f * v.mean(), v.std(ddof=1), n, ALPHA_HOLM)
                cells.append(f"{p1:.2f}/{p2:.2f}")
            emit(f"  {n:>4} | " + " | ".join(f"{c:^28}" for c in cells))

    if {"DP_dhat", "AR9_raw"} <= set(obs):
        emit("\n[P1 CONJUNCTION] both simple effects at Holm-alpha (independence => conservative):")
        for n in N_GRID:
            row = []
            for f in EFFECT_FRACS:
                pa = power_onesided(f * obs["DP_dhat"]["mean"] if obs["DP_dhat"]["mean"] > 0 else f * abs(obs["DP_dhat"]["mean"]),
                                    obs["DP_dhat"]["sd"], n, ALPHA_HOLM)
                pb = power_onesided(f * obs["AR9_raw"]["mean"], obs["AR9_raw"]["sd"], n, ALPHA_HOLM)
                row.append(f"{pa*pb:.2f}")
            emit(f"  n={n:>2}: " + "  ".join(f"{int(f*100):3d}%eff->{r}" for f, r in zip(EFFECT_FRACS, row)))
        emit("  NOTE: DP_dhat here proxies V(dp_dhat)-V(dp_raw) with the dp sharing effect; replace with a")
        emit("  direct dhat-vs-raw dp contrast when available (Study 1 has no dp_raw arm -- disclosed).")

    if "AR9_raw" in obs:
        emit(f"\n[P2 GAMMA = V_raw(clip12) - V_raw(inf)] UNOBSERVED cell: sensitivity curves")
        emit(f"  effect grid = fractions of observed AR9_raw V ({obs['AR9_raw']['mean']:+.1f}); sd proxy = AR9_raw sd x inflation")
        emit(f"  {'n':>4} | " + " | ".join(f"sd x{m:.1f}" for m in SD_INFLATE) + "   (cells: pow@100/50/25% eff, Holm-alpha)")
        for n in N_GRID:
            cols = []
            for m in SD_INFLATE:
                ps = [power_onesided(f * obs["AR9_raw"]["mean"], m * obs["AR9_raw"]["sd"], n, ALPHA_HOLM)
                      for f in EFFECT_FRACS]
                cols.append("/".join(f"{p:.2f}" for p in ps))
            emit(f"  {n:>4} | " + " | ".join(f"{c:^14}" for c in cols))

    for name, da, db, keys in a.tost_proxy:
        ks = keys if keys == "all" else [k.strip() for k in keys.split(",")]
        v, base, seeds = paired_v(da, db, ks)          # arm-vs-arm paired contrast
        band_base = obs.get("AR9_raw", {}).get("base", base)
        band = a.band_frac * band_base
        emit(f"\n[H-REP TOST proxy: {name}] contrast sd={v.std(ddof=1):.1f} (n={len(seeds)}); "
             f"band=+/-{band:.1f}; true diff assumed 0; sd sensitivity x{SD_INFLATE}")
        emit(f"  {'n':>4} | " + " | ".join(f"sd x{m:.1f}" for m in SD_INFLATE))
        for n in N_GRID:
            emit(f"  {n:>4} | " + " | ".join(f"{power_tost(m * v.std(ddof=1), n, band):>7.2f}" for m in SD_INFLATE))

    emit("\n" + "=" * 96)
    emit("REGISTERED DECISION RULE (write the chosen n into prereg v1.3 BEFORE hashing):")
    emit(f"  n* = smallest n in {list(N_GRID)[1:]} with (i) P1-conjunction power >= {TARGET_PRIMARY} at 50% effect,")
    emit(f"  (ii) Gamma power >= {TARGET_PRIMARY} at (50% effect, sd x1.5), (iii) TOST power >= {TARGET_TOST} at sd x1.5.")
    emit("  If n* = 15 -> seeds 30-44. If n* = 20 -> seeds 30-49. If n* = 25 -> seeds 30-54.")
    emit("  Pilot seeds (>=50 used in development) are excluded from confirmatory analysis regardless.")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\n[written] {a.out}")


if __name__ == "__main__":
    main()