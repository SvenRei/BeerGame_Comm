"""
comm_stats.py -- cross-seed value-of-sharing aggregator for the SIGNAL study.

Elevates the H1 communication result from a point estimate to a defended claim. The unit of
analysis is the independent training seed: one comm arm and one no-comm arm trained at the same
seed, each scored deterministically per-lambda on the eval seeds (CRN), then CRN-paired by seed.

INPUT (produced by `python agents/eval_signal.py --dump-comm DIR`, one DIR per arm):
  DIR/seed{S}.json        {lambda(float): mean_cost(float)}                 # the arm's cost vector
  DIR/seed{S}_ferr.json   {lambda(float): {echelon: upstream_forecast_err}} # the Lee mechanism

REPORTS:
  * V_cost(seed) = mean_lambda(no_comm[seed]) - mean_lambda(comm[seed])   (+ => communication is
    cheaper => sharing has value). Mean V_cost with bootstrap CI, paired Wilcoxon/sign test, and a
    TOST equivalence test against a +/-band (the H1 null: is comm value within a negligible band?).
  * per-echelon forecast-error delta = ferr(no_comm) - ferr(comm), upstream echelons (the
    mechanism: sharing should cut upstream forecast error -- Lee-So-Tang 2000).

Reuses the inference primitives in scripts.c1_stats (bootstrap_ci / paired / tost), so the math is
shared with the C1 report.

REGISTERED-SEED WHITELIST (v2.1 audit fix -- the n=25-vs-n=30 defect)
---------------------------------------------------------------------
The loaders previously swept EVERY seed*.json present in a directory. Directories that also held
the quarantined dev/pilot seed files (50-54) therefore contaminated every family, curve and
intervention table with 5 unregistered observations (n=30 instead of the registered n=25), while
the confirmatory pipeline (confirmatory_v2.load_cell) was unaffected because it loads an exact
seed vector fail-closed. This module now applies the same discipline:

  Resolution order for the whitelist (first match wins):
    1. explicit `seeds=` argument to a loader / analysis function
       (CLI: --seeds "25-49" on the report / curve / interventions subcommands)
    2. the SIGNAL_SEEDS environment variable (exported by sweep_all_hypotheses.sh and
       auto_campaign2.sh from the registered $SEEDS vector)
    3. none -> legacy load-everything behaviour (ad-hoc use), with the seed set actually
       used printed on every report so contamination can never again be silent.

  Spec grammar: "25-49" | "25..49" | "25,26,27" | "25 26 27" | any mix.

  SIGNAL_SEEDS_STRICT=1 (or --strict-seeds) makes a whitelist FAIL-CLOSED per directory: a
  directory that exists but is missing a whitelisted seed aborts (mirrors confirmatory_v2).
  A directory that is absent/empty remains a soft skip (feature not run is not contamination).

Run:
  python scripts/comm_stats.py                                           # self-test (numpy only)
  python scripts/comm_stats.py report --comm-dir results/comm_on --nocomm-dir results/comm_off \
                                      --seeds "25-49"
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

SEEDS_ENV = "SIGNAL_SEEDS"                                          # registered whitelist (spec grammar above)
STRICT_ENV = "SIGNAL_SEEDS_STRICT"                                  # "1" => missing whitelisted seed aborts


# ==============================================================================
# Registered-seed whitelist
# ==============================================================================
def parse_seed_spec(spec):
    """'25-49' | '25..49' | '25,26 27' | mixes -> sorted unique [int]; None/'' -> None."""
    if spec is None:
        return None
    s = str(spec).strip()
    if not s:
        return None
    out = set()
    for tok in s.replace(",", " ").split():
        if ".." in tok:
            a, b = tok.split("..", 1)
            out.update(range(int(a), int(b) + 1))
        elif "-" in tok[1:]:                                        # allow negative singletons, not ranges of them
            a, b = tok.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(tok))
    if not out:
        raise ValueError(f"empty seed spec: {spec!r}")
    return sorted(out)


def active_whitelist(seeds=None):
    """Explicit arg > SIGNAL_SEEDS env > None. Accepts a spec string or an int iterable."""
    if seeds is not None:
        if isinstance(seeds, (str, bytes)):
            return parse_seed_spec(seeds)
        return sorted({int(s) for s in seeds})
    return parse_seed_spec(os.environ.get(SEEDS_ENV))


def _strict(strict=None):
    if strict is not None:
        return bool(strict)
    return os.environ.get(STRICT_ENV, "0").strip() in ("1", "true", "TRUE", "yes")


def seed_range_str(seeds):
    """Compact human form: contiguous -> '25..49'; else the explicit sorted list."""
    ss = sorted(int(s) for s in seeds)
    if not ss:
        return "(none)"
    if ss == list(range(ss[0], ss[-1] + 1)):
        return f"{ss[0]}..{ss[-1]}" if len(ss) > 1 else f"{ss[0]}"
    return ",".join(str(s) for s in ss)


def _apply_whitelist(found, wl, d, what, strict):
    """Filter {seed: payload} to the whitelist; print an audit line on any exclusion; fail-closed
    on missing whitelisted seeds when strict. `found` empty stays empty (soft skip upstream)."""
    if wl is None or not found:
        return found
    extras = sorted(set(found) - set(wl))
    kept = {s: found[s] for s in wl if s in found}
    missing = sorted(set(wl) - set(kept))
    if extras:
        print(f"  [seeds] {d} ({what}): kept {len(kept)}/{len(wl)} registered; "
              f"EXCLUDED unregistered {seed_range_str(extras)}")
    if missing:
        msg = (f"  [seeds] {d} ({what}): MISSING registered seed(s) {seed_range_str(missing)} "
               f"({len(kept)}/{len(wl)} present)")
        if strict:
            sys.exit(f"FAIL-CLOSED{msg}\n  (dir exists but is incomplete against the whitelist; "
                     f"unset {STRICT_ENV} only for exploratory partial reads)")
        print(msg + "  [non-strict: proceeding on the present subset]")
    return kept


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


def load_cost_dir(d, seeds=None, strict=None):
    """DIR/seed{S}.json -> {seed: {lambda: cost}} (skips the _ferr/_bw/_censor/_iv siblings).
    Applies the registered-seed whitelist (explicit `seeds` > SIGNAL_SEEDS env > none)."""
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "seed*.json"))):
        if p.endswith(_SIBLING_SUFFIXES):
            continue
        s = _seed_of(p)
        if s is None:
            continue
        with open(p) as f:
            out[s] = {float(k): float(v) for k, v in json.load(f).items()}
    return _apply_whitelist(out, active_whitelist(seeds), d, "cost", _strict(strict))


def load_ferr_dir(d, seeds=None, strict=None):
    """DIR/seed{S}_ferr.json -> {seed: {lambda: {echelon: forecast_error}}}. Whitelist as above."""
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "seed*_ferr.json"))):
        s = _seed_of(p)
        if s is None:
            continue
        with open(p) as f:
            out[s] = {float(k): {a: float(b) for a, b in v.items()} for k, v in json.load(f).items()}
    return _apply_whitelist(out, active_whitelist(seeds), d, "ferr", _strict(strict))


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
def substitution_curve(budget_dirs, lambdas=None, n_boot=10000, seeds=None, strict=None):
    """V as a function of training budget, from the budget-milestone checkpoints train_signal
    writes (signal_checkpoint_budget{N}.pt), dumped per budget into paired dirs.

    budget_dirs : {budget_episodes(int) -> (comm_dir, nocomm_dir)}   (same seed set per budget)

    Claim this feeds: inference capacity and information sharing are substitutes -- the classical
    redundancy null (Raghunathan 2001) is the infinite-inference limit. Under substitution,
    V(budget) declines as the no-comm agent's learned inference improves; the opposite sign
    (V rises with budget: exploiting a channel must itself be learned) is a complementarity
    finding, equally reportable. The curve decides.

    Statistic (exploratory-registered): per-seed OLS slope of V_s over log2(budget) across seeds
    with all budgets present, plus a tie-naive per-seed Spearman; bootstrap-t CI over seeds.
    Seeds honour the registered whitelist (explicit `seeds` > SIGNAL_SEEDS env)."""
    budgets = sorted(int(b) for b in budget_dirs)
    per_seed, rows = {}, []
    for b in budgets:
        comm = load_cost_dir(budget_dirs[b][0], seeds=seeds, strict=strict)
        nocomm = load_cost_dir(budget_dirs[b][1], seeds=seeds, strict=strict)
        sds, d, base = _paired_diffs(comm, nocomm, lambdas)
        for s, v in zip(sds, d):
            per_seed.setdefault(s, {})[b] = float(v)
        rows.append({"budget": b, "n_seeds": len(sds), "v_mean": float(d.mean()),
                     "v_ci": bootstrap_ci(d, n_boot=n_boot),
                     "v_pct": float(100.0 * d.mean() / max(1e-9, base))})
    full = {s: bv for s, bv in per_seed.items() if len(bv) == len(budgets)}
    x = np.log2(np.asarray(budgets, float))

    def _spear(y):                                   # Spearman via scipy (community-validated)
        y = np.asarray(y, float)
        if y.size < 2:
            return float("nan")
        # Constant series (e.g. an arm with V identically ~0 at machine precision at every
        # budget) has no defined rank correlation; report 0.0 without tripping scipy's
        # ConstantInputWarning into the result sheet.
        if np.allclose(y, y[0]) or np.allclose(x, x[0]):
            return 0.0
        from scipy.stats import spearmanr
        res = spearmanr(x, y)
        r = getattr(res, "statistic", getattr(res, "correlation", float("nan")))
        return float(r) if np.isfinite(r) else 0.0

    ys = [np.array([full[s][b] for b in budgets]) for s in sorted(full)]
    slopes = np.array([np.polyfit(x, y, 1)[0] for y in ys]) if ys else np.array([])
    spears = np.array([_spear(y) for y in ys]) if ys else np.array([])
    return {"rows": rows, "budgets": budgets, "n_trend_seeds": len(full),
            "trend_seeds": sorted(full),
            "slope_per_log2_mean": float(slopes.mean()) if slopes.size else float("nan"),
            "slope_ci": bootstrap_ci(slopes, n_boot=n_boot) if slopes.size else (float("nan"),) * 2,
            "spearman_mean": float(spears.mean()) if spears.size else float("nan")}


def print_curve(sc):
    print("=" * 78)
    print(f"SUBSTITUTION CURVE  V(training budget)  |  trend over {sc['n_trend_seeds']} complete "
          f"seed-series  |  seeds {seed_range_str(sc.get('trend_seeds', []))}")
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
def load_iv_dir(d, seeds=None, strict=None):
    """DIR/seed{S}_iv.json -> {seed: payload} (written by eval_signal.py --dump-iv).
    Applies the registered-seed whitelist (explicit `seeds` > SIGNAL_SEEDS env)."""
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "seed*_iv.json"))):
        s = _seed_of(p)
        if s is None:
            continue
        with open(p) as f:
            out[s] = json.load(f)
    return _apply_whitelist(out, active_whitelist(seeds), d, "iv", _strict(strict))


def interventions_summary(iv, n_boot=10000, ci=0.95):
    """Cross-seed aggregation of the per-checkpoint message-intervention probe. The per-seed dumps
    carry episode-mean costs under do(m): honest / shuffled / cross / zeroed. The seed is the unit
    (episodes inside a checkpoint are the instrument's Monte Carlo, not replication): per seed s,
    d_x(s) = cost_x(s) - cost_honest(s); report mean d_x, bootstrap CI (auto -> studentized
    bootstrap-t for the seed mean), Wilcoxon; content_share = ratio of mean deltas
    d_shuffled/d_zeroed (pooled, robust to near-zero per-seed denominators).
    Registered gate (prereg content_attribution_rule): a positive V is attributed to message
    content only if the cross-seed d_shuffled CI excludes 0 from below (shuffling hurts)."""
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
    print(f"MESSAGE-INTERVENTION SUMMARY (cross-seed)  |  {sm['n_seeds']} seeds "
          f"({seed_range_str(sm['seeds'])})  |  honest mean cost {sm['honest_mean']:.1f}")
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


# ==============================================================================
# Report
# ==============================================================================
def print_report(vs, fd=None):
    def _ci1(t): return f"[{t[0]:.1f}, {t[1]:.1f}]"
    print("=" * 78)
    print(f"VALUE-OF-SHARING REPORT  |  {vs['n_seeds']} seed-pairs "
          f"({seed_range_str(vs['seeds'])})  |  lambdas "
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
    ap.add_argument("--seeds", default=None,
                    help=f"registered seed whitelist, e.g. '25-49' (default: ${SEEDS_ENV})")
    ap.add_argument("--strict-seeds", action="store_true",
                    help="fail-closed if a whitelisted seed is missing from an existing dir")
    args = ap.parse_args(argv)
    strict = True if args.strict_seeds else None
    comm = load_cost_dir(args.comm_dir, seeds=args.seeds, strict=strict)
    nocomm = load_cost_dir(args.nocomm_dir, seeds=args.seeds, strict=strict)
    if not comm or not nocomm:
        print(f"no seed*.json found (comm={len(comm)}, nocomm={len(nocomm)}). "
              f"Generate them with `eval_signal.py --dump-comm`.")
        return
    vs = value_of_sharing(comm, nocomm, lambdas=args.lambdas, band_frac=args.band_frac)
    cf = load_ferr_dir(args.comm_dir, seeds=args.seeds, strict=strict)
    nf = load_ferr_dir(args.nocomm_dir, seeds=args.seeds, strict=strict)
    fd = forecast_delta(cf, nf, lambdas=args.lambdas) if (cf and nf) else None
    print_report(vs, fd)


# ==============================================================================
# Self-test (numpy only; scipy optional for the p-values)
# ==============================================================================
def _selftest():
    # The self-test fabricates seeds 0..N and must be immune to a whitelist exported in the
    # invoking shell; save/clear/restore the env.
    _saved = {k: os.environ.pop(k, None) for k in (SEEDS_ENV, STRICT_ENV)}
    try:
        _selftest_body()
    finally:
        for k, v in _saved.items():
            if v is not None:
                os.environ[k] = v


def _selftest_body():
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

    # D2) constant-V series: Spearman must come back 0.0 with NO scipy ConstantInputWarning.
    import warnings
    tmp2 = tempfile.mkdtemp(prefix="curveconst_")
    pairs2 = {}
    for b in (1000, 2000, 4000, 8000):
        cd, nd = _os.path.join(tmp2, f"c{b}"), _os.path.join(tmp2, f"n{b}")
        _os.makedirs(cd); _os.makedirs(nd)
        for s in range(8):
            _json.dump({l: 3000.0 for l in lams}, open(_os.path.join(cd, f"seed{s}.json"), "w"))
            _json.dump({l: 3000.0 for l in lams}, open(_os.path.join(nd, f"seed{s}.json"), "w"))
        pairs2[b] = (cd, nd)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                # any warning -> failure
        sc2 = substitution_curve(pairs2, n_boot=500)
    assert sc2["spearman_mean"] == 0.0
    print("  D2) constant-V series: Spearman 0.0, no ConstantInputWarning  PASS")

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

    # F) REGISTERED-SEED WHITELIST: dirs hold seeds 0..29; whitelist 0-24 must keep exactly 25,
    #    exclude the 5 extras, and be identical whether given explicitly or via SIGNAL_SEEDS.
    wd_c, wd_n = _os.path.join(tmp, "wl_c"), _os.path.join(tmp, "wl_n")
    _os.makedirs(wd_c); _os.makedirs(wd_n)
    for s in range(30):
        no = {l: 3000.0 + rng.normal(0, 20) for l in lams}
        co = {l: no[l] - 100.0 + rng.normal(0, 10) for l in lams}
        _json.dump(co, open(_os.path.join(wd_c, f"seed{s}.json"), "w"))
        _json.dump(no, open(_os.path.join(wd_n, f"seed{s}.json"), "w"))
        _json.dump({l: {"honest": 1.0} for l in lams}, open(_os.path.join(wd_c, f"seed{s}_ferr.json"), "w"))
    c_wl = load_cost_dir(wd_c, seeds="0-24")
    n_wl = load_cost_dir(wd_n, seeds=[i for i in range(25)])
    assert sorted(c_wl) == list(range(25)) and sorted(n_wl) == list(range(25))
    vs_wl = value_of_sharing(c_wl, n_wl, n_boot=1000)
    assert vs_wl["n_seeds"] == 25 and vs_wl["seeds"] == list(range(25))
    os.environ[SEEDS_ENV] = "0..24"                                # env route must match
    assert sorted(load_cost_dir(wd_c)) == list(range(25))
    assert sorted(load_ferr_dir(wd_c)) == list(range(25))
    del os.environ[SEEDS_ENV]
    assert len(load_cost_dir(wd_c)) == 30                          # no whitelist -> legacy load-all
    assert parse_seed_spec("25-49") == list(range(25, 50))
    assert parse_seed_spec("25..49") == list(range(25, 50))
    assert parse_seed_spec("1,2 3") == [1, 2, 3]
    assert seed_range_str(range(25, 50)) == "25..49" and seed_range_str([1, 3]) == "1,3"
    print("  F) whitelist: 30-file dir -> n=25 kept, 5 extras excluded (arg + env routes)  PASS")

    # G) STRICT: an existing dir missing a whitelisted seed must fail-closed.
    _os.remove(_os.path.join(wd_c, "seed3.json"))
    try:
        load_cost_dir(wd_c, seeds="0-24", strict=True)
        raise AssertionError("strict whitelist did not fail-closed on a missing seed")
    except SystemExit:
        pass
    kept = load_cost_dir(wd_c, seeds="0-24", strict=False)         # non-strict: proceed on subset
    assert sorted(kept) == [s for s in range(25) if s != 3]
    print("  G) strict whitelist: missing registered seed -> FAIL-CLOSED; non-strict proceeds  PASS")

    print_report(vs, fd)
    print("\ncomm_stats self-test PASS")


def _curve_cli(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="comm_stats.py curve")
    ap.add_argument("--budget", nargs=3, action="append", required=True,
                    metavar=("EPISODES", "COMM_DIR", "NOCOMM_DIR"),
                    help="repeat per milestone, e.g. --budget 2000 results/h1b2000_comm results/h1b2000_nocomm")
    ap.add_argument("--lambdas", nargs="+", type=float, default=None)
    ap.add_argument("--seeds", default=None,
                    help=f"registered seed whitelist, e.g. '25-49' (default: ${SEEDS_ENV})")
    ap.add_argument("--strict-seeds", action="store_true")
    args = ap.parse_args(argv)
    pairs = {int(b): (c, n) for b, c, n in args.budget}
    print_curve(substitution_curve(pairs, lambdas=args.lambdas, seeds=args.seeds,
                                   strict=(True if args.strict_seeds else None)))


def _interventions_cli(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="comm_stats.py interventions")
    ap.add_argument("--dir", required=True, help="dir of seed{S}_iv.json (eval_signal --dump-iv)")
    ap.add_argument("--seeds", default=None,
                    help=f"registered seed whitelist, e.g. '25-49' (default: ${SEEDS_ENV})")
    ap.add_argument("--strict-seeds", action="store_true")
    args = ap.parse_args(argv)
    iv = load_iv_dir(args.dir, seeds=args.seeds, strict=(True if args.strict_seeds else None))
    if not iv:
        print(f"no seed*_iv.json in {args.dir}. Generate with `eval_signal.py --ar1 --dump-iv DIR`.")
        return
    print_interventions(interventions_summary(iv))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        _report_cli(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "curve":
        _curve_cli(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "interventions":
        _interventions_cli(sys.argv[2:])
    else:
        _selftest()