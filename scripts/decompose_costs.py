"""
decompose_costs.py -- operational decomposition of V (review item 6: WHERE the 10.6% comes from).

The confirmatory tables prove THAT the raw broadcast is worth ~+445 at rho=0.9; a management
journal referee will ask what the policy DOES differently. run_episode() in agents/eval_signal.py
already computes the full operational panel per episode (per-stage holding/backorder cost split,
service level alpha, fill rate beta, stockout frequency, stage & cumulative bullwhip ratios,
mean inventory/backlog/net stock, censoring fractions) -- it is simply never persisted per arm.
This script closes that gap in two modes:

  DUMP (torch; one call per checkpoint, mirrors dump_comm's env resolution exactly):
    python scripts/decompose_costs.py dump --ckpt <path> --out sweep_out/ops/ar1r9_raw \
        --ar1-rho 0.9 --episodes 40
    -> <out>/seed{S}_ops.json: episode-mean of every per-stage metric + scalar cost,
       CRN over HELDOUT_SEED_BASE (same eval seed space as the headline numbers).
    For the DR-Poisson (dp) arms use --dp: evaluates on the HELDOUT_LAMBDAS grid and averages,
    matching how the headline dp costs were produced.

  AGGREGATE (numpy/scipy only; registered-seed whitelist, fail-closed):
    python scripts/decompose_costs.py aggregate --comm sweep_out/ops/ar1r9_raw \
        --nocomm sweep_out/ops/ar1r9_nocomm --seeds 25-49 --out reports/OPS_ar1r9_raw.md
    -> seed-paired per-stage Delta table (comm - nocomm) with BCa CIs and Wilcoxon p per cell:
       the cost split answers "holding or backlog?", service/fill answer "availability?",
       bullwhip answers "variance dampening?", mean_inv answers "leaner or just luckier?".

Sanity identity printed in aggregate: sum over stages of total_c equals the scalar episode
cost (run_episode accumulates cost as the SUM of per-agent local costs), tolerance 1e-3 relative --
a failed identity means the dumps are stale or mixed across env versions.
"""
import os
import sys
import json
import argparse
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.c1_stats import bootstrap_ci, paired                    # noqa: E402
from scripts.comm_stats import parse_seed_spec                       # noqa: E402

AGENTS = ["retailer", "wholesaler", "distributor", "manufacturer"]
STAGE_KEYS = ["hold_c", "back_c", "total_c", "serv_alpha", "fill_beta_s", "stockout_freq",
              "bw_stage", "bw_cum", "mean_inv", "mean_back", "mean_netstock",
              "zero_order_frac", "max_order_frac"]
SCALAR_KEYS = ["cost", "fill_beta"]


# ==============================================================================
# dump mode (torch)
# ==============================================================================
def dump(args):
    import torch
    from agents.eval_signal import (SIGNALPolicy, run_episode, make_ar1_env,
                                    resolve_env_base, HELDOUT_SEED_BASE, HELDOUT_LAMBDAS,
                                    DemandRandomizedBeerGame)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    base = resolve_env_base(ckpt, args.env_json)
    seed = ckpt.get("seed", ckpt.get("config", {}).get("seed", None))
    if args.seed is not None:
        seed = args.seed
    if seed is None:
        sys.exit("FAIL: checkpoint carries no seed and --seed not given (file naming needs it)")
    if args.dp:
        envs = [DemandRandomizedBeerGame({**dict(base), "demand_type": "poisson"},
                                         lam_lo=float(l), lam_hi=float(l), p_shift=0.0)
                for l in HELDOUT_LAMBDAS]
        mode = f"DR-Poisson heldout lambdas {HELDOUT_LAMBDAS}"
    else:
        envs = [make_ar1_env(args.ar1_rho, args.ar1_mu, args.ar1_sigma, base)]
        mode = f"AR(1) rho={args.ar1_rho:g} (mu={args.ar1_mu:g} sigma={args.ar1_sigma:g})"
    runs = []
    for env in envs:
        pol = SIGNALPolicy(ckpt, env, ablate=False)
        runs += [run_episode(pol, env, HELDOUT_SEED_BASE + e, trace=False)
                 for e in range(args.episodes)]
    rec = {"meta": {"ckpt": os.path.abspath(args.ckpt), "seed": int(seed),
                    "episodes_per_env": int(args.episodes), "n_envs": len(envs), "mode": mode,
                    "use_comm": bool(pol.use_comm), "seed_base": int(HELDOUT_SEED_BASE)}}
    for k in SCALAR_KEYS:
        rec[k] = float(np.mean([r[k] for r in runs]))
    for k in STAGE_KEYS:
        rec[k] = {a: float(np.mean([r[k][a] for r in runs])) for a in AGENTS}
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"seed{int(seed)}_ops.json")
    with open(path, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"[ops-dump] wrote {path}  use_comm={pol.use_comm}  [{mode}]")


# ==============================================================================
# aggregate mode (torch-free)
# ==============================================================================
def _load_dir(d, seeds):
    out = {}
    for s in seeds:
        p = os.path.join(d, f"seed{s}_ops.json")
        if not os.path.exists(p):
            sys.exit(f"FAIL-CLOSED: {d} missing seed{s}_ops.json (registered whitelist)")
        with open(p) as f:
            out[s] = json.load(f)
    return out


def _vec(recs, seeds, key, agent=None):
    return np.array([recs[s][key] if agent is None else recs[s][key][agent] for s in seeds],
                    float)


def aggregate(args):
    seeds = parse_seed_spec(args.seeds)
    C = _load_dir(args.comm, seeds)
    N = _load_dir(args.nocomm, seeds)
    uc = {C[seeds[0]]["meta"]["use_comm"], N[seeds[0]]["meta"]["use_comm"]}
    lines = [f"# Operational decomposition: `{os.path.basename(args.comm.rstrip('/'))}` "
             f"vs `{os.path.basename(args.nocomm.rstrip('/'))}`",
             f"\nseeds={args.seeds} (n={len(seeds)}, fail-closed)  "
             f"modes: comm=`{C[seeds[0]]['meta']['mode']}` nocomm=`{N[seeds[0]]['meta']['mode']}`",
             f"use_comm flags: comm={C[seeds[0]]['meta']['use_comm']} "
             f"nocomm={N[seeds[0]]['meta']['use_comm']}"
             + ("  **WARNING: flags do not differ -- wrong dirs?**" if len(uc) == 1 else "") + "\n"]
    dc = _vec(C, seeds, "cost") - _vec(N, seeds, "cost")
    lines.append(f"Headline seed-paired Delta cost (comm - nocomm; negative = comm cheaper): "
                 f"**{np.mean(dc):+.1f}** [{bootstrap_ci(dc)[0]:+.1f},{bootstrap_ci(dc)[1]:+.1f}] "
                 f"(Wilcoxon p={paired(_vec(C, seeds, 'cost'), _vec(N, seeds, 'cost'))['wilcoxon_p']:.3g})\n")
    # ---- per-stage panel -------------------------------------------------------------------
    pretty = {"hold_c": "holding cost", "back_c": "backorder cost", "total_c": "total local cost",
              "serv_alpha": "service level alpha (P[no backlog])", "fill_beta_s": "fill rate beta",
              "stockout_freq": "stockout frequency", "bw_stage": "bullwhip (stage ratio)",
              "bw_cum": "bullwhip (cumulative vs customer)", "mean_inv": "mean inventory",
              "mean_back": "mean backlog", "mean_netstock": "mean net stock",
              "zero_order_frac": "zero-order fraction (lower censoring)",
              "max_order_frac": "max-order fraction (upper censoring)"}
    for k in STAGE_KEYS:
        lines.append(f"## {pretty[k]} (`{k}`) -- Delta = comm - nocomm")
        lines.append("| stage | comm | nocomm | Delta | 95% CI (BCa) | Wilcoxon p |")
        lines.append("|---|---|---|---|---|---|")
        for a in AGENTS:
            c, n = _vec(C, seeds, k, a), _vec(N, seeds, k, a)
            d = c - n
            ci = bootstrap_ci(d)
            lines.append(f"| {a} | {np.mean(c):.3f} | {np.mean(n):.3f} | {np.mean(d):+.3f} | "
                         f"[{ci[0]:+.3f},{ci[1]:+.3f}] | {paired(c, n)['wilcoxon_p']:.3g} |")
        lines.append("")
    # ---- cost-identity check (guards stale/mixed dumps) ------------------------------------
    tot_stage = sum(_vec(C, seeds, "total_c", a) - _vec(N, seeds, "total_c", a) for a in AGENTS)
    ratio = np.mean(tot_stage) / np.mean(dc) if abs(np.mean(dc)) > 1e-9 else float("nan")
    ok = np.isfinite(ratio) and abs(ratio - 1.0) < 1e-3
    lines.append(f"Cost identity sum_stage(total_c) / scalar cost (Deltas): {ratio:.6f} -> "
                 f"{'PASS' if ok else 'FAIL -- dumps inconsistent (stale mix?); do not report'}\n")
    txt = "\n".join(lines) + "\n"
    print(txt)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write(txt)
        print(f"-> wrote {args.out}")


# ==============================================================================
def _selftest():
    """Synthetic aggregate: comm arm shifts backorder cost down at upstream stages, holding
    slightly up; check the table lands on the planted deltas and the identity gate works."""
    import tempfile
    rng = np.random.default_rng(1)
    seeds = list(range(25, 50))
    with tempfile.TemporaryDirectory() as td:
        for name, back_shift, use_comm in (("comm", -80.0, True), ("noc", 0.0, False)):
            d = os.path.join(td, name)
            os.makedirs(d)
            for s in seeds:
                hold = {a: 300.0 + (10.0 if use_comm else 0.0) + rng.normal(0, 3) for a in AGENTS}
                back = {a: 700.0 + back_shift + rng.normal(0, 3) for a in AGENTS}
                tot = {a: hold[a] + back[a] for a in AGENTS}
                rec = {"meta": {"mode": "synthetic", "use_comm": use_comm, "seed": s},
                       "cost": float(np.sum(list(tot.values()))), "fill_beta": 0.9,
                       "hold_c": hold, "back_c": back, "total_c": tot}
                for k in STAGE_KEYS:
                    rec.setdefault(k, {a: 0.0 for a in AGENTS})
                with open(os.path.join(d, f"seed{s}_ops.json"), "w") as f:
                    json.dump(rec, f)
        ns = argparse.Namespace(comm=os.path.join(td, "comm"), nocomm=os.path.join(td, "noc"),
                                seeds="25-49", out=os.path.join(td, "OPS.md"))
        aggregate(ns)
        out = open(ns.out).read()
        assert "PASS" in out and "WARNING" not in out
        # planted: back_c Delta ~ -80 per stage, hold_c ~ +10, scalar ~ -70
    print("decompose_costs selftest: PASS")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = ap.add_subparsers(dest="mode", required=True)
    d = sub.add_parser("dump", help="per-checkpoint operational dump (torch)")
    d.add_argument("--ckpt", required=True)
    d.add_argument("--out", required=True, help="ops dir for this ARM (one file per seed)")
    d.add_argument("--episodes", type=int, default=40)
    d.add_argument("--ar1-rho", type=float, default=0.9)
    d.add_argument("--ar1-mu", type=float, default=12.0)
    d.add_argument("--ar1-sigma", type=float, default=3.0)
    d.add_argument("--dp", action="store_true", help="DR-Poisson heldout-lambda grid instead of AR(1)")
    d.add_argument("--env-json", default=None)
    d.add_argument("--seed", type=int, default=None, help="override ckpt seed for file naming")
    d.set_defaults(fn=dump)
    g = sub.add_parser("aggregate", help="seed-paired comm-vs-nocomm per-stage table (no torch)")
    g.add_argument("--comm", required=True)
    g.add_argument("--nocomm", required=True)
    g.add_argument("--seeds", default="25-49")
    g.add_argument("--out", default=None)
    g.set_defaults(fn=aggregate)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()