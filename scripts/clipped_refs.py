"""
clipped_refs.py -- AR(1) base-stock references under obs_order_clip (the P2 worlds).

WHY (review item: "Add clipped-world optimality diagnostics -- mandatory if P2 appears in the
main paper"). The registered AR reference AR_BestBS = min(CondBS, StaticBS) was computed in the
UNCLIPPED environment. Whether it transfers to the clip-12/clip-20 worlds is not a judgment
call -- it is decidable from what each policy reads:

  * ARCondBSPolicy (privileged): conditions every echelon on the RETAILER's observed d_t
    (obs["retailer"][3]). The env clips o[3] only for agents != retailer, and the policy's only
    other input is IP = o[0]-o[1]+o[2] (never clipped). obs_order_clip is observation-side only
    (env dynamics untouched), so under CRN the privileged policy's trajectories -- and
    therefore its cost -- are IDENTICAL at every clip level.        -> clip-INVARIANT.
  * ARStaticBSPolicy: reads only IP.                                 -> clip-INVARIANT.
  * AR1BaseStockPolicy ("ownobs" here): the SAME conditional-base-stock rule, but each echelon
    conditions on its OWN o[3] -- which IS clipped upstream.         -> clip-DEPENDENT.

The privileged/ownobs pair holds the decision rule fixed (identical AR(1) cumulative-forecast
critical-fractile order-up-to) and varies ONLY the conditioning variable: retailer's true d_t
vs the local, possibly-garbled order stream. Neither is a multi-echelon optimum (upstream order
streams are not AR(1)); the wedge is a frontier gap WITHIN the registered conditional-base-stock
family, in the registered reference currency.

This script (a) VERIFIES the two invariance claims empirically under CRN (exact-equality check,
not a statistical one -- any deviation is itself a finding), and (b) produces the number the P2
identification is missing: AR_CondBS_ownobs at clip in {inf, 20, 12}, i.e. the conditional-
base-stock cost available to agents restricted to their own garbled order streams.

The resulting decomposition, consumed by scripts/p2_decompose.py:

  WEDGE(clip) = cost[AR_CondBS_ownobs](clip) - cost[AR_CondBS_privileged]
              = the information value a PERFECT conditional-base-stock user of the raw retailer
                broadcast could recover under that clip (the message channel is never clipped).

  If WEDGE(12) is large while the learned clip12_raw arm captured ~0 of it, the P2 reversal is
  a LEARNING failure under garbled complementary observations, not information redundancy. If
  WEDGE(12) itself collapses, the reversal is a property of the decision problem. Either way the
  registered frontier (privileged rungs) provably did not move, so "clipping made the
  environment harder" is off the table: only the observations changed.

Run (repo root):
  python scripts/clipped_refs.py --episodes 200 --rhos 0.9 --clips inf,20,12 \
      --out results/baselines_ar_clip_v2.json
Analytic policies, no training, no selection step: pure-CPU minutes.
"""
import os
import sys
import json
import argparse
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.baselines import (ENV_BASE, EVAL_SEED_BASE, rollout,                    # noqa: E402
                               ARCondBSPolicy, ARStaticBSPolicy, AR1BaseStockPolicy)
from scripts.demand_families import make_demand_family_envs                          # noqa: E402
from envs.beer_game_env import BeerGameParallelEnv                                   # noqa: E402

INVARIANT_TOL = 1e-6                     # CRN trajectories must match to roundoff, not to noise


def _make_env(rho, mu, sigma, clip, base):
    AR1Env = make_demand_family_envs(BeerGameParallelEnv)[0]
    cfg = {**base, "demand_type": "poisson", "family": "ar1",
           "ar1_mu": mu, "ar1_rho": float(rho), "ar1_sigma": sigma}
    if clip is not None:
        cfg["obs_order_clip"] = float(clip)
    return AR1Env(cfg)


def _mean_cost(env, policy, episodes, seed_base):
    # rollout() calls policy.reset() itself at each episode start.
    return float(np.mean([rollout(env, policy, seed_base + e)[0] for e in range(episodes)]))


def run(rhos, clips, episodes, mu, sigma, out_path, seed_base=EVAL_SEED_BASE, env_cfg=None):
    base = dict(ENV_BASE if env_cfg is None else env_cfg)
    h = float(base.get("holding_cost", 0.5))
    b = float(base.get("backorder_cost", 1.0))
    clip_keys = ["inf" if c is None else f"{int(c)}" for c in clips]
    out = {"meta": {"mu": mu, "sigma": sigma, "rhos": [float(r) for r in rhos],
                    "clips": clip_keys, "h": h, "b": b,
                    "eval_seed_base": int(seed_base), "eval_episodes": int(episodes),
                    "invariant_tol": INVARIANT_TOL,
                    "note": ("AR_CondBS/AR_StaticBS are the REGISTERED privileged rungs "
                             "(clip-invariant by construction; verified below). "
                             "AR_CondBS_ownobs = AR1BaseStockPolicy: identical conditional "
                             "base-stock rule conditioned on each echelon's OWN o[3] (clipped "
                             "upstream) -- the clip-dependent own-observation rung. WEDGE = "
                             "ownobs - privileged cost = raw-broadcast-recoverable value "
                             "within the conditional-base-stock family.")},
           "rhos": {}}
    all_pass = True
    for rho in rhos:
        pols = {"AR_CondBS": ARCondBSPolicy(mu, rho, sigma, h=h, b=b),
                "AR_StaticBS": ARStaticBSPolicy(mu, rho, sigma, h=h, b=b),
                "AR_CondBS_ownobs": AR1BaseStockPolicy(mu=mu, rho=rho, sigma=sigma, h=h, b=b)}
        costs = {name: {} for name in pols}
        print(f"== rho={rho:g}  ({episodes} eps/point, CRN seed_base={seed_base}) ==")
        for c, ck in zip(clips, clip_keys):
            env = _make_env(rho, mu, sigma, c, base)
            for name, pol in pols.items():
                costs[name][ck] = _mean_cost(env, pol, episodes, seed_base)
            print(f"  clip={ck:>4}: " + "  ".join(f"{n}={costs[n][ck]:.1f}" for n in pols))
        # ---- invariance verification (exact under CRN; deviation is a finding, not noise) ----
        inv = {}
        for name in ("AR_CondBS", "AR_StaticBS"):
            vals = [costs[name][ck] for ck in clip_keys]
            spread = float(max(vals) - min(vals))
            ok = spread <= INVARIANT_TOL * max(1.0, abs(vals[0]))
            inv[name] = {"costs": {ck: costs[name][ck] for ck in clip_keys},
                         "max_abs_spread": spread, "pass": bool(ok)}
            all_pass &= ok
            print(f"  INVARIANCE {name:<12}: max spread {spread:.3g} "
                  f"-> {'PASS (clip-invariant, as derived)' if ok else 'FAIL -- unexpected clip leak into a non-clipped read path; INVESTIGATE'}")
        best = {ck: min(costs["AR_CondBS"][ck], costs["AR_StaticBS"][ck]) for ck in clip_keys}
        # WEDGE sign convention: ownobs has weakly less information, so ownobs cost >= privileged
        # cost; report wedge = ownobs - privileged >= 0 = recoverable value of the raw broadcast.
        wedge = {ck: costs["AR_CondBS_ownobs"][ck] - costs["AR_CondBS"][ck] for ck in clip_keys}
        print("  WEDGE (ownobs - privileged = raw-broadcast-recoverable value under clip):")
        for ck in clip_keys:
            print(f"    clip={ck:>4}: wedge={wedge[ck]:+.1f}  "
                  f"(ownobs {costs['AR_CondBS_ownobs'][ck]:.1f} vs privileged {costs['AR_CondBS'][ck]:.1f})")
        out["rhos"][f"{float(rho):g}"] = {"costs": costs, "AR_BestBS": best,
                                          "invariance": inv, "wedge_ownobs_minus_priv": wedge}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"-> wrote {out_path}   (invariance overall: {'PASS' if all_pass else 'FAIL'})")
    return out


def _parse_clips(spec):
    out = []
    for tok in str(spec).replace(",", " ").split():
        out.append(None if tok.lower() in ("inf", "none") else float(tok))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rhos", default="0.9", help="comma/space list, e.g. '0.9' or '0.6 0.9'")
    ap.add_argument("--clips", default="inf,20,12")
    ap.add_argument("--episodes", type=int, default=200,
                    help="CRN episodes per (rho, clip, policy) point (refs precision)")
    ap.add_argument("--mu", type=float, default=12.0)
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--seed-base", type=int, default=EVAL_SEED_BASE)
    ap.add_argument("--out", default="results/baselines_ar_clip_v2.json")
    a = ap.parse_args()
    rhos = [float(t) for t in a.rhos.replace(",", " ").split()]
    run(rhos, _parse_clips(a.clips), a.episodes, a.mu, a.sigma, a.out, seed_base=a.seed_base)