#!/usr/bin/env python3
"""qmix_dump.py -- per-seed cost dumps for Phase-G QMIX checkpoints (sign-concordance producers).

eval_signal.SIGNALPolicy loads MAPPO checkpoints only; this is the QMIX counterpart, writing the
SAME schema the analysis loaders read: <out>/seed{S}.json = {key: mean_team_cost} over
--episodes CRN episodes (HELDOUT_SEED_BASE + e, identical to eval_signal / baselines), where key is
the AR(1) rho (--ar1-rho) or each DP test lambda (--dp). The env is rebuilt from the checkpoint's
stored env base, so clip / cost-model / max_order settings travel with the run automatically.
Usage:
  python scripts/qmix_dump.py --ckpt path.pt --out DIR (--ar1-rho 0.9 | --dp) [--episodes 200]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.qmix_agent import qmix_greedy_costs                      # noqa: E402
from envs.beer_game_env import BeerGameParallelEnv                   # noqa: E402
from envs.demand_randomization import DemandRandomizedBeerGame       # noqa: E402
from scripts.demand_families import make_demand_family_envs          # noqa: E402

SEED_BASE = 500000  # FINAL-EVAL space (review 2.0 fix #1; == eval_signal/baselines)                                                   # == eval_signal / baselines CRN base
DP_TEST_LAMBDAS = [6.0, 10.0, 14.0, 18.0, 22.0]                      # C1 test grid (scored post-hoc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ar1-rho", type=float, default=None)
    ap.add_argument("--dp", action="store_true")
    ap.add_argument("--episodes", type=int, default=200)
    a = ap.parse_args()
    if (a.ar1_rho is None) == (not a.dp):
        ap.error("exactly one of --ar1-rho / --dp")

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    A, base = ck["config"], dict(ck["env"])                          # env base carries obs_order_clip etc.
    seeds = [SEED_BASE + e for e in range(a.episodes)]
    out = {}
    if a.dp:
        for lam in DP_TEST_LAMBDAS:
            env = DemandRandomizedBeerGame({**base, "demand_type": "poisson"},
                                           lam_lo=lam, lam_hi=lam, p_shift=0.0)
            out[f"{lam:g}"] = float(np.mean(qmix_greedy_costs(ck, env, seeds)))
    else:
        AR1, _, _ = make_demand_family_envs(BeerGameParallelEnv)
        env = AR1({**base, "demand_type": "poisson", "family": "ar1",
                   "ar1_mu": A.get("ar1_mu", 12.0), "ar1_rho": a.ar1_rho,
                   "ar1_sigma": A.get("ar1_sigma", 3.0)})
        out[f"{a.ar1_rho:g}"] = float(np.mean(qmix_greedy_costs(ck, env, seeds)))

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"seed{int(ck.get('seed', 0))}.json")
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"[qmix_dump] {path}  {out}")


if __name__ == "__main__":
    main()