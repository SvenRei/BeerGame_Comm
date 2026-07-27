"""
qmix_grid_benchmark.py -- the C11 fair benchmark: how much of the QMIX competence gap is the
ACTION GRID's fault, before blaming the learner?

QMIX actuates order = clip(S - IP, 0, max_order) with S restricted to linspace(0, s_max, n).
The executed G2 arms ran n=41, s_max=160 (4-unit spacing) -- NOT the 21/200 code default.
This script evaluates the privileged AR conditional base-stock policy (and its static
counterpart) PROJECTED onto each candidate grid: identical decision rule and information,
S snapped to the nearest grid point, identical actuation formula. The gap

    expressiveness_gap(grid) = cost[GridCondBS(grid)] - cost[CondBS(continuous)]

is a floor on what ANY policy confined to that grid loses, under the same CRN episodes as
every registered reference (EVAL_SEED_BASE). Whatever remains of the measured QMIX gap
(qmix nocomm ~6603 vs continuous privileged 3747.6) after subtracting this floor is
attributable to LEARNING, not expressiveness -- that split decides how much of the spec's
Part-C program is actually needed.

Run (repo root, pure CPU, ~a minute):
  python scripts/qmix_grid_benchmark.py --episodes 200 \
      --out results/qmix_grid_benchmark.json
"""
import os
import sys
import json
import argparse
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.baselines import (ENV_BASE, EVAL_SEED_BASE, AGENTS, rollout,            # noqa: E402
                               ARCondBSPolicy, ARStaticBSPolicy, _action, _ip)
from scripts.demand_families import make_demand_family_envs                          # noqa: E402
from envs.beer_game_env import BeerGameParallelEnv                                   # noqa: E402

# (n_actions, s_max): code default / EXECUTED G2 / two finer candidates for the variant.
GRIDS = [(21, 200.0), (41, 160.0), (81, 120.0), (161, 120.0)]


class GridProjected:
    """Snap a conditional/static base-stock policy's S to the nearest grid point, then apply
    the EXACT QMIX actuation (order = clip(S - IP, 0, max_order)). Nearest-point projection is
    the canonical (and generous) choice: a learner could at best match it pointwise."""

    def __init__(self, inner, n_actions, s_max):
        self.inner = inner
        self.grid = np.linspace(0.0, float(s_max), int(n_actions))

    def reset(self):
        if hasattr(self.inner, "reset"):
            self.inner.reset()

    def _snap(self, S):
        return float(self.grid[int(np.argmin(np.abs(self.grid - S)))])

    def act(self, obs, env):
        out = {}
        if isinstance(self.inner, ARCondBSPolicy):
            d_t = float(obs[AGENTS[0]][3])
            for i, a in enumerate(AGENTS):
                S = (self.inner.tau[i] * self.inner.mu
                     + self.inner._phi[i] * (d_t - self.inner.mu) + self.inner._safety[i])
                out[a] = _action(max(0.0, self._snap(S) - _ip(obs[a])), env.max_order)
        else:                                           # ARStaticBSPolicy: fixed S per echelon
            for i, a in enumerate(AGENTS):
                out[a] = _action(max(0.0, self._snap(self.inner.S[i]) - _ip(obs[a])),
                                 env.max_order)
        return out


def run(episodes, mu, sigma, rho, out_path):
    AR1 = make_demand_family_envs(BeerGameParallelEnv)[0]
    base = dict(ENV_BASE)
    h, b = float(base.get("holding_cost", 0.5)), float(base.get("backorder_cost", 1.0))
    env = AR1({**base, "demand_type": "poisson", "family": "ar1",
               "ar1_mu": mu, "ar1_rho": rho, "ar1_sigma": sigma})
    cond = ARCondBSPolicy(mu, rho, sigma, h=h, b=b)
    stat = ARStaticBSPolicy(mu, rho, sigma, h=h, b=b)

    def mc(policy):
        return float(np.mean([rollout(env, policy, EVAL_SEED_BASE + e)[0]
                              for e in range(episodes)]))

    c_cont, s_cont = mc(cond), mc(stat)
    print(f"== fair grid benchmark  rho={rho:g}  ({episodes} CRN eps, "
          f"seed_base {EVAL_SEED_BASE}) ==")
    print(f"  continuous:  CondBS {c_cont:.1f}   StaticBS {s_cont:.1f}")
    out = {"meta": {"rho": rho, "mu": mu, "sigma": sigma, "h": h, "b": b,
                    "episodes": episodes, "seed_base": EVAL_SEED_BASE,
                    "note": ("expressiveness_gap = GridCondBS - continuous CondBS: the floor "
                             "any grid-confined policy pays. Executed G2 grid = (41, 160).")},
           "continuous": {"CondBS": c_cont, "StaticBS": s_cont}, "grids": {}}
    for n, smax in GRIDS:
        key = f"n{n}_smax{int(smax)}"
        gc = mc(GridProjected(cond, n, smax))
        gs = mc(GridProjected(stat, n, smax))
        gap = gc - c_cont
        spacing = smax / (n - 1)
        tag = " <- EXECUTED G2" if (n, smax) == (41, 160.0) else ""
        out["grids"][key] = {"n_actions": n, "s_max": smax, "spacing": spacing,
                             "GridCondBS": gc, "GridStaticBS": gs,
                             "expressiveness_gap": gap}
        print(f"  grid n={n:>3} s_max={smax:>5.0f} (spacing {spacing:>5.2f}): "
              f"GridCondBS {gc:.1f}  gap {gap:+8.1f}  GridStaticBS {gs:.1f}{tag}")
    # learning-vs-expressiveness split for the EXECUTED grid, against the measured QMIX numbers
    g2 = out["grids"]["n41_smax160"]
    measured_qmix_nocomm = 6603.0                        # diagnostic mean (documented input)
    learn_gap = measured_qmix_nocomm - g2["GridCondBS"]
    out["split_executed_grid"] = {
        "measured_qmix_nocomm": measured_qmix_nocomm,
        "grid_floor_GridCondBS": g2["GridCondBS"],
        "expressiveness_gap": g2["expressiveness_gap"],
        "residual_learning_gap": learn_gap,
        "read": ("Of the total gap to the continuous frontier, expressiveness_gap is the "
                 "grid's share and residual_learning_gap is QMIX training's share.")}
    print(f"\n  SPLIT @ executed grid (41,160): total qmix-nocomm gap "
          f"{measured_qmix_nocomm - c_cont:+.1f} = expressiveness {g2['expressiveness_gap']:+.1f} "
          f"+ learning {learn_gap:+.1f}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"-> wrote {out_path}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--mu", type=float, default=12.0)
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--rho", type=float, default=0.9)
    ap.add_argument("--out", default="results/qmix_grid_benchmark.json")
    a = ap.parse_args()
    run(a.episodes, a.mu, a.sigma, a.rho, a.out)
