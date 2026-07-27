"""
forecast_pretrain.py -- pretrain + certify + freeze the repaired dhat forecaster (spec A8).

Pipeline:  sample demand streams from the EXACT env DGP  ->  train DemandForecaster on
one-step prediction (early stop on held-out val)  ->  certify against the empirical
conditional benchmark (A14/A16)  ->  save the certified, frozen artifact.

The streams are produced by stepping the real BeerGameParallelEnv AR(1) family with a trivial
policy and recording the retailer's realized customer demand from infos (demand is exogenous:
the actions cannot influence it, so any policy yields the true DGP). This guarantees the
forecaster is trained and certified on the rounded/truncated process the policy will face --
not on an idealized Gaussian AR(1).

Seed spaces (declared, disjoint from everything registered):
  train streams   seed base 700000
  val streams     seed base 710000
  benchmark fit   seed base 720000   (empirical conditional table)
  test/cert       seed base 730000   (certification metrics computed here ONLY)
None of these touch training seeds 25-54 or the eval bases (500000+).

Run (repo root):
  python scripts/forecast_pretrain.py --rho 0.9 --mu 12 --sigma 3 \
      --out results/forecaster_ar1r9.pt
Prints the certification report; exits nonzero if the gate FAILS (so campaign scripts can
fail-closed on an uncertified forecaster).
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agents.demand_forecaster import (DemandForecaster, make_one_step_dataset,       # noqa: E402
                                      empirical_conditional_benchmark, linear_predictor,
                                      forecast_metrics, certify, certification_report,
                                      save_certified)
from envs.beer_game_env import BeerGameParallelEnv                                    # noqa: E402
from scripts.demand_families import make_demand_family_envs                           # noqa: E402
from scripts.baselines import ENV_BASE                                                # noqa: E402

SEED_TRAIN, SEED_VAL, SEED_BENCH, SEED_TEST = 700000, 710000, 720000, 730000


def sample_demand_streams(rho, mu, sigma, episodes, seed_base, horizon=None):
    AR1 = make_demand_family_envs(BeerGameParallelEnv)[0]
    cfg = {**ENV_BASE, "demand_type": "poisson", "family": "ar1",
           "ar1_mu": mu, "ar1_rho": float(rho), "ar1_sigma": sigma}
    if horizon:
        cfg["horizon"] = int(horizon)
    env = AR1(cfg)
    streams = []
    for e in range(episodes):
        obs, _ = env.reset(seed=seed_base + e)
        ds, done = [], False
        while not done:
            obs, _, term, trunc, infos = env.step({a: [0.0] for a in env.agents})
            ds.append(float(infos["retailer"]["training_targets"]["demand"]))
            done = any(term.values()) or any(trunc.values())
        streams.append(np.asarray(ds, dtype=np.float32))
    return streams


def train_forecaster(train_streams, val_streams, hidden, initial_mean, lr, clip,
                     max_epochs, patience, log_every=25):
    Xtr, Ytr = make_one_step_dataset(train_streams)
    Xva, Yva = make_one_step_dataset(val_streams)
    f = DemandForecaster(hidden=hidden, initial_mean=initial_mean)
    opt = torch.optim.Adam(f.parameters(), lr=lr)
    best_val, best_state, since = float("inf"), None, 0
    for ep in range(1, max_epochs + 1):
        f.train()
        opt.zero_grad()
        pred, _ = f(Xtr)
        loss = nn.functional.mse_loss(pred, Ytr)
        loss.backward()
        nn.utils.clip_grad_norm_(f.parameters(), clip)
        opt.step()
        f.eval()
        with torch.no_grad():
            vpred, _ = f(Xva)
            vmse = float(nn.functional.mse_loss(vpred, Yva))
        if vmse < best_val - 1e-4:
            best_val, since = vmse, 0
            best_state = {k: v.detach().clone() for k, v in f.state_dict().items()}
        else:
            since += 1
        if ep % log_every == 0 or since >= patience:
            print(f"  epoch {ep:4d}  train MSE {float(loss):7.3f}  val MSE {vmse:7.3f}"
                  f"  (best {best_val:7.3f}, patience {since}/{patience})")
        if since >= patience:
            break
    f.load_state_dict(best_state)
    f.eval()
    return f, best_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, default=0.9)
    ap.add_argument("--mu", type=float, default=12.0)
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--train-episodes", type=int, default=300)
    ap.add_argument("--val-episodes", type=int, default=80)
    ap.add_argument("--bench-episodes", type=int, default=400)
    ap.add_argument("--test-episodes", type=int, default=120)
    ap.add_argument("--max-epochs", type=int, default=1500)
    ap.add_argument("--patience", type=int, default=60)
    ap.add_argument("--out", default="results/forecaster_ar1r9.pt")
    a = ap.parse_args()
    torch.manual_seed(0)

    print(f"[streams] sampling exact-DGP demand (rho={a.rho} mu={a.mu} sigma={a.sigma})...")
    tr = sample_demand_streams(a.rho, a.mu, a.sigma, a.train_episodes, SEED_TRAIN)
    va = sample_demand_streams(a.rho, a.mu, a.sigma, a.val_episodes, SEED_VAL)
    be = sample_demand_streams(a.rho, a.mu, a.sigma, a.bench_episodes, SEED_BENCH)
    te = sample_demand_streams(a.rho, a.mu, a.sigma, a.test_episodes, SEED_TEST)
    print(f"[streams] train {len(tr)}x{len(tr[0])}  val {len(va)}  bench {len(be)}  test {len(te)}")

    f, best_val = train_forecaster(tr, va, a.hidden, a.mu, a.lr, a.clip,
                                   a.max_epochs, a.patience)

    bench = empirical_conditional_benchmark(be, mu=a.mu, rho=a.rho)
    lin = linear_predictor(a.mu, a.rho)
    Xte, Yte = make_one_step_dataset(te)
    with torch.no_grad():
        pte, _ = f(Xte)
    prev = Xte[..., 0].numpy().reshape(-1)
    targ = Yte[..., 0].numpy().reshape(-1)
    preds = pte[..., 0].numpy().reshape(-1)
    met = forecast_metrics(preds, targ, bench(prev))
    met["linear_predictor_mse"] = float(np.mean((lin(prev) - targ) ** 2))
    cert = certify(met)
    print("\n== CERTIFICATION (held-out test streams, seed base 730000) ==")
    print(certification_report(met, cert))
    print(f"  (linear predictor MSE {met['linear_predictor_mse']:.3f}; "
          f"empirical benchmark MSE {met['bench_mse']:.3f})")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    meta = {"rho": a.rho, "mu": a.mu, "sigma": a.sigma, "hidden": a.hidden, "lr": a.lr,
            "grad_clip": a.clip, "seed_bases": {"train": SEED_TRAIN, "val": SEED_VAL,
                                                "bench": SEED_BENCH, "test": SEED_TEST},
            "best_val_mse": best_val, "env_base": dict(ENV_BASE),
            "label": "POST-UNBLINDING REPAIR: dedicated forecaster, policy-independent"}
    save_certified(a.out, f, met, cert, meta)
    with open(a.out.replace(".pt", "_metrics.json"), "w") as fh:
        json.dump({"metrics": met, "certification": cert, "meta": meta}, fh, indent=2)
    print(f"-> wrote {a.out} (+ _metrics.json)   CERTIFIED={cert['pass']}")
    sys.exit(0 if cert["pass"] else 1)


if __name__ == "__main__":
    main()
