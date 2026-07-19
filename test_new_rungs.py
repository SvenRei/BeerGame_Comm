#!/usr/bin/env python3
"""test_new_rungs.py -- proof suite for the v1.3 rungs {eps, oracle, raw_lag1, raw_lag2}.
Run from the repo root: python test_new_rungs.py   (exit 0 = all proofs pass)
T1 hand-table exactness (incl. step-0 zeros + mu-padding)  T2 end-to-end identity on the REAL
trainer+env for ALL steps/agents across TWO episodes (scale+timing+episode-reset in one)
T3 DP oracle == this episode's true lambda  T4 eval-path parity  T5 dimension contract."""
import sys
import numpy as np
import torch
sys.path.insert(0, ".")
from agents.signal_agent import SIGNALActor, SIGNALTrainer, _FIXED_MSG_DIMS
from envs.beer_game_env import BeerGameParallelEnv
from envs.demand_randomization import DemandRandomizedBeerGame
from scripts.demand_families import make_demand_family_envs

P = F = 0
def ck(name, cond):
    global P, F
    print(("  [PASS] " if cond else "  [FAIL] ") + name)
    P += bool(cond); F += (not cond)

RUNGS = ("eps", "oracle", "raw_lag1", "raw_lag2")
MU, RHO = 12.0, 0.9
BASE = {"horizon": 18, "max_order": 100, "holding_cost": 0.5, "backorder_cost": 1.0}
AR1cls, _NB, _Fam = make_demand_family_envs(BeerGameParallelEnv)
def ar1_env():
    return AR1cls({**BASE, "demand_type": "poisson", "family": "ar1",
                   "ar1_mu": MU, "ar1_rho": RHO, "ar1_sigma": 3.0})

def expected_stream(obs_d, mu, rho, rung, lam=None):
    """The registered rules, implemented independently of the agent code."""
    out = []
    for t in range(len(obs_d)):
        if t == 0:
            out.append(0.0); continue
        d0 = obs_d[t - 1] if t >= 2 else mu
        d1 = obs_d[t - 2] if t >= 3 else mu
        if rung == "eps":      out.append(obs_d[t] - (mu + rho * (d0 - mu)))
        elif rung == "oracle": out.append(float(lam) if lam is not None else mu + rho * (obs_d[t] - mu))
        elif rung == "raw_lag1": out.append(d0)
        else:                    out.append(d1)
    return out

print("== T1: hand-table exactness (mu=12, rho=0.9; slot-3 sequence 0,11,14,17,9) ==")
seq = [0.0, 11.0, 14.0, 17.0, 9.0]
hand = {"eps":      [0.0, -1.0, 14 - (12 + .9 * -1.0), 17 - (12 + .9 * 2.0), 9 - (12 + .9 * 5.0)],
        "oracle":   [0.0, 12 + .9 * -1, 12 + .9 * 2, 12 + .9 * 5, 12 + .9 * -3],
        "raw_lag1": [0.0, 12.0, 11.0, 14.0, 17.0],
        "raw_lag2": [0.0, 12.0, 12.0, 11.0, 14.0]}
for r in RUNGS:
    ck(f"T1a rule impls agree ({r})", np.allclose(hand[r], expected_stream(seq, MU, RHO, r), atol=1e-9))
for rung in RUNGS:
    a = SIGNALActor(obs_dim=4, msg_dim=1, hidden=8, content=rung)
    a.demand_mu, a.demand_rho = MU, RHO
    got, dprev, o_last = [], None, None
    for d in seq:
        o = torch.tensor([[3.0, 1.0, 2.0, d]])          # other slots arbitrary: must be ignored
        if o_last is not None:
            dprev = (torch.full((1, 2), MU) if dprev is None
                     else torch.cat([o_last[:, 3:4], dprev[:, 0:1]], dim=1))
        o_last = o
        got.append(float(a.message(o, torch.zeros(1, 1, 8), dprev=dprev).reshape(-1)[0]))
    ck(f"T1b {rung} builder == hand values", np.allclose(got, hand[rung], atol=1e-6))

print("== T2: REAL trainer + REAL AR(1) env; identity for ALL t,i; TWO episodes (reset proof) ==")
for rung in RUNGS:
    tr = SIGNALTrainer({"msg_content": rung, "hidden": 16, "ar1_mu": MU, "ar1_rho": RHO},
                       n_agents=4, obs_dim=4, state_dim=133, adj=np.ones((4, 4)) / 4.0)
    tr.roll_obs = []
    ok = True
    for ep, seed in enumerate((123, 456)):
        d = tr.collect(ar1_env(), seed=seed)
        msg, obs = tr.roll_obs[-1]["msg_out"], d["obs"].numpy()
        for i in range(4):
            exp = expected_stream(list(obs[:, i, 3]), MU, RHO, rung)
            if not np.allclose(msg[:, i, 0], exp, atol=1e-5):
                ok = False; print(f"    MISMATCH {rung} ep{ep} agent{i}: {msg[:4,i,0]} vs {exp[:4]}")
    ck(f"T2 {rung}: every message == f(stored obs) (scale+timing+episode-reset)", ok)

print("== T3: DR-Poisson oracle == this episode's true lambda ==")
tr = SIGNALTrainer({"msg_content": "oracle", "hidden": 16}, n_agents=4, obs_dim=4,
                   state_dim=133, adj=np.ones((4, 4)) / 4.0)
tr.roll_obs = []
lams, ok3 = [], True
for seed in (11, 12, 13):
    env = DemandRandomizedBeerGame({**BASE, "demand_type": "poisson"},
                                   lam_lo=4.0, lam_hi=24.0, p_shift=0.0, shift_scale=2.5)
    tr.collect(env, seed=seed)
    lam = float(env._dr_lambda); lams.append(lam)
    m = tr.roll_obs[-1]["msg_out"]
    ok3 &= bool(np.allclose(m[0], 0.0)) and bool(np.allclose(m[1:], lam, atol=1e-6))
ck(f"T3 oracle emits 0.0 at t=0 then the episode's true lambda (lams={['%.2f' % l for l in lams]})", ok3)
ck("T3 lambda varies across episodes (per-episode context, not a constant)", len(set(lams)) >= 2)

print("== T4: eval-path parity (SIGNALPolicy.act reproduces the identity) ==")
import subprocess, glob, os
if not glob.glob("weights_signal/run_signal_*_rungtest_eps/signal_checkpoint_best.pt"):
    subprocess.run([sys.executable, "agents/train_signal.py", "agent=signal", "seed=5",
                    "total_episodes=9", "agent.hidden=16", "agent.heldout_every=4",
                    "agent.heldout_episodes=1", "agent.batch_episodes=4", "agent.log_every=0",
                    "agent.train_env=ar1", "agent.ar1_rho=0.9", "agent.heldout_mode=ar1",
                    "agent.msg_content=eps", "agent.algorithm=rungtest_eps"],
                   env={**os.environ, "WANDB_MODE": "disabled"}, capture_output=True)
ck_path = glob.glob("weights_signal/run_signal_*_rungtest_eps/signal_checkpoint_best.pt")
ck("T4a eps checkpoint trained (the rung survives the FULL training pipeline)", bool(ck_path))
if ck_path:
    from agents.eval_signal import SIGNALPolicy, run_episode, AGENTS
    env = ar1_env()
    ckd = torch.load(ck_path[0], map_location="cpu", weights_only=False)
    pol = SIGNALPolicy(ckd, env)
    r = run_episode(pol, env, 777)                       # untouched full path must not crash
    ck("T4b run_episode completes on the eps policy", np.isfinite(r["cost"]))
    env = ar1_env(); obs, _ = env.reset(seed=888); pol.reset()
    cfgd = getattr(env, "_config", {}) or {}
    for _a in pol.actors:
        _a.demand_mu = float(cfgd.get("ar1_mu", 12.0)); _a.demand_rho = float(cfgd.get("ar1_rho", 0.0))
    pol._lam = None
    obs_hist, msg_hist = [], []
    while True:
        o_np = np.stack([obs[a] for a in AGENTS]); obs_hist.append(o_np[:, 3].copy())
        acts = pol.act(obs)
        msg_hist.append(pol.m_buf.clone().numpy()[:, 0])   # emitted this step (delay buffer)
        obs, _, _, trunc, _ = env.step({a: np.asarray(acts[a]).reshape(1) for a in AGENTS})
        if any(trunc.values()): break
    obs_hist, msg_hist = np.array(obs_hist), np.array(msg_hist)
    ok4 = all(np.allclose(msg_hist[:, i],
                          expected_stream(list(obs_hist[:, i]), MU, RHO, "eps"), atol=1e-5)
              for i in range(4))
    ck("T4c eval messages == f(obs stream) for all agents (eval lag state + ctx correct)", ok4)

print("== T5: dimension contract ==")
ck("T5 all four rungs are 1-dim in _FIXED_MSG_DIMS",
   all(_FIXED_MSG_DIMS[r] == 1 for r in RUNGS))
print(f"\nRESULT: {P} passed, {F} failed"); sys.exit(1 if F else 0)