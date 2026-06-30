"""
train_signal.py -- Hydra/W&B training entry point for the MINIMAL SIGNAL agent.
================================================================================
SIGNAL (Strategic Information-sharing for Game-theoretic, Networked, Adaptive
Logistics). The agent (signal_agent.SIGNALTrainer) owns the rollout (collect) and
the MAPPO update; THIS file owns only: env construction, the training loop, the
held-out-lambda gate, checkpointing, and W&B logging.

Dependency set (self-contained; no legacy DRACO modules):
  agents.signal_agent        the agent + trainer + AGENTS (self-contained: torch only)
  envs.beer_game_env         the env (interface verified: reset->(obs,info); action is a
                             FRACTION in [0,1]; step->(obs,rew,term,trunc,infos);
                             infos[a]['local_cost'], infos[a]['training_targets']['demand'];
                             obs=[inv,backlog,on_order,last_demand]; get_global_state()=133-d)
  envs.demand_randomization  DemandRandomizedBeerGame (per-episode Poisson-rate randomization)
  scripts.demand_families    AR(1)/NegBin/family envs (the autocorrelation study, H2)
  agents.topologies          ADJ matrices (the comm-topology sweep)
  scripts.c1_stats           load_rungs/mean_refs (BAR/CEILING ref loader)

The two study knobs ride on the agent config and need no code change here:
  agent.msg_content {dhat|ip|dhat_ip|learned}, agent.comm_topology, agent.use_comm  (communication)
  agent.srdqn_beta, agent.tau                                                        (economics)

Run examples:
  # cooperative, neighbour topology, share demand belief, stationary regime-inference task
  python agents/train_signal.py agent=signal agent.msg_content=dhat agent.comm_topology=neighbor \
         seed=10 agent.algorithm=signal_dhat_s10 total_episodes=15000
  # no-comm control (identical architecture, ADJ zeroed)
  python agents/train_signal.py agent=signal agent.use_comm=false seed=10 agent.algorithm=signal_nocomm_s10
  # self-interested + coordinating contract on the AR(1) regime
  python agents/train_signal.py agent=signal agent.train_env=ar1 agent.ar1_rho=0.9 \
         agent.srdqn_beta=0.0 agent.tau=9.5 seed=10 agent.algorithm=signal_contract_s10
================================================================================
"""
import os
import sys
import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

# Put the project ROOT on sys.path BEFORE importing agents./envs./scripts. -- every entry-point in
# this repo does this. This file lives in agents/, so two dirs up == the repo root. Without this line,
# `python agents/train_signal.py` cannot resolve the absolute package imports below (ModuleNotFoundError).
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.signal_agent import SIGNALTrainer, AGENTS                # the agent (self-contained: torch only)
from envs.beer_game_env import BeerGameParallelEnv
from envs.demand_randomization import DemandRandomizedBeerGame       # per-episode Poisson-rate randomization
from scripts.demand_families import make_demand_family_envs, make_ar1_rho_envs
from agents.topologies import get_adj

# Held-out CRN seed base -- MUST equal baselines.py / heldout_eval.py SEED_BASE so per-lambda
# costs are directly comparable to `python scripts/baselines.py regime`.
SEED_BASE = 100000
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------ env builders
def make_train_env(base, A):
    """TRAINING env from agent.train_env:
       dr_poisson (default): DemandRandomizedBeerGame, lambda ~ U[lo,hi]/episode (regime inference).
       ar1                 : AR(1) demand at a fixed rho (Lee-So-Tang autocorrelation regime).
       family              : per-episode family randomization (Poisson/NegBin/AR1)."""
    mode = A.get("train_env", "dr_poisson")
    if mode == "dr_poisson":
        return DemandRandomizedBeerGame(
            {**base, "demand_type": "poisson"},
            lam_lo=A.get("dr_lambda_lo", 4.0), lam_hi=A.get("dr_lambda_hi", 24.0),
            p_shift=A.get("dr_p_shift", 0.0), shift_scale=A.get("dr_shift_scale", 2.5))
    AR1, _NegBin, Family = make_demand_family_envs(BeerGameParallelEnv)
    if mode == "ar1":
        return AR1({**base, "demand_type": "poisson", "family": "ar1",
                    "ar1_mu": A.get("ar1_mu", 12.0), "ar1_rho": A.get("ar1_rho", 0.6),
                    "ar1_sigma": A.get("ar1_sigma", 3.0)})
    if mode == "family":
        keys = ["dr_families", "dr_lambda_lo", "dr_lambda_hi", "nb_mu_lo", "nb_mu_hi",
                "nb_dispersion_lo", "nb_dispersion_hi", "ar1_mu_lo", "ar1_mu_hi",
                "ar1_rho_lo", "ar1_rho_hi", "ar1_sigma"]
        return Family({**base, "demand_type": "poisson", **{k: A[k] for k in keys if k in A}})
    raise ValueError(f"unknown agent.train_env '{mode}'")


def make_heldout_envs(base, A):
    """Stationary held-out regimes for the gate. Default: poisson at VALIDATION lambdas (leakage-safe;
    the C1/eval TEST lambdas are scored post-hoc, never used for selection). For the AR(1) study,
    gate on held-out rho (disjoint from the test rho grid)."""
    if A.get("heldout_mode", "poisson") == "ar1":
        AR1, _, _ = make_demand_family_envs(BeerGameParallelEnv)
        return make_ar1_rho_envs(AR1, base, rhos=tuple(A.get("heldout_ar1_rhos", [0.15, 0.45, 0.75])),
                                 mu=A.get("ar1_mu", 12.0), sigma=A.get("ar1_sigma", 3.0))
    return {float(l): DemandRandomizedBeerGame({**base, "demand_type": "poisson"},
                                               lam_lo=float(l), lam_hi=float(l), p_shift=0.0)
            for l in A.get("heldout_lambdas", [8, 12, 16, 20])}


# ------------------------------------------------------------------ refs + gate
def load_refs(A):
    """BAR (fixed_ref) and CEILING (oracle_ref) from results/baselines_regime_v2.json, averaged over
    the gate lambdas. Falls back to config scalars if the json is absent. Regenerate the json with the
    SAME cost model (env.penalty_at_retailer_only) as training, or the gap is meaningless."""
    path = os.path.join(_ROOT, "results", "baselines_regime_v2.json")
    try:
        from scripts.c1_stats import load_rungs, mean_refs
        m = mean_refs(load_rungs(path), A.get("heldout_lambdas", None))
        bar = m.get("BAR_static", A.get("heldout_fixed_ref", 4726.0))
        ceil = m.get("Oracle", A.get("heldout_oracle_ref", 2202.0))
        print(f"[signal] refs from baselines_regime_v2.json: BAR={bar:.1f}  CEILING={ceil:.1f}")
        return float(bar), float(ceil)
    except Exception as ex:                                          # noqa: BLE001 -- want any failure -> fallback
        bar = float(A.get("heldout_fixed_ref", 4726.0)); ceil = float(A.get("heldout_oracle_ref", 2202.0))
        print(f"[signal] refs json unavailable ({ex}); using config scalars BAR={bar:.1f} CEILING={ceil:.1f}")
        return bar, ceil


@torch.no_grad()
def heldout_eval(trainer, heldout_envs, episodes, fixed_ref, oracle_ref):
    """Deterministic TEAM cost per held-out regime vs BAR/CEILING, via the agent's OWN
    collect(deterministic=True) (byte-for-byte the trained policy). Team cost per episode =
    sum over agents,steps of env local_cost == the TRUE objective (NOT the shaped reward). CRN:
    reset at SEED_BASE + e (matches baselines). Gap_Recovered = (BAR - cost)/(BAR - CEILING):
    >=0 beats the deployable fixed bar (the result exists), 1 matches the oracle, <0 no inference."""
    per = {}
    for key, env in heldout_envs.items():
        costs = [float(trainer.collect(env, seed=SEED_BASE + e, deterministic=True)["cost"].sum().item())
                 for e in range(episodes)]
        per[key] = float(np.mean(costs))
    mean_cost = float(np.mean(list(per.values())))
    denom = max(1e-6, fixed_ref - oracle_ref)
    log = {f"Eval/{('rho_' if isinstance(k, float) and k <= 1.0 else 'lam_')}{k:g}_Cost": v
           for k, v in per.items()}
    log["Eval/Mean_Cost"] = mean_cost
    log["Eval/Gap_Recovered"] = (fixed_ref - mean_cost) / denom
    log["Eval/vs_BAR_pct"] = 100.0 * (fixed_ref - mean_cost) / max(1e-6, fixed_ref)
    return log, mean_cost


# ------------------------------------------------------------------ main loop
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    A = OmegaConf.to_container(cfg.agent, resolve=True)             # plain dict (the env + trainer want .get)
    base = OmegaConf.to_container(cfg.env, resolve=True)
    run = wandb.init(project="BeerGame_Research", name=A["algorithm"],
                     config=OmegaConf.to_container(cfg, resolve=True))
    device = "cuda" if (A.get("use_gpu", False) and torch.cuda.is_available()) else "cpu"

    # --- envs ---
    train_env = make_train_env(base, A)
    heldout_envs = make_heldout_envs(base, A)
    obs0, _ = train_env.reset(seed=cfg.seed)                        # reset BEFORE get_global_state()
    obs_dim = int(np.asarray(obs0[AGENTS[0]]).shape[0])            # 4
    state_dim = int(train_env.get_global_state().shape[0])         # 133 (CTDE critic input)

    # --- communication wiring: real ADJ, or a ZERO ADJ for the no-comm control (identical
    #     architecture, incoming == 0, so comm-vs-no-comm differ ONLY in whether messages flow) ---
    n = len(AGENTS)
    adj = (get_adj(A.get("comm_topology", "neighbor")).numpy() if A.get("use_comm", True)
           else np.zeros((n, n), dtype=np.float32))

    trainer = SIGNALTrainer(A, n_agents=n, obs_dim=obs_dim, state_dim=state_dim, adj=adj, device=device)
    print(f"[signal] obs_dim={obs_dim} state_dim={state_dim} msg_content={A.get('msg_content')} "
          f"topology={A.get('comm_topology') if A.get('use_comm', True) else 'NONE(zeroADJ)'} "
          f"beta={A.get('srdqn_beta')} tau={A.get('tau')} params="
          f"{sum(p.numel() for p in trainer.params):,}")

    fixed_ref, oracle_ref = load_refs(A)
    run_dir = os.path.join(_ROOT, "weights_signal", f"run_signal_{run.id}_{A['algorithm']}")
    os.makedirs(run_dir, exist_ok=True)

    batch_eps = int(A.get("batch_episodes", 8))
    warm_up = int(A.get("warm_up_episodes", 0))                     # 0 for SIGNAL: belief trains with policy
    eval_every = int(A.get("heldout_every", 200))
    eval_eps = int(A.get("heldout_episodes", 8))
    best = float("inf")
    train_rng = np.random.default_rng(cfg.seed + 12345)            # training seeds, DISJOINT from SEED_BASE

    batch = []
    for ep in range(int(cfg.total_episodes)):
        batch.append(trainer.collect(train_env, seed=int(train_rng.integers(0, 2**31 - 1))))
        log = {}
        if len(batch) >= batch_eps:
            if ep >= warm_up:
                a_loss, c_loss = trainer.update(batch)
                log.update({"train/actor_loss": a_loss, "train/critic_loss": c_loss})
            batch = []

        if ep > warm_up and ep % eval_every == 0:
            elog, mean_cost = heldout_eval(trainer, heldout_envs, eval_eps, fixed_ref, oracle_ref)
            log.update(elog)
            if mean_cost < best:                                   # checkpoint on the held-out gate ONLY
                best = mean_cost
                torch.save({"actors": [ac.state_dict() for ac in trainer.actors],
                            "critic": trainer.critic.state_dict(),
                            "config": A, "adj": adj.tolist(),
                            "obs_dim": obs_dim, "state_dim": state_dim,
                            "msg_content": A.get("msg_content"), "episode": ep,
                            "best_heldout_cost": best},
                           os.path.join(run_dir, "signal_checkpoint_best.pt"))
                log["Eval/best_heldout_cost"] = best
                print(f"[signal] ep {ep}: held-out mean cost {mean_cost:.1f}  "
                      f"Gap_Recovered {elog['Eval/Gap_Recovered']:+.3f}  (checkpoint saved)")

        if log:
            log["episode"] = ep
            wandb.log(log)

    wandb.finish()
    print(f"[signal] done. best held-out mean cost = {best:.1f}   ->  {run_dir}")


if __name__ == "__main__":
    main()