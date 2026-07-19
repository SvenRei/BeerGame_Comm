import os
import sys
from collections import deque
import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

# Put the project root on sys.path before importing agents/envs/scripts (this file lives in agents/,
# so two directories up is the root). Without it, `python agents/train_signal.py` cannot resolve the
# absolute package imports below.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.signal_agent import SIGNALTrainer, AGENTS                # the agent (self-contained: torch only)
from envs.beer_game_env import BeerGameParallelEnv
from envs.demand_randomization import DemandRandomizedBeerGame       # per-episode Poisson-rate randomization
from scripts.demand_families import make_demand_family_envs, make_ar1_rho_envs
from agents.topologies import get_adj
from agents import signal_csvlog                                    # scientific scalar logger (default-off)

# Held-out CRN seed base; must equal baselines.py SEED_BASE and eval_signal.py HELDOUT_SEED_BASE so
# per-lambda costs are directly comparable to `python scripts/baselines.py regime`.
SEED_BASE = 100000   # GATE (checkpoint-selection) space ONLY. Final eval lives at 500000+
#                      (eval_signal/baselines/qmix_dump) -- three disjoint seed spaces (fix #1).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------ env builders
def make_train_env(base, A):
    """Training env selected by agent.train_env:
       dr_poisson (default): DemandRandomizedBeerGame, lambda ~ U[lo,hi] per episode (regime inference).
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
    """Stationary held-out regimes for the gate. Default: Poisson at the validation lambdas (leakage-
    safe; the C1/eval test lambdas are scored post-hoc, never used for selection). For the AR(1) study,
    gate on held-out rho disjoint from the test rho grid."""
    if A.get("heldout_mode", "poisson") == "ar1":
        AR1, _, _ = make_demand_family_envs(BeerGameParallelEnv)
        return make_ar1_rho_envs(AR1, base, rhos=tuple(A.get("heldout_ar1_rhos", [0.15, 0.45, 0.75])),
                                 mu=A.get("ar1_mu", 12.0), sigma=A.get("ar1_sigma", 3.0))
    return {float(l): DemandRandomizedBeerGame({**base, "demand_type": "poisson"},
                                               lam_lo=float(l), lam_hi=float(l), p_shift=0.0)
            for l in A.get("heldout_lambdas", [8, 12, 16, 20])}


# ------------------------------------------------------------------ refs + gate
def load_refs(A):
    """BAR (fixed_ref) and CEILING (oracle_ref) for the held-out gate, resolved in three tiers so the
    printed refs are never silently wrong:
      1. per-lambda means over the gate lambdas from results/baselines_regime_v2.json (used when those
         exact lambdas are present);
      2. otherwise the json's overall means over whatever lambdas it holds (real benchmark numbers on a
         different lambda grid, flagged as such);
      3. otherwise the config scalars heldout_fixed_ref/heldout_oracle_ref (json absent/unreadable).
    These refs are only a monotone normalizer for checkpoint selection; the headline gap is scored
    post-hoc by eval_signal on the test lambdas. Regenerate the json with the same cost model
    (env.penalty_at_retailer_only) as training, or the gap is meaningless."""
    path = os.path.join(_ROOT, "results", "baselines_regime_v2.json")
    cfg_bar = float(A.get("heldout_fixed_ref", 4726.0)); cfg_ceil = float(A.get("heldout_oracle_ref", 2202.0))
    try:
        from scripts.c1_stats import load_rungs, mean_refs
        rungs = load_rungs(path)
        gate_lams = A.get("heldout_lambdas", None)
        m = mean_refs(rungs, gate_lams)                              # tier 1: gate lambdas present in json
        if "BAR_static" in m and "Oracle" in m:
            bar, ceil = float(m["BAR_static"]), float(m["Oracle"])
            print(f"[signal] refs from baselines_regime_v2.json @ gate lambdas {gate_lams}: "
                  f"BAR={bar:.1f}  CEILING={ceil:.1f}")
            return bar, ceil
        full = mean_refs(rungs, None)                               # tier 2: json's own lambda grid
        if "BAR_static" in full and "Oracle" in full:
            bar, ceil = float(full["BAR_static"]), float(full["Oracle"])
            json_lams = sorted(next(iter(rungs.values())))
            print(f"[signal] gate lambdas {gate_lams} not in baselines_regime_v2.json; using its "
                  f"OVERALL means (lambdas {[f'{l:g}' for l in json_lams]}): BAR={bar:.1f} CEILING={ceil:.1f}")
            return bar, ceil
        raise KeyError("no BAR_static/Oracle rungs in baselines_regime_v2.json")
    except Exception as ex:                                          # noqa: BLE001 -- any failure -> config scalars
        print(f"[signal] baselines refs unavailable ({type(ex).__name__}: {ex}); "
              f"using config scalars BAR={cfg_bar:.1f} CEILING={cfg_ceil:.1f}")
        return cfg_bar, cfg_ceil


@torch.no_grad()
def heldout_eval(trainer, heldout_envs, episodes, fixed_ref, oracle_ref, collect_stats=False):
    """Deterministic team cost per held-out regime vs BAR/CEILING, via the agent's own
    collect(deterministic=True) (byte-for-byte the trained policy). Team cost per episode is the sum
    over agents and steps of env local_cost, i.e. the true objective (not the shaped reward). CRN:
    reset at SEED_BASE + e (matches baselines). Gap_Recovered = (BAR - cost)/(BAR - CEILING): >=0 beats
    the deployable fixed bar, 1 matches the oracle, <0 means no inference.

    collect_stats (SIGNAL_CSVLOG only) additionally returns the raw eval buffers/rolls for the scalar
    logger. The per-regime mean and mean_cost use the same collect() calls with the same seeds either
    way, so gating (best-checkpoint selection) is unaffected; the extra data is pure observation off
    the deterministic (no-sampling) rollout."""
    per = {}
    stats = ({"per_ep_total": [], "per_echelon": [], "buffers": [], "rolls": []}
             if collect_stats else None)
    for key, env in heldout_envs.items():
        if not collect_stats:
            costs = [float(trainer.collect(env, seed=SEED_BASE + e, deterministic=True)["cost"].sum().item())
                     for e in range(episodes)]
        else:
            costs = []
            for e in range(episodes):
                trainer.roll_obs = []                          # collect() appends one obs-only record
                buf = trainer.collect(env, seed=SEED_BASE + e, deterministic=True)
                costs.append(float(buf["cost"].sum().item()))
                stats["per_ep_total"].append(costs[-1])
                stats["per_echelon"].append(buf["cost"].sum(0).cpu().numpy())     # [N] per-echelon $
                stats["buffers"].append(buf)
                stats["rolls"].append(trainer.roll_obs[0] if trainer.roll_obs else None)
            trainer.roll_obs = None
        per[key] = float(np.mean(costs))
    mean_cost = float(np.mean(list(per.values())))
    denom = max(1e-6, fixed_ref - oracle_ref)
    log = {f"Eval/{('rho_' if isinstance(k, float) and k <= 1.0 else 'lam_')}{k:g}_Cost": v
           for k, v in per.items()}
    log["Eval/Mean_Cost"] = mean_cost
    log["Eval/Gap_Recovered"] = (fixed_ref - mean_cost) / denom
    log["Eval/vs_BAR_pct"] = 100.0 * (fixed_ref - mean_cost) / max(1e-6, fixed_ref)
    return (log, mean_cost, stats) if collect_stats else (log, mean_cost)


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

    # --- communication wiring: the real ADJ, or a zero ADJ for the no-comm control (identical
    #     architecture with incoming == 0, so comm vs no-comm differ only in whether messages flow) ---
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

    # --- Scalar logger (default off; SIGNAL_CSVLOG=1 to enable). Pure observation: reads scalars the
    #     update/eval paths already computed and changes no registered number.
    csvlog = None
    if signal_csvlog.csvlog_enabled():
        meta = signal_csvlog.make_run_meta(OmegaConf.to_container(cfg, resolve=True), A, run.id,
                                           sum(p.numel() for p in trainer.params),
                                           obs_dim, state_dim, fixed_ref, oracle_ref)
        csvlog = signal_csvlog.SignalCSVLogger(run_dir, meta)
        print(f"[signal] SIGNAL_CSVLOG=1: scalar logger -> {run_dir}/metrics_*.csv", flush=True)

    batch_eps = int(A.get("batch_episodes", 8))
    warm_up = int(A.get("warm_up_episodes", 0))                     # 0 for SIGNAL: belief trains with policy
    eval_every = int(A.get("heldout_every", 200))
    eval_eps = int(A.get("heldout_episodes", 8))
    log_every = int(A.get("log_every", 100))                        # heartbeat cadence in episodes (0 = silent between gates)
    patience = int(A.get("patience", 0))                            # early stop: episodes w/o held-out improvement (0 = off)
    best = float("inf"); best_ep = -1                               # we KEEP the best-gated checkpoint, never the last one
    best_payload = None                                             # the best checkpoint dict, kept for budget milestones
    # Budget milestones (substitution curve): at each milestone M, snapshot the deployable-at-budget-M
    # model (the best-gated checkpoint so far) to signal_checkpoint_budget{M}.pt. V(budget) from these
    # feeds the sharing-vs-inference substitution analysis (comm_stats.py curve). Milestones never
    # change training; they only copy the current best payload.
    milestones = sorted(int(m) for m in (A.get("budget_milestones") or []))
    mi = 0

    def _save_budget(m, ep_now, truncated=False):
        if best_payload is None:
            print(f"[signal] budget milestone {m}: no gated checkpoint yet (gate cadence "
                  f"{eval_every} > milestone?) -- skipped", flush=True)
            return
        torch.save({**best_payload, "budget_episodes": int(m), "trained_episodes": int(ep_now),
                    "budget_truncated": bool(truncated)},
                   os.path.join(run_dir, f"signal_checkpoint_budget{int(m)}.pt"))
        print(f"[signal] budget milestone {m}: snapshot of best@ep{best_ep} saved"
              f"{' (training ended earlier; deployable-at-budget = final best)' if truncated else ''}",
              flush=True)

    train_rng = np.random.default_rng(cfg.seed + 12345)            # training seeds, DISJOINT from SEED_BASE

    batch = []
    recent_cost = deque(maxlen=50)                                  # running TRAIN-episode team cost (liveness + learning signal)
    last_a = last_c = float("nan")
    ep = -1                                                         # defined even if total_episodes == 0 (tail milestones)
    for ep in range(int(cfg.total_episodes)):
        buf = trainer.collect(train_env, seed=int(train_rng.integers(0, 2**31 - 1)))
        recent_cost.append(float(buf["cost"].sum().item()))
        batch.append(buf)
        log = {}
        if len(batch) >= batch_eps:
            if ep >= warm_up:
                pre_dial = None
                if csvlog is not None:                         # arm the pure-observation sinks
                    pre_dial = signal_csvlog.snapshot_dial(trainer)
                    trainer.upd_obs = {}
                last_a, last_c = trainer.update(batch)
                log.update({"train/actor_loss": last_a, "train/critic_loss": last_c})
                if csvlog is not None:
                    csvlog.log_update(ep, last_a, last_c, trainer.upd_obs, pre_dial, trainer)
                    trainer.upd_obs = None
            batch = []

        # Heartbeat, so a long improvement-free stretch between gates is not mistaken for a hang.
        if log_every and ep % log_every == 0:
            print(f"[signal] ep {ep}/{int(cfg.total_episodes)}  train_team_cost~{np.mean(recent_cost):.0f}"
                  f"  a_loss={last_a:+.3f} c_loss={last_c:.0f}", flush=True)

        if ep > warm_up and ep % eval_every == 0:
            if csvlog is not None:
                elog, mean_cost, gate_stats = heldout_eval(
                    trainer, heldout_envs, eval_eps, fixed_ref, oracle_ref, collect_stats=True)
            else:
                elog, mean_cost = heldout_eval(trainer, heldout_envs, eval_eps, fixed_ref, oracle_ref)
            log.update(elog)
            improved = mean_cost < best
            if improved:                                           # save the best held-out point (not the last)
                best = mean_cost; best_ep = ep
                # Clone the tensors: state_dict() values alias the live parameters, which Adam mutates
                # in place, so an un-cloned payload would drift to the current weights and later budget
                # snapshots would serialize live-at-milestone weights while claiming best@ep (measured:
                # 1.5e-3 max-abs drift within 5 episodes). The clone freezes the gated best at one extra
                # parameter copy per improving gate.
                best_payload = {"actors": [{k: v.clone() for k, v in ac.state_dict().items()}
                                           for ac in trainer.actors],
                                "critic": {k: v.clone() for k, v in trainer.critic.state_dict().items()},
                                "config": A, "adj": adj.tolist(),
                                "env": base,                # cfg.env as trained (cost model, lead times, ...);
                                #                             eval_signal rebuilds its envs from this, so a
                                #                             canonical-cost (penalty_at_retailer_only) or
                                #                             lead-time run is otherwise scored on the default
                                #                             env against the wrong references.
                                "obs_dim": obs_dim, "state_dim": state_dim,
                                "msg_content": A.get("msg_content"), "episode": ep,
                                "seed": int(cfg.seed),      # top-level Hydra seed (not in the agent cfg); per-seed
                                #                             dumps key filenames off this, so without it every
                                #                             checkpoint mislabels as seed0 and CRN pairing breaks.
                                "best_heldout_cost": best}
                torch.save(best_payload, os.path.join(run_dir, "signal_checkpoint_best.pt"))
                log["Eval/best_heldout_cost"] = best
                print(f"[signal] ep {ep}: held-out mean cost {mean_cost:.1f}  "
                      f"Gap_Recovered {elog['Eval/Gap_Recovered']:+.3f}  (checkpoint saved)")
            # Record the held-out row on every gate, unconditionally (before any early stop), so the CSV
            # holds the true learning curve rather than the best-only monotone envelope that parsing the
            # improvement-gated stdout would yield.
            if csvlog is not None:
                csvlog.log_gate(ep, elog, mean_cost, gate_stats, best, best_ep,
                                (ep - best_ep) if best_ep >= 0 else -1, trainer, A)
            if (not improved) and patience and best_ep >= 0 and (ep - best_ep) >= patience:
                print(f"[signal] EARLY STOP at ep {ep}: no held-out improvement for {ep - best_ep} eps "
                      f"(best {best:.1f} @ ep {best_ep}; checkpoint kept)", flush=True)
                break

        while mi < len(milestones) and (ep + 1) >= milestones[mi]:  # budget milestone crossed
            _save_budget(milestones[mi], ep + 1)
            mi += 1

        if log:
            log["episode"] = ep
            wandb.log(log)

    # Milestones beyond the last trained episode (early stop / short run): the deployable-at-budget-M
    # model is the final best (training would not have improved further), stamped budget_truncated.
    while mi < len(milestones):
        _save_budget(milestones[mi], ep + 1, truncated=True)
        mi += 1

    if csvlog is not None:
        csvlog.close()
    wandb.finish()
    print(f"[signal] done. best held-out mean cost = {best:.1f} @ ep {best_ep}   ->  {run_dir}", flush=True)


if __name__ == "__main__":
    main()