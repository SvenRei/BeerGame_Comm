"""
Evaluation and measurement layer for the SIGNAL agent.
================================================================================
SIGNAL (Strategic Information-sharing for Game-theoretic, Networked, Adaptive
Logistics). Loads a `signal_checkpoint_best.pt` produced by train_signal.py into a
SIGNALPolicy and runs the paper's measurements. The eval is keyed to the SIGNAL actor
architecture (GRU belief + linear base-stock head + demand readout + message ladder),
which has no encoder, latent z, structured-vs-mlp head, or plug-in belief.

The policy is the trained policy: SIGNALPolicy.act() reproduces SIGNALTrainer.collect()'s
per-step forward exactly -- incoming = ADJ @ m_prev (one-step delay), per-agent GRU belief
update, base-stock head -> S, order = clip(S - IP), message = ladder(d_hat, IP). The saved
ADJ (already zeroed for a use_comm=false checkpoint) is used directly, so messages route
exactly as trained; ablate=True is the comm-off counterfactual on the same weights.

The env is the trained env: every probe rebuilds its env from the env config persisted in
the checkpoint (train_signal saves cfg.env), resolved by resolve_env_base with precedence
CLI --env-json > ckpt['env'] > module ENV_BASE fallback (pre-fix checkpoints only, with a
warning). Without this, a canonical-cost (penalty_at_retailer_only) or lead-time run would
be silently scored on the default env against the wrong references.

What it computes (each part names the question it answers):
  * Standard benchmark (per regime poisson/black_swan/extreme_chaos): mean/CVaR cost,
    bullwhip, Type-1 (alpha) service, Type-2 (beta) fill, jitter, and the zero-message
    comm value (paired Wilcoxon).
  * Per-stage dashboard: local + cumulative bullwhip, net-stock amplification, on-hand/
    backlog, service/fill, per-echelon cost (Chen 2000; Lee 1997; Disney-Towill 2003).
  * Regime-uncertainty (C1): per-lambda cost + Gap_Recovered vs the BAR/Oracle references
    from `python scripts/baselines.py regime`.
  * Comm-value decomposition (the Lee mechanism): comm ON vs OFF, CRN-paired -- per-stage
    bullwhip/inventory/backlog split and the per-echelon upstream forecast-error delta
    vs customer demand. delta>0 => the shared signal cuts upstream forecast error.
  * Honesty probe: each message component vs the sender's true demand / inventory position.
  * Positive listening: causal dS/d(told) per message component (components swept one at a
    time, the others held at their actually-received values).
  * Message intervention (--interventions): do(m) on the channel -- honest vs time-shuffled
    vs cross-episode vs zeroed message streams, CRN-paired. The causal content test: if
    shuffled/cross degrade toward zeroed, the value is state-correlated information, not
    channel presence. Includes an identity-replay self-check (replaying the honest stream
    reproduces its cost).
  * Producers: --dump-c1 (per-seed {lambda: cost} for scripts/c1_stats.py) and --dump-comm
    (per-seed {lambda: cost} + per-echelon forecast error for scripts/comm_stats.py).
    --dump-comm + --dump-ar1 "0,0.3,0.6,0.9" keys the same dump format by AR(1) rho instead
    of Poisson lambda (the H2 producer; comm_stats' loaders are key-agnostic).

Two seed spaces (do not conflate):
  SEED_BASE=2000          -> standard-benchmark / message rollouts.
  HELDOUT_SEED_BASE=1e5   -> C1 / dump rollouts; == baselines.py SEED_BASE, so SIGNAL and the
                             reference rungs are scored on the same demand draws (CRN).

Usage:
  python agents/eval_signal.py --ckpt weights_signal/run_signal_<id>_<algo>/signal_checkpoint_best.pt
  python agents/eval_signal.py --ckpt ... --episodes 100 --full
  python agents/eval_signal.py --ckpt ... --dump-c1   results/signal_c1
  python agents/eval_signal.py --ckpt ... --dump-comm results/comm_on
  python agents/eval_signal.py --ckpt ... --dump-comm results/comm_on_ar1 --dump-ar1 "0,0.3,0.6,0.9"
  python agents/eval_signal.py --ckpt ... --env-json conf/env_canonical.json   # override the saved env
  python agents/eval_signal.py --ckpt ... --ar1 --interventions                # causal content test, AR(1)
  python agents/eval_signal.py --ckpt ... --ar1 --episodes 40 --dump-iv results/iv_arm   # per-seed gate dump
      # then cross-seed: python scripts/comm_stats.py interventions --dir results/iv_arm
"""
import os
import sys
import json
import argparse
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.beer_game_env import BeerGameParallelEnv
from envs.demand_randomization import DemandRandomizedBeerGame
from agents.signal_agent import SIGNALActor, order_from_S, msg_dim_of, AGENTS
from agents.topologies import get_adj
from scripts.demand_families import make_demand_family_envs

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCENARIOS = ["poisson", "black_swan", "extreme_chaos"]
SEED_BASE = 2000
HELDOUT_SEED_BASE = 100000          # matches baselines.py SEED_BASE (CRN with the BAR/CEILING)
HELDOUT_LAMBDAS = [6.0, 10.0, 14.0, 18.0, 22.0]
ENV_BASE = {"horizon": 50, "max_order": 100}   # fallback only: probes use resolve_env_base(ckpt)
H_COST = 0.5     # fallback only: run_episode reads h from the live env (env.h); used only when an
#                  env object lacks that attribute.
B_COST = 1.0     # fallback only: as above (env.b), including the canonical-cost flag.

# Message-component layout per content type (which slot carries which named signal).
# 'dhat' and 'ip' messages ride at natural units (see SIGNALActor.message), so the honesty probe
# compares the component directly to demand / IP with _MSG_SCALE = 1.0 (a prior /100 scaling made
# the message causally inert; see positive_listening).
_MSG_SCALE = 1.0


def resolve_env_base(ckpt, env_json=None):
    """Resolve the env config the checkpoint was trained on, with explicit precedence:
         CLI --env-json  >  ckpt['env'] (train_signal persists cfg.env)  >  module ENV_BASE.
    The fallback exists only for pre-fix checkpoints and warns, because a canonical-cost
    (penalty_at_retailer_only) or lead-time run would otherwise be silently scored on the
    default env against the wrong references. demand_type is stripped so each probe sets its
    own demand regime on top of this base."""
    if env_json:
        with open(env_json) as f:
            base = json.load(f)
        print(f"  env config <- {env_json} (CLI override)")
    elif isinstance(ckpt.get("env"), dict) and ckpt["env"]:
        base = dict(ckpt["env"])
        print(f"  env config <- checkpoint (as trained): h={base.get('holding_cost', 0.5)} "
              f"b={base.get('backorder_cost', 1.0)} "
              f"penalty_at_retailer_only={base.get('penalty_at_retailer_only', False)}")
    else:
        base = dict(ENV_BASE)
        print("  [WARN] checkpoint has no saved env config (pre-fix trainer); using module ENV_BASE "
              "defaults. If this run used a non-default env (canonical cost, lead-time ranges), "
              "re-run with --env-json or retrain with the updated train_signal.py.")
    base.pop("demand_type", None)               # each probe sets its own demand regime
    return base


def make_ar1_env(rho, mu=12.0, sigma=3.0, env_base=None):
    """AR(1) eval env at a fixed rho, built exactly as train_signal builds the ar1 train env
    (demand_type='poisson' + family='ar1'). Measures an AR(1)-trained checkpoint in its training
    regime; evaluating it on Poisson (the default probes) tests white noise, where comm value is
    null by construction (Axsater-Rosling)."""
    base = dict(ENV_BASE if env_base is None else env_base)
    AR1, _NegBin, _Family = make_demand_family_envs(BeerGameParallelEnv)
    return AR1({**base, "demand_type": "poisson", "family": "ar1",
                "ar1_mu": float(mu), "ar1_rho": float(rho), "ar1_sigma": float(sigma)})


def _safe_ratio(num, den):
    return float(num / den) if den and np.isfinite(den) and den != 0 else float("nan")


def _component_layout(content):
    """Return {component_name: column_index} for a message content type. 'learned' has no
    named ground truth, so its channels are reported against both demand and IP."""
    if content == "raw":
        return {"raw": 0}
    if content == "dhat":
        return {"dhat": 0}
    if content == "ip":
        return {"ip": 0}
    if content == "dhat_ip":
        return {"dhat": 0, "ip": 1}
    return {}                                       # learned: handled separately


# ==============================================================================
# SIGNAL policy (reproduces SIGNALTrainer.collect's per-step forward)
# ==============================================================================
class SIGNALPolicy:
    def __init__(self, ckpt, env, ablate=False, deterministic=True):
        self.env = env
        self.ablate = ablate
        self.deterministic = deterministic
        self.max_order = float(env.max_order)
        self.N = len(AGENTS)
        cfg = ckpt.get("config", {})                # train_signal saves the agent config dict directly
        self.cfg = cfg
        self.msg_override = None                    # optional [T,N,msg] array: do(m_{t-1}) intervention

        self.hidden = int(cfg.get("hidden", 64))
        self.content = ckpt.get("msg_content", cfg.get("msg_content", "dhat"))
        self.learned_msg_dim = int(cfg.get("learned_msg_dim", 3))
        self.msg_dim = msg_dim_of(self.content, self.learned_msg_dim)
        self.use_comm = bool(cfg.get("use_comm", True))
        self.comm_topology = cfg.get("comm_topology", "neighbor")
        self.obs_dim = int(ckpt.get("obs_dim", env.observation_space(AGENTS[0]).shape[0]))

        # ADJ: prefer the exact matrix saved at train time (already zeroed for a no-comm run);
        # else rebuild from the topology name, zeroed if use_comm=false. ablate zeros it live.
        if ckpt.get("adj") is not None:
            self.adj = torch.tensor(ckpt["adj"], dtype=torch.float32, device=DEVICE)
        else:
            self.adj = (get_adj(self.comm_topology).to(DEVICE) if self.use_comm
                        else torch.zeros(self.N, self.N, device=DEVICE))

        self.use_dhat_head = bool(cfg.get("use_dhat_head", False))   # must match training (changes head input size)
        self.actors = []
        for sd in ckpt["actors"]:
            a = SIGNALActor(self.obs_dim, self.msg_dim, self.hidden, self.content,
                            self.learned_msg_dim, use_dhat_head=self.use_dhat_head).to(DEVICE)
            a.load_state_dict(sd)
            a.eval()
            self.actors.append(a)
        self.reset()

    def reset(self):
        self.h = [torch.zeros(1, 1, self.hidden, device=DEVICE) for _ in range(self.N)]
        self.m_buf = torch.zeros(self.N, self.msg_dim, device=DEVICE)
        self._t = 0                                 # step counter for msg_override indexing
        # self.msg_override is deliberately not cleared here: run_episode() calls reset()
        # internally, so the intervention runner sets the override before run_episode and clears it
        # after. override[t] replaces the previous-step messages m_{t-1} the router would deliver.
        self.last_S = np.zeros(self.N)
        self.last_dhat = np.full(self.N, np.nan)
        self.last_msg = np.zeros((self.N, self.msg_dim))
        self.last_ctx = None                        # (o_t, [hi per agent], incoming) for the listening probe

    @torch.no_grad()
    def act(self, obs):
        o_t = torch.tensor(np.stack([obs[a] for a in AGENTS]), dtype=torch.float32, device=DEVICE)  # [N,obs]
        if self.msg_override is not None and self._t < len(self.msg_override):
            m_prev = torch.as_tensor(self.msg_override[self._t], dtype=torch.float32,
                                     device=DEVICE).view(self.N, self.msg_dim)
        else:
            m_prev = self.m_buf
        incoming = self.adj @ m_prev                # [N,msg]
        if self.ablate:
            incoming = torch.zeros_like(incoming)

        S = torch.zeros(self.N, 1, device=DEVICE)
        dhat = np.full(self.N, np.nan)
        m_out = torch.zeros(self.N, self.msg_dim, device=DEVICE)
        hi_list = []
        for i in range(self.N):
            h_seq, self.h[i] = self.actors[i].belief(o_t[i].view(1, 1, -1),
                                                     incoming[i].view(1, 1, -1), self.h[i])
            hi = h_seq[-1]                           # [1,hidden]
            hi_list.append(hi)
            S_mu, S_std = self.actors[i].base_stock(o_t[i:i + 1], hi, incoming[i:i + 1])
            S[i] = S_mu if self.deterministic else torch.distributions.Normal(S_mu, S_std).sample()
            dhat[i] = float(self.actors[i].demand_estimate(hi).reshape(-1)[0].item())
            m_out[i] = self.actors[i].message(o_t[i:i + 1], hi).reshape(-1)

        order, _ = order_from_S(S, o_t, self.max_order)
        frac = (order / self.max_order).clamp(0.0, 1.0)
        self.last_S = S.detach().cpu().numpy().reshape(-1)
        self.last_dhat = dhat
        self.last_msg = m_out.detach().cpu().numpy()
        self.last_ctx = (o_t.detach(), hi_list, incoming.detach())
        self.m_buf = m_out.detach()                 # ONE-STEP message delay (matches collect)
        self._t += 1
        return {a: float(frac[i, 0].item()) for i, a in enumerate(AGENTS)}

    @torch.no_grad()
    def probe_S(self, agent_idx, msg_value, component=0):
        """Order-up-to S that actor `agent_idx` would output at the last step's (obs, belief) if
        component `component` of its incoming message were `msg_value` (message natural units),
        all other components held at their actually-received values. dS/d(message component)
        measures causal positive listening per named signal. Only the swept column is changed:
        setting every column to the scalar would, for dhat_ip, conflate dS/d(dhat) with dS/d(ip)."""
        o_t, hi_list, inc = self.last_ctx
        m = inc[agent_idx:agent_idx + 1].clone()    # observed incoming as the baseline
        m[0, int(component)] = float(msg_value)
        s_mu, _ = self.actors[agent_idx].base_stock(o_t[agent_idx:agent_idx + 1], hi_list[agent_idx], m)
        return float(s_mu.reshape(-1)[0].item())


# ==============================================================================
# Episode rollout + metrics (optional per-step trace for the message analyses)
# ==============================================================================
def run_episode(policy, env, seed, trace=False):
    torch.manual_seed(seed)
    obs, _ = env.reset(seed=seed)
    policy.reset()
    orders = {a: [] for a in AGENTS}
    demand = {a: [] for a in AGENTS}
    inv = {a: [] for a in AGENTS}
    back = {a: [] for a in AGENTS}
    ip = {a: [] for a in AGENTS}
    fill_d = {a: 0.0 for a in AGENTS}
    fill_m = {a: 0.0 for a in AGENTS}
    tot_cost = 0.0
    cust_series = []
    tr_dhat, tr_msg, tr_cust = [], [], []
    tr_dem, tr_ip = [], []
    while True:
        acts = policy.act(obs)
        for a in AGENTS:
            orders[a].append(int(np.floor(np.clip(acts[a], 0, 1) * env.max_order + 0.5)))
            o = obs[a]
            ip[a].append(float(o[0]) - float(o[1]) + float(o[2]))         # inventory position (pre-step)
        obs, _r, _t, truncs, infos = env.step({a: [acts[a]] for a in AGENTS})
        cust = float(env.current_incoming_order["retailer"])              # realized customer demand this step
        cust_series.append(cust)
        for a in AGENTS:
            tot_cost += infos[a]["local_cost"]
            inv[a].append(float(obs[a][0]))
            back[a].append(float(obs[a][1]))
            demand[a].append(float(env.current_incoming_order[a]))        # this stage's realized incoming
            tt = infos[a].get("training_targets", {})
            fill_d[a] += tt.get("demand", 0.0)
            fill_m[a] += tt.get("demand_met", 0.0)
        if trace:
            tr_dhat.append(policy.last_dhat.copy())
            tr_msg.append(policy.last_msg.copy())
            tr_cust.append(cust)
            tr_dem.append([demand[a][-1] for a in AGENTS])
            tr_ip.append([ip[a][-1] for a in AGENTS])
        if any(truncs.values()):
            break

    all_inv = np.concatenate([inv[a] for a in AGENTS])
    all_back = np.concatenate([back[a] for a in AGENTS])
    cust_arr = np.asarray(cust_series, float)
    cust_var = float(np.var(cust_arr))
    # Cost-split parameters from the live env (single source of truth; canonical-cost aware).
    # The per-stage cost split must reproduce what the env actually charged: under
    # penalty_at_retailer_only, upstream stages pay holding but not backorder penalties.
    h_cost = float(getattr(env, "h", H_COST))
    b_cost = float(getattr(env, "b", B_COST))
    pen_ret_only = bool(getattr(env, "_penalty_at_retailer_only", False))
    bw_stage, ovar, dvar, bw_cum, nsamp = {}, {}, {}, {}, {}
    mean_inv, mean_back, mean_netstock = {}, {}, {}
    serv_alpha, fill_beta_s, stockout_freq, hold_c, back_c, total_c = {}, {}, {}, {}, {}, {}
    zero_of, max_of = {}, {}
    for a in AGENTS:
        o = np.asarray(orders[a], float); d = np.asarray(demand[a], float)
        # Censoring fingerprint (Proposition 1 mechanism): with a base-stock head the order is
        # clip(S - IP, 0, max_order), so o==0 is the lower-censoring event (S <= IP) and o==max
        # the upper one. Censoring makes the order stream a non-invertible statistic of demand --
        # exactly where the Raghunathan/GGS redundancy null breaks. P(censor) should rise with
        # rho (bullwhip widens the order distribution) and predict V across seeds.
        zero_of[a] = float(np.mean(o == 0)); max_of[a] = float(np.mean(o >= float(env.max_order)))
        iv = np.asarray(inv[a], float); bk = np.asarray(back[a], float)
        ns = iv - bk
        ovar[a] = float(np.var(o)); dvar[a] = float(np.var(d))
        bw_stage[a] = _safe_ratio(ovar[a], dvar[a])
        bw_cum[a] = _safe_ratio(ovar[a], cust_var)
        nsamp[a] = _safe_ratio(float(np.var(ns)), cust_var)
        mean_inv[a] = float(iv.mean()); mean_back[a] = float(bk.mean()); mean_netstock[a] = float(ns.mean())
        serv_alpha[a] = float(np.mean(bk == 0))
        fill_beta_s[a] = _safe_ratio(fill_m[a], fill_d[a])
        stockout_freq[a] = float(np.mean(bk > 0))
        hold_c[a] = h_cost * float(iv.sum())
        bc = b_cost * float(bk.sum())
        back_c[a] = 0.0 if (pen_ret_only and a != "retailer") else bc    # matches env local_cost
        total_c[a] = hold_c[a] + back_c[a]
    hold_total = float(sum(hold_c.values())); back_total = float(sum(back_c.values()))

    out = {
        "cost": tot_cost,
        "cust_sum": float(cust_arr.sum()),
        "cost_per_demand": _safe_ratio(tot_cost, float(cust_arr.sum())),
        "bw_overall": _safe_ratio(np.var(orders["manufacturer"]), np.var(demand["retailer"])),
        "avg_inv": float(all_inv.mean()), "avg_back": float(all_back.mean()),
        "ret_service_alpha": float(np.mean(np.array(back["retailer"]) == 0)),
        "fill_beta": _safe_ratio(fill_m["retailer"], fill_d["retailer"]),
        "jitter": float(np.mean([np.mean(np.abs(np.diff(orders[a]))) if len(orders[a]) > 1 else 0.0
                                 for a in AGENTS])),
        "hold_total": hold_total, "back_total": back_total,
        "bw_stage": bw_stage, "ovar": ovar, "dvar": dvar, "bw_cum": bw_cum, "nsamp": nsamp,
        "mean_inv": mean_inv, "mean_back": mean_back, "mean_netstock": mean_netstock,
        "serv_alpha": serv_alpha, "fill_beta_s": fill_beta_s, "stockout_freq": stockout_freq,
        "hold_c": hold_c, "back_c": back_c, "total_c": total_c,
        "zero_order_frac": zero_of, "max_order_frac": max_of,
    }
    if trace:
        out["trace"] = {"dhat": np.array(tr_dhat), "msg": np.array(tr_msg),
                        "cust": np.array(tr_cust), "demand": np.array(tr_dem), "ip": np.array(tr_ip)}
    return out


def cvar(costs, alpha):
    c = np.sort(np.asarray(costs))[::-1]
    k = max(1, int(np.ceil(alpha * len(c))))
    return float(c[:k].mean())


def evaluate(policy, env, episodes, trace=False):
    return [run_episode(policy, env, SEED_BASE + e, trace=trace) for e in range(episodes)]


# ==============================================================================
# Regime-uncertainty table (C1): per-lambda cost + Gap_Recovered vs BAR/CEILING
# ==============================================================================
def regime_uncertainty(ckpt, episodes, bar, ceiling, lambdas=HELDOUT_LAMBDAS, env_base=None):
    base = dict(ENV_BASE if env_base is None else env_base)
    print(f"\n  regime-uncertainty (C1)   BAR={bar:.0f}  CEILING={ceiling:.0f}"
          f"   [{episodes} eps/lambda, CRN seeds {HELDOUT_SEED_BASE}+]")
    print(f"    {'lambda':>7}{'mean cost':>12}{'S_mean':>9}")
    per = {}
    for lam in lambdas:
        env = DemandRandomizedBeerGame({**base, "demand_type": "poisson"},
                                       lam_lo=float(lam), lam_hi=float(lam), p_shift=0.0)
        pol = SIGNALPolicy(ckpt, env, ablate=False)
        costs, smeans = [], []
        for e in range(episodes):
            r = run_episode(pol, env, HELDOUT_SEED_BASE + e)
            costs.append(r["cost"]); smeans.append(float(np.mean(pol.last_S)))
        per[lam] = float(np.mean(costs))
        print(f"    {lam:>7g}{per[lam]:>12.1f}{np.mean(smeans):>9.1f}")
    mean_cost = float(np.mean(list(per.values())))
    gap = (bar - mean_cost) / max(1e-6, bar - ceiling)
    verdict = "BEATS the fixed bar (C1)" if gap > 0 else "below the fixed bar (no C1 yet)"
    print(f"    {'MEAN':>7}{mean_cost:>12.1f}   Gap_Recovered={gap:+.3f}  ({verdict})")
    return {"per_lambda": per, "mean_cost": mean_cost, "gap_recovered": gap}


# ==============================================================================
# Comm-value decomposition (the Lee mechanism): ON vs OFF, CRN-paired
# ==============================================================================
def _upstream_forecast_error(traces):
    """Per-echelon mean |d_hat_i - customer demand| pooled over a list of trace dicts."""
    dhat = np.concatenate([t["dhat"] for t in traces], 0)      # [T_tot, N]
    cust = np.concatenate([t["cust"] for t in traces], 0)      # [T_tot]
    return {a: float(np.nanmean(np.abs(dhat[:, i] - cust))) for i, a in enumerate(AGENTS)}


def ar1_comm_vs_rho(ckpt, episodes, rhos, mu=12.0, sigma=3.0, env_base=None):
    """Comm value as a function of AR(1) rho. Deterministic team cost with messages ON vs OFF
    (same weights, CRN-paired seeds) at each rho. Lee-So-Tang (2000) predicts comm value (OFF-ON)
    ~ 0 at rho=0 and rising with rho. For a use_comm=false checkpoint ON==OFF (ADJ already zero),
    so the table just reports that model's cost per rho (cross-model reference)."""
    print(f"\n  AR(1) message-RELIANCE vs autocorrelation   [{episodes} eps/rho, CRN seeds {SEED_BASE}+, "
          f"mu={mu:g} sigma={sigma:g}]")
    print(f"    {'rho':>6}{'cost_ON':>10}{'cost_OFF':>10}{'reliance%':>13}")
    out = {}
    for rho in rhos:
        env = make_ar1_env(rho, mu, sigma, env_base)
        pol_on = SIGNALPolicy(ckpt, env, ablate=False)
        pol_off = SIGNALPolicy(ckpt, env, ablate=True)
        c_on = float(np.mean([run_episode(pol_on, env, SEED_BASE + e)["cost"] for e in range(episodes)]))
        c_off = float(np.mean([run_episode(pol_off, env, SEED_BASE + e)["cost"] for e in range(episodes)]))
        cv = 100.0 * (c_off - c_on) / max(1e-6, c_off)
        out[float(rho)] = {"cost_on": c_on, "cost_off": c_off, "reliance_pct": cv}
        print(f"    {rho:>6g}{c_on:>10.1f}{c_off:>10.1f}{cv:>+12.2f}%")
    print("    (reliance% = (OFF-ON)/OFF for the SAME comm-trained model = how much it DEPENDS on its")
    print("     messages. This is NOT the value of sharing V -- a comm-trained model over-relies on the")
    print("     channel, so this OVERSTATES V. True V = cross-arm: this model's cost_ON vs a SEPARATELY")
    print("     no-comm-TRAINED model's cost (use comm_stats.py on dumps of both arms).)")
    return out


def comm_value_decomposition(ckpt, episodes=40, scenario="poisson", env=None, label=None, env_base=None):
    if env is None:
        env = BeerGameParallelEnv({**(ENV_BASE if env_base is None else env_base),
                                   "demand_type": scenario})
    label = label or scenario
    pol_on = SIGNALPolicy(ckpt, env, ablate=False)
    if not pol_on.use_comm:
        print(f"\n  comm decomposition ({label}): use_comm=false checkpoint. skipped.")
        return None
    pol_off = SIGNALPolicy(ckpt, env, ablate=True)
    on = [run_episode(pol_on, env, SEED_BASE + e, trace=True) for e in range(episodes)]
    off = [run_episode(pol_off, env, SEED_BASE + e, trace=True) for e in range(episodes)]

    def agg(rs, key, a): return float(np.nanmean([r[key][a] for r in rs]))
    print(f"\n  comm decomposition ({label}, {episodes} eps, ON vs OFF paired, topology={pol_on.comm_topology})")
    print(f"    {'echelon':<13}{'BW_ON':>8}{'BW_OFF':>8}{'inv_ON':>8}{'inv_OFF':>8}{'back_ON':>8}{'back_OFF':>9}")
    for a in AGENTS:
        print(f"    {a:<13}{agg(on,'bw_stage',a):>8.2f}{agg(off,'bw_stage',a):>8.2f}"
              f"{agg(on,'mean_inv',a):>8.1f}{agg(off,'mean_inv',a):>8.1f}"
              f"{agg(on,'mean_back',a):>8.1f}{agg(off,'mean_back',a):>9.1f}")
    h_on = np.mean([r["hold_total"] for r in on]); h_off = np.mean([r["hold_total"] for r in off])
    b_on = np.mean([r["back_total"] for r in on]); b_off = np.mean([r["back_total"] for r in off])
    c_on = np.mean([r["cost"] for r in on]); c_off = np.mean([r["cost"] for r in off])
    print(f"    cost (ON->OFF): total {c_on:.0f}->{c_off:.0f}   holding {h_on:.0f}->{h_off:.0f}   "
          f"backorder {b_on:.0f}->{b_off:.0f}")

    e_on = _upstream_forecast_error([r["trace"] for r in on])
    e_off = _upstream_forecast_error([r["trace"] for r in off])
    print(f"    {'upstream':<13}{'fErr_ON':>9}{'fErr_OFF':>10}{'delta(help)':>13}  (target = CUSTOMER demand)")
    for a in AGENTS:
        if a == "retailer":
            continue                                            # retailer observes demand directly
        print(f"    {a:<13}{e_on[a]:>9.2f}{e_off[a]:>10.2f}{e_off[a] - e_on[a]:>+13.2f}")
    print("    (delta>0 = sharing the clean retailer demand cuts UPSTREAM forecast error = Lee see-through-")
    print("     bullwhip. NOTE d_hat is grounded to each stage's OWN incoming demand, so a small/negative")
    print("     delta in a stable regime is a legitimate Axsater-Rosling result, not a bug.)")
    return {"comm_value_pct": float(100.0 * (c_off - c_on) / max(1e-6, c_off)),
            "fErr_on": e_on, "fErr_off": e_off}


# ==============================================================================
# Honesty probe: each message COMPONENT vs the sender's true demand / inventory position
# ==============================================================================
def _corr(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) < 1e-8 or np.std(y[ok]) < 1e-8:
        return float("nan")
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def honesty_probe(ckpt, episodes=40, scenario="poisson", env=None, label=None, env_base=None):
    """Does each message component report the sender's truth? For 'dhat' compare the component
    (scaled by _MSG_SCALE) to the sender's own realized demand; for 'ip' to the sender's true
    inventory position (recoverable from its obs, hence honest by construction); 'learned' channels
    are scored against both demand and IP."""
    if env is None:
        env = BeerGameParallelEnv({**(ENV_BASE if env_base is None else env_base),
                                   "demand_type": scenario})
    label = label or scenario
    pol = SIGNALPolicy(ckpt, env, ablate=False)
    if not pol.use_comm:
        print(f"\n  honesty probe ({label}): use_comm=false checkpoint. skipped.")
        return None
    traces = [run_episode(pol, env, SEED_BASE + e, trace=True)["trace"] for e in range(episodes)]
    msg = np.concatenate([t["msg"] for t in traces], 0)         # [T_tot, N, msg_dim]
    dem = np.concatenate([t["demand"] for t in traces], 0)      # [T_tot, N] sender own demand
    ipv = np.concatenate([t["ip"] for t in traces], 0)          # [T_tot, N] sender IP
    layout = _component_layout(pol.content)
    print(f"\n  honesty probe ({label}, content={pol.content}, {episodes} eps)")
    out = {}

    if pol.content == "learned":
        print(f"    {'sender':<13}{'channel':>8}{'|msg|':>8}{'sat>0.9':>9}{'corr(demand)':>14}{'corr(IP)':>10}")
        for i, a in enumerate(AGENTS):
            gain = float(pol.actors[i].msg_gain) if pol.actors[i].msg_gain is not None else 1.0
            for c in range(pol.msg_dim):
                ch = msg[:, i, c]
                sat = float(np.mean(np.abs(ch) > 0.9 * abs(gain)))      # tanh saturates near +-gain, not +-1
                print(f"    {a:<13}{c:>8}{np.mean(np.abs(ch)):>8.3f}{sat:>9.2f}"
                      f"{_corr(ch, dem[:, i]):>14.3f}{_corr(ch, ipv[:, i]):>10.3f}")
        print("    (learned has no named ground truth; high corr(demand) on a channel = it encodes demand.")
        print("     sat>0.9 is vs the learned gain |g|; the message rides at +-g, not +-1.)")
        return out

    print(f"    {'sender':<13}{'component':>10}{'mean(msg*100)':>14}{'mean(truth)':>12}"
          f"{'bias':>9}{'corr':>8}")
    for name, col in layout.items():
        truth_src = dem if name in ("dhat", "raw") else ipv
        for i, a in enumerate(AGENTS):
            reported = msg[:, i, col] * _MSG_SCALE
            truth = truth_src[:, i]
            bias = float(np.nanmean(reported - truth))
            corr = _corr(reported, truth)
            out[f"{a}/{name}"] = {"bias": bias, "corr": corr,
                                  "mean_reported": float(np.nanmean(reported)),
                                  "mean_truth": float(np.nanmean(truth))}
            print(f"    {a:<13}{name:>10}{np.nanmean(reported):>14.2f}{np.nanmean(truth):>12.2f}"
                  f"{bias:>+9.2f}{corr:>8.3f}")
    print("    (corr~1, bias~0 => the component faithfully reports the named signal. 'ip' and 'raw' are")
    print("     deterministic obs readouts, honest by construction; 'dhat' is the learned, falsifiable case.")
    print("     For 'raw' the truth is the sender's demand at the SAME step; corr<1 only reflects the")
    print("     one-step reporting alignment, not dishonesty.)")
    return out


# ==============================================================================
# Does the agent listen? Two reports that separate "comm redundant" from "comm ignored"
# (cost-based comm value cannot tell these apart; these can).
# ==============================================================================
def positive_listening(ckpt, episodes=20, env=None, label=None, demand_grid=None, env_base=None):
    """Causal listening probe, per message component. At every state actually visited, hold the
    receiver's (obs, belief) fixed and sweep one component of its incoming message across a value
    grid (0..30, the component's natural units), all other components held at their actually-
    received values, recording the order-up-to S it would emit (probe_S). Components are swept one
    at a time: sweeping every column together would conflate dS/d(dhat) with dS/d(ip) for dhat_ip.
    dS/d(told) on the dhat component ~ (lead+1) means the receiver fully acts on the demand message;
    ~0 means deaf to it. The IP component has no such textbook target -- its slope is reported for
    sign and size (the VMI channel; a negative slope, ordering less when the sender holds more, is
    the theory-consistent direction). This is the test cost cannot do: a 0 here with honest messages
    means the agent is deaf, not the channel empty. (Probes the direct head path; the recurrent
    belief path is covered by the comm-on/off cost ablation.)"""
    if env is None:
        env = BeerGameParallelEnv({**(ENV_BASE if env_base is None else env_base),
                                   "demand_type": "poisson"})
    label = label or "poisson"
    if demand_grid is None:
        demand_grid = np.array([0., 5., 10., 15., 20., 25., 30.])
    msg_grid = demand_grid / _MSG_SCALE                      # probe_S expects the message's native scale
    pol = SIGNALPolicy(ckpt, env, ablate=False)
    if not pol.use_comm:
        print(f"\n  positive-listening ({label}): use_comm=false checkpoint. skipped.")
        return None
    if pol.content == "learned":                                     # no demand semantics, so the told-demand sweep does not apply
        print(f"\n  positive-listening ({label}): content=learned has no demand semantics, so the injected")
        print("    told-demand sweep does not apply. Use message_weight_audit (scale-agnostic) as the")
        print("    listening check for this rung, and the comm-on/off cost ablation for the net effect.")
        return None
    layout = _component_layout(pol.content)                          # {name: column}
    receivers = [i for i in range(pol.N) if float(pol.adj[i].abs().sum()) > 0]   # only stages that hear someone
    slopes = {i: {nm: [] for nm in layout} for i in receivers}
    swings = {i: {nm: [] for nm in layout} for i in receivers}
    for e in range(episodes):
        obs, _ = env.reset(seed=SEED_BASE + e)
        pol.reset()
        while True:
            acts = pol.act(obs)                              # sets pol.last_ctx for this step
            for i in receivers:
                for nm, col in layout.items():
                    S_vals = np.array([pol.probe_S(i, m, component=col) for m in msg_grid])
                    slopes[i][nm].append(float(np.polyfit(demand_grid, S_vals, 1)[0]))   # dS per unit told-value
                    swings[i][nm].append(float(S_vals[-1] - S_vals[0]))                  # S(told=30) - S(told=0)
            obs, _r, _t, truncs, _info = env.step({a: [acts[a]] for a in AGENTS})
            if any(truncs.values()):
                break
    print(f"\n  positive-listening probe ({label}, {episodes} eps; per-component sweep, told "
          f"{demand_grid[0]:g}..{demand_grid[-1]:g}, other components at received values)")
    print(f"    {'receiver':<13}{'component':>10}{'dS/dTold':>10}{'S swing 0->30':>15}")
    out = {}
    for i, a in enumerate(AGENTS):
        if i not in receivers:
            print(f"    {a:<13}{'(hears no one)':>10}")
            continue
        out[a] = {}
        for nm in layout:
            ms = float(np.mean(slopes[i][nm])); sw = float(np.mean(swings[i][nm]))
            out[a][nm] = {"dS_dTold": ms, "S_swing": sw}
            print(f"    {a:<13}{nm:>10}{ms:>10.3f}{sw:>15.2f}")
    print("    (dhat component: dS/dTold ~ (lead+1) => the receiver fully acts on the demand message;")
    print("     raw component: told value is ONE noisy observation, so the rational slope is SHRUNK below")
    print("     (lead+1) by the signal-to-noise ratio -- expect raw slope < dhat slope (forecast > data);")
    print("     ~0 => deaf to content. ip component: sign/size only (negative = VMI-consistent).")
    print("     Honest messages (high corr in the probe above) + dS/dTold~0 = the agent is IGNORING a")
    print("     usable signal -- an optimization/architecture finding, NOT 'communication has no value'.)")
    return out


def message_weight_audit(ckpt, env=None, env_base=None):
    """Structural companion: the L2 norm the trained actor places on its message inputs relative to
    its obs inputs, at (1) the GRU input gate and (2) the base-stock head. A ratio ~ 0 means the
    optimizer pruned the channel (dead by construction); a sizeable ratio means the channel is wired
    in and the null must be redundancy, not pruning. Weights-only, no rollout."""
    if env is None:
        env = BeerGameParallelEnv({**(ENV_BASE if env_base is None else env_base),
                                   "demand_type": "poisson"})
    pol = SIGNALPolicy(ckpt, env, ablate=False)
    if not pol.use_comm:
        print("\n  message-weight audit: use_comm=false checkpoint. skipped.")
        return None
    od, md, H = pol.obs_dim, pol.msg_dim, pol.hidden
    print(f"\n  message-weight audit  (||W_msg|| / ||W_obs||;  ~0 => optimizer ignored the channel)")
    print(f"    {'receiver':<13}{'GRU-in':>9}{'head':>9}")
    out = {}
    for i, a in enumerate(AGENTS):
        act = pol.actors[i]
        Wih = act.gru.weight_ih_l0.detach()                 # [3H, od+md]; msg cols od:od+md
        g_obs = float(Wih[:, :od].norm()); g_msg = float(Wih[:, od:od + md].norm())
        Wh = act.head.weight.detach()                       # [1, od+H+md(+1)]; msg cols od+H:od+H+md
        h_obs = float(Wh[:, :od].norm()); h_msg = float(Wh[:, od + H:od + H + md].norm())
        gr = g_msg / max(1e-9, g_obs); hr = h_msg / max(1e-9, h_obs)
        out[a] = {"gru_ratio": gr, "head_ratio": hr}
        print(f"    {a:<13}{gr:>9.3f}{hr:>9.3f}")
    print("    (ratio compares per-input weight mass; obs has 4 inputs, message 1-2, so ~0.2-0.5 is")
    print("     'wired in'. Values ~0 at every receiver = the message inputs were driven to zero.)")
    return out

def message_intervention_probe(ckpt, episodes=20, env=None, label=None, env_base=None):
    """Causal content test by intervention on the channel, do(m) (Jaques et al. 2019; the design
    licensed by Lowe et al. 2019): re-run the same CRN episodes with the message stream replaced by
      honest   : live messages (reference)
      shuffled : the episode's own honest messages, time-permuted  (same marginals, timing destroyed)
      cross    : another episode's honest messages                 (same process, wrong realization)
      zeroed   : channel off (the existing ablation)
    If the channel's value is content (state-correlated information), cost(shuffled)/cost(cross)
    degrade toward cost(zeroed); if only presence/scale matters, shuffled ~ honest. This upgrades
    the H3 mechanism claim from correlational to interventional and is the registered gate for
    attributing a positive V to content. Self-check: replaying episode 0's own honest stream
    (identity intervention) must reproduce its cost within tolerance -- printed and asserted.
    The unit of analysis here is the episode within one checkpoint (an instrument, not the cross-seed
    confirmatory statistic); deltas are episode-paired with bootstrap-t CIs."""
    from scripts.c1_stats import bootstrap_ci, paired
    if env is None:
        env = BeerGameParallelEnv({**(ENV_BASE if env_base is None else env_base),
                                   "demand_type": "poisson"})
    label = label or "poisson"
    pol = SIGNALPolicy(ckpt, env, ablate=False)
    if not pol.use_comm:
        print(f"\n  message-intervention ({label}): use_comm=false checkpoint. skipped.")
        return None
    N, md = pol.N, pol.msg_dim

    def _prev_of(msgs):                              # emitted m_t -> the m_{t-1} stream act() consumes
        z = np.zeros((1, N, md), dtype=np.float64)
        return np.concatenate([z, msgs[:-1]], axis=0)

    # pass 1: honest runs (trace records the emitted messages)
    honest = [run_episode(pol, env, SEED_BASE + e, trace=True) for e in range(episodes)]
    c_hon = np.array([r["cost"] for r in honest])
    msgs = [r["trace"]["msg"] for r in honest]       # [T,N,md] each

    # identity replay self-check on episode 0 (must reproduce the honest cost within tolerance)
    pol.msg_override = _prev_of(msgs[0])
    c_replay = run_episode(pol, env, SEED_BASE + 0)["cost"]
    pol.msg_override = None
    replay_ok = abs(c_replay - c_hon[0]) <= 1e-4 * max(1.0, abs(c_hon[0]))
    assert replay_ok, f"identity replay mismatch: {c_replay} vs {c_hon[0]} (override wiring broken)"

    # pass 2: interventions, CRN-paired to the honest episodes
    c_shuf, c_cross = np.zeros(episodes), np.zeros(episodes)
    for e in range(episodes):
        rng = np.random.default_rng(777_000 + e)
        pol.msg_override = _prev_of(msgs[e][rng.permutation(len(msgs[e]))])
        c_shuf[e] = run_episode(pol, env, SEED_BASE + e)["cost"]
        partner = msgs[(e + 1) % episodes]
        T = min(len(partner), len(msgs[e]))
        pol.msg_override = _prev_of(partner[:T])
        c_cross[e] = run_episode(pol, env, SEED_BASE + e)["cost"]
    pol.msg_override = None
    pol_off = SIGNALPolicy(ckpt, env, ablate=True)
    c_zero = np.array([run_episode(pol_off, env, SEED_BASE + e)["cost"] for e in range(episodes)])

    print(f"\n  message-intervention probe ({label}, {episodes} CRN eps; identity-replay check PASS)")
    print(f"    {'condition':<12}{'mean cost':>11}{'delta vs honest':>17}{'95% CI':>20}{'p(Wilcoxon)':>13}")
    out = {"honest": float(c_hon.mean())}
    for nm, c in [("shuffled", c_shuf), ("cross", c_cross), ("zeroed", c_zero)]:
        d = c - c_hon                                # + => intervention hurts => content mattered
        pr = paired(d, np.zeros_like(d))
        lo, hi = bootstrap_ci(d)
        out[nm] = {"mean": float(c.mean()), "delta": float(d.mean()),
                   "ci": (lo, hi), "p": pr["wilcoxon_p"]}
        print(f"    {nm:<12}{c.mean():>11.1f}{d.mean():>+17.1f}"
              f"   [{lo:>7.1f},{hi:>8.1f}]{pr['wilcoxon_p']:>13.3g}")
    dz = out["zeroed"]["delta"]
    if abs(dz) > 1e-6:
        share = out["shuffled"]["delta"] / dz
        print(f"    content share = delta(shuffled)/delta(zeroed) = {share:+.2f}  "
              f"(~1: value IS the state-correlated content; ~0: only presence/scale)")
        out["content_share"] = float(share)
    else:
        print("    content share: n/a (zeroing the channel does not change cost -> channel worthless here)")
    print("    (registered gate: a positive V is attributed to CONTENT only if delta(shuffled) CI > 0.)")
    return out


# ==============================================================================
# Per-stage dashboard (OM dashboard; reuses a rollout if given)
# ==============================================================================
def per_stage_dashboard(ckpt, episodes, scenario, eps=None, env_base=None):
    if eps is None:
        env = BeerGameParallelEnv({**(ENV_BASE if env_base is None else env_base),
                                   "demand_type": scenario})
        eps = evaluate(SIGNALPolicy(ckpt, env, ablate=False), env, episodes)
    n = len(eps)

    def m(key, a): return float(np.nanmean([e[key][a] for e in eps]))
    print(f"\n  per-stage dashboard ({scenario}, {n} eps)")
    print(f"    {'echelon':<13}{'BW_loc':>8}{'BW_cum':>8}{'NSAmp':>8}{'inv':>7}{'back':>7}{'netstk':>8}"
          f"{'alpha':>7}{'beta':>7}{'P(stk)':>8}{'total$':>9}")
    for a in AGENTS:
        print(f"    {a:<13}{m('bw_stage',a):>8.2f}{m('bw_cum',a):>8.2f}{m('nsamp',a):>8.2f}"
              f"{m('mean_inv',a):>7.1f}{m('mean_back',a):>7.1f}{m('mean_netstock',a):>8.1f}"
              f"{m('serv_alpha',a):>7.2f}{m('fill_beta_s',a):>7.2f}{m('stockout_freq',a):>8.2f}{m('total_c',a):>9.0f}")
    print("    (BW_loc=Var(ord)/Var(own-incoming); BW_cum=Var(ord)/Var(customer demand) [Lee/Chen];")
    print("     NSAmp=Var(net stock)/Var(customer demand) [Disney-Towill]; total$=holding+backorder.)")


# ==============================================================================
# Producers: per-seed dumps for the cross-seed aggregators
# ==============================================================================
def dump_c1(ckpt, episodes, out_dir, seed=None, lambdas=HELDOUT_LAMBDAS, env_base=None):
    """Producer for scripts/c1_stats.py: write {lambda: mean_cost} for this checkpoint to
    out_dir/seed{S}.json (+ seed{S}_bw.json), scored on the eval seeds (CRN with baselines)."""
    base = dict(ENV_BASE if env_base is None else env_base)
    if seed is None:
        seed = ckpt.get("seed", ckpt.get("config", {}).get("seed", 0))   # top-level seed first (see train_signal save)
    per, per_bw = {}, {}
    for lam in lambdas:
        env = DemandRandomizedBeerGame({**base, "demand_type": "poisson"},
                                       lam_lo=float(lam), lam_hi=float(lam), p_shift=0.0)
        pol = SIGNALPolicy(ckpt, env, ablate=False)
        rs = [run_episode(pol, env, HELDOUT_SEED_BASE + e) for e in range(episodes)]
        per[float(lam)] = float(np.mean([r["cost"] for r in rs]))
        per_bw[float(lam)] = {a: float(np.nanmean([r["bw_cum"][a] for r in rs])) for a in AGENTS}
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"seed{int(seed)}.json")
    with open(path, "w") as f:
        json.dump(per, f, indent=2)
    with open(os.path.join(out_dir, f"seed{int(seed)}_bw.json"), "w") as f:
        json.dump(per_bw, f, indent=2)
    print(f"\n  [dump-c1] wrote {path} (+ seed{int(seed)}_bw.json)")
    print(f"    lambdas {[f'{l:g}' for l in lambdas]}  mean_cost {[round(per[float(l)],1) for l in lambdas]}")
    return path


def dump_comm(ckpt, episodes, out_dir, seed=None, lambdas=HELDOUT_LAMBDAS, env_base=None,
              ar1_rhos=None, ar1_mu=12.0, ar1_sigma=3.0):
    """Producer for scripts/comm_stats.py: write per-seed {key: cost} (the comm/no-comm arm's
    cost vector) and the per-echelon upstream forecast error to out_dir/seed{S}.json /
    seed{S}_ferr.json. Run once per checkpoint into a comm-on dir and a comm-off (use_comm=false)
    dir; comm_stats CRN-pairs them across seeds.
    Keys: Poisson lambdas by default. Pass ar1_rhos=[0,0.3,0.6,0.9] (CLI --dump-ar1) to key the
    same format by AR(1) rho instead -- the H2 producer. comm_stats' loaders and value_of_sharing
    are key-agnostic, and prereg.h2_slope consumes the per-rho structure."""
    base = dict(ENV_BASE if env_base is None else env_base)
    if seed is None:
        seed = ckpt.get("seed", ckpt.get("config", {}).get("seed", 0))   # top-level seed first (see train_signal save)
    if ar1_rhos:
        keyed_envs = {float(r): make_ar1_env(r, ar1_mu, ar1_sigma, base) for r in ar1_rhos}
        mode = f"AR(1) rhos {[f'{r:g}' for r in ar1_rhos]} (mu={ar1_mu:g} sigma={ar1_sigma:g})"
    else:
        keyed_envs = {float(lam): DemandRandomizedBeerGame({**base, "demand_type": "poisson"},
                                                           lam_lo=float(lam), lam_hi=float(lam),
                                                           p_shift=0.0)
                      for lam in lambdas}
        mode = f"Poisson lambdas {[f'{l:g}' for l in lambdas]}"
    per, per_ferr, per_censor = {}, {}, {}
    for key, env in keyed_envs.items():
        pol = SIGNALPolicy(ckpt, env, ablate=False)
        rs = [run_episode(pol, env, HELDOUT_SEED_BASE + e, trace=True) for e in range(episodes)]
        per[key] = float(np.mean([r["cost"] for r in rs]))
        per_ferr[key] = _upstream_forecast_error([r["trace"] for r in rs])
        # Censoring fingerprint on the same CRN episodes as the cost (Proposition 1 signature):
        # per-stage fraction of lower-censored (order==0) and upper-censored (order==max) steps.
        per_censor[key] = {a: {"zero": float(np.mean([r["zero_order_frac"][a] for r in rs])),
                               "max": float(np.mean([r["max_order_frac"][a] for r in rs]))}
                           for a in AGENTS}
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"seed{int(seed)}.json")
    with open(path, "w") as f:
        json.dump(per, f, indent=2)
    with open(os.path.join(out_dir, f"seed{int(seed)}_ferr.json"), "w") as f:
        json.dump(per_ferr, f, indent=2)
    with open(os.path.join(out_dir, f"seed{int(seed)}_censor.json"), "w") as f:
        json.dump(per_censor, f, indent=2)
    print(f"\n  [dump-comm] wrote {path} (+ seed{int(seed)}_ferr.json, seed{int(seed)}_censor.json)"
          f"  use_comm={pol.use_comm}  [{mode}]")
    return path


# ==============================================================================
# main
# ==============================================================================
def _resolve_refs(args, root):
    """C1 references: explicit CLI > --refs-json (baselines_regime_v2.json) > hardcoded fallback."""
    bar, ceiling = args.bar, args.ceiling
    refs_path = args.refs_json if os.path.isabs(args.refs_json) else os.path.join(root, args.refs_json)
    if (bar is None or ceiling is None) and os.path.exists(refs_path):
        try:
            from scripts.c1_stats import load_rungs, mean_refs
            m = mean_refs(load_rungs(refs_path), HELDOUT_LAMBDAS)
            if bar is None:
                bar = m.get("BAR_static")
            if ceiling is None:
                ceiling = m.get("Oracle")
            print(f"  C1 refs <- {os.path.basename(refs_path)}: BAR={bar} Oracle={ceiling}")
        except Exception as e:                       # noqa: BLE001
            print(f"  (refs-json load failed: {type(e).__name__}: {e}; using CLI/defaults)")
    return (bar if bar is not None else 4726.0), (ceiling if ceiling is not None else 2202.0)


def main():
    ap = argparse.ArgumentParser(description="SIGNAL beer-game eval (policy loader + measurement probes).")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--cvar", type=float, default=0.2, help="tail level for CVaR cost")
    ap.add_argument("--refs-json", default="results/baselines_regime_v2.json",
                    help="four-rung refs from `python scripts/baselines.py regime` (derives BAR/CEILING)")
    ap.add_argument("--bar", type=float, default=None, help="override fixed-policy BAR")
    ap.add_argument("--ceiling", type=float, default=None, help="override per-lambda CEILING")
    ap.add_argument("--env-json", default=None, metavar="PATH",
                    help="override the checkpoint's saved env config with this json (rare; the "
                         "default is the env AS TRAINED, persisted in the checkpoint)")
    ap.add_argument("--regime-episodes", type=int, default=20, help="episodes per held-out lambda")
    ap.add_argument("--messages", action="store_true", help="run the comm decomposition + honesty probe")
    ap.add_argument("--interventions", action="store_true",
                    help="run the message-intervention probe (honest vs shuffled/cross/zeroed messages; "
                         "the causal content test gating content-attribution of a positive V)")
    ap.add_argument("--full", action="store_true", help="standard benchmark + dashboard + C1 + messages")
    ap.add_argument("--ar1", action="store_true",
                    help="AR(1) mode: run the comm-value-vs-rho table + decomposition/honesty ON AR(1) demand "
                         "(use this for ar1-trained checkpoints; the default probes are Poisson = off-regime)")
    ap.add_argument("--ar1-rho", type=float, default=0.9, help="AR(1) rho for the decomposition + honesty probe")
    ap.add_argument("--ar1-rhos", default="0.0,0.3,0.6,0.9", help="AR(1) rho grid for the comm-value-vs-rho table")
    ap.add_argument("--ar1-mu", type=float, default=12.0, help="AR(1) mean demand (match training agent.ar1_mu)")
    ap.add_argument("--ar1-sigma", type=float, default=3.0, help="AR(1) innovation sigma (match training agent.ar1_sigma)")
    ap.add_argument("--dump-c1", default=None, metavar="DIR",
                    help="PRODUCER: write per-seed {lambda: cost} to DIR for scripts/c1_stats.py, then exit")
    ap.add_argument("--dump-comm", default=None, metavar="DIR",
                    help="PRODUCER: write per-seed {lambda: cost}+forecast-error to DIR for comm_stats.py, then exit")
    ap.add_argument("--dump-ar1", default=None, metavar="RHOS",
                    help="with --dump-comm: key the dump by AR(1) rho instead of Poisson lambda, e.g. "
                         "\"0,0.3,0.6,0.9\" (the H2 producer; uses --ar1-mu/--ar1-sigma)")
    ap.add_argument("--dump-iv", default=None, metavar="DIR",
                    help="run the message-intervention probe and write seed{S}_iv.json to DIR "
                         "(cross-seed aggregation: scripts/comm_stats.py interventions --dir DIR). "
                         "With --ar1 the probe runs on the AR(1) env at --ar1-rho (the sweep mode).")
    ap.add_argument("--dump-episodes", type=int, default=200, help="episodes/lambda for the producers")
    ap.add_argument("--seed", type=int, default=None, help="seed label for the producers (default: ckpt's seed)")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    cfg = ckpt.get("config", {})
    env_base = resolve_env_base(ckpt, args.env_json)      # the env AS TRAINED, for every probe below

    if args.dump_c1:
        dump_c1(ckpt, args.dump_episodes, args.dump_c1, seed=args.seed, env_base=env_base)
        return
    if args.dump_comm:
        rhos = ([float(x) for x in args.dump_ar1.split(",") if x.strip() != ""]
                if args.dump_ar1 else None)
        dump_comm(ckpt, args.dump_episodes, args.dump_comm, seed=args.seed, env_base=env_base,
                  ar1_rhos=rhos, ar1_mu=args.ar1_mu, ar1_sigma=args.ar1_sigma)
        return
    if args.dump_iv:
        # Producer for the registered content-attribution gate: one seed{S}_iv.json per checkpoint
        # (episode-mean cost under do(m): honest/shuffled/cross/zeroed). Cross-seed inference lives
        # in scripts/comm_stats.py `interventions` (the seed is the unit; this file is the instrument).
        if args.ar1:
            iv_env = make_ar1_env(args.ar1_rho, args.ar1_mu, args.ar1_sigma, env_base)
            lbl = f"ar1 rho={args.ar1_rho:g}"
        else:
            iv_env, lbl = None, "poisson"
        res = message_intervention_probe(ckpt, episodes=args.episodes, env=iv_env, label=lbl,
                                         env_base=env_base)
        if res is None:
            print("  [dump-iv] use_comm=false checkpoint -> no intervention dump (by design).")
            return
        seed = args.seed if args.seed is not None else ckpt.get("seed", cfg.get("seed", 0))
        os.makedirs(args.dump_iv, exist_ok=True)
        payload = {"label": lbl, "episodes": int(args.episodes), **res}
        path = os.path.join(args.dump_iv, f"seed{int(seed)}_iv.json")
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  [dump-iv] wrote {path}")
        return

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bar, ceiling = _resolve_refs(args, root)

    print(f"\nSIGNAL eval  |  ckpt={os.path.basename(args.ckpt)}  |  msg_content={cfg.get('msg_content')}  "
          f"|  comm={cfg.get('use_comm')}/{cfg.get('comm_topology','neighbor')}  |  episodes={args.episodes}\n")

    # ---- standard benchmark across the fixed regimes + zero-message comm value ----
    from scipy import stats
    header = (f"{'scenario':<15}{'mean cost':>12}{'CVaR cost':>12}{'bullwhip':>11}"
              f"{'srv-alpha':>11}{'fill-beta':>11}{'jitter':>9}   comm value")
    print(header); print("-" * len(header))
    eps_by_scenario = {}
    for scenario in SCENARIOS:
        env = BeerGameParallelEnv({**env_base, "demand_type": scenario})
        eps = evaluate(SIGNALPolicy(ckpt, env, ablate=False), env, args.episodes)
        eps_by_scenario[scenario] = eps
        costs = np.array([e["cost"] for e in eps])
        abl = evaluate(SIGNALPolicy(ckpt, env, ablate=True), env, args.episodes)
        c_abl = np.array([e["cost"] for e in abl])
        with np.errstate(invalid="ignore", divide="ignore"):
            comm_value = float(np.nanmean((c_abl - costs) / np.where(costs == 0, np.nan, costs)) * 100.0)
        try:
            p = float(stats.wilcoxon(c_abl, costs).pvalue) if not np.all(c_abl - costs == 0) else 1.0
        except Exception:
            p = float("nan")
        print(f"{scenario:<15}{costs.mean():>12.1f}{cvar(costs, args.cvar):>12.1f}"
              f"{np.nanmean([e['bw_overall'] for e in eps]):>11.2f}"
              f"{np.mean([e['ret_service_alpha'] for e in eps]):>11.2f}"
              f"{np.nanmean([e['fill_beta'] for e in eps]):>11.2f}"
              f"{np.mean([e['jitter'] for e in eps]):>9.1f}   {comm_value:+.1f}%  (p={p:.1e})")
    print("\ncomm value = % cost change when messages are zeroed (paired, same seeds); positive => comm helps.")

    if args.full:
        for scenario in SCENARIOS:
            per_stage_dashboard(ckpt, args.episodes, scenario, eps=eps_by_scenario[scenario],
                                env_base=env_base)

    regime_uncertainty(ckpt, args.regime_episodes, bar, ceiling, env_base=env_base)

    if args.ar1:
        rhos = [float(x) for x in args.ar1_rhos.split(",") if x.strip() != ""]
        ar1_comm_vs_rho(ckpt, args.regime_episodes, rhos, mu=args.ar1_mu, sigma=args.ar1_sigma,
                        env_base=env_base)
        ar1_env = make_ar1_env(args.ar1_rho, args.ar1_mu, args.ar1_sigma, env_base)
        lbl = f"ar1 rho={args.ar1_rho:g}"
        comm_value_decomposition(ckpt, episodes=min(40, args.episodes), env=ar1_env, label=lbl)
        honesty_probe(ckpt, episodes=min(40, args.episodes), env=ar1_env, label=lbl)
        positive_listening(ckpt, episodes=min(20, args.episodes), env=ar1_env, label=lbl)
        message_weight_audit(ckpt, env=ar1_env)
        if args.interventions:
            message_intervention_probe(ckpt, episodes=min(20, args.episodes), env=ar1_env, label=lbl)
    elif args.messages or args.full:
        comm_value_decomposition(ckpt, episodes=min(40, args.episodes), env_base=env_base)
        honesty_probe(ckpt, episodes=min(40, args.episodes), env_base=env_base)
        positive_listening(ckpt, episodes=min(20, args.episodes), env_base=env_base)
        message_weight_audit(ckpt, env_base=env_base)
        if args.interventions:
            message_intervention_probe(ckpt, episodes=min(20, args.episodes), env_base=env_base)
    elif args.interventions:
        message_intervention_probe(ckpt, episodes=min(20, args.episodes), env_base=env_base)
    print()


if __name__ == "__main__":
    main()