"""
eval_signal.py -- evaluation / measurement layer for the SIGNAL agent.
================================================================================
SIGNAL (Strategic Information-sharing for Game-theoretic, Networked, Adaptive
Logistics). Loads a `signal_checkpoint_best.pt` produced by train_signal.py into a
SIGNALPolicy and runs the paper's measurements. It REPLACES the legacy DRACO eval:
the SIGNAL agent has no encoder/z/structured-vs-mlp head/plug-in belief, so this is
a from-scratch port keyed to the SIGNAL actor (GRU belief + linear base-stock head +
demand readout + message ladder).

THE POLICY IS THE TRAINED POLICY. SIGNALPolicy.act() reproduces SIGNALTrainer.collect()'s
per-step forward byte-for-byte: incoming = ADJ @ m_prev (one-step delay), per-agent GRU
belief update, base-stock head -> S, order = clip(S - IP), message = ladder(d_hat, IP).
The saved ADJ (already zeroed for a use_comm=false checkpoint) is used directly, so
messages route exactly as trained; ablate=True is the comm-OFF counterfactual on the
SAME weights.

WHAT IT COMPUTES (each part names the question it answers):
  * STANDARD BENCHMARK (per regime poisson/black_swan/extreme_chaos): mean/CVaR cost,
    bullwhip, Type-1 (alpha) service, Type-2 (beta) fill, jitter, and the zero-message
    comm value (paired Wilcoxon).
  * PER-STAGE DASHBOARD: local + cumulative bullwhip, net-stock amplification, on-hand/
    backlog, service/fill, per-echelon cost (Chen 2000; Lee 1997; Disney-Towill 2003).
  * REGIME-UNCERTAINTY (C1): per-lambda cost + Gap_Recovered vs the BAR/Oracle references
    from `python scripts/baselines.py regime`.
  * COMM-VALUE DECOMPOSITION (the Lee mechanism): comm ON vs OFF, CRN-paired -- per-stage
    bullwhip/inventory/backlog split and the per-echelon UPSTREAM forecast-error delta
    vs CUSTOMER demand. delta>0 => the shared signal cuts upstream forecast error.
  * HONESTY PROBE: each message COMPONENT vs the sender's true demand / inventory position.
  * PRODUCERS: --dump-c1 (per-seed {lambda: cost} for scripts/c1_stats.py) and --dump-comm
    (per-seed {lambda: cost} + per-echelon forecast error for scripts/comm_stats.py).

TWO SEED SPACES (do not conflate):
  SEED_BASE=2000          -> standard-benchmark / message rollouts.
  HELDOUT_SEED_BASE=1e5   -> C1 / dump rollouts; == baselines.py SEED_BASE, so SIGNAL and the
                             reference rungs are scored on the SAME demand draws (CRN).

Usage:
  python agents/eval_signal.py --ckpt weights_signal/run_signal_<id>_<algo>/signal_checkpoint_best.pt
  python agents/eval_signal.py --ckpt ... --episodes 100 --full
  python agents/eval_signal.py --ckpt ... --dump-c1   results/signal_c1
  python agents/eval_signal.py --ckpt ... --dump-comm results/comm_on
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCENARIOS = ["poisson", "black_swan", "extreme_chaos"]
SEED_BASE = 2000
HELDOUT_SEED_BASE = 100000          # matches baselines.py SEED_BASE (CRN with the BAR/CEILING)
HELDOUT_LAMBDAS = [6.0, 10.0, 14.0, 18.0, 22.0]
ENV_BASE = {"horizon": 50, "max_order": 100}
H_COST = 0.5     # holding cost/unit/week -- MUST match the env/config (config.yaml: holding_cost)
B_COST = 1.0     # backorder cost/unit/week -- MUST match the env/config (config.yaml: backorder_cost)

# message-component layout per content type (which slot carries which named signal).
# 'dhat' and 'ip' messages are broadcast in a /100 scale (see SIGNALActor.message), so the
# honesty probe multiplies the component back by 100 before comparing to demand / IP units.
_MSG_SCALE = 100.0


def _safe_ratio(num, den):
    return float(num / den) if den and np.isfinite(den) and den != 0 else float("nan")


def _component_layout(content):
    """Return {component_name: column_index} for a message content type. 'learned' has no
    named ground truth, so its channels are reported against both demand and IP."""
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
        cfg = ckpt.get("config", {})                # train_signal saves the AGENT dict directly
        self.cfg = cfg

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

        self.actors = []
        for sd in ckpt["actors"]:
            a = SIGNALActor(self.obs_dim, self.msg_dim, self.hidden, self.content,
                            self.learned_msg_dim).to(DEVICE)
            a.load_state_dict(sd)
            a.eval()
            self.actors.append(a)
        self.reset()

    def reset(self):
        self.h = [torch.zeros(1, 1, self.hidden, device=DEVICE) for _ in range(self.N)]
        self.m_buf = torch.zeros(self.N, self.msg_dim, device=DEVICE)
        self.last_S = np.zeros(self.N)
        self.last_dhat = np.full(self.N, np.nan)
        self.last_msg = np.zeros((self.N, self.msg_dim))
        self.last_ctx = None                        # (o_t, [hi per agent], incoming) for the listening probe

    @torch.no_grad()
    def act(self, obs):
        o_t = torch.tensor(np.stack([obs[a] for a in AGENTS]), dtype=torch.float32, device=DEVICE)  # [N,obs]
        incoming = self.adj @ self.m_buf            # [N,msg]
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
        return {a: float(frac[i, 0].item()) for i, a in enumerate(AGENTS)}

    @torch.no_grad()
    def probe_S(self, agent_idx, msg_value):
        """Order-up-to S that actor `agent_idx` WOULD output at the LAST step's (obs, belief) if its
        incoming message were `msg_value` (in the /100 broadcast scale). dS/d(message) measures
        causal positive listening: does the receiver act on message CONTENT?"""
        o_t, hi_list, _inc = self.last_ctx
        m = torch.full((1, self.msg_dim), float(msg_value), device=DEVICE)
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
    bw_stage, ovar, dvar, bw_cum, nsamp = {}, {}, {}, {}, {}
    mean_inv, mean_back, mean_netstock = {}, {}, {}
    serv_alpha, fill_beta_s, stockout_freq, hold_c, back_c, total_c = {}, {}, {}, {}, {}, {}
    for a in AGENTS:
        o = np.asarray(orders[a], float); d = np.asarray(demand[a], float)
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
        hold_c[a] = H_COST * float(iv.sum())
        back_c[a] = B_COST * float(bk.sum())
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
def regime_uncertainty(ckpt, episodes, bar, ceiling, lambdas=HELDOUT_LAMBDAS):
    print(f"\n  regime-uncertainty (C1)   BAR={bar:.0f}  CEILING={ceiling:.0f}"
          f"   [{episodes} eps/lambda, CRN seeds {HELDOUT_SEED_BASE}+]")
    print(f"    {'lambda':>7}{'mean cost':>12}{'S_mean':>9}")
    per = {}
    for lam in lambdas:
        env = DemandRandomizedBeerGame({**ENV_BASE, "demand_type": "poisson"},
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


def comm_value_decomposition(ckpt, episodes=40, scenario="poisson"):
    env = BeerGameParallelEnv({**ENV_BASE, "demand_type": scenario})
    pol_on = SIGNALPolicy(ckpt, env, ablate=False)
    if not pol_on.use_comm:
        print(f"\n  comm decomposition ({scenario}): use_comm=false checkpoint. skipped.")
        return None
    pol_off = SIGNALPolicy(ckpt, env, ablate=True)
    on = [run_episode(pol_on, env, SEED_BASE + e, trace=True) for e in range(episodes)]
    off = [run_episode(pol_off, env, SEED_BASE + e, trace=True) for e in range(episodes)]

    def agg(rs, key, a): return float(np.nanmean([r[key][a] for r in rs]))
    print(f"\n  comm decomposition ({scenario}, {episodes} eps, ON vs OFF paired, topology={pol_on.comm_topology})")
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


def honesty_probe(ckpt, episodes=40, scenario="poisson"):
    """Does each message component report the sender's truth? For 'dhat' compare component*100 to
    the sender's OWN realized demand; for 'ip' to the sender's true inventory position (recoverable
    from its obs, so this is honest by construction); 'learned' channels are scored against both."""
    env = BeerGameParallelEnv({**ENV_BASE, "demand_type": scenario})
    pol = SIGNALPolicy(ckpt, env, ablate=False)
    if not pol.use_comm:
        print(f"\n  honesty probe ({scenario}): use_comm=false checkpoint. skipped.")
        return None
    traces = [run_episode(pol, env, SEED_BASE + e, trace=True)["trace"] for e in range(episodes)]
    msg = np.concatenate([t["msg"] for t in traces], 0)         # [T_tot, N, msg_dim]
    dem = np.concatenate([t["demand"] for t in traces], 0)      # [T_tot, N] sender own demand
    ipv = np.concatenate([t["ip"] for t in traces], 0)          # [T_tot, N] sender IP
    layout = _component_layout(pol.content)
    print(f"\n  honesty probe ({scenario}, content={pol.content}, {episodes} eps)")
    out = {}

    if pol.content == "learned":
        print(f"    {'sender':<13}{'channel':>8}{'|msg|':>8}{'sat>0.9':>9}{'corr(demand)':>14}{'corr(IP)':>10}")
        for i, a in enumerate(AGENTS):
            for c in range(pol.msg_dim):
                ch = msg[:, i, c]
                print(f"    {a:<13}{c:>8}{np.mean(np.abs(ch)):>8.3f}{np.mean(np.abs(ch) > 0.9):>9.2f}"
                      f"{_corr(ch, dem[:, i]):>14.3f}{_corr(ch, ipv[:, i]):>10.3f}")
        print("    (learned has no named ground truth; high corr(demand) on a channel = it encodes demand.)")
        return out

    print(f"    {'sender':<13}{'component':>10}{'mean(msg*100)':>14}{'mean(truth)':>12}"
          f"{'bias':>9}{'corr':>8}")
    for name, col in layout.items():
        truth_src = dem if name == "dhat" else ipv
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
    print("    (corr~1, bias~0 => the component faithfully reports the named signal. 'ip' is a deterministic")
    print("     readout of the obs, so it is honest by construction; 'dhat' is the learned, falsifiable case.)")
    return out


# ==============================================================================
# Per-stage dashboard (OM dashboard; reuses a rollout if given)
# ==============================================================================
def per_stage_dashboard(ckpt, episodes, scenario, eps=None):
    if eps is None:
        env = BeerGameParallelEnv({**ENV_BASE, "demand_type": scenario})
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
def dump_c1(ckpt, episodes, out_dir, seed=None, lambdas=HELDOUT_LAMBDAS):
    """PRODUCER for scripts/c1_stats.py: write {lambda: mean_cost} for THIS checkpoint to
    out_dir/seed{S}.json (+ seed{S}_bw.json), scored on the EVAL seeds (CRN with baselines)."""
    if seed is None:
        seed = ckpt.get("config", {}).get("seed", 0)
    per, per_bw = {}, {}
    for lam in lambdas:
        env = DemandRandomizedBeerGame({**ENV_BASE, "demand_type": "poisson"},
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


def dump_comm(ckpt, episodes, out_dir, seed=None, lambdas=HELDOUT_LAMBDAS):
    """PRODUCER for scripts/comm_stats.py: write per-seed {lambda: cost} (the comm/no-comm arm's
    cost vector) AND the per-echelon upstream forecast error to out_dir/seed{S}.json /
    seed{S}_ferr.json. Run once per checkpoint into a comm-on dir and a comm-off (use_comm=false)
    dir; comm_stats CRN-pairs them across seeds."""
    if seed is None:
        seed = ckpt.get("config", {}).get("seed", 0)
    per, per_ferr = {}, {}
    for lam in lambdas:
        env = DemandRandomizedBeerGame({**ENV_BASE, "demand_type": "poisson"},
                                       lam_lo=float(lam), lam_hi=float(lam), p_shift=0.0)
        pol = SIGNALPolicy(ckpt, env, ablate=False)
        rs = [run_episode(pol, env, HELDOUT_SEED_BASE + e, trace=True) for e in range(episodes)]
        per[float(lam)] = float(np.mean([r["cost"] for r in rs]))
        per_ferr[float(lam)] = _upstream_forecast_error([r["trace"] for r in rs])
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"seed{int(seed)}.json")
    with open(path, "w") as f:
        json.dump(per, f, indent=2)
    with open(os.path.join(out_dir, f"seed{int(seed)}_ferr.json"), "w") as f:
        json.dump(per_ferr, f, indent=2)
    print(f"\n  [dump-comm] wrote {path} (+ seed{int(seed)}_ferr.json)  use_comm={pol.use_comm}")
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
    ap.add_argument("--regime-episodes", type=int, default=20, help="episodes per held-out lambda")
    ap.add_argument("--messages", action="store_true", help="run the comm decomposition + honesty probe")
    ap.add_argument("--full", action="store_true", help="standard benchmark + dashboard + C1 + messages")
    ap.add_argument("--dump-c1", default=None, metavar="DIR",
                    help="PRODUCER: write per-seed {lambda: cost} to DIR for scripts/c1_stats.py, then exit")
    ap.add_argument("--dump-comm", default=None, metavar="DIR",
                    help="PRODUCER: write per-seed {lambda: cost}+forecast-error to DIR for comm_stats.py, then exit")
    ap.add_argument("--dump-episodes", type=int, default=200, help="episodes/lambda for the producers")
    ap.add_argument("--seed", type=int, default=None, help="seed label for the producers (default: ckpt's seed)")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    cfg = ckpt.get("config", {})

    if args.dump_c1:
        dump_c1(ckpt, args.dump_episodes, args.dump_c1, seed=args.seed)
        return
    if args.dump_comm:
        dump_comm(ckpt, args.dump_episodes, args.dump_comm, seed=args.seed)
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
        env = BeerGameParallelEnv({**ENV_BASE, "demand_type": scenario})
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
            per_stage_dashboard(ckpt, args.episodes, scenario, eps=eps_by_scenario[scenario])

    regime_uncertainty(ckpt, args.regime_episodes, bar, ceiling)

    if args.messages or args.full:
        comm_value_decomposition(ckpt, episodes=min(40, args.episodes))
        honesty_probe(ckpt, episodes=min(40, args.episodes))
    print()


if __name__ == "__main__":
    main()
