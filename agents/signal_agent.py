"""
signal_agent.py
================================================================================
SIGNAL (Strategic Information-sharing for Game-theoretic, Networked, Adaptive
Logistics) -- a deliberately MINIMAL communicating MAPPO agent for the value-of-
information-sharing study. The agent is an INSTRUMENT, not the contribution: it
is kept as simple as possible so that any measured effect of communication is
attributable to the message channel, not to architectural cleverness.

Three parts, and nothing else:
  (1) BELIEF  : one small GRU per agent over its local-observation history. The
                hidden state IS the demand-regime belief. Trained end-to-end by
                the policy gradient -- no separate ELBO/KL/reconstruction.
  (2) HEAD    : ONE linear layer over [obs, belief, message] (and, when use_dhat_head,
                the smoothed demand estimate d_hat) -> an order-up-to level S. softplus is
                only a positivity link. d_hat is appended because gradient descent alone
                failed to learn the demand-dependence within budget; its head weight is
                warm-started near the textbook (lead+1) protection-interval multiplier.
                It remains a fully-learnable linear layer -- no frozen lead/safety/corr prior.
  (3) MESSAGE : the semantic ladder {dhat | ip | dhat_ip | learned}, routed along
                a topology with a one-step delay. This is the ONLY part we engineer
                with care, because it is the object of study.
                TRAINING OF THE LADDER (deliberate asymmetry, declared in the paper):
                the NAMED rungs (dhat/ip/dhat_ip) have FIXED semantics -- their channel is
                detached, and their content trains only through the sender's own objectives
                (aux grounding, own S-head), which is what keeps d_hat an honest forecast.
                The 'learned' rung is BY DEFINITION content optimized end-to-end for team
                benefit, so its channel is DIFFERENTIABLE in the update (DIAL, Foerster et
                al. 2016) -- see SIGNALTrainer._coupled_forward. Without that, msg_head/
                msg_gain receive no gradient and the rung is a frozen random projection.
Underneath: textbook CTDE-MAPPO (centralized per-agent critic on the global state
during training, decentralized execution; PPO-clip; GAE). The economics layer
(srdqn_beta cost-sharing + the coordinating transfer tau) rides on the reward in a
few lines and is independent of the agent.

DATA FLOW (one timestep t), per agent i in [retailer, wholesaler, distributor, manufacturer]:

        local obs_i,t ----+
        (4 scalars)       |
                          v
   incoming msg_i,t --> [ GRU ] --> belief h_i,t --+--> [ linear head ] --> S_i,t --> order_i,t = clip(S - IP)
   (routed, 1-step delay)                          |
                                                   +--> [ linear d-readout ] --> d_hat_i,t
                                                                                   |
                                          message m_i,t = ladder(d_hat, IP, learned) --> routed by ADJ (delay) --> incoming_*,t+1

   TRAINING ONLY: global state s_t --> [ MLP critic ] --> V_i(s_t)  (baseline; not used at execution)
   REWARD:        r_i,t = -(own_cost_i + beta*others_cost) - tau*backlog_i (+ credit to customer)   <-- economics layer

Honest status: this file is SYNTAX-checked (py_compile) and ships a shape self-test
(`python signal_agent.py`, needs torch but NO env). The self-test now ALSO asserts the
'learned' channel receives gradient (msg_head/msg_gain move after one update -- the DIAL
fix). Smoke-test against the real env before trusting numbers. Pre-DIAL 'learned'
checkpoints were trained with a FROZEN random-projection channel and must be retrained.
================================================================================
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

AGENTS = ["retailer", "wholesaler", "distributor", "manufacturer"]

# message dimensionality per content type ("learned" uses cfg.learned_msg_dim)
_FIXED_MSG_DIMS = {"dhat": 1, "ip": 1, "dhat_ip": 2}


def msg_dim_of(content, learned_dim):
    return int(learned_dim) if content == "learned" else _FIXED_MSG_DIMS[content]


def order_from_S(S, obs, max_order):
    """Convert an order-up-to level S into the env order. Inventory position
    IP = on_hand - backlog + on_order; order = clip(S - IP, 0, max_order)."""
    inv, back, onord = obs[..., 0:1], obs[..., 1:2], obs[..., 2:3]
    IP = inv - back + onord
    return torch.clamp(S - IP, 0.0, max_order), IP


# ============================================================================ #
# Actor: GRU belief + linear base-stock head + linear demand readout + message  #
# ============================================================================ #
class SIGNALActor(nn.Module):
    def __init__(self, obs_dim, msg_dim, hidden, content, learned_msg_dim=3, s_init=50.0,
                 log_std_init=-0.5, use_dhat_head=False, dhat_coef=5.0, dhat_init=0.0,
                 learned_msg_gain=10.0):
        super().__init__()
        self.content = content
        self.msg_dim = msg_dim
        self.hidden = hidden
        self.use_dhat_head = bool(use_dhat_head)
        # (1) BELIEF. Input = [local obs, received message] so communication can update the belief.
        #     One GRU layer; its hidden state is the entire memory the agent has. No KL, no ELBO.
        self.gru = nn.GRU(obs_dim + msg_dim, hidden)                 # (seq, batch, feat)
        # (2) BASE-STOCK HEAD. ONE linear layer -> scalar S (softplus = positivity link only).
        #     use_dhat_head appends the SMOOTHED, grounded demand estimate d_hat to the head input so S
        #     can condition on the demand regime: S ~= dhat_coef*d_hat + safety (textbook (lead+1)*demand
        #     + safety). Using d_hat -- not raw last_demand -- keeps Poisson noise OUT of S (no bullwhip).
        #     This is the deliberate demand->S coupling (OFF by default): the sweep showed optimization
        #     alone cannot make the constant base-stock adapt, and the retailer fails to scale S even
        #     with perfect local demand. It also gives COMM a path to S: a received message updates the
        #     belief h -> d_hat -> S, so sharing can change ordering (testable with topology/AR1).
        head_in = obs_dim + hidden + msg_dim + (1 if self.use_dhat_head else 0)
        self.head = nn.Linear(head_in, 1)
        nn.init.constant_(self.head.bias, float(s_init))            # safety floor / constant level
        if self.use_dhat_head:
            with torch.no_grad():
                self.head.weight[0, -1] = float(dhat_coef)         # weight on the appended d_hat ~ lead+1
        self.log_std = nn.Parameter(torch.zeros(1) + float(log_std_init))   # exploration noise on S (init std=exp(log_std_init))
        # (3) DEMAND READOUT. Linear belief -> d_hat (>=0). The 'dhat' message, grounded to realized
        #     demand by the aux loss, AND (if use_dhat_head) the driver of the base-stock above.
        self.d_head = nn.Linear(hidden, 1)
        if dhat_init:                                               # start d_hat ~ mean demand (softplus link)
            nn.init.constant_(self.d_head.bias, float(dhat_init))
        # (4) LEARNED message head: ONLY built for the interpretability-control rung. A learnable GAIN
        #     lifts the tanh output (bounded [-1,1]) to the SAME O(10) natural-unit scale as the dhat/ip
        #     messages, so the receiver can weight it comparably -- without it the learned rung is
        #     scale-handicapped (the /100 bug, but for learned) and a null result would be uninterpretable
        #     (deaf receiver vs redundant content). Init = learned_msg_gain; the optimizer can adjust it.
        #     NOTE: these parameters are trained ONLY via the DIAL path in SIGNALTrainer.update()
        #     (_coupled_forward); they appear in no other loss.
        self.msg_head = (nn.Linear(obs_dim + hidden, learned_msg_dim)
                         if content == "learned" else None)
        self.msg_gain = (nn.Parameter(torch.tensor(float(learned_msg_gain)))
                         if content == "learned" else None)

    def belief(self, obs_seq, msg_seq, h0=None):
        """obs_seq [T,B,obs], msg_seq [T,B,msg] -> (h_seq [T,B,hidden], hN [1,B,hidden])."""
        return self.gru(torch.cat([obs_seq, msg_seq], dim=-1), h0)

    def demand_estimate(self, h):                                   # belief -> nonneg demand forecast
        return F.softplus(self.d_head(h))

    def base_stock(self, obs, h, msg):                             # -> (S_mu>=0, S_std)
        feats = [obs, h, msg] + ([self.demand_estimate(h)] if self.use_dhat_head else [])
        S_mu = F.softplus(self.head(torch.cat(feats, dim=-1)))
        return S_mu, self.log_std.exp().clamp(1e-2, 3.0)

    def message(self, obs, h):
        """The semantic ladder. Each rung is a NAMED supply-chain signal except 'learned'.
        SCALE: the named messages ride at NATURAL units (demand/inventory, ~O(10)) so they are
        COMMENSURATE with the raw-unit observations they sit beside in the receiver's GRU input and
        base-stock head. An earlier /100 scaling made the message ~100x smaller than obs, so the
        receiver could not give it usable weight -- the positive-listening probe read dS/dmsg ~ 0
        (honest messages, but the agent was deaf). See eval_signal.positive_listening + _MSG_SCALE."""
        IP = (obs[..., 0:1] - obs[..., 1:2] + obs[..., 2:3])              # inventory-position signal (VMI), natural units
        dh = self.demand_estimate(h)                                     # demand signal (Lee), natural units
        if self.content == "dhat":     return dh
        if self.content == "ip":       return IP
        if self.content == "dhat_ip":  return torch.cat([dh, IP], dim=-1)
        # 'learned' = gain * tanh(...) so its bounded code rides at the SAME O(10) scale as dhat/ip and is
        # audible to the receiver (see msg_gain). tanh keeps it bounded; the gain restores the magnitude.
        if self.content == "learned":  return self.msg_gain * torch.tanh(self.msg_head(torch.cat([obs, h], dim=-1)))
        raise ValueError(self.content)


# ============================================================================ #
# Critic: centralized, per-agent value of the global state (CTDE, training only) #
# ============================================================================ #
class SIGNALCritic(nn.Module):
    def __init__(self, state_dim, hidden, n_agents):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_agents))                            # one value per agent (per-agent rewards)

    def forward(self, s):                                          # s [.., state_dim] -> [.., N]
        return self.net(s)


# ============================================================================ #
# Trainer: plain MAPPO (PPO-clip + GAE) + the economics (beta cost-share, tau)   #
# ============================================================================ #
class SIGNALTrainer:
    def __init__(self, cfg, n_agents, obs_dim, state_dim, adj, device="cpu"):
        self.device = torch.device(device)
        self.N = n_agents
        self.max_order = float(cfg.get("max_order", 100.0))
        self.content = cfg.get("msg_content", "dhat")
        self.msg_dim = msg_dim_of(self.content, cfg.get("learned_msg_dim", 3))
        self.adj = torch.tensor(adj, dtype=torch.float32, device=self.device)     # row-stochastic topology
        H = int(cfg.get("hidden", 64))
        s_init = float(cfg.get("s_init", 50.0))                    # base-stock head bias / safety term (see SIGNALActor)
        log_std_init = float(cfg.get("log_std_init", -0.5))        # initial exploration std on S = exp(log_std_init)
        use_dhat_head = bool(cfg.get("use_dhat_head", False))      # condition S on the smoothed d_hat (regime-adaptive)
        dhat_coef = float(cfg.get("dhat_coef", 5.0))               # init weight on d_hat in the S-head (~ lead+1)
        dhat_init = float(cfg.get("dhat_init", 0.0))               # init d_hat ~ mean demand so S starts sensible
        self.actors = nn.ModuleList([                              # per-agent actors (chain is heterogeneous)
            SIGNALActor(obs_dim, self.msg_dim, H, self.content, cfg.get("learned_msg_dim", 3),
                        s_init=s_init, log_std_init=log_std_init, use_dhat_head=use_dhat_head,
                        dhat_coef=dhat_coef, dhat_init=dhat_init,
                        learned_msg_gain=float(cfg.get("learned_msg_gain", 10.0)))
            for _ in range(n_agents)]).to(self.device)
        self.critic = SIGNALCritic(state_dim, H, n_agents).to(self.device)
        # --- optimisation (textbook PPO) ---
        self.gamma = float(cfg.get("gamma", 0.99)); self.lam = float(cfg.get("gae_lambda", 0.95))
        self.eps = float(cfg.get("eps_clip", 0.2)); self.k_epochs = int(cfg.get("k_epochs", 4))
        self.ent_coef = float(cfg.get("entropy_coef", 0.0)); self.vf_coef = float(cfg.get("vf_coef", 0.5))
        self.aux_coef = float(cfg.get("aux_coef", 0.1))           # d_hat grounding (message interpretability)
        self.max_grad_norm = float(cfg.get("max_grad_norm", 0.5))
        # reward_scale DIVIDES the shaped reward -> shrinks the critic's target/loss/gradient so it no
        # longer dominates the shared grad-clip and starve the (standardized-advantage) actor. The gate
        # still scores RAW env cost, so it stays comparable across reward_scale values. 1.0 = unchanged.
        self.reward_scale = float(cfg.get("reward_scale", 1.0))
        self.params = list(self.actors.parameters()) + list(self.critic.parameters())
        self.opt = torch.optim.Adam(self.params, lr=float(cfg.get("lr", 3e-4)))
        # --- economics layer (independent of the agent) ---
        self.srdqn_beta = float(cfg.get("srdqn_beta", 1.0))       # 1=cooperative, 0=self-interested
        tau = float(cfg.get("tau", 0.0))                          # coordinating transfer (0=no contract)
        self.tau_vec = torch.tensor([0.0] + [tau] * (n_agents - 1),
                                    device=self.device).view(1, -1)  # charged on UPSTREAM backorders only

    # ---------------------------------------------------------------- rollout
    @torch.no_grad()
    def collect(self, env, seed, deterministic=False):
        """Roll ONE episode. Messages are stored as RECEIVED and DETACHED. In the update they are
        fixed extended observations for the NAMED contents (plain MAPPO -- a clean, defensible
        default that keeps d_hat/ip semantics grounded). For content='learned' the update IGNORES
        the stored messages and recomputes them IN-GRAPH (_coupled_forward), so the channel trains
        (DIAL). The rollout itself is identical either way: deterministic messages, one-step delay."""
        dev = self.device
        obs, _ = env.reset(seed=seed)
        h = [torch.zeros(1, 1, self.actors[i].hidden, device=dev) for i in range(self.N)]
        m_prev = torch.zeros(self.N, self.msg_dim, device=dev)
        O, MIN, GS, SA, LP, C, DT, DN = ([] for _ in range(8))
        while True:
            o = torch.tensor(np.stack([obs[a] for a in AGENTS]), dtype=torch.float32, device=dev)  # [N,obs]
            incoming = self.adj @ m_prev                                       # [N,msg] (delivered next step)
            S = torch.zeros(self.N, 1, device=dev)
            logp = torch.zeros(self.N, device=dev)
            m_out = torch.zeros(self.N, self.msg_dim, device=dev)
            for i in range(self.N):
                h_seq, h[i] = self.actors[i].belief(o[i].view(1, 1, -1), incoming[i].view(1, 1, -1), h[i])
                hi = h_seq[-1]                                                 # [1,hidden]
                S_mu, S_std = self.actors[i].base_stock(o[i:i + 1], hi, incoming[i:i + 1])
                Si = S_mu if deterministic else Normal(S_mu, S_std).sample()
                logp[i] = Normal(S_mu, S_std).log_prob(Si).sum()
                S[i] = Si
                m_out[i] = self.actors[i].message(o[i:i + 1], hi).reshape(-1)
            order, _ = order_from_S(S, o, self.max_order)
            frac = (order / self.max_order).clamp(0.0, 1.0)
            gstate = torch.tensor(env.get_global_state(), dtype=torch.float32, device=dev).view(-1)
            nobs, _r, terms, truncs, infos = env.step({a: [float(frac[i, 0])] for i, a in enumerate(AGENTS)})
            cost = torch.tensor([infos[a]["local_cost"] for a in AGENTS], dtype=torch.float32, device=dev)
            dtgt = torch.tensor([infos[a]["training_targets"]["demand"] for a in AGENTS],
                                dtype=torch.float32, device=dev)
            done = bool(any(terms.values()) or any(truncs.values()))
            O.append(o); MIN.append(incoming.detach()); GS.append(gstate)
            SA.append(S.detach()); LP.append(logp.detach()); C.append(cost); DT.append(dtgt)
            DN.append(torch.tensor(1.0 if done else 0.0, device=dev))
            m_prev = m_out.detach()                                            # ONE-STEP message delay
            obs = nobs
            if done:
                break
        return dict(obs=torch.stack(O), msg_in=torch.stack(MIN), gstate=torch.stack(GS),
                    S=torch.stack(SA), logp=torch.stack(LP), cost=torch.stack(C),
                    dtgt=torch.stack(DT), done=torch.stack(DN))                # all [T, N, *] or [T,*]

    # ------------------------------------------------------------------- GAE
    def _gae(self, rew, V, done):
        # NOTE (truncation vs termination): episodes end by TIME-LIMIT truncation (the env's
        # terminations are always False), but `done` marks the final step, so the last step is treated
        # as terminal -- no value bootstrap from V(s_T). At the fixed horizon this is a small CONSTANT
        # bias; bootstrap the truncated tail if you ever vary the horizon or need the absolute return scale.
        T = rew.size(0); adv = torch.zeros_like(rew); last = torch.zeros((), device=rew.device)
        for t in reversed(range(T)):
            nonterm = 1.0 - done[t]
            v_next = V[t + 1] if t + 1 < T else torch.zeros((), device=rew.device)
            delta = rew[t] + self.gamma * v_next * nonterm - V[t]
            last = delta + self.gamma * self.lam * nonterm * last
            adv[t] = last
        return adv

    # ------------------------------------------- DIAL forward (learned rung ONLY)
    def _coupled_forward(self, obs_seq):
        """Joint IN-GRAPH forward of all agents, coupled through ADJ with the SAME one-step message
        delay as collect(): incoming_t = ADJ @ msg_{t-1}, msg_t = f_i(obs_t, h_t). Used by update()
        ONLY when content='learned', so gradients flow  receiver-loss -> incoming -> ADJ -> sender
        msg_gain / msg_head / GRU  (DIAL; Foerster et al. 2016) and the channel actually TRAINS.

        [FIX: previously the learned message was collected DETACHED and never re-entered any loss,
        so msg_head/msg_gain received no gradient and stayed at initialization forever -- a frozen
        random projection. The H5 'unconstrained learned channel' rung was therefore vacuous.
        Pre-fix 'learned' checkpoints must be retrained.]

        FRAMING (why this does not change PPO's semantics): with the one-step delay, the four actors
        + channel are ONE joint recurrent policy; the messages are cross-agent hidden state.
        Recomputing them across the k_epochs is exactly what recurrent PPO already does with h --
        at epoch 1 the recomputed messages equal the stored rollout messages bit-for-bit (same
        parameters, deterministic channel), and later epochs drift the same way h drifts.

        WHY the NAMED rungs are excluded: dhat must remain a demand forecast grounded by the aux
        loss -- a team-gradient through the channel would co-opt it into an arbitrary control signal
        and void the honesty probe's premise; ip is a parameter-free obs readout with nothing to
        train. Detaching the named channels is therefore load-bearing, not a shortcut.

        COST: stepwise T*N single-step GRU calls instead of fused per-agent sequences (~0.3-0.5 s
        per update at T=50, N=4, hidden=64 on CPU) -- learned arm only; named arms keep the fast
        path byte-identical.

        obs_seq [T,N,obs] -> (h_all [T,N,hidden], msg_in_live [T,N,msg])."""
        T = obs_seq.size(0)
        dev = obs_seq.device
        h = [torch.zeros(1, 1, self.actors[i].hidden, device=dev) for i in range(self.N)]
        m_prev = torch.zeros(self.N, self.msg_dim, device=dev)
        H_all, MIN = [], []
        for t in range(T):
            incoming = self.adj @ m_prev                          # [N,msg] (t-1 messages, IN-graph)
            o = obs_seq[t]                                        # [N,obs]
            h_t, m_t = [], []
            for i in range(self.N):
                h_seq, h[i] = self.actors[i].belief(o[i].view(1, 1, -1),
                                                    incoming[i].view(1, 1, -1), h[i])
                hi = h_seq[-1]                                    # [1,hidden]
                h_t.append(hi.squeeze(0))                         # [hidden]
                m_t.append(self.actors[i].message(o[i:i + 1], hi).reshape(-1))
            H_all.append(torch.stack(h_t))                        # [N,hidden]
            MIN.append(incoming)                                  # [N,msg]
            m_prev = torch.stack(m_t)                             # NOT detached -- the DIAL edge
        return torch.stack(H_all), torch.stack(MIN)               # [T,N,hidden], [T,N,msg]

    # ---------------------------------------------------------------- update
    def update(self, episodes):
        dev = self.device
        # (A) shaped reward = own + beta*others + coordinating transfer (money-conserving) ----------
        for d in episodes:
            c = d["cost"]                                          # [T,N] per-agent local cost
            others = c.sum(-1, keepdim=True) - c
            back = d["obs"][..., 1]                                # [T,N] backlog (obs index 1)
            transfer = self.tau_vec * back                        # paid BY each stage for its backorders
            credit = torch.zeros_like(transfer); credit[..., :-1] = transfer[..., 1:]  # credit its customer
            d["rew"] = (-(c + self.srdqn_beta * others) - transfer + credit) / self.reward_scale
        # (B) advantages from the centralized critic (per-agent) ------------------------------------
        with torch.no_grad():
            for d in episodes:
                V = self.critic(d["gstate"])                      # [T,N]
                adv = torch.stack([self._gae(d["rew"][:, i], V[:, i], d["done"]) for i in range(self.N)], dim=1)
                d["V"], d["adv"] = V, adv
                d["vtarget"] = (adv + V).detach()                 # GAE lambda-return
            all_adv = torch.cat([d["adv"] for d in episodes], 0)  # standardize advantages (PPO practice)
            a_mu, a_sd = all_adv.mean(0), all_adv.std(0) + 1e-8
            for d in episodes:
                d["adv"] = (d["adv"] - a_mu) / a_sd
        # (C) PPO-clip actor + MSE critic, re-running the GRU over stored sequences (BPTT). ---------
        #     For content='learned' the MESSAGES are also recomputed in-graph via _coupled_forward
        #     (DIAL), so the channel is trainable. Named contents keep the stored, DETACHED messages
        #     (fixed semantics; grounded content). Same PPO surrogate either way.
        a_loss = c_loss = 0.0
        for _ in range(self.k_epochs):
            self.opt.zero_grad()
            total = torch.zeros((), device=dev)
            for d in episodes:
                # critic (shared across agents; messages do not enter the critic)
                V = self.critic(d["gstate"])                      # [T,N]
                closs = F.mse_loss(V, d["vtarget"])
                ploss = torch.zeros((), device=dev)
                if self.content == "learned":                     # DIAL: one joint in-graph forward
                    h_live, min_live = self._coupled_forward(d["obs"])   # [T,N,hid], [T,N,msg]
                for i in range(self.N):
                    if self.content == "learned":
                        h = h_live[:, i, :]                       # [T,hidden]  (channel in-graph)
                        msg_i_used = min_live[:, i, :]            # [T,msg]     (channel in-graph)
                    else:
                        obs_i = d["obs"][:, i, :].unsqueeze(1)    # [T,1,obs]
                        msg_i = d["msg_in"][:, i, :].unsqueeze(1) # [T,1,msg]  (fixed/detached from rollout)
                        h_seq, _ = self.actors[i].belief(obs_i, msg_i)
                        h = h_seq.squeeze(1)                      # [T,hidden]
                        msg_i_used = d["msg_in"][:, i, :]
                    S_mu, S_std = self.actors[i].base_stock(d["obs"][:, i, :], h, msg_i_used)
                    logp_new = Normal(S_mu, S_std).log_prob(d["S"][:, i, :]).sum(-1)   # [T]
                    ratio = torch.exp(logp_new - d["logp"][:, i])                       # pi_new/pi_old
                    A = d["adv"][:, i]
                    surr = torch.min(ratio * A, torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * A)
                    ent = Normal(S_mu, S_std).entropy().mean()
                    ploss = ploss - surr.mean() - self.ent_coef * ent
                    # d_hat grounding: keeps the demand message INTERPRETABLE as a forecast (light)
                    dhat = self.actors[i].demand_estimate(h)                            # [T,1]
                    ploss = ploss + self.aux_coef * F.mse_loss(dhat, d["dtgt"][:, i].unsqueeze(-1))
                total = total + (ploss + self.vf_coef * closs) / len(episodes)
                a_loss += float(ploss.item()); c_loss += float(closs.item())
            total.backward()
            nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)
            self.opt.step()
        k = max(1, self.k_epochs * len(episodes))
        return a_loss / k, c_loss / k


# ============================================================================ #
# Shape self-test (run: `python signal_agent.py`). Needs torch, NOT the env.     #
# Now also asserts the DIAL fix: for content='learned', msg_head/msg_gain MOVE   #
# after one update (they receive gradient through the coupled in-graph channel). #
# ============================================================================ #
if __name__ == "__main__":
    torch.manual_seed(0)
    OBS, N, H, T = 4, 4, 64, 20
    for content in ["dhat", "ip", "dhat_ip", "learned"]:
        md = msg_dim_of(content, 3)
        STATE = OBS * N
        cfg = {"msg_content": content, "hidden": H, "learned_msg_dim": 3,
               "srdqn_beta": 0.0, "tau": 9.5, "k_epochs": 2}
        adj = np.array([[0, 1, 0, 0], [.5, 0, .5, 0], [0, .5, 0, .5], [0, 0, 1, 0]], float)
        tr = SIGNALTrainer(cfg, N, OBS, STATE, adj, device="cpu")
        # fabricate a buffer with the exact shapes collect() produces, then run an update
        ep = {"obs": torch.rand(T, N, OBS) * 20, "msg_in": torch.rand(T, N, md),
              "gstate": torch.rand(T, STATE) * 20, "S": torch.rand(T, N, 1) * 40,
              "logp": torch.randn(T, N), "cost": torch.rand(T, N) * 50,
              "dtgt": torch.rand(T, N) * 20, "done": torch.zeros(T)}
        ep["done"][-1] = 1.0
        # DIAL gradient-flow assertion: snapshot the learned channel's params before the update.
        pre_w = pre_g = None
        if content == "learned":
            pre_w = tr.actors[0].msg_head.weight.detach().clone()
            pre_g = float(tr.actors[0].msg_gain.detach())
            hh, mm = tr._coupled_forward(ep["obs"])                 # shape contract of the DIAL path
            assert hh.shape == (T, N, H) and mm.shape == (T, N, md), (hh.shape, mm.shape)
            assert mm.requires_grad, "coupled messages must be IN-graph (DIAL edge)"
        else:
            assert tr.actors[0].msg_head is None and tr.actors[0].msg_gain is None
        a, c = tr.update([ep, dict(ep)])
        dial_note = ""
        if content == "learned":
            dw = float((tr.actors[0].msg_head.weight.detach() - pre_w).abs().max())
            dg = abs(float(tr.actors[0].msg_gain.detach()) - pre_g)
            assert dw > 1e-9 and dg > 1e-12, \
                f"DIAL broken: learned channel got no gradient (dW={dw:.2e}, dGain={dg:.2e})"
            dial_note = f"  DIAL: dW={dw:.1e} dGain={dg:.1e} (channel trains)"
        # exercise a single-agent forward at one step
        h0 = torch.zeros(1, 1, H)
        hs, _ = tr.actors[0].belief(torch.rand(1, 1, OBS), torch.rand(1, 1, md), h0)
        Smu, Ssd = tr.actors[0].base_stock(torch.rand(1, OBS), hs[-1], torch.rand(1, md))
        m = tr.actors[0].message(torch.rand(1, OBS), hs[-1])
        assert Smu.shape == (1, 1) and m.shape[-1] == md
        n_params = sum(p.numel() for p in tr.params)
        print(f"  content={content:<8} msg_dim={md}  params={n_params:>7,}  "
              f"a_loss={a:+.3f} c_loss={c:+.3f}  S={float(Smu):.1f}  msg_dim_out={m.shape[-1]}  OK{dial_note}")
    print("signal_agent shape self-test PASS (all four message rungs build, update, and act;")
    print("                                  'learned' channel verified TRAINABLE via DIAL)")