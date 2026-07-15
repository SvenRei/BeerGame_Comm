
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

AGENTS = ["retailer", "wholesaler", "distributor", "manufacturer"]

# Message dimensionality per content type ("learned" uses cfg.learned_msg_dim).
# "raw" = the sender's last realized incoming demand (obs slot 3): point-of-sale data sharing
# (Cachon-Fisher 2000), the data counterpart to the "dhat" forecast rung (Aviv 2001/2007).
_FIXED_MSG_DIMS = {"raw": 1, "dhat": 1, "ip": 1, "dhat_ip": 2}


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
        # (1) Belief: a single GRU over [local obs, received message]; its hidden state is the agent's
        #     entire memory and demand-regime belief (trained end-to-end; no KL/ELBO objective).
        self.gru = nn.GRU(obs_dim + msg_dim, hidden)                 # (seq, batch, feat)
        # (2) Base-stock head: one linear layer -> scalar S (softplus is a positivity link only).
        #     With use_dhat_head, the smoothed demand estimate d_hat is appended so S conditions on the
        #     demand regime: S ~= dhat_coef*d_hat + safety (textbook (lead+1)*mean-demand + safety).
        #     Using d_hat rather than the raw last demand keeps sampling noise out of S (no bullwhip).
        #     This demand->S coupling also gives communication a path to ordering (message -> belief h ->
        #     d_hat -> S); disabling it recovers a constant base-stock level.
        head_in = obs_dim + hidden + msg_dim + (1 if self.use_dhat_head else 0)
        self.head = nn.Linear(head_in, 1)
        nn.init.constant_(self.head.bias, float(s_init))            # safety floor / constant level
        if self.use_dhat_head:
            with torch.no_grad():
                self.head.weight[0, -1] = float(dhat_coef)         # weight on the appended d_hat ~ lead+1
        self.log_std = nn.Parameter(torch.zeros(1) + float(log_std_init))   # exploration noise on S (init std=exp(log_std_init))
        # (3) Demand readout: linear belief -> d_hat (>=0). Serves as the "dhat" message (grounded to
        #     realized demand by the aux loss) and, if use_dhat_head, drives the base-stock head above.
        self.d_head = nn.Linear(hidden, 1)
        if dhat_init:                                               # start d_hat ~ mean demand (softplus link)
            nn.init.constant_(self.d_head.bias, float(dhat_init))
        # (4) Learned message head: built only for the interpretability-control rung. A learnable gain
        #     lifts the tanh output ([-1,1]) to the O(10) natural-unit scale of the dhat/ip messages so
        #     the receiver can weight it comparably; without it the learned rung is scale-handicapped and
        #     a null result would be ambiguous (deaf receiver vs. redundant content). These parameters
        #     train only via the DIAL path in SIGNALTrainer.update() (_coupled_forward).
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
        """Semantic message ladder; every rung is a named supply-chain signal except "learned".
        The named messages ride at natural units (demand/inventory, ~O(10)) so they are commensurate
        with the raw-unit observations beside them in the receiver's GRU input and base-stock head.
        Scaling the message far below obs magnitude leaves the receiver unable to weight it (positive-
        listening dS/dmsg ~ 0 despite honest messages); see eval_signal.positive_listening."""
        IP = (obs[..., 0:1] - obs[..., 1:2] + obs[..., 2:3])              # inventory-position signal (VMI), natural units
        dh = self.demand_estimate(h)                                     # demand signal (Lee), natural units
        # "raw" = the sender's last realized incoming demand (obs slot 3): unprocessed point-of-sale
        # data (Cachon-Fisher 2000). A parameter-free obs readout, honest by construction, at natural
        # units (like "ip"). Emitted at step t it carries d_{t-1}, so the honesty probe's same-step
        # correlation is <1 from one-step alignment, not misreporting. Registered contrast: dhat vs raw
        # = forecast vs data.
        if self.content == "raw":      return obs[..., 3:4]
        if self.content == "dhat":     return dh
        if self.content == "ip":       return IP
        if self.content == "dhat_ip":  return torch.cat([dh, IP], dim=-1)
        # "learned" = gain * tanh(...): a bounded code rescaled to the O(10) scale of dhat/ip so it is
        # audible to the receiver (see msg_gain). tanh bounds it; the gain restores the magnitude.
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
        # reward_scale divides the shaped reward, shrinking the critic's target/loss/gradient so it no
        # longer dominates the shared gradient clip and starves the (standardized-advantage) actor. The
        # gate scores raw env cost, so results stay comparable across reward_scale values (1.0 = off).
        self.reward_scale = float(cfg.get("reward_scale", 1.0))
        self.params = list(self.actors.parameters()) + list(self.critic.parameters())
        self.opt = torch.optim.Adam(self.params, lr=float(cfg.get("lr", 3e-4)))
        # --- economics layer (independent of the agent) ---
        self.srdqn_beta = float(cfg.get("srdqn_beta", 1.0))       # 1=cooperative, 0=self-interested
        tau = float(cfg.get("tau", 0.0))                          # coordinating transfer (0=no contract)
        self.tau_vec = torch.tensor([0.0] + [tau] * (n_agents - 1),
                                    device=self.device).view(1, -1)  # charged on UPSTREAM backorders only
        # --- Pure-observation logging sinks (see agents/signal_csvlog.py) ---------------------------
        # None unless train_signal runs with SIGNAL_CSVLOG=1, which sets them before update()/collect()
        # and reads them after. Every write below is guarded by `is not None` and only reads tensors
        # already computed here (detached, under no_grad): no random draw, no reordering, no change to
        # any registered number (loss, gradient, Adam step, checkpoint, RNG stream). Unset => this class
        # is byte-identical to the frozen instrument.
        self.upd_obs = None       # dict : per-update PPO diagnostics (approx_kl, clip_frac, grad_norm, ...)
        self.roll_obs = None      # list : per-collect rollout tensors (emitted msg, d_hat, belief h)

    # ---------------------------------------------------------------- rollout
    @torch.no_grad()
    def collect(self, env, seed, deterministic=False):
        """Roll one episode. Messages are stored as received and detached: in the update they are
        fixed extended observations for the named contents (plain MAPPO, which keeps d_hat/ip semantics
        grounded). For content="learned" the update ignores the stored messages and recomputes them
        in-graph (_coupled_forward) so the channel trains (DIAL). The rollout is identical either way:
        deterministic messages with a one-step delay."""
        dev = self.device
        obs, _ = env.reset(seed=seed)
        h = [torch.zeros(1, 1, self.actors[i].hidden, device=dev) for i in range(self.N)]
        m_prev = torch.zeros(self.N, self.msg_dim, device=dev)
        O, MIN, GS, SA, LP, C, DT, DN = ([] for _ in range(8))
        R_MSG, R_DHAT, R_H = ([], [], []) if self.roll_obs is not None else (None, None, None)  # obs-only sink
        while True:
            o = torch.tensor(np.stack([obs[a] for a in AGENTS]), dtype=torch.float32, device=dev)  # [N,obs]
            incoming = self.adj @ m_prev                                       # [N,msg] (delivered next step)
            S = torch.zeros(self.N, 1, device=dev)
            logp = torch.zeros(self.N, device=dev)
            m_out = torch.zeros(self.N, self.msg_dim, device=dev)
            _h_step = [] if self.roll_obs is not None else None
            _d_step = [] if self.roll_obs is not None else None
            for i in range(self.N):
                h_seq, h[i] = self.actors[i].belief(o[i].view(1, 1, -1), incoming[i].view(1, 1, -1), h[i])
                hi = h_seq[-1]                                                 # [1,hidden]
                S_mu, S_std = self.actors[i].base_stock(o[i:i + 1], hi, incoming[i:i + 1])
                Si = S_mu if deterministic else Normal(S_mu, S_std).sample()
                logp[i] = Normal(S_mu, S_std).log_prob(Si).sum()
                S[i] = Si
                m_out[i] = self.actors[i].message(o[i:i + 1], hi).reshape(-1)
                if self.roll_obs is not None:                                 # read-only: forecast + belief on hand
                    _h_step.append(hi.reshape(-1))
                    _d_step.append(self.actors[i].demand_estimate(hi).reshape(-1))
            if self.roll_obs is not None:
                R_MSG.append(m_out.detach().clone())                          # [N,msg] emitted this step
                R_H.append(torch.stack(_h_step))                             # [N,hidden] belief that fed the head
                R_DHAT.append(torch.cat(_d_step))                            # [N] demand estimate
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
        if self.roll_obs is not None:                                          # one record per collect() call
            self.roll_obs.append({"msg_out": torch.stack(R_MSG).cpu().numpy(),   # [T,N,msg]
                                  "dhat":    torch.stack(R_DHAT).cpu().numpy(),   # [T,N]
                                  "h":       torch.stack(R_H).cpu().numpy()})     # [T,N,hidden]
        return dict(obs=torch.stack(O), msg_in=torch.stack(MIN), gstate=torch.stack(GS),
                    S=torch.stack(SA), logp=torch.stack(LP), cost=torch.stack(C),
                    dtgt=torch.stack(DT), done=torch.stack(DN))                # all [T, N, *] or [T,*]

    # ------------------------------------------------------------------- GAE
    def _gae(self, rew, V, done):
        # Truncation vs termination: episodes end by time-limit truncation (env terminations are always
        # False), but `done` marks the final step, so the last step is treated as terminal with no value
        # bootstrap from V(s_T). At a fixed horizon this is a small constant bias; bootstrap the
        # truncated tail if the horizon varies or the absolute return scale is needed.
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
        """Joint in-graph forward of all agents, coupled through ADJ with the same one-step message
        delay as collect(): incoming_t = ADJ @ msg_{t-1}, msg_t = f_i(obs_t, h_t). Used by update()
        only when content="learned", so gradients flow receiver-loss -> incoming -> ADJ -> sender
        (msg_gain / msg_head / GRU); this is DIAL (Foerster et al. 2016) and is what trains the channel.
        Without it the learned message is detached at collect time and never re-enters a loss, so its
        parameters stay at initialization (a frozen random projection) and the learned rung is vacuous.

        This does not change PPO's semantics: with the one-step delay the four actors plus channel form
        one joint recurrent policy whose messages are cross-agent hidden state, recomputed across the
        k_epochs exactly as recurrent PPO recomputes h. At epoch 1 the recomputed messages equal the
        stored rollout messages (same parameters, deterministic channel); later epochs drift as h does.

        The named rungs are excluded deliberately: dhat must stay a demand forecast grounded by the aux
        loss (a team gradient through the channel would co-opt it and void the honesty probe), and ip/raw
        are parameter-free obs readouts with nothing to train.

        Cost: T*N single-step GRU calls instead of fused per-agent sequences (~0.3-0.5 s per update at
        T=50, N=4, hidden=64 on CPU), for the learned arm only; named arms keep the fast path.

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
            if self.upd_obs is not None:          # OBSERVATION: critic fit at update time (read-only)
                V_all = torch.cat([d["V"].reshape(-1) for d in episodes])
                vt_all = torch.cat([d["vtarget"].reshape(-1) for d in episodes])
                var_t = float(vt_all.var(unbiased=False))
                self.upd_obs["explained_variance"] = (
                    1.0 - float((vt_all - V_all).var(unbiased=False)) / var_t) if var_t > 1e-12 else float("nan")
        # (C) PPO-clip actor + MSE critic, re-running the GRU over stored sequences (BPTT). For
        #     content="learned" the messages are also recomputed in-graph via _coupled_forward (DIAL),
        #     making the channel trainable; named contents keep the stored, detached messages (fixed,
        #     grounded semantics). Same PPO surrogate either way.
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
                    # d_hat grounding: keeps the demand message interpretable as a forecast (light aux loss)
                    dhat = self.actors[i].demand_estimate(h)                            # [T,1]
                    aux = F.mse_loss(dhat, d["dtgt"][:, i].unsqueeze(-1))               # (named for observation only)
                    ploss = ploss + self.aux_coef * aux
                    if self.upd_obs is not None:   # OBSERVATION: PPO trust-region health (read-only)
                        with torch.no_grad():
                            ob = self.upd_obs
                            ob.setdefault("approx_kl", []).append(float((d["logp"][:, i] - logp_new).mean()))
                            ob.setdefault("clip_frac", []).append(
                                float((torch.abs(ratio - 1.0) > self.eps).float().mean()))
                            ob.setdefault("entropy", []).append(float(ent))
                            ob.setdefault("aux_loss", []).append(float(aux))
                            ob.setdefault("S_mean", []).append(float(S_mu.mean()))
                            ob.setdefault("action_std", []).append(float(S_std.mean()))
                total = total + (ploss + self.vf_coef * closs) / len(episodes)
                a_loss += float(ploss.item()); c_loss += float(closs.item())
            total.backward()
            gnorm = nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)   # RETURNS the pre-clip norm
            if self.upd_obs is not None:          # OBSERVATION: dead-gradient signature (read-only)
                self.upd_obs.setdefault("grad_norm", []).append(float(gnorm))
            self.opt.step()
        k = max(1, self.k_epochs * len(episodes))
        return a_loss / k, c_loss / k


# ============================================================================ #
# Shape self-test (run: `python signal_agent.py`). Needs torch, not the env.     #
# Also asserts DIAL: for content="learned", msg_head/msg_gain move after one      #
# update (they receive gradient through the coupled in-graph channel).            #
# ============================================================================ #
if __name__ == "__main__":
    torch.manual_seed(0)
    OBS, N, H, T = 4, 4, 64, 20
    for content in ["raw", "dhat", "ip", "dhat_ip", "learned"]:
        md = msg_dim_of(content, 3)
        STATE = OBS * N
        cfg = {"msg_content": content, "hidden": H, "learned_msg_dim": 3,
               "srdqn_beta": 0.0, "tau": 9.5, "k_epochs": 2}
        adj = np.array([[0, 1, 0, 0], [.5, 0, .5, 0], [0, .5, 0, .5], [0, 0, 1, 0]], float)
        tr = SIGNALTrainer(cfg, N, OBS, STATE, adj, device="cpu")
        # Fabricate a buffer with the exact shapes collect() produces, then run an update. S and logp
        # must come from the actual policy: fully fabricated values make the PPO ratio underflow to 0
        # (stored S ~ U(0,40) vs S_mu ~ 50 at std ~ 0.6 -> logp_new ~ -1e3 -> exp() == 0), so the
        # surrogate carries no gradient and a dead policy-gradient path would go undetected.
        ep = {"obs": torch.rand(T, N, OBS) * 20, "msg_in": torch.rand(T, N, md),
              "gstate": torch.rand(T, STATE) * 20, "cost": torch.rand(T, N) * 50,
              "dtgt": torch.rand(T, N) * 20, "done": torch.zeros(T)}
        ep["done"][-1] = 1.0
        with torch.no_grad():
            Ss, LPs = [], []
            for i in range(N):
                h_seq, _ = tr.actors[i].belief(ep["obs"][:, i, :].unsqueeze(1),
                                               ep["msg_in"][:, i, :].unsqueeze(1))
                mu, sd = tr.actors[i].base_stock(ep["obs"][:, i, :], h_seq.squeeze(1),
                                                 ep["msg_in"][:, i, :])
                Si = Normal(mu, sd).sample()
                Ss.append(Si)
                LPs.append(Normal(mu, sd).log_prob(Si).sum(-1))
            ep["S"] = torch.stack(Ss, dim=1)                  # [T,N,1] true old actions
            ep["logp"] = torch.stack(LPs, dim=1)              # [T,N]   true old log-probs -> ratio ~ 1 at epoch 1
        # PPO-path assertion: with true old log-probs the surrogate gradient is nonzero, so the
        # base-stock head must move after an update; a frozen head signals a dead policy-gradient path.
        pre_head = tr.actors[0].head.weight.detach().clone()
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
        d_head = float((tr.actors[0].head.weight.detach() - pre_head).abs().max())
        assert d_head > 1e-9, f"policy-gradient path DEAD: base-stock head unchanged (dW={d_head:.2e})"
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
        o1 = torch.rand(1, OBS) * 20
        Smu, Ssd = tr.actors[0].base_stock(o1, hs[-1], torch.rand(1, md))
        m = tr.actors[0].message(o1, hs[-1])
        assert Smu.shape == (1, 1) and m.shape[-1] == md
        if content == "raw":                      # semantic check: raw IS the last realized incoming
            assert torch.allclose(m, o1[..., 3:4]), "raw message must equal obs[..., 3:4] verbatim"
        n_params = sum(p.numel() for p in tr.params)
        print(f"  content={content:<8} msg_dim={md}  params={n_params:>7,}  "
              f"a_loss={a:+.3f} c_loss={c:+.3f}  S={float(Smu.detach()):.1f}  msg_dim_out={m.shape[-1]}  OK{dial_note}")
    print("signal_agent shape self-test PASS (all FIVE message rungs build, update, and act;")
    print("                                  policy-gradient path verified LIVE (head moves);")
    print("                                  'learned' channel verified TRAINABLE via DIAL)")