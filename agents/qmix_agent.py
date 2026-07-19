"""qmix_agent.py -- recurrent QMIX for the SIGNAL beer game (value-based CTDE robustness arm).

WHY THIS EXISTS
  The editorial review made a second, value-based learner a *blocking* requirement for the primary
  claims: MAPPO (policy-gradient, continuous base-stock) and QMIX (value-based, monotonic value
  factorization) have fundamentally different training objectives, so sign concordance across them is
  the evidence that a finding reflects the supply chain and not one optimizer.

DESIGN (what is shared vs. necessarily different)
  SHARED with SIGNAL, on purpose, so the comparison is fair:
    * the environment, the 4 local-scalar observation, and the ONE-STEP message delay;
    * the received-message belief input: a GRU over [obs, incoming_message] (recurrent memory PRESERVED
      -- the editor insisted the recurrent capacity be kept, so a frame-stacked MLP is NOT used);
    * the message ladder cells the primaries need -- raw (obs slot 3, parameter-free) and dhat (a
      grounded GRU demand readout, aux-MSE to realized demand, exactly SIGNAL's grounding); nocomm is
      the zero-ADJ control (identical net, channel carries zeros);
    * the global state (get_global_state, 133-d) as the mixer's conditioning input (CTDE);
    * the objective: minimize team cost = sum of env local_cost (reward = -team cost).
  NECESSARILY DIFFERENT (this is what makes it a different method, not a reimplementation):
    * actions are DISCRETE order-up-to levels S in a fixed grid; order = clip(S - IP, 0, max_order)
      (same base-stock -> order map as SIGNAL, discretized). Q_i(tau_i, a_i) over the grid.
    * a monotonic mixing network combines per-agent chosen-action Q-values into Q_tot conditioned on
      the global state; nonneg hypernet weights guarantee argmax_a Q_tot == per-agent argmax (IGM),
      which is what licenses decentralized greedy execution.

SCOPE
  raw + dhat + nocomm only (the crossover and clip primaries use exactly these). eps/condmean/
  true_lambda/learned are NOT ported: they are not on the primary path, and keeping the surface small
  keeps the robustness arm auditable. CSV logging (SIGNAL-specific) is unsupported here by design.

The QMIXTrainer mirrors SIGNALTrainer's PUBLIC interface (.collect/.update/.actors/.critic/.params) so
agents/train_signal.py drives it with a one-line learner switch and no change to the gate / best-
checkpoint / budget-milestone machinery.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

AGENTS = ["retailer", "wholesaler", "distributor", "manufacturer"]
_QMIX_MSG_DIMS = {"raw": 1, "dhat": 1}          # the only rungs on the primary path


def order_from_S(S, obs, max_order):
    """Identical base-stock -> order map as signal_agent.order_from_S (kept local for self-containment)."""
    inv, back, onord = obs[..., 0:1], obs[..., 1:2], obs[..., 2:3]
    IP = inv - back + onord
    return torch.clamp(S - IP, 0.0, max_order), IP


# ============================================================================ #
# Per-agent recurrent Q-network (DRQN) + optional grounded dhat message head     #
# ============================================================================ #
class DRQNAgent(nn.Module):
    def __init__(self, obs_dim, msg_dim, hidden, n_actions, content="raw"):
        super().__init__()
        self.hidden = hidden
        self.content = content
        self.msg_dim = msg_dim
        self.gru = nn.GRU(obs_dim + msg_dim, hidden)                 # belief over [obs, incoming msg]
        self.q_head = nn.Linear(hidden, n_actions)                  # Q(a) over discrete order-up-to grid
        # dhat message head: a grounded demand readout off the recurrent belief (softplus >=0), trained
        # by a light aux MSE to realized demand -- SIGNAL's exact convention, so the dhat message means
        # the same thing (a grounded GRU forecast) under QMIX. Present only for the dhat rung.
        self.d_head = nn.Linear(hidden, 1) if content == "dhat" else None

    def step(self, obs_t, inc_t, h):
        """One timestep. obs_t/inc_t [1,feat]; h [1,1,hidden] -> (q [1,n_actions], dhat [1,1] or None, h')."""
        x = torch.cat([obs_t, inc_t], dim=-1).view(1, 1, -1)
        y, h = self.gru(x, h)
        q = self.q_head(y[-1])
        dhat = F.softplus(self.d_head(y[-1])) if self.d_head is not None else None
        return q, dhat, h

    def q_sequence(self, obs_seq, msg_seq):
        """Whole-episode Q for the update. obs_seq/msg_seq [T,B,*] -> (Q [T,B,n_actions], dhat [T,B,1])."""
        y, _ = self.gru(torch.cat([obs_seq, msg_seq], dim=-1))
        q = self.q_head(y)
        dhat = F.softplus(self.d_head(y)) if self.d_head is not None else None
        return q, dhat

    def emit(self, obs_t, dhat_t):
        """The message this agent SENDS (natural units, like SIGNAL)."""
        if self.content == "dhat":
            return dhat_t.reshape(1, -1)
        return obs_t[..., 3:4].reshape(1, -1)                        # raw: last realized incoming demand


# ============================================================================ #
# Monotonic mixing network (hypernetwork-parameterized; enforces IGM)            #
# ============================================================================ #
class QMixer(nn.Module):
    def __init__(self, n_agents, state_dim, embed=32):
        super().__init__()
        self.n = n_agents
        self.embed = embed
        self.state_norm = nn.LayerNorm(state_dim)                  # bound hypernet inputs (stability)
        self.hyper_w1 = nn.Linear(state_dim, embed * n_agents)      # |.| -> nonneg  (monotonicity)
        self.hyper_b1 = nn.Linear(state_dim, embed)
        self.hyper_w2 = nn.Linear(state_dim, embed)                 # |.| -> nonneg
        self.hyper_b2 = nn.Sequential(nn.Linear(state_dim, embed), nn.ReLU(), nn.Linear(embed, 1))

    def forward(self, q_agents, state):
        """q_agents [B,N] (chosen-action Q per agent), state [B,state_dim] -> Q_tot [B,1]."""
        B = q_agents.size(0)
        state = self.state_norm(state)                            # normalize before every hypernet
        w1 = torch.abs(self.hyper_w1(state)).view(B, self.n, self.embed)   # [B,N,E] >= 0
        b1 = self.hyper_b1(state).view(B, 1, self.embed)
        hidden = F.elu(torch.bmm(q_agents.view(B, 1, self.n), w1) + b1)    # [B,1,E]
        w2 = torch.abs(self.hyper_w2(state)).view(B, self.embed, 1)        # [B,E,1] >= 0
        b2 = self.hyper_b2(state).view(B, 1, 1)
        return (torch.bmm(hidden, w2) + b2).view(B, 1)                     # [B,1]


# ============================================================================ #
# QMIX trainer: episodic replay, epsilon-greedy collect, recurrent TD update     #
# ============================================================================ #
class QMIXTrainer:
    def __init__(self, cfg, n_agents, obs_dim, state_dim, adj, device="cpu"):
        self.device = torch.device(device)
        self.N = n_agents
        self.max_order = float(cfg.get("max_order", 100.0))
        self.content = cfg.get("msg_content", "raw")
        if self.content not in _QMIX_MSG_DIMS:
            raise ValueError(f"QMIX supports msg_content in {list(_QMIX_MSG_DIMS)} (primary-path rungs); "
                             f"got '{self.content}'. eps/condmean/true_lambda/learned are MAPPO-only.")
        self.msg_dim = _QMIX_MSG_DIMS[self.content]
        self.adj = torch.tensor(adj, dtype=torch.float32, device=self.device)
        H = int(cfg.get("hidden", 64))
        # discrete order-up-to grid: order = clip(S - IP, 0, max_order); S in [0, s_max] on n_actions pts
        self.n_actions = int(cfg.get("qmix_n_actions", 21))
        self.s_max = float(cfg.get("qmix_s_max", 200.0))
        self.S_grid = torch.linspace(0.0, self.s_max, self.n_actions, device=self.device)  # [A]
        self.actors = nn.ModuleList([DRQNAgent(obs_dim, self.msg_dim, H, self.n_actions, self.content)
                                     for _ in range(n_agents)]).to(self.device)
        self.mixer = QMixer(n_agents, state_dim, int(cfg.get("qmix_embed", 32))).to(self.device)
        # target nets (hard-updated)
        self.tgt_actors = nn.ModuleList([DRQNAgent(obs_dim, self.msg_dim, H, self.n_actions, self.content)
                                         for _ in range(n_agents)]).to(self.device)
        self.tgt_mixer = QMixer(n_agents, state_dim, int(cfg.get("qmix_embed", 32))).to(self.device)
        self._sync_targets()
        # SIGNAL-interface aliases so train_signal's payload-save works unchanged
        self.critic = self.mixer                                    # "critic" slot in the checkpoint == mixer
        self.params = list(self.actors.parameters()) + list(self.mixer.parameters())
        self.opt = torch.optim.Adam(self.params, lr=float(cfg.get("qmix_lr", 5e-4)))          # review fix #5: namespaced
        self.gamma = float(cfg.get("gamma", 0.99))
        self.reward_scale = float(cfg.get("qmix_reward_scale", 100.0))  # fix #5: namespaced (MAPPO reward_scale=1.0 must NOT leak in)
        self.aux_coef = float(cfg.get("aux_coef", 0.1))             # dhat grounding (same role as SIGNAL)
        self.grad_clip = float(cfg.get("qmix_max_grad_norm", 10.0))     # fix #5: namespaced (MAPPO 0.5 would strangle TD gradients)
        self.batch_size = int(cfg.get("qmix_batch", 8))             # episodes sampled per update
        self.buf_cap = int(cfg.get("qmix_buffer", 500))             # replay capacity (episodes)
        self.target_every = int(cfg.get("qmix_target_update", 20))  # GRAD STEPS between hard target syncs
        self.grad_steps = int(cfg.get("qmix_grad_steps", 4))        # sampled minibatches per update() call
        # epsilon schedule (linear anneal over updates)
        self.eps_start = float(cfg.get("qmix_eps_start", 1.0))
        self.eps_end = float(cfg.get("qmix_eps_end", 0.05))
        self.eps_anneal = int(cfg.get("qmix_eps_anneal", 4000))     # TRAINING EPISODES to reach eps_end
        #   (clock = collect() calls, not updates: train_signal updates once per batch_episodes, so an
        #    update-based clock would leave epsilon ~0.5 after 8000 episodes -- measured, not guessed.)
        self._buffer = []
        self._updates = 0
        self._collects = 0                                          # epsilon clock (training rollouts only)
        # Exploration RNG: defaults to the run's torch seed (train_signal calls torch.manual_seed(
        # cfg.seed) before construction), so every campaign seed explores differently; a constant
        # default would couple exploration noise across seeds.
        self._rng = np.random.default_rng(int(cfg.get("qmix_seed", torch.initial_seed() % (2 ** 31))))
        # SIGNAL CSV sinks are unsupported for QMIX; keep the attributes so guards in train_signal skip.
        self.upd_obs = None
        self.roll_obs = None

    def _sync_targets(self):
        for t, s in zip(self.tgt_actors, self.actors):
            t.load_state_dict(s.state_dict())
        self.tgt_mixer.load_state_dict(self.mixer.state_dict())

    @property
    def epsilon(self):
        frac = min(1.0, self._collects / max(1, self.eps_anneal))
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    # ---------------------------------------------------------------- rollout
    @torch.no_grad()
    def collect(self, env, seed, deterministic=False):
        """Roll one episode. deterministic=True -> greedy (gate/eval, NOT stored); else eps-greedy
        (training, stored to replay). Returns the SIGNAL buffer contract incl. 'cost' [T,N] so the
        existing gate and best-checkpoint logic are reused verbatim."""
        dev = self.device
        obs, _ = env.reset(seed=seed)
        h = [torch.zeros(1, 1, self.actors[i].hidden, device=dev) for i in range(self.N)]
        m_prev = torch.zeros(self.N, self.msg_dim, device=dev)
        eps = 0.0 if deterministic else self.epsilon
        if not deterministic:
            self._collects += 1
        OBS, MIN, ACT, STATE, REW, COST, DTG, DONE = ([] for _ in range(8))
        while True:
            o = torch.tensor(np.stack([obs[a] for a in AGENTS]), dtype=torch.float32, device=dev)  # [N,obs]
            incoming = self.adj @ m_prev                                     # [N,msg] one-step delay
            state = torch.tensor(env.get_global_state(), dtype=torch.float32, device=dev).view(-1)
            S = torch.zeros(self.N, 1, device=dev)
            a_idx = torch.zeros(self.N, dtype=torch.long, device=dev)
            m_out = torch.zeros(self.N, self.msg_dim, device=dev)
            for i in range(self.N):
                q, dhat, h[i] = self.actors[i].step(o[i:i + 1], incoming[i:i + 1], h[i])
                if eps > 0 and self._rng.random() < eps:
                    ai = int(self._rng.integers(0, self.n_actions))
                else:
                    ai = int(torch.argmax(q, dim=-1).item())
                a_idx[i] = ai
                S[i] = self.S_grid[ai]
                m_out[i] = self.actors[i].emit(o[i:i + 1], dhat)
            order, _ = order_from_S(S, o, self.max_order)
            frac = (order / self.max_order).clamp(0.0, 1.0)
            nobs, _r, terms, truncs, infos = env.step({a: [float(frac[i, 0])] for i, a in enumerate(AGENTS)})
            cost = torch.tensor([infos[a]["local_cost"] for a in AGENTS], dtype=torch.float32, device=dev)
            dtg = torch.tensor([infos[a]["training_targets"]["demand"] for a in AGENTS],
                               dtype=torch.float32, device=dev)
            done = bool(any(terms.values()) or any(truncs.values()))
            OBS.append(o); MIN.append(incoming); ACT.append(a_idx); STATE.append(state)
            REW.append(-cost.sum() / self.reward_scale)              # team reward = -team cost (scaled)
            COST.append(cost); DTG.append(dtg)
            DONE.append(torch.tensor(1.0 if done else 0.0, device=dev))
            m_prev = m_out
            obs = nobs
            if done:
                break
        ep = dict(obs=torch.stack(OBS), msg_in=torch.stack(MIN), actions=torch.stack(ACT),
                  state=torch.stack(STATE), reward=torch.stack(REW), cost=torch.stack(COST),
                  dtgt=torch.stack(DTG), done=torch.stack(DONE))
        if not deterministic:                                       # only training rollouts populate replay
            self._buffer.append({k: v.detach() for k, v in ep.items()})
            if len(self._buffer) > self.buf_cap:
                self._buffer.pop(0)
        return ep

    # ---------------------------------------------------------------- TD update
    def update(self, batch):
        """QMIX recurrent TD step on a minibatch of episodes sampled from replay. `batch` (the recent
        SIGNAL-style buffers train_signal passes) is ignored -- collect() already stored proper
        episodes. Returns (aux_loss, td_loss) to match SIGNALTrainer.update's 2-tuple."""
        if len(self._buffer) < self.batch_size:
            return float("nan"), float("nan")
        aux_out = td_out = float("nan")
        for _g in range(self.grad_steps):
            aux_out, td_out = self._one_grad_step()
        return aux_out, td_out

    def _one_grad_step(self):
        idx = self._rng.choice(len(self._buffer), self.batch_size, replace=False)
        eps_batch = [self._buffer[i] for i in idx]
        Ts = {int(e["obs"].shape[0]) for e in eps_batch}
        if len(Ts) != 1:                                            # fixed-horizon env => equal T; be loud if not
            raise RuntimeError(f"QMIX replay episodes of unequal length {sorted(Ts)}; "
                               "stacked recurrent update assumes a fixed horizon")
        obs = torch.stack([e["obs"] for e in eps_batch], dim=1)     # [T,B,N,obs]
        msg = torch.stack([e["msg_in"] for e in eps_batch], dim=1)  # [T,B,N,msg]
        act = torch.stack([e["actions"] for e in eps_batch], dim=1) # [T,B,N]
        state = torch.stack([e["state"] for e in eps_batch], dim=1) # [T,B,state]
        rew = torch.stack([e["reward"] for e in eps_batch], dim=1)  # [T,B]
        dtg = torch.stack([e["dtgt"] for e in eps_batch], dim=1)    # [T,B,N]
        done = torch.stack([e["done"] for e in eps_batch], dim=1)   # [T,B]
        T, B = obs.shape[0], obs.shape[1]

        q_chosen, q_tgt_max, aux = [], [], torch.zeros((), device=self.device)
        for i in range(self.N):
            qi, dhat_i = self.actors[i].q_sequence(obs[:, :, i, :], msg[:, :, i, :])    # [T,B,A]
            qi_taken = torch.gather(qi, -1, act[:, :, i].unsqueeze(-1)).squeeze(-1)     # [T,B]
            q_chosen.append(qi_taken)
            with torch.no_grad():
                qti, _ = self.tgt_actors[i].q_sequence(obs[:, :, i, :], msg[:, :, i, :])
                q_tgt_max.append(qti.max(dim=-1).values)            # [T,B] greedy (vanilla QMIX target)
            if dhat_i is not None:                                  # dhat grounding (same as SIGNAL aux)
                aux = aux + F.mse_loss(dhat_i.squeeze(-1), dtg[:, :, i])
        q_chosen = torch.stack(q_chosen, dim=-1)                    # [T,B,N]
        q_tgt_max = torch.stack(q_tgt_max, dim=-1)                  # [T,B,N]

        # mix per timestep (flatten T,B for the mixer)
        qtot = self.mixer(q_chosen.reshape(T * B, self.N), state.reshape(T * B, -1)).view(T, B)
        with torch.no_grad():
            qtot_tgt = self.tgt_mixer(q_tgt_max.reshape(T * B, self.N),
                                      state.reshape(T * B, -1)).view(T, B)
            # y_t = r_t + gamma * (1-done_t) * Qtot_target(next); next uses the shifted state sequence.
            qtot_next = torch.zeros_like(qtot_tgt)
            qtot_next[:-1] = qtot_tgt[1:]
            y = rew + self.gamma * (1.0 - done) * qtot_next
        td = F.smooth_l1_loss(qtot, y)      # Huber: caps gradient under large early TD errors (stability)
        loss = td + self.aux_coef * aux

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.params, self.grad_clip)
        self.opt.step()
        self._updates += 1
        if self._updates % self.target_every == 0:
            self._sync_targets()
        return float(aux.item()) if isinstance(aux, torch.Tensor) else 0.0, float(td.item())


# ============================================================================ #
# Greedy evaluation (checkpoint -> per-seed team costs; the sign-concordance input)
# ============================================================================ #
@torch.no_grad()
def qmix_greedy_costs(ckpt, env, seeds):
    """Load a QMIX checkpoint and return per-seed total team costs on the given seeds (greedy).
    This is the QMIX counterpart of heldout_eval's cost sum -- the quantity whose SIGN (comm vs
    nocomm) is compared to MAPPO. ckpt: a loaded dict (as saved by train_signal)."""
    A = ckpt["config"]
    adj = np.array(ckpt["adj"], dtype=np.float32)
    obs_dim, state_dim = int(ckpt["obs_dim"]), int(ckpt["state_dim"])
    tr = QMIXTrainer(A, n_agents=len(AGENTS), obs_dim=obs_dim, state_dim=state_dim, adj=adj)
    for ac, sd in zip(tr.actors, ckpt["actors"]):
        ac.load_state_dict(sd)
    tr.mixer.load_state_dict(ckpt["critic"])
    return [float(tr.collect(env, seed=s, deterministic=True)["cost"].sum().item()) for s in seeds]


# ============================================================================ #
# Self-test: python qmix_agent.py                                               #
# ============================================================================ #
if __name__ == "__main__":
    torch.manual_seed(0); np.random.seed(0)
    cfg = {"hidden": 32, "qmix_n_actions": 11, "qmix_s_max": 160.0, "qmix_batch": 4,
           "msg_content": "dhat", "qmix_target_update": 5, "qmix_eps_anneal": 50}
    N, OB, ST = 4, 4, 133
    tr = QMIXTrainer(cfg, N, OB, ST, adj=np.ones((N, N)) / N)

    class _Env:                                                     # minimal stand-in for the self-test
        def reset(self, seed=0): self._t = 0; return {a: np.zeros(OB, np.float32) for a in AGENTS}, {}
        def get_global_state(self): return np.zeros(ST, np.float32)
        def step(self, acts):
            self._t += 1
            info = {a: {"local_cost": float(np.random.rand() * 10),
                        "training_targets": {"demand": 12.0}} for a in AGENTS}
            return ({a: np.random.rand(OB).astype(np.float32) for a in AGENTS},
                    {}, {a: self._t >= 8 for a in AGENTS}, {a: False for a in AGENTS}, info)
    env = _Env()
    losses = []
    for k in range(40):
        tr.collect(env, seed=k)
        a, td = tr.update(None)
        if not np.isnan(td):
            losses.append(td)
    assert len(losses) > 5 and np.isfinite(losses).all(), "TD loss not produced/finite"

    # IGM: argmax of Q_tot over the joint == per-agent argmax (monotone mixer property), checked
    # numerically on a random state by enumerating one agent's actions with others greedy.
    st = torch.zeros(1, ST)
    qv = [torch.randn(1, tr.n_actions) for _ in range(N)]
    greedy = [int(q.argmax()) for q in qv]
    base = torch.tensor([[qv[i][0, greedy[i]] for i in range(N)]])
    qtot_greedy = tr.mixer(base, st).item()
    ok_igm = True
    for i in range(N):
        for a_alt in range(tr.n_actions):
            v = base.clone(); v[0, i] = qv[i][0, a_alt]
            if tr.mixer(v, st).item() > qtot_greedy + 1e-5:
                ok_igm = False
    assert ok_igm, "IGM violated: a per-agent deviation raised Q_tot (mixer not monotone)"

    # monotonicity directly: dQtot/dQ_i >= 0
    qin = torch.randn(3, N, requires_grad=True)
    out = tr.mixer(qin, torch.zeros(3, ST)).sum()
    g, = torch.autograd.grad(out, qin)
    assert (g >= -1e-6).all(), "mixer gradient negative -> not monotone"

    print("qmix_agent self-test PASS (recurrent Q + grounded dhat + monotone mixer + IGM; "
          f"TD finite over {len(losses)} updates, last={losses[-1]:.3f})")