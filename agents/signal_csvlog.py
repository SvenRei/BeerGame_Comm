"""
signal_csvlog.py -- scalar logger for train_signal.py.
================================================================================
Rationale. W&B is disabled for the air-gapped, 48-96-way parallel sweep
(WANDB_MODE=disabled), so the only record of a run would be parsed stdout, and the
trainer prints the held-out line only inside `if mean_cost < best:`. That yields the
monotone-decreasing envelope of the learning curve rather than the curve itself. This
module instead writes the true per-gate curve (every held-out gate, unconditionally)
plus the PPO and communication diagnostics that reveal a frozen-actor failure, to
per-run CSVs.

Design contract (safe against the frozen, SHA-hashed instrument):
  * Pure observation: reads only scalars/tensors the update and eval paths already
    computed (via SIGNALTrainer.upd_obs / .roll_obs, both None and inert unless this
    logger enables them). No random draw, no reordering, no change to any registered number.
  * One CSV per run, in that run's run_dir, append + flush() after every row. No shared
    writer, no network, stdlib csv only, so 48-96 processes each own their files.
  * Default off: gated on SIGNAL_CSVLOG=1; unset => train_signal is byte-identical to the
    frozen instrument.

Files written into run_dir:
  metrics_heldout.csv   one row per held-out gate (the learning curve and estimand)
  metrics_update.csv    one row per PPO update (convergence / failure-mode diagnostics)
  run_meta.json         one provenance manifest (self-describes the CSVs)

An optional TensorBoard SummaryWriter behind a second flag (SIGNAL_TBLOG=1) is a
live-watching convenience only; the CSV is the artifact.
"""
import os
import csv
import json
import time
import hashlib
import subprocess

import numpy as np
import torch

from agents.signal_agent import AGENTS

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------ flags
def csvlog_enabled():
    """Master switch. Unset/0 => the logger is never constructed and the trainer is
    byte-identical to the frozen instrument."""
    return os.environ.get("SIGNAL_CSVLOG", "0") not in ("", "0", "false", "False")


def _tb_enabled():
    return os.environ.get("SIGNAL_TBLOG", "0") not in ("", "0", "false", "False")


def _update_every():
    try:
        return max(1, int(os.environ.get("SIGNAL_UPDATE_EVERY", "1")))
    except ValueError:
        return 1


# ------------------------------------------------------------------ provenance helpers
def _git_sha():
    for cmd in (["git", "rev-parse", "HEAD"],):
        try:
            out = subprocess.run(cmd, cwd=_ROOT, capture_output=True, text=True, timeout=5)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        except Exception:                                   # noqa: BLE001 -- git absent / not a repo
            pass
    return os.environ.get("GIT_SHA", "unknown")


def _prereg_hash():
    try:
        from scripts.prereg import registration_hash
        return registration_hash()
    except Exception:                                       # noqa: BLE001 -- prereg module optional
        return os.environ.get("SIGNAL_PREREG_HASH", "unknown")


def _config_hash(cfg_container):
    """Deterministic SHA256 over the fully-resolved config -- groups the 435 runs and lets a
    referee verify two runs share an arm definition."""
    blob = json.dumps(cfg_container, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def make_run_meta(cfg_container, A, run_id, n_params, obs_dim, state_dim,
                  fixed_ref, oracle_ref):
    """The self-describing manifest for one run (sidecar to the CSVs)."""
    use_comm = bool(A.get("use_comm", True))
    return {
        "run_id": run_id,
        "algorithm": A.get("algorithm"),                    # arm+seed tag, e.g. dp_upstream_s12
        "seed": int(cfg_container.get("seed", -1)),
        "topology": (A.get("comm_topology", "neighbor") if use_comm else "none_zeroADJ"),
        "use_comm": use_comm,
        "msg_content": A.get("msg_content"),
        "train_env": A.get("train_env"),
        "heldout_mode": A.get("heldout_mode", "poisson"),
        "rho": A.get("ar1_rho"),                            # meaningful only for the AR(1) arms
        "beta": A.get("srdqn_beta"),
        "tau": A.get("tau"),
        "budget_milestones": list(A.get("budget_milestones") or []),
        "fixed_ref_BAR": float(fixed_ref),
        "oracle_ref_CEILING": float(oracle_ref),
        "n_params": int(n_params),
        "obs_dim": int(obs_dim),
        "state_dim": int(state_dim),
        "config_hash": _config_hash(cfg_container),
        "prereg_hash": _prereg_hash(),
        "git_sha": _git_sha(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "start_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


# ------------------------------------------------------------------ metric helpers (pure, no rollout)
def _corr(x, y):
    x = np.asarray(x, float).ravel(); y = np.asarray(y, float).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) < 1e-8 or np.std(y[ok]) < 1e-8:
        return float("nan")
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def _mean(seq):
    a = np.asarray([v for v in seq if v == v], float)       # drop NaN
    return float(a.mean()) if a.size else float("nan")


@torch.no_grad()
def _positive_listening(trainer, buffers, rolls, component=0, lo=0.0, hi=20.0):
    """Causal dS/d(msg) on the direct head path, averaged over receivers and visited states.
    Reproduces eval_signal.positive_listening's head probe from tensors already on hand (the stored
    obs, received incoming, and the belief h that fed the head); no rollout is re-run. Sweeps one
    message component from `lo` to `hi`, holding the others at their received values, and reads the
    order-up-to S the base-stock head would emit. A slope near 0 with honest messages indicates a deaf
    receiver (signaling without listening), which cost alone cannot distinguish from redundancy."""
    dev = trainer.device
    adj = trainer.adj
    receivers = [i for i in range(trainer.N) if float(adj[i].abs().sum()) > 0]
    if not receivers:
        return float("nan")
    slopes = []
    for buf, roll in zip(buffers, rolls):
        if roll is None:
            continue
        obs = buf["obs"].to(dev)                            # [T,N,obs]
        inc = buf["msg_in"].to(dev)                         # [T,N,msg] received baseline
        h = torch.as_tensor(roll["h"], dtype=torch.float32, device=dev)   # [T,N,hidden]
        for i in receivers:
            m_lo = inc[:, i, :].clone(); m_lo[:, component] = lo
            m_hi = inc[:, i, :].clone(); m_hi[:, component] = hi
            s_lo, _ = trainer.actors[i].base_stock(obs[:, i, :], h[:, i, :], m_lo)
            s_hi, _ = trainer.actors[i].base_stock(obs[:, i, :], h[:, i, :], m_hi)
            slopes.append(float(((s_hi - s_lo) / (hi - lo)).mean()))
    return _mean(slopes)


def compute_gate_metrics(trainer, gate_stats, A):
    """Reduce the raw eval buffers/rolls captured during the gate into the CSV scalars. All values
    are computed from tensors the eval collect ALREADY produced (costs, obs, received messages,
    d_hat, belief h); the only extra compute is the finite-difference head probe for listening."""
    per_ep = np.asarray(gate_stats["per_ep_total"], float)
    ech = np.stack(gate_stats["per_echelon"], 0) if gate_stats["per_echelon"] else np.zeros((1, len(AGENTS)))
    ech_mean = ech.mean(0)                                  # [N] mean per-echelon cost over eval eps

    buffers = gate_stats["buffers"]; rolls = gate_stats["rolls"]
    # pool emitted messages, forecasts and truths across all eval episodes
    msg_all, fe_terms, sig_by_agent = [], [], {i: {"m": [], "d": []} for i in range(trainer.N)}
    S_all = []
    for buf, roll in zip(buffers, rolls):
        S_all.append(buf["S"].reshape(-1).cpu().numpy())
        if roll is None:
            continue
        msg_out = roll["msg_out"]                           # [T,N,msg] emitted
        dhat = roll["dhat"]                                 # [T,N]
        dtgt = buf["dtgt"].cpu().numpy()                    # [T,N] each stage's realized incoming
        cust = dtgt[:, 0]                                   # retailer incoming == end-customer demand
        msg_all.append(msg_out.reshape(-1))
        fe_terms.append(np.abs(dhat - cust[:, None]).reshape(-1))   # |d_hat_i - customer demand|
        for i in range(trainer.N):
            sig_by_agent[i]["m"].append(msg_out[:, i, 0])  # first message component
            sig_by_agent[i]["d"].append(dtgt[:, i])        # sender's own demand

    msg_flat = np.concatenate(msg_all) if msg_all else np.array([np.nan])
    fe = float(np.concatenate(fe_terms).mean()) if fe_terms else float("nan")
    # positive signalling: does the emitted message co-vary with the sender's own state? (Lowe 2019
    # speaker-consistency proxy; corr, averaged over senders). Near 0 => message carries no info.
    sig = _mean([_corr(np.concatenate(sig_by_agent[i]["m"]), np.concatenate(sig_by_agent[i]["d"]))
                 for i in range(trainer.N) if sig_by_agent[i]["m"]])
    listen = _positive_listening(trainer, buffers, rolls)

    action_std = float(np.mean([float(a.log_std.detach().exp().clamp(1e-2, 3.0).mean())
                                for a in trainer.actors]))
    S_pool = np.concatenate(S_all) if S_all else np.array([np.nan])

    m = {
        "heldout_cost_std": float(per_ep.std()) if per_ep.size else float("nan"),
        "n_eval_eps": int(per_ep.size),
        "heldout_mode": A.get("heldout_mode", "poisson"),
        "heldout_S_mean": float(S_pool.mean()),
        "heldout_action_std": action_std,
        "msg_mean": float(np.nanmean(msg_flat)),
        "msg_std": float(np.nanstd(msg_flat)),
        "positive_signaling": sig,
        "positive_listening": listen,
        "forecast_error": fe,
    }
    for i, a in enumerate(AGENTS):
        m[f"cost_{a}"] = float(ech_mean[i]) if i < ech_mean.size else float("nan")
    return m


# ------------------------------------------------------------------ dial (learned-rung) gradient certs
def snapshot_dial(trainer):
    """Snapshot the learned channel's params before an update; None for the named rungs (nothing to
    train). Paired with dial_delta(), this certifies the DIAL channel receives gradient (dW/dGain
    move) rather than remaining a frozen random projection."""
    if trainer.actors[0].msg_head is None:
        return None
    return [{"w": a.msg_head.weight.detach().clone(), "g": float(a.msg_gain.detach())}
            for a in trainer.actors]


def dial_delta(trainer, pre):
    if pre is None:
        return float("nan"), float("nan")
    dW = max(float((a.msg_head.weight.detach() - p["w"]).abs().max()) for a, p in zip(trainer.actors, pre))
    dG = max(abs(float(a.msg_gain.detach()) - p["g"]) for a, p in zip(trainer.actors, pre))
    return dW, dG


# ------------------------------------------------------------------ the logger
_GATE_COLS = (["episode", "heldout_mean_cost", "heldout_cost_std", "n_eval_eps"]
              + [f"cost_{a}" for a in AGENTS]
              + ["gap_recovered", "heldout_mode", "vs_bar_pct",
                 "best_so_far", "best_ep", "eps_since_improvement",
                 "heldout_S_mean", "heldout_action_std",
                 "msg_mean", "msg_std", "positive_signaling", "positive_listening", "forecast_error"])

_UPDATE_COLS = ["episode", "update_idx", "policy_loss", "value_loss", "entropy", "aux_loss",
                "approx_kl", "clip_fraction", "grad_norm", "explained_variance",
                "S_mean", "action_std", "dial_dW", "dial_dgain"]


class SignalCSVLogger:
    def __init__(self, run_dir, meta):
        self.run_dir = run_dir
        self._update_idx = 0
        self._update_every = _update_every()
        with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        self._gate_f, self._gate_w = self._open("metrics_heldout.csv", _GATE_COLS)
        self._upd_f, self._upd_w = self._open("metrics_update.csv", _UPDATE_COLS)
        self._tb = self._init_tb()

    def _open(self, name, header):
        path = os.path.join(self.run_dir, name)
        fresh = (not os.path.exists(path)) or os.path.getsize(path) == 0
        f = open(path, "a", newline="")                     # append; each process owns this file
        w = csv.writer(f)
        if fresh:
            w.writerow(header)
            f.flush()
        return f, w

    def _init_tb(self):
        if not _tb_enabled():
            return None
        try:
            from torch.utils.tensorboard import SummaryWriter
            return SummaryWriter(log_dir=os.path.join(self.run_dir, "tb"))
        except Exception as ex:                             # noqa: BLE001 -- convenience only, never fatal
            print(f"[csvlog] SIGNAL_TBLOG set but TensorBoard unavailable ({type(ex).__name__}); "
                  f"CSV logging continues.", flush=True)
            return None

    # -------------------------------------------------------------- per PPO update
    def log_update(self, episode, policy_loss, value_loss, upd_obs, pre_dial, trainer):
        self._update_idx += 1
        if (self._update_idx % self._update_every) != 0:
            return
        upd_obs = upd_obs or {}
        dW, dG = dial_delta(trainer, pre_dial)
        row = {
            "episode": episode, "update_idx": self._update_idx,
            "policy_loss": policy_loss, "value_loss": value_loss,
            "entropy": _mean(upd_obs.get("entropy", [])),
            "aux_loss": _mean(upd_obs.get("aux_loss", [])),
            "approx_kl": _mean(upd_obs.get("approx_kl", [])),
            "clip_fraction": _mean(upd_obs.get("clip_frac", [])),
            "grad_norm": _mean(upd_obs.get("grad_norm", [])),
            "explained_variance": upd_obs.get("explained_variance", float("nan")),
            "S_mean": _mean(upd_obs.get("S_mean", [])),
            "action_std": _mean(upd_obs.get("action_std", [])),
            "dial_dW": dW, "dial_dgain": dG,
        }
        self._upd_w.writerow([row[c] for c in _UPDATE_COLS])
        self._upd_f.flush()
        if self._tb is not None:
            for c in _UPDATE_COLS[2:]:
                v = row[c]
                if isinstance(v, float) and v == v:
                    self._tb.add_scalar(f"update/{c}", v, episode)

    # -------------------------------------------------------------- per held-out gate
    def log_gate(self, episode, elog, mean_cost, gate_stats, best, best_ep,
                 eps_since_improvement, trainer, A):
        """Written on every gate, unconditionally, decoupled from the improvement branch, so the CSV
        holds the true (non-monotone) learning curve rather than the best-only envelope."""
        gm = compute_gate_metrics(trainer, gate_stats, A)
        row = {
            "episode": episode,
            "heldout_mean_cost": float(mean_cost),
            "heldout_cost_std": gm["heldout_cost_std"],
            "n_eval_eps": gm["n_eval_eps"],
            "gap_recovered": float(elog.get("Eval/Gap_Recovered", float("nan"))),
            "heldout_mode": gm["heldout_mode"],
            "vs_bar_pct": float(elog.get("Eval/vs_BAR_pct", float("nan"))),
            "best_so_far": float(best),
            "best_ep": int(best_ep),
            "eps_since_improvement": int(eps_since_improvement),
            "heldout_S_mean": gm["heldout_S_mean"],
            "heldout_action_std": gm["heldout_action_std"],
            "msg_mean": gm["msg_mean"], "msg_std": gm["msg_std"],
            "positive_signaling": gm["positive_signaling"],
            "positive_listening": gm["positive_listening"],
            "forecast_error": gm["forecast_error"],
        }
        for a in AGENTS:
            row[f"cost_{a}"] = gm[f"cost_{a}"]
        self._gate_w.writerow([row[c] for c in _GATE_COLS])
        self._gate_f.flush()
        if self._tb is not None:
            for c in _GATE_COLS:
                v = row.get(c)
                if isinstance(v, float) and v == v:
                    self._tb.add_scalar(f"heldout/{c}", v, episode)

    def close(self):
        for f in (self._gate_f, self._upd_f):
            try:
                f.flush(); f.close()
            except Exception:                               # noqa: BLE001
                pass
        if self._tb is not None:
            try:
                self._tb.flush(); self._tb.close()
            except Exception:                               # noqa: BLE001
                pass
