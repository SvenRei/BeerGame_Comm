#!/usr/bin/env python3
"""Checkpoint-only diagnostics for the SIGNAL paper.

No training is performed. Three subcommands answer the pre-retraining questions:

1. dhat
   Is the MAPPO retailer demand head a competent one-step forecast under the actual
   rounded/truncated AR(1) data-generating process?

2. qmix-channel
   Is a trained QMIX communication channel collapsed, alive but neutral, useful,
   or actively maladaptive? Uses CRN-paired honest/shuffled/cross/zeroed rollouts.

3. convergence
   Did the MAPPO raw arm plateau on held-out cost, remain stable after its best
   checkpoint, and reach the supplied privileged frontier?

Run from the repository root. Examples are printed by --help.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.eval_signal import (  # noqa: E402
    AGENTS,
    HELDOUT_SEED_BASE,
    SIGNALPolicy,
    make_ar1_env,
    resolve_env_base,
    run_episode,
)
from agents.qmix_agent import QMIXTrainer, order_from_S  # noqa: E402
from envs.beer_game_env import BeerGameParallelEnv  # noqa: E402
from scripts.demand_families import ar1_step, make_demand_family_envs  # noqa: E402


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def _expand(patterns: Sequence[str]) -> List[str]:
    out: List[str] = []
    for p in patterns:
        hits = sorted(glob.glob(p))
        out.extend(hits if hits else [p])
    out = sorted(dict.fromkeys(out))
    missing = [p for p in out if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing input(s): {missing[:5]}")
    return out


def _mean_ci(x: Sequence[float], level: float = 0.95, n_boot: int = 10000,
             seed: int = 0) -> Tuple[float, float, float]:
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(a.mean())
    if a.size == 1 or float(a.std()) == 0.0:
        return m, m, m
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    bm = a[idx].mean(axis=1)
    alpha = 1.0 - level
    lo, hi = np.quantile(bm, [alpha / 2.0, 1.0 - alpha / 2.0])
    return m, float(lo), float(hi)


def _regression(y: np.ndarray, x: np.ndarray) -> Tuple[float, float]:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) < 1e-12:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(x[ok], y[ok], 1)
    return float(intercept), float(slope)


def _metrics(pred: np.ndarray, actual: np.ndarray) -> Dict[str, float]:
    pred = np.asarray(pred, float)
    actual = np.asarray(actual, float)
    ok = np.isfinite(pred) & np.isfinite(actual)
    pred, actual = pred[ok], actual[ok]
    err = pred - actual
    corr = (float(np.corrcoef(pred, actual)[0, 1])
            if pred.size >= 3 and np.std(pred) > 1e-12 and np.std(actual) > 1e-12
            else float("nan"))
    intercept, slope = _regression(actual, pred)  # actual = intercept + slope * prediction
    return {
        "n": int(pred.size),
        "mean_pred": float(pred.mean()),
        "mean_actual": float(actual.mean()),
        "std_pred": float(pred.std()),
        "bias": float(err.mean()),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mse": float(np.mean(err ** 2)),
        "corr": corr,
        "cal_intercept": intercept,
        "cal_slope": slope,
    }


def _write_json(payload: dict, path: Optional[str]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text + "\n", encoding="utf-8")
        print(f"[diagnostic] wrote {path}")


# ---------------------------------------------------------------------------
# 1. d-hat quality diagnostic
# ---------------------------------------------------------------------------

def _empirical_conditional_mean(mu: float, rho: float, sigma: float,
                                draws: int, burn: int, seed: int) -> Dict[int, float]:
    """Estimate E[D_t | D_{t-1}=k] under the exact rounded/truncated DGP.

    This is more accurate than calling mu + rho(D_{t-1}-mu) the 'true conditional
    mean'. The latter is only the linear predictor because the observed process is
    rounded and truncated and the latent state is not observed.
    """
    rng = np.random.default_rng(seed)
    latent = float(mu)
    prev = None
    sums: Dict[int, float] = {}
    counts: Dict[int, int] = {}
    for t in range(burn + draws):
        d, latent = ar1_step(latent, mu, rho, sigma, rng)
        di = int(round(d))
        if t >= burn and prev is not None:
            sums[prev] = sums.get(prev, 0.0) + float(di)
            counts[prev] = counts.get(prev, 0) + 1
        prev = di
    return {k: sums[k] / counts[k] for k in sums if counts[k] >= 25}


def diagnose_dhat(args: argparse.Namespace) -> None:
    ckpts = _expand(args.ckpt)
    lookup = _empirical_conditional_mean(
        args.mu, args.rho, args.sigma, args.oracle_draws, args.oracle_burn, args.oracle_seed
    )
    seed_rows = []
    pooled = {"dhat": [], "actual": [], "linear": [], "empirical": [], "mean": []}

    for path in ckpts:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        content = ck.get("msg_content", ck.get("config", {}).get("msg_content"))
        if content != "dhat":
            raise ValueError(f"{path}: expected msg_content=dhat, found {content!r}")
        base = resolve_env_base(ck, args.env_json)
        env = make_ar1_env(args.rho, args.mu, args.sigma, base)
        pol = SIGNALPolicy(ck, env, ablate=False, deterministic=True)

        dh_all, y_all, lin_all, emp_all, mean_all = [], [], [], [], []
        for e in range(args.episodes):
            r = run_episode(pol, env, args.seed_base + e, trace=True)
            tr = r["trace"]
            y = np.asarray(tr["cust"], float)
            dh = np.asarray(tr["dhat"], float)[:, 0]  # retailer forecast; retailer is the broadcaster
            prev = np.concatenate([[0.0], y[:-1]])
            linear = np.maximum(0.0, args.mu + args.rho * (prev - args.mu))
            empirical = np.array([lookup.get(int(round(v)), linear[j]) for j, v in enumerate(prev)], float)
            # Step 0 is the environment's zero-filled convention and has no real lagged observation.
            start = 1 if args.drop_step0 else 0
            dh_all.extend(dh[start:]); y_all.extend(y[start:]); lin_all.extend(linear[start:])
            emp_all.extend(empirical[start:]); mean_all.extend(np.full(y[start:].shape, args.mu))

        md = _metrics(np.asarray(dh_all), np.asarray(y_all))
        ml = _metrics(np.asarray(lin_all), np.asarray(y_all))
        me = _metrics(np.asarray(emp_all), np.asarray(y_all))
        mm = _metrics(np.asarray(mean_all), np.asarray(y_all))
        ratio = md["mse"] / me["mse"] if me["mse"] > 0 else float("nan")
        skill_mean = 1.0 - md["mse"] / mm["mse"] if mm["mse"] > 0 else float("nan")
        skill_emp = 1.0 - md["mse"] / me["mse"] if me["mse"] > 0 else float("nan")
        seed_rows.append({
            "checkpoint": path,
            "train_seed": int(ck.get("seed", ck.get("config", {}).get("seed", -1))),
            "dhat": md,
            "linear_predictor": ml,
            "empirical_E_Dt_given_Dtm1": me,
            "constant_mean": mm,
            "mse_ratio_dhat_to_empirical": ratio,
            "skill_vs_constant_mean": skill_mean,
            "skill_vs_empirical": skill_emp,
        })
        pooled["dhat"].extend(dh_all); pooled["actual"].extend(y_all); pooled["linear"].extend(lin_all)
        pooled["empirical"].extend(emp_all); pooled["mean"].extend(mean_all)

    ratios = np.array([r["mse_ratio_dhat_to_empirical"] for r in seed_rows], float)
    biases = np.array([abs(r["dhat"]["bias"]) for r in seed_rows], float)
    slopes = np.array([r["dhat"]["cal_slope"] for r in seed_rows], float)
    stds = np.array([r["dhat"]["std_pred"] for r in seed_rows], float)

    median_ratio = float(np.nanmedian(ratios))
    median_bias = float(np.nanmedian(biases))
    median_slope = float(np.nanmedian(slopes))
    median_std = float(np.nanmedian(stds))

    if (median_ratio <= args.healthy_mse_ratio and median_bias <= args.max_abs_bias
            and args.cal_slope_lo <= median_slope <= args.cal_slope_hi and median_std >= args.min_pred_std):
        verdict = "HEALTHY_FORECAST_HEAD"
        interpretation = "dhat is close enough to the DGP benchmark; its low economic value is a finding, not a weak-head explanation."
    elif median_ratio > args.broken_mse_ratio or median_std < args.min_pred_std:
        verdict = "LIKELY_FORECAST_HEAD_PROBLEM"
        interpretation = "dhat is materially worse than the observation-conditional benchmark or nearly constant; retrain only dhat-related arms after fixing grounding."
    else:
        verdict = "BORDERLINE_FORECAST_QUALITY"
        interpretation = "forecast quality is neither clearly competent nor clearly broken; inspect calibration and seed heterogeneity before retraining."

    payload = {
        "diagnostic": "dhat_forecast_quality",
        "definition_note": (
            "The code reports the registered AR linear predictor and an empirical E[D_t|D_{t-1}] "
            "under the exact rounded/truncated DGP. The linear predictor is not mislabeled as the exact conditional mean."
        ),
        "parameters": {k: v for k, v in vars(args).items() if k != "func"},
        "per_checkpoint": seed_rows,
        "aggregate": {
            "n_checkpoints": len(seed_rows),
            "median_mse_ratio_dhat_to_empirical": median_ratio,
            "median_abs_bias": median_bias,
            "median_calibration_slope": median_slope,
            "median_prediction_std": median_std,
            "pooled_dhat": _metrics(np.asarray(pooled["dhat"]), np.asarray(pooled["actual"])),
            "pooled_linear_predictor": _metrics(np.asarray(pooled["linear"]), np.asarray(pooled["actual"])),
            "pooled_empirical_conditional": _metrics(np.asarray(pooled["empirical"]), np.asarray(pooled["actual"])),
            "verdict": verdict,
            "interpretation": interpretation,
        },
    }
    _write_json(payload, args.out)


# ---------------------------------------------------------------------------
# 2. QMIX channel diagnostic
# ---------------------------------------------------------------------------

@dataclass
class QMixRoll:
    cost: float
    emitted: np.ndarray      # [T,N,msg]
    incoming: np.ndarray     # [T,N,msg]
    actions: np.ndarray      # [T,N]


class QMixProbe:
    def __init__(self, ckpt: dict, env):
        self.ckpt = ckpt
        self.env = env
        A = ckpt["config"]
        self.tr = QMIXTrainer(
            A,
            n_agents=len(AGENTS),
            obs_dim=int(ckpt["obs_dim"]),
            state_dim=int(ckpt["state_dim"]),
            adj=np.asarray(ckpt["adj"], np.float32),
            device="cpu",
        )
        for actor, sd in zip(self.tr.actors, ckpt["actors"]):
            actor.load_state_dict(sd)
            actor.eval()
        self.tr.mixer.load_state_dict(ckpt["critic"])
        self.tr.mixer.eval()
        self.tr.max_order = float(env.max_order)

    @torch.no_grad()
    def run(self, seed: int, override_prev: Optional[np.ndarray] = None) -> QMixRoll:
        obs, _ = self.env.reset(seed=seed)
        h = [torch.zeros(1, 1, a.hidden) for a in self.tr.actors]
        m_prev = torch.zeros(self.tr.N, self.tr.msg_dim)
        emitted, incoming_hist, actions = [], [], []
        total = 0.0
        t = 0
        while True:
            o = torch.tensor(np.stack([obs[a] for a in AGENTS]), dtype=torch.float32)
            if override_prev is not None and t < len(override_prev):
                m_prev = torch.tensor(override_prev[t], dtype=torch.float32).view(self.tr.N, self.tr.msg_dim)
            incoming = self.tr.adj @ m_prev
            S = torch.zeros(self.tr.N, 1)
            m_out = torch.zeros(self.tr.N, self.tr.msg_dim)
            a_idx = torch.zeros(self.tr.N, dtype=torch.long)
            for i in range(self.tr.N):
                q, dhat, h[i] = self.tr.actors[i].step(o[i:i + 1], incoming[i:i + 1], h[i])
                ai = int(torch.argmax(q, dim=-1).item())
                a_idx[i] = ai
                S[i] = self.tr.S_grid[ai]
                m_out[i] = self.tr.actors[i].emit(o[i:i + 1], dhat)
            order, _ = order_from_S(S, o, self.tr.max_order)
            frac = (order / self.tr.max_order).clamp(0.0, 1.0)
            nobs, _r, terms, truncs, infos = self.env.step(
                {a: [float(frac[i, 0])] for i, a in enumerate(AGENTS)}
            )
            total += sum(float(infos[a]["local_cost"]) for a in AGENTS)
            emitted.append(m_out.numpy().copy())
            incoming_hist.append(incoming.numpy().copy())
            actions.append(a_idx.numpy().copy())
            m_prev = m_out
            obs = nobs
            t += 1
            if any(terms.values()) or any(truncs.values()):
                break
        return QMixRoll(total, np.asarray(emitted), np.asarray(incoming_hist), np.asarray(actions))


def _prev_stream(emitted: np.ndarray) -> np.ndarray:
    z = np.zeros((1,) + emitted.shape[1:], dtype=np.float32)
    return np.concatenate([z, emitted[:-1]], axis=0)


def _weight_ratios_qmix(probe: QMixProbe) -> Dict[str, dict]:
    out = {}
    od, md = int(probe.ckpt["obs_dim"]), probe.tr.msg_dim
    for i, name in enumerate(AGENTS):
        W = probe.tr.actors[i].gru.weight_ih_l0.detach().numpy()
        obs_norm = float(np.linalg.norm(W[:, :od]))
        msg_norm = float(np.linalg.norm(W[:, od:od + md]))
        out[name] = {
            "obs_weight_norm": obs_norm,
            "msg_weight_norm": msg_norm,
            "msg_to_obs_ratio": msg_norm / max(obs_norm, 1e-12),
        }
    return out


def _discrete_mi(x: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Plug-in mutual information in nats after quantile-binning continuous x."""
    x = np.asarray(x, float).reshape(-1)
    y = np.asarray(y, int).reshape(-1)
    ok = np.isfinite(x)
    x, y = x[ok], y[ok]
    if x.size < 20 or np.std(x) < 1e-12:
        return 0.0
    edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    if edges.size < 3:
        return 0.0
    xb = np.digitize(x, edges[1:-1], right=False)
    ux, cx = np.unique(xb, return_counts=True)
    uy, cy = np.unique(y, return_counts=True)
    px = {u: c / x.size for u, c in zip(ux, cx)}
    py = {u: c / y.size for u, c in zip(uy, cy)}
    mi = 0.0
    for xv in ux:
        for yv in uy:
            pxy = float(np.mean((xb == xv) & (y == yv)))
            if pxy > 0:
                mi += pxy * math.log(pxy / (px[xv] * py[yv]))
    return float(mi)


def diagnose_qmix(args: argparse.Namespace) -> None:
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    base = dict(ck.get("env", {}))
    A = ck["config"]
    AR1, _, _ = make_demand_family_envs(BeerGameParallelEnv)
    env = AR1({**base, "demand_type": "poisson", "family": "ar1",
               "ar1_mu": args.mu, "ar1_rho": args.rho, "ar1_sigma": args.sigma})
    probe = QMixProbe(ck, env)

    honest = [probe.run(args.seed_base + e) for e in range(args.episodes)]
    c_h = np.array([r.cost for r in honest], float)
    emitted = [r.emitted for r in honest]

    cond_costs: Dict[str, List[float]] = {"shuffled": [], "cross": [], "zeroed": []}
    cond_flips: Dict[str, List[float]] = {"shuffled": [], "cross": [], "zeroed": []}
    for e, ref in enumerate(honest):
        rng = np.random.default_rng(args.shuffle_seed + e)
        sh = emitted[e][rng.permutation(len(emitted[e]))]
        partner = emitted[(e + 1) % len(emitted)]
        cross = partner[:len(ref.emitted)]
        if len(cross) < len(ref.emitted):
            reps = int(np.ceil(len(ref.emitted) / len(cross)))
            cross = np.tile(cross, (reps, 1, 1))[:len(ref.emitted)]
        zero = np.zeros_like(ref.emitted)
        for name, stream in (("shuffled", sh), ("cross", cross), ("zeroed", zero)):
            rr = probe.run(args.seed_base + e, _prev_stream(stream))
            cond_costs[name].append(rr.cost)
            T = min(len(rr.actions), len(ref.actions))
            cond_flips[name].append(float(np.mean(rr.actions[:T] != ref.actions[:T])))

    weights = _weight_ratios_qmix(probe)
    incoming = np.concatenate([r.incoming for r in honest], axis=0)
    actions = np.concatenate([r.actions for r in honest], axis=0)
    receivers = [i for i in range(len(AGENTS)) if np.abs(np.asarray(ck["adj"])[i]).sum() > 0]
    receiver_stats = {}
    for i in receivers:
        msg = incoming[:, i, 0]
        receiver_stats[AGENTS[i]] = {
            "incoming_mean": float(np.mean(msg)),
            "incoming_std": float(np.std(msg)),
            "mi_message_action_nats": _discrete_mi(msg, actions[:, i], bins=args.mi_bins),
            **weights[AGENTS[i]],
        }

    interventions = {}
    for name in ("shuffled", "cross", "zeroed"):
        c = np.asarray(cond_costs[name], float)
        d = c - c_h  # positive: intervention hurts; negative: removing/corrupting channel helps
        mean, lo, hi = _mean_ci(d, n_boot=args.bootstrap, seed=args.bootstrap_seed)
        interventions[name] = {
            "mean_cost": float(c.mean()),
            "delta_vs_honest": mean,
            "delta_ci95": [lo, hi],
            "mean_action_flip_rate": float(np.mean(cond_flips[name])),
        }

    max_flip = max(v["mean_action_flip_rate"] for v in interventions.values())
    avg_weight = float(np.mean([receiver_stats[a]["msg_to_obs_ratio"] for a in receiver_stats])) if receiver_stats else 0.0
    avg_msg_std = float(np.mean([receiver_stats[a]["incoming_std"] for a in receiver_stats])) if receiver_stats else 0.0
    zero_delta = interventions["zeroed"]["delta_vs_honest"]
    zero_lo, zero_hi = interventions["zeroed"]["delta_ci95"]
    cost_scale = max(1.0, float(c_h.mean()))

    nocomm = None
    if args.nocomm_ckpt:
        nk = torch.load(args.nocomm_ckpt, map_location="cpu", weights_only=False)
        nprobe = QMixProbe(nk, env)
        nc = np.array([nprobe.run(args.seed_base + e).cost for e in range(args.episodes)], float)
        econ = nc - c_h  # positive: trained communication policy is economically better
        em, elo, ehi = _mean_ci(econ, n_boot=args.bootstrap, seed=args.bootstrap_seed + 1)
        nocomm = {"mean_cost": float(nc.mean()), "V_nocomm_minus_comm": em, "ci95": [elo, ehi]}

    active = (avg_msg_std >= args.min_msg_std and
              (max_flip >= args.min_flip_rate or avg_weight >= args.min_weight_ratio))
    material = abs(zero_delta) >= args.material_cost_frac * cost_scale

    if not active and not material:
        verdict = "CHANNEL_COLLAPSED_OR_IGNORED"
        interpretation = "Messages vary, but receivers scarcely change actions/cost; fix QMIX communication before interpreting its economic sign."
    elif zero_hi < 0 and max_flip >= args.min_flip_rate:
        verdict = "CHANNEL_ALIVE_BUT_MALADAPTIVE"
        interpretation = "Zeroing the channel improves the same trained policy; QMIX uses the signal in a cost-increasing way."
    elif zero_lo > 0:
        if nocomm and nocomm["V_nocomm_minus_comm"] < 0:
            verdict = "RELIANCE_WITHOUT_ECONOMIC_VALUE"
            interpretation = "The trained QMIX policy relies on messages, but the separately trained no-communication policy is better."
        else:
            verdict = "CHANNEL_ALIVE_AND_USEFUL_WITHIN_POLICY"
            interpretation = "Zeroing messages hurts the trained policy; compare with the separately trained no-communication checkpoint for economic value."
    elif active:
        verdict = "CHANNEL_ALIVE_COST_EFFECT_UNCERTAIN"
        interpretation = "Messages alter behavior, but their cost consequence is not precise at this episode count."
    else:
        verdict = "BORDERLINE_CHANNEL_HEALTH"
        interpretation = "Evidence is mixed; increase evaluation episodes before changing training code."

    payload = {
        "diagnostic": "qmix_channel_health",
        "channel_note": (
            "This repository's QMIX raw/dhat channels are continuous deterministic messages, not discrete tokens. "
            "Therefore token-count entropy is not the correct primary diagnostic; use input variance, weights, interventions, and action flips."
        ),
        "checkpoint": args.ckpt,
        "content": A.get("msg_content"),
        "episodes": args.episodes,
        "honest_mean_cost": float(c_h.mean()),
        "receiver_stats": receiver_stats,
        "interventions": interventions,
        "separately_trained_nocomm": nocomm,
        "aggregate": {
            "mean_receiver_msg_std": avg_msg_std,
            "mean_receiver_weight_ratio": avg_weight,
            "max_intervention_action_flip_rate": max_flip,
            "verdict": verdict,
            "interpretation": interpretation,
        },
    }
    _write_json(payload, args.out)


# ---------------------------------------------------------------------------
# 3. Convergence diagnostic
# ---------------------------------------------------------------------------

def _read_csv(path: str) -> List[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _f(row: dict, key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def diagnose_convergence(args: argparse.Namespace) -> None:
    paths = _expand(args.csv)
    per_seed = []
    slopes = []
    last_best_gaps = []
    boundary = []

    for path in paths:
        rows = _read_csv(path)
        ep = np.array([_f(r, "episode") for r in rows], float)
        cost = np.array([_f(r, args.metric) for r in rows], float)
        ok = np.isfinite(ep) & np.isfinite(cost)
        ep, cost = ep[ok], cost[ok]
        if ep.size < 3:
            raise ValueError(f"{path}: fewer than 3 finite gate rows")
        order = np.argsort(ep); ep, cost = ep[order], cost[order]
        end = float(ep.max())
        sel = ep >= end - args.tail_episodes
        if sel.sum() < 3:
            sel = np.arange(ep.size) >= max(0, ep.size - 3)
        slope = float(np.polyfit(ep[sel], cost[sel], 1)[0])
        pct_per_1000 = 100.0 * slope * 1000.0 / max(1e-9, float(np.mean(cost[sel])))
        best_idx = int(np.argmin(cost))
        best_cost = float(cost[best_idx]); best_ep = float(ep[best_idx]); last_cost = float(cost[-1])
        gap = 100.0 * (last_cost - best_cost) / max(1e-9, best_cost)
        at_boundary = best_ep >= end - args.boundary_window
        row = {
            "csv": path,
            "max_episode": end,
            "best_episode_from_curve": best_ep,
            "best_cost_from_curve": best_cost,
            "last_cost": last_cost,
            "last_minus_best_pct": gap,
            "tail_slope_cost_per_episode": slope,
            "tail_slope_pct_per_1000": pct_per_1000,
            "best_at_boundary": bool(at_boundary),
        }
        per_seed.append(row); slopes.append(pct_per_1000); last_best_gaps.append(gap); boundary.append(at_boundary)

    sm, slo, shi = _mean_ci(slopes, n_boot=args.bootstrap, seed=args.bootstrap_seed)
    gm, glo, ghi = _mean_ci(last_best_gaps, n_boot=args.bootstrap, seed=args.bootstrap_seed + 1)
    boundary_frac = float(np.mean(boundary))
    plateau = abs(sm) <= args.max_abs_slope_pct_per_1000 and slo <= 0.0 <= shi
    stable = gm <= args.max_last_best_gap_pct

    frontier = None
    frontier_pass = None
    if args.achieved_cost is not None and args.frontier is not None:
        gap_pct = 100.0 * (args.achieved_cost - args.frontier) / max(1e-9, args.frontier)
        frontier_pass = gap_pct <= args.max_frontier_gap_pct
        frontier = {
            "achieved_cost": args.achieved_cost,
            "frontier_cost": args.frontier,
            "gap_pct": gap_pct,
            "pass": frontier_pass,
            "note": "Use costs evaluated on comparable demand streams and cost accounting.",
        }

    signals = [plateau, stable]
    if frontier_pass is not None:
        signals.append(bool(frontier_pass))
    if all(signals):
        verdict = "CONVERGED"
        interpretation = "The raw arm is flat late in training, stable relative to its best gate, and (if supplied) at the privileged frontier."
    elif frontier_pass and (plateau or stable):
        verdict = "PRACTICALLY_CONVERGED"
        interpretation = "One curve diagnostic is imperfect, but frontier performance makes a material undertraining explanation unlikely."
    elif not plateau and sm < -args.max_abs_slope_pct_per_1000:
        verdict = "STILL_IMPROVING"
        interpretation = "Held-out cost is still falling materially near the budget boundary; targeted longer training is warranted."
    elif not stable and gm > args.max_last_best_gap_pct:
        verdict = "UNSTABLE_OR_LATE_REGRESSION"
        interpretation = "The last gates deteriorate materially from the best checkpoint; inspect PPO stability and checkpoint selection."
    else:
        verdict = "MIXED_CONVERGENCE_EVIDENCE"
        interpretation = "No clear retraining decision; inspect seed-level curves and budget checkpoints."

    payload = {
        "diagnostic": "raw_arm_convergence",
        "metric": args.metric,
        "n_seed_curves": len(per_seed),
        "per_seed": per_seed,
        "aggregate": {
            "tail_slope_pct_per_1000_mean": sm,
            "tail_slope_ci95": [slo, shi],
            "last_minus_best_pct_mean": gm,
            "last_minus_best_ci95": [glo, ghi],
            "best_at_boundary_fraction": boundary_frac,
            "plateau_pass": plateau,
            "stability_pass": stable,
            "frontier": frontier,
            "verdict": verdict,
            "interpretation": interpretation,
        },
    }
    _write_json(payload, args.out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dhat", help="MAPPO retailer forecast-head quality")
    d.add_argument("--ckpt", action="append", required=True,
                   help="Checkpoint path or glob; repeat for multiple patterns")
    d.add_argument("--episodes", type=int, default=200)
    d.add_argument("--seed-base", type=int, default=HELDOUT_SEED_BASE + 70000,
                   help="Fresh diagnostic evaluation stream; do not reuse it for claims")
    d.add_argument("--rho", type=float, default=0.9)
    d.add_argument("--mu", type=float, default=12.0)
    d.add_argument("--sigma", type=float, default=3.0)
    d.add_argument("--env-json", default=None)
    d.add_argument("--drop-step0", action=argparse.BooleanOptionalAction, default=True)
    d.add_argument("--oracle-draws", type=int, default=1_000_000)
    d.add_argument("--oracle-burn", type=int, default=10_000)
    d.add_argument("--oracle-seed", type=int, default=314159)
    d.add_argument("--healthy-mse-ratio", type=float, default=1.10)
    d.add_argument("--broken-mse-ratio", type=float, default=1.25)
    d.add_argument("--max-abs-bias", type=float, default=0.5)
    d.add_argument("--cal-slope-lo", type=float, default=0.8)
    d.add_argument("--cal-slope-hi", type=float, default=1.2)
    d.add_argument("--min-pred-std", type=float, default=0.5)
    d.add_argument("--out", default=None)
    d.set_defaults(func=diagnose_dhat)

    q = sub.add_parser("qmix-channel", help="QMIX communication health and intervention test")
    q.add_argument("--ckpt", required=True, help="QMIX communication checkpoint")
    q.add_argument("--nocomm-ckpt", default=None, help="Matched separately trained QMIX no-comm checkpoint")
    q.add_argument("--episodes", type=int, default=100)
    q.add_argument("--seed-base", type=int, default=HELDOUT_SEED_BASE + 80000)
    q.add_argument("--rho", type=float, default=0.9)
    q.add_argument("--mu", type=float, default=12.0)
    q.add_argument("--sigma", type=float, default=3.0)
    q.add_argument("--shuffle-seed", type=int, default=777000)
    q.add_argument("--mi-bins", type=int, default=10)
    q.add_argument("--bootstrap", type=int, default=10000)
    q.add_argument("--bootstrap-seed", type=int, default=11)
    q.add_argument("--min-msg-std", type=float, default=0.25)
    q.add_argument("--min-weight-ratio", type=float, default=0.05)
    q.add_argument("--min-flip-rate", type=float, default=0.01)
    q.add_argument("--material-cost-frac", type=float, default=0.01)
    q.add_argument("--out", default=None)
    q.set_defaults(func=diagnose_qmix)

    c = sub.add_parser("convergence", help="Held-out curve and frontier convergence diagnostic")
    c.add_argument("--csv", action="append", required=True,
                   help="metrics_heldout.csv path or glob; repeat for multiple patterns")
    c.add_argument("--metric", default="heldout_mean_cost")
    c.add_argument("--tail-episodes", type=float, default=2000.0)
    c.add_argument("--boundary-window", type=float, default=200.0)
    c.add_argument("--max-abs-slope-pct-per-1000", type=float, default=0.5)
    c.add_argument("--max-last-best-gap-pct", type=float, default=1.0)
    c.add_argument("--achieved-cost", type=float, default=None,
                   help="Final-evaluation achieved cost, e.g. 3745.8")
    c.add_argument("--frontier", type=float, default=None,
                   help="Comparable privileged/reference cost, e.g. 3747.6")
    c.add_argument("--max-frontier-gap-pct", type=float, default=2.0)
    c.add_argument("--bootstrap", type=int, default=10000)
    c.add_argument("--bootstrap-seed", type=int, default=22)
    c.add_argument("--out", default=None)
    c.set_defaults(func=diagnose_convergence)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()