#!/usr/bin/env python3
"""
plot_train_curves.py -- the REAL training curve: supply-chain team cost at the TRAINING regime.

Why this exists
---------------
Every figure produced so far (fig1..fig13) plots `heldout_mean_cost`, i.e. the SELECTION GATE:
a deterministic evaluation at rho in {0.15, 0.45, 0.75}, sampled every `heldout_every` (200) eps,
and it stops when the run early-stops. That is the right series for checkpoint selection and the
wrong series for the question "is the agent learning to run the supply chain?".

The trainer also prints, every `log_every` (100) episodes:

    [signal] ep 1300/8000  train_team_cost~3611  a_loss=... c_loss=...

`train_team_cost` = np.mean(deque(maxlen=50)) of the TOTAL team cost of the last 50 TRAINING
episodes, on the training env (AR(1), rho=0.9 -- the deployment regime), under the STOCHASTIC
(exploration) policy. Dense, at the right rho, and it shows the rise-fall-degradation shape that
the sparse gate series cannot. This script parses it out of the trainlogs and plots it.

Caveats printed on the figure, because they matter for interpretation:
  * stochastic policy => a level offset ABOVE the deterministic eval of the same weights;
  * 50-episode rolling mean => lags the true instantaneous policy by ~25 episodes;
  * training regime rho=0.9 is NOT the gate regime, so this curve can fall while the gate is flat.
    That divergence is itself a finding, not an error -- see the printed gate/train comparison.

Trainlogs APPEND across re-runs of the same arm+seed, so only the LAST `$ `-delimited block is
parsed (matching the newest run directory, same rule as the curves stage).

Usage:
    python plot_train_curves.py                       # dev seeds 60,61, all ladder arms
    python plot_train_curves.py --seeds 60            # one seed
    python plot_train_curves.py --arms r4_raw r4_nocomm
    python plot_train_curves.py --csv                 # also dump the parsed series to CSV
"""
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
W = os.path.join(ROOT, "weights_signal")
FIGS = os.path.join(ROOT, "repair_out", "figures")

LADDER = ["r4_nocomm", "r4_raw", "r4_arpred", "r4_dhatc", "r4_learned", "r4_ip"]
COLORS = {"r4_nocomm": "#1f77b4", "r4_raw": "#ff7f0e", "r4_arpred": "#2ca02c",
          "r4_dhatc": "#d62728", "r4_learned": "#9467bd", "r4_ip": "#8c564b"}

RE_TRAIN = re.compile(r"ep (\d+)/\d+\s+train_team_cost~([\d.]+)")
RE_GATE = re.compile(r"ep (\d+): held-out mean cost ([\d.]+)")
RE_STOP = re.compile(r"EARLY STOP at ep (\d+)")


def logfile(tag, seed):
    for pat in (f".trainlog_{tag}_s{seed}.txt", f".sweeplog_{tag}_s{seed}.txt"):
        p = os.path.join(W, pat)
        if os.path.exists(p):
            return p
    return None


def parse(tag, seed):
    """Return (episodes, train_cost, gate_eps, gate_cost, stop_ep) from the LAST run block."""
    p = logfile(tag, seed)
    if not p:
        return None
    txt = open(p, errors="ignore").read()
    blocks = [b for b in txt.split("\n$ ") if "train_team_cost" in b or "held-out" in b]
    if not blocks:
        return None
    b = blocks[-1]                                     # newest attempt wins (logs append)
    tr = [(int(e), float(c)) for e, c in RE_TRAIN.findall(b)]
    ga = [(int(e), float(c)) for e, c in RE_GATE.findall(b)]
    st = RE_STOP.findall(b)
    if not tr:
        return None
    return ([e for e, _ in tr], [c for _, c in tr],
            [e for e, _ in ga], [c for _, c in ga],
            int(st[-1]) if st else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="60,61")
    ap.add_argument("--arms", nargs="*", default=LADDER)
    ap.add_argument("--csv", action="store_true")
    ap.add_argument("--diag", action="store_true",
                    help="also render fig15 diagnostic panels per run from metrics_*.csv "
                         "(gate+bests, entropy vs annealed coef, action-std sharpening, "
                         "approx_kl, value_loss, grad_norm)")
    a = ap.parse_args()
    os.makedirs(FIGS, exist_ok=True)
    seeds = [int(s) for s in a.seeds.split(",")]

    for seed in seeds:
        series = {}
        for arm in a.arms:
            r = parse(arm, seed)
            if r:
                series[arm] = r
        if not series:
            print(f"  seed {seed}: no trainlogs found -- skipped")
            continue

        fig, ax = plt.subplots(figsize=(9, 5.5))
        for arm, (eps, cost, _ge, _gc, stop) in series.items():
            ax.plot(eps, cost, lw=1.6, color=COLORS.get(arm), label=arm.replace("r4_", ""))
            lo = min(range(len(cost)), key=lambda i: cost[i])
            ax.plot(eps[lo], cost[lo], "o", ms=5, color=COLORS.get(arm), zorder=5)
        ax.axhline(3747.6, ls="--", lw=1.2, color="k", alpha=.65)
        ax.axhline(4988.6, ls=":", lw=1.2, color="gray", alpha=.8)
        ax.text(ax.get_xlim()[1], 3747.6, " AR_CondBS", va="center", fontsize=8)
        ax.text(ax.get_xlim()[1], 4988.6, " AR_StaticBS", va="center", fontsize=8, color="gray")
        ax.set_xlabel("Training episode")
        ax.set_ylabel("Team cost per episode  (50-ep rolling mean, training env)")
        ax.set_title(f"TRAINING curve -- supply-chain team cost at rho=0.9  (dev seed {seed})")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=.25)
        fig.text(0.5, 0.005,
                 "stochastic (exploration) policy => level offset above deterministic eval; "
                 "dots = per-arm minimum",
                 ha="center", fontsize=7.5, color="#555555")
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        out = os.path.join(FIGS, f"fig14_train_cost_s{seed}.pdf")
        fig.savefig(out); plt.close(fig)
        print(f"  wrote {out}")

        print(f"\n-- seed {seed}: TRAIN cost (rho=0.9) vs GATE cost (rho .15/.45/.75) --")
        print(f"   {'arm':12s} {'train@start':>11s} {'train@min':>10s} {'@ep':>6s} "
              f"{'train@end':>10s} {'gate best@ep':>13s} {'stop':>6s}")
        for arm, (eps, cost, ge, gc, stop) in series.items():
            lo = min(range(len(cost)), key=lambda i: cost[i])
            gbest = min(range(len(gc)), key=lambda i: gc[i]) if gc else None
            print(f"   {arm:12s} {cost[0]:11.0f} {cost[lo]:10.0f} {eps[lo]:6d} "
                  f"{cost[-1]:10.0f} "
                  f"{(str(ge[gbest]) if gbest is not None else '-'):>13s} "
                  f"{(str(stop) if stop else '-'):>6s}")
        print("   (train@min episode >> gate best episode  =>  the gate stopped a run that was "
              "still improving\n    at the DEPLOYMENT regime. Same episode => gate and training "
              "agree.)")

        if a.diag:
            for arm in a.arms:
                _diag_panel(arm, seed)

        if a.csv:
            cp = os.path.join(ROOT, "repair_out", "curves", f"train_cost_s{seed}.csv")
            os.makedirs(os.path.dirname(cp), exist_ok=True)
            with open(cp, "w") as f:
                f.write("arm,episode,train_team_cost\n")
                for arm, (eps, cost, _a, _b, _c) in series.items():
                    for e, c in zip(eps, cost):
                        f.write(f"{arm},{e},{c}\n")
            print(f"   csv -> {cp}")



def _newest_dir(tag):
    import glob as _g
    ds = [d for d in _g.glob(os.path.join(W, f"run_signal_*_{tag}")) if os.path.isdir(d)]
    return max(ds, key=os.path.getmtime) if ds else None


def _diag_panel(arm, seed):
    """fig15: per-run training diagnostics from metrics_update.csv + metrics_heldout.csv."""
    import csv as _csv
    tag = f"{arm}_s{seed}"
    d = _newest_dir(tag)
    if not d:
        print(f"  diag {tag}: no run dir -- skipped"); return
    try:
        U = list(_csv.DictReader(open(os.path.join(d, "metrics_update.csv"))))
        H = list(_csv.DictReader(open(os.path.join(d, "metrics_heldout.csv"))))
    except FileNotFoundError as e:
        print(f"  diag {tag}: {e} -- skipped (SIGNAL_CSVLOG off for this run?)"); return
    if len(U) < 3:
        print(f"  diag {tag}: metrics_update has {len(U)} rows -- skipped"); return
    g = lambda r, k: (float(r[k]) if r.get(k) not in (None, "", "nan") else float("nan"))
    ue = [g(r, "episode") for r in U]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    # 1 gate + bests
    he = [g(r, "episode") for r in H]; hc = [g(r, "heldout_mean_cost") for r in H]
    ax[0, 0].plot(he, hc, lw=1.4, color="#1d4ed8")
    best = float("inf"); bx = []; by = []
    for e_, c_ in zip(he, hc):
        if c_ < best - 1e-9: best = c_; bx.append(e_); by.append(c_)
    ax[0, 0].plot(bx, by, "o", ms=4, color="#dc2626")
    ax[0, 0].set_title("gate cost (rho .15/.45/.75) + new bests")
    # 2 entropy + the coefficient actually applied
    ax[0, 1].plot(ue, [g(r, "entropy") for r in U], lw=1.2, color="#15803d",
                  label="policy entropy")
    ax2 = ax[0, 1].twinx()
    # reconstruct the schedule from the run config header if present, else leave blank
    ax[0, 1].set_title("policy entropy (sharpening trace)")
    ax[0, 1].legend(fontsize=8)
    # 3 action std
    ax[0, 2].plot(ue, [g(r, "action_std") for r in U], lw=1.2, color="#b45309")
    ax[0, 2].set_title("action std (grid spread; uniform 41 bins ~= 46)")
    # 4 approx_kl
    ax[1, 0].plot(ue, [g(r, "approx_kl") for r in U], lw=1.0, color="#6d28d9")
    ax[1, 0].axhline(0.02, ls="--", lw=1, color="gray")
    ax[1, 0].set_title("approx KL (PPO target ~0.02 dashed)")
    ax[1, 0].set_yscale("symlog", linthresh=0.05)
    # 5 value loss
    ax[1, 1].plot(ue, [g(r, "value_loss") for r in U], lw=1.0, color="#0f766e")
    ax[1, 1].set_title("critic loss"); ax[1, 1].set_yscale("log")
    # 6 grad norm
    ax[1, 2].plot(ue, [g(r, "grad_norm") for r in U], lw=1.0, color="#334155",
                  label="actor")
    cg = [g(r, "critic_grad_norm") for r in U]
    if any(x == x for x in cg):
        ax[1, 2].plot(ue, cg, lw=1.0, color="#94a3b8", label="critic")
        ax[1, 2].legend(fontsize=8)
    ax[1, 2].set_title("pre-clip grad norm"); ax[1, 2].set_yscale("log")
    for row in ax:
        for a_ in row:
            a_.grid(alpha=.25); a_.set_xlabel("episode")
    fig.suptitle(f"training diagnostics -- {tag}  ({os.path.basename(d)})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(FIGS, f"fig15_diag_{tag}.pdf")
    fig.savefig(out); plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
