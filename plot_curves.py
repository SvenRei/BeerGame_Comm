#!/usr/bin/env python
"""
plot_curves.py -- publishable seed-aggregated learning curves from the SIGNAL CSV logs.
================================================================================
Ingests the per-seed `metrics_heldout.csv` files that train_signal.py writes under
SIGNAL_CSVLOG=1 (one per run_dir), groups them into ARMS, aligns on `episode`, and
plots the seed-aggregated mean of `heldout_mean_cost` with an uncertainty band, to a
vector PDF in a colorblind-safe palette.

This is a STANDALONE analysis tool: stdlib csv for reading, numpy + matplotlib for the
figure, NO experiment tracker and NO dependency on the training code. The CSV is the
artifact; this script is one deterministic step from the artifact to the figure.

BAND (parameterized, default bootstrap-95%-CI):
  --band bootstrap-ci : studentized bootstrap-t CI of the across-seed MEAN (default). This
                        matches the studentized bootstrap-t used for the confirmatory tables
                        (scripts/c1_stats, prereg.py), so the figure and the tables tell one
                        statistical story.
  --band iqr          : 25th-75th percentile across seeds (distribution-free spread).

USAGE:
  # one arm (glob over its per-seed run dirs), then two arms overlaid:
  python plot_curves.py --arm upstream "weights_signal/run_signal_*_dp_upstream_s*/metrics_heldout.csv" \
                        --out fig_upstream.pdf
  python plot_curves.py --arm comm    "weights_signal/*_ar1r9_upstream_s*/metrics_heldout.csv" \
                        --arm nocomm  "weights_signal/*_ar1r9_nocomm_s*/metrics_heldout.csv" \
                        --metric heldout_mean_cost --band bootstrap-ci --out fig_h2.pdf

Each --arm takes LABEL GLOB. Any column in the CSV can be the y metric (--metric),
e.g. gap_recovered, forecast_error, positive_listening, msg_std.
"""
import os
import sys
import csv
import glob
import argparse

import numpy as np

# Wong (2011) colorblind-safe qualitative palette (Nature Methods 8:441). Order chosen so the
# first two arms (the usual comm/no-comm contrast) are the maximally-distinguishable blue/orange.
_PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000"]


# ------------------------------------------------------------------ CSV ingest (stdlib only)
def _to_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def load_series(path, xkey, ykey):
    """Return (episodes[np], values[np]) from one metrics_heldout.csv, sorted by episode."""
    xs, ys = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if xkey not in row or ykey not in row:
                raise KeyError(f"{path}: missing column '{xkey}' or '{ykey}' "
                               f"(has {list(row.keys())})")
            xs.append(_to_float(row[xkey])); ys.append(_to_float(row[ykey]))
    x = np.asarray(xs, float); y = np.asarray(ys, float)
    order = np.argsort(x)
    return x[order], y[order]


def load_arm(pattern, xkey, ykey):
    """Load every per-seed CSV matching `pattern` into {episode -> [values across seeds]},
    keeping only episodes present in EVERY seed (intersection -> no imputation/extrapolation)."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        return None, 0, []
    per_seed = []
    for p in paths:
        x, y = load_series(p, xkey, ykey)
        per_seed.append(dict(zip(x.tolist(), y.tolist())))
    common = set(per_seed[0])
    for d in per_seed[1:]:
        common &= set(d)
    eps = np.array(sorted(common), float)
    mat = np.array([[d[e] for e in eps] for d in per_seed], float)  # [n_seeds, n_eps]
    truncated = [len(d) for d in per_seed if len(d) != len(eps)]
    return eps, len(paths), (mat, truncated)


# ------------------------------------------------------------------ uncertainty bands
def _bootstrap_t_ci(x, n_boot, alpha, rng):
    """Studentized bootstrap-t CI of the mean (same family as the confirmatory tables)."""
    x = np.asarray([v for v in x if v == v], float)
    n = x.size
    theta = float(x.mean()) if n else float("nan")
    if n < 2:
        return theta, theta
    se = x.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return theta, theta
    t = np.empty(n_boot)
    for b in range(n_boot):
        xb = rng.choice(x, n, replace=True)
        seb = xb.std(ddof=1) / np.sqrt(n)
        t[b] = (xb.mean() - theta) / seb if seb > 0 else 0.0
    t_lo, t_hi = np.percentile(t, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return theta - t_hi * se, theta - t_lo * se       # pivot inversion


def aggregate(mat, band, n_boot, alpha, seed):
    """mat [n_seeds, n_eps] -> (center[n_eps], lo[n_eps], hi[n_eps])."""
    rng = np.random.default_rng(seed)
    center = np.nanmean(mat, axis=0)
    lo = np.empty(mat.shape[1]); hi = np.empty(mat.shape[1])
    for j in range(mat.shape[1]):
        col = mat[:, j]
        if band == "iqr":
            lo[j], hi[j] = np.nanpercentile(col, [25, 75])
        else:                                          # bootstrap-ci (default)
            lo[j], hi[j] = _bootstrap_t_ci(col, n_boot, alpha, rng)
    return center, lo, hi


# ------------------------------------------------------------------ figure
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", nargs=2, action="append", metavar=("LABEL", "GLOB"), required=True,
                    help="LABEL and a glob over per-seed metrics_heldout.csv files. Repeatable.")
    ap.add_argument("--metric", default="heldout_mean_cost", help="y-axis column (default: the estimand)")
    ap.add_argument("--xkey", default="episode")
    ap.add_argument("--band", choices=["bootstrap-ci", "iqr"], default="bootstrap-ci",
                    help="uncertainty band (default bootstrap-95%%-CI, consistent with the tables)")
    ap.add_argument("--ci", type=float, default=0.95, help="CI level for --band bootstrap-ci")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0, help="bootstrap RNG seed (reproducible band)")
    ap.add_argument("--out", default="learning_curves.pdf", help="vector output (.pdf recommended)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--ylabel", default=None)
    ap.add_argument("--figsize", nargs=2, type=float, default=[7.0, 4.5])
    args = ap.parse_args(argv)

    try:
        import matplotlib
        matplotlib.use("Agg")                          # headless / air-gapped safe
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("plot_curves.py needs matplotlib (pip install matplotlib). The CSVs are the "
                 "artifact; only this plotting step requires it.")

    alpha = 1.0 - args.ci
    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    plotted = 0
    for k, (label, pattern) in enumerate(args.arm):
        eps, n_seeds, payload = load_arm(pattern, args.xkey, args.metric)
        if eps is None or len(eps) == 0:
            print(f"[plot] arm '{label}': no CSVs / no common episodes for glob {pattern!r} -- skipped")
            continue
        mat, truncated = payload
        if truncated:
            print(f"[plot] arm '{label}': {len(truncated)} seed(s) shorter than the common grid "
                  f"-> aligned on the {len(eps)} shared episodes (no imputation).")
        center, lo, hi = aggregate(mat, args.band, args.n_boot, alpha, args.seed)
        color = _PALETTE[k % len(_PALETTE)]
        ax.plot(eps, center, color=color, lw=2.0, label=f"{label} (n={n_seeds})", zorder=3)
        ax.fill_between(eps, lo, hi, color=color, alpha=0.20, lw=0, zorder=2)
        plotted += 1

    if not plotted:
        sys.exit("[plot] nothing to plot -- check the --arm globs.")

    band_name = ("bootstrap-%d%% CI" % round(args.ci * 100)) if args.band == "bootstrap-ci" else "IQR"
    ax.set_xlabel("Training episode")
    ax.set_ylabel(args.ylabel or args.metric.replace("_", " "))
    ax.set_title(args.title or f"Held-out learning curve  (mean ± {band_name})")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out)                              # vector when args.out ends in .pdf/.svg
    print(f"[plot] wrote {args.out}  ({plotted} arm(s), metric={args.metric}, band={band_name})")


if __name__ == "__main__":
    main()
