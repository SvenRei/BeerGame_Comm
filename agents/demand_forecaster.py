"""
demand_forecaster.py -- dedicated, policy-independent one-step demand forecaster (dhat repair).

WHY THIS FILE EXISTS (post-unblinding diagnostic, seeds 25-49): the original SIGNAL "dhat" is a
linear readout of the POLICY GRU hidden state, trained by an aux MSE that is ADDED to the PPO
loss under one shared Adam (signal_agent.py: demand_estimate(h); ploss += aux_coef*aux; one
optimizer over all params). With use_dhat_head, dhat also feeds the base-stock head (init
weight 5.0), so PPO's preferred order-up-to level back-propagates INTO the forecast. Measured
result: pred SD 0.50 vs benchmark 5.52, median MSE ratio 4.29, bias +1.68 -- a biased constant,
not a forecast. Every raw-vs-dhat contrast in the original campaign is therefore a raw-vs-
(degenerate-latent) contrast and is labeled as such; this module is the repair.

DESIGN (mirrors the external repair spec, Part A):
  * separate GRU + head; NO dependence on any policy state;
  * input = realized demand history only (demand is exogenous to the supply-chain policy, so
    pretrain->certify->freeze is the clean mode);
  * softplus output with bias initialized to inverse-softplus(initial_mean) (A9);
  * certification gate (A14) computed against the EMPIRICAL conditional benchmark of the exact
    rounded/truncated DGP (A16) -- the linear predictor mu+rho(d-mu) is reported alongside but
    is NOT called the exact conditional mean, because integer rounding and truncation at 0 make
    that claim false;
  * no economic interpretation is permitted for a forecaster that fails the gate: certify()
    returns pass/fail per criterion and the caller must enforce it.

This module is deliberately import-light (torch + numpy) and is NOT wired into train_signal
yet; integration (detached input, separate optimizer, dhat_used in the rollout buffer) is a
separate reviewed patch.
"""
import math
import json
import numpy as np
import torch
import torch.nn as nn

# ---- A14 certification thresholds (AR(1) rho=0.9 defaults; override per environment). ------
DEFAULT_GATE = {
    "max_mse_ratio": 1.10,          # vs empirical conditional benchmark
    "max_abs_bias": 0.50,
    "calibration_slope_low": 0.80,
    "calibration_slope_high": 1.20,
    "min_correlation": 0.80,
    "min_std_fraction": 0.50,       # pred SD >= 50% of benchmark pred SD (anti-constant)
}


def inv_softplus(y):
    """Bias b with softplus(b)=y (for y>0): b = log(exp(y)-1). Numerically safe for y>~1."""
    y = float(y)
    return math.log(math.expm1(y)) if y < 20 else y


class DemandForecaster(nn.Module):
    """One-step-ahead demand forecaster. Input: demand history d_{<=t-1}; output: dhat_t.

    forward(seq): seq [B, T, 1] of PAST demands -> preds [B, T, 1] where preds[:, k] is the
    forecast of the element AFTER seq[:, k] (i.e., trained target for preds[:, k] is
    seq[:, k+1] shifted in the dataset builder -- see make_one_step_dataset, which owns the
    temporal alignment and is unit-tested for it).
    step(d_prev, h): online API for rollout use later (d_prev [B,1] the just-realized demand).
    """

    def __init__(self, hidden=32, initial_mean=12.0):
        super().__init__()
        self.hidden = int(hidden)
        self.gru = nn.GRU(1, self.hidden, batch_first=True)
        self.head = nn.Linear(self.hidden, 1)
        nn.init.zeros_(self.head.weight)                       # start as the constant prior...
        nn.init.constant_(self.head.bias, inv_softplus(initial_mean))   # ...= initial_mean (A9)
        self.initial_mean = float(initial_mean)

    def forward(self, seq, h0=None):
        out, hT = self.gru(seq, h0)
        return nn.functional.softplus(self.head(out)), hT      # nonneg forecasts

    @torch.no_grad()
    def step(self, d_prev, h):
        out, h2 = self.gru(d_prev.view(-1, 1, 1), h)
        return nn.functional.softplus(self.head(out)).view(-1, 1), h2

    def init_hidden(self, batch=1):
        return torch.zeros(1, batch, self.hidden)


# ==============================================================================
# dataset construction -- OWNS the temporal alignment (tested in tests/test_forecaster.py)
# ==============================================================================
def make_one_step_dataset(streams):
    """streams: list of 1-D demand arrays (one per episode). Returns (X, Y) with
    X [B, T-1, 1] = d_0..d_{T-2} and Y [B, T-1, 1] = d_1..d_{T-1}: the model prediction at
    position k (having seen d_0..d_k) is paired with target d_{k+1}. Episodes are independent
    (hidden state resets per row), which is the reset-boundary handling."""
    T = min(len(s) for s in streams)
    if T < 2:
        raise ValueError("streams must have length >= 2")
    X = np.stack([np.asarray(s[: T - 1], dtype=np.float32) for s in streams])[..., None]
    Y = np.stack([np.asarray(s[1:T], dtype=np.float32) for s in streams])[..., None]
    return torch.from_numpy(X), torch.from_numpy(Y)


# ==============================================================================
# benchmarks (A16): empirical conditional mean of the EXACT (rounded, truncated) DGP
# ==============================================================================
def empirical_conditional_benchmark(train_streams, mu, rho):
    """Binned E[d_t | d_{t-1}] estimated from the true DGP's own samples, with the linear
    predictor mu + rho*(d_prev - mu) (clipped at 0) as fallback for unseen bins. Demand is
    integer-valued in this env, so bins are exact integer values of d_{t-1}."""
    from collections import defaultdict
    sums, cnts = defaultdict(float), defaultdict(int)
    for s in train_streams:
        s = np.asarray(s)
        for a, b in zip(s[:-1], s[1:]):
            k = int(round(a))
            sums[k] += float(b)
            cnts[k] += 1
    table = {k: sums[k] / cnts[k] for k in cnts}

    def predict(d_prev):
        d_prev = np.asarray(d_prev, dtype=float)
        out = np.maximum(0.0, mu + rho * (d_prev - mu))         # linear fallback (NOT exact CM)
        flat = out.reshape(-1)
        for i, v in enumerate(np.round(d_prev).astype(int).reshape(-1)):
            if v in table and cnts[v] >= 25:                    # trust bins with support
                flat[i] = table[v]
        return flat.reshape(d_prev.shape)

    return predict


def linear_predictor(mu, rho):
    """The registered linear predictor mu + rho*(d-mu), clipped at 0. Reported alongside the
    empirical benchmark; deliberately NOT labeled the exact conditional mean (rounding and
    truncation in the DGP make that false)."""
    def predict(d_prev):
        return np.maximum(0.0, mu + rho * (np.asarray(d_prev, dtype=float) - mu))
    return predict


# ==============================================================================
# certification (A13 metrics + A14 gate)
# ==============================================================================
def forecast_metrics(preds, targets, bench_preds):
    p = np.asarray(preds, float).reshape(-1)
    y = np.asarray(targets, float).reshape(-1)
    b = np.asarray(bench_preds, float).reshape(-1)
    mse, bmse = float(np.mean((p - y) ** 2)), float(np.mean((b - y) ** 2))
    const_mse = float(np.mean((np.mean(y) - y) ** 2))
    slope, intercept = (np.polyfit(p, y, 1) if np.std(p) > 1e-9 else (float("nan"),) * 2)
    return {
        "mse": mse, "rmse": math.sqrt(mse), "mae": float(np.mean(np.abs(p - y))),
        "bias": float(np.mean(p - y)), "pred_sd": float(np.std(p)),
        "target_sd": float(np.std(y)), "corr": (float(np.corrcoef(p, y)[0, 1])
                                                if np.std(p) > 1e-9 else 0.0),
        "calib_slope": float(slope), "calib_intercept": float(intercept),
        "bench_mse": bmse, "bench_rmse": math.sqrt(bmse), "bench_pred_sd": float(np.std(b)),
        "const_mse": const_mse, "mse_ratio_vs_bench": mse / bmse if bmse > 0 else float("inf"),
        "mse_ratio_vs_const": mse / const_mse if const_mse > 0 else float("inf"),
        "n": int(p.size),
    }


def certify(metrics, gate=None):
    g = dict(DEFAULT_GATE, **(gate or {}))
    checks = {
        "mse_ratio": metrics["mse_ratio_vs_bench"] <= g["max_mse_ratio"],
        "bias": abs(metrics["bias"]) <= g["max_abs_bias"],
        "calibration": (g["calibration_slope_low"] <= metrics["calib_slope"]
                        <= g["calibration_slope_high"]),
        "correlation": metrics["corr"] >= g["min_correlation"],
        "variance": metrics["pred_sd"] >= g["min_std_fraction"] * metrics["bench_pred_sd"],
    }
    return {"pass": all(checks.values()), "checks": checks, "gate": g}


def certification_report(metrics, cert):
    lines = [f"  forecast RMSE {metrics['rmse']:.3f} vs benchmark {metrics['bench_rmse']:.3f} "
             f"(MSE ratio {metrics['mse_ratio_vs_bench']:.3f})",
             f"  bias {metrics['bias']:+.3f}  pred SD {metrics['pred_sd']:.3f} "
             f"(benchmark {metrics['bench_pred_sd']:.3f})  corr {metrics['corr']:.3f}  "
             f"calib slope {metrics['calib_slope']:.3f}",
             "  gate: " + "  ".join(f"{k}={'PASS' if v else 'FAIL'}"
                                    for k, v in cert["checks"].items()),
             f"  => CERTIFIED: {'YES' if cert['pass'] else 'NO'}"]
    return "\n".join(lines)


def save_certified(path, model, metrics, cert, meta):
    torch.save({"forecaster_state": model.state_dict(),
                "hidden": model.hidden, "initial_mean": model.initial_mean,
                "metrics": metrics, "certification": cert, "meta": meta,
                "forecaster_version": 1}, path)


def load_certified(path, require_pass=True):
    d = torch.load(path, map_location="cpu", weights_only=False)
    m = DemandForecaster(hidden=d["hidden"], initial_mean=d["initial_mean"])
    m.load_state_dict(d["forecaster_state"])
    m.eval()
    if require_pass and not d["certification"]["pass"]:
        raise RuntimeError(f"forecaster at {path} FAILED certification; "
                           f"refusing to load for policy training (require_pass=True)")
    return m, d
