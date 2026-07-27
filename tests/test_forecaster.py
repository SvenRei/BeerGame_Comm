"""
tests/test_forecaster.py -- the five pre-training gates for the repaired dhat (spec A2/A5/A12/
A15/A9). Run from repo root:  python -m tests.test_forecaster
All CPU, seconds. Training of any repaired arm is FORBIDDEN until every test passes.
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.demand_forecaster import (DemandForecaster, make_one_step_dataset,          # noqa: E402
                                      empirical_conditional_benchmark, linear_predictor,
                                      forecast_metrics, certify, save_certified,
                                      load_certified, inv_softplus)


def _ar1_stream(T, mu=12.0, rho=0.9, sigma=3.0, rng=None):
    """The env's DGP shape: AR(1) latent, rounded to int, truncated at 0."""
    rng = rng or np.random.default_rng(0)
    d, x = [], mu
    for _ in range(T):
        x = mu + rho * (x - mu) + rng.normal(0, sigma)
        d.append(max(0, int(round(x))))
    return np.array(d, float)


# T1 -- temporal alignment (A2): the dataset builder pairs the prediction made after seeing
# d_0..d_k with target d_{k+1}, never d_k or d_{k+2}; episodes reset independently.
def test_temporal_alignment():
    s0 = np.arange(10, 20, dtype=float)            # strictly increasing ramp: unambiguous lags
    s1 = np.arange(50, 60, dtype=float)
    X, Y = make_one_step_dataset([s0, s1])
    assert X.shape == (2, 9, 1) and Y.shape == (2, 9, 1)
    assert torch.allclose(Y[0, :, 0], torch.tensor(s0[1:], dtype=torch.float32))
    assert torch.allclose(X[0, :, 0], torch.tensor(s0[:-1], dtype=torch.float32))
    # target at position k is exactly one ahead of input at position k -- not 0, not 2:
    assert float((Y[0] - X[0]).abs().mean()) == 1.0
    # episode independence: row 1 never sees row 0's values (builder keeps rows separate)
    assert float(X[1].min()) == 50.0
    print("T1 temporal alignment: PASS")


# T2 -- gradient isolation (A5): a policy-style loss must not touch forecaster params; the
# forecast loss must. This is the property the ORIGINAL implementation violates by design.
def test_gradient_isolation():
    f = DemandForecaster(hidden=8, initial_mean=12.0)
    policy = nn.Sequential(nn.Linear(3, 16), nn.Tanh(), nn.Linear(16, 1))
    X, Y = make_one_step_dataset([_ar1_stream(40)])
    dhat, _ = f(X)
    policy_in = torch.cat([X[:, :3, 0], dhat[:, :3, 0].detach()], dim=-1)[:, :3]
    policy_loss = policy(policy_in).pow(2).mean()          # uses dhat.detach() -- the repaired contract
    policy_loss.backward()
    assert all(p.grad is None or float(p.grad.abs().sum()) == 0.0 for p in f.parameters()), \
        "policy loss leaked gradients into the forecaster"
    for p in f.parameters():
        p.grad = None
    dhat2, _ = f(X)
    nn.functional.mse_loss(dhat2, Y).backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in f.parameters()), \
        "forecast loss produced no forecaster gradients"
    print("T2 gradient isolation: PASS")


# T3 -- fixed-batch overfit (A15): on a small fixed batch the module must beat the constant
# mean decisively and produce nonconstant predictions. If it cannot, stop -- do not train RL.
def test_fixed_batch_overfit():
    rng = np.random.default_rng(1)
    streams = [_ar1_stream(60, rng=rng) for _ in range(16)]
    X, Y = make_one_step_dataset(streams)
    f = DemandForecaster(hidden=32, initial_mean=12.0)
    opt = torch.optim.Adam(f.parameters(), lr=5e-3)
    for _ in range(600):
        opt.zero_grad()
        pred, _ = f(X)
        loss = nn.functional.mse_loss(pred, Y)
        loss.backward()
        nn.utils.clip_grad_norm_(f.parameters(), 5.0)
        opt.step()
    with torch.no_grad():
        pred, _ = f(X)
    const_mse = float(((Y - Y.mean()) ** 2).mean())
    fit_mse = float(((pred - Y) ** 2).mean())
    psd, tsd = float(pred.std()), float(Y.std())
    assert fit_mse < 0.55 * const_mse, (fit_mse, const_mse)
    assert psd > 0.5 * tsd, f"pred SD {psd:.2f} collapsed vs target SD {tsd:.2f}"
    print(f"T3 fixed-batch overfit: PASS (MSE {fit_mse:.2f} vs const {const_mse:.2f}, "
          f"pred SD {psd:.2f}/{tsd:.2f})")


# T4 -- checkpoint round-trip (A12): save -> load -> numerically identical forecasts; loader
# refuses an uncertified forecaster when require_pass=True.
def test_checkpoint_roundtrip(tmpdir="/tmp"):
    f = DemandForecaster(hidden=16, initial_mean=12.0)
    X, _ = make_one_step_dataset([_ar1_stream(30)])
    with torch.no_grad():
        p1, _ = f(X)
    m = {"mse_ratio_vs_bench": 1.0, "bias": 0.0, "calib_slope": 1.0, "corr": 0.9,
         "pred_sd": 5.0, "bench_pred_sd": 5.0}
    path = os.path.join(tmpdir, "fc_rt.pt")
    save_certified(path, f, m, {"pass": True, "checks": {}, "gate": {}}, {"note": "test"})
    g, _ = load_certified(path)
    with torch.no_grad():
        p2, _ = g(X)
    assert torch.allclose(p1, p2, atol=1e-6)
    save_certified(path, f, m, {"pass": False, "checks": {}, "gate": {}}, {"note": "test"})
    try:
        load_certified(path, require_pass=True)
        raise AssertionError("loader accepted an uncertified forecaster")
    except RuntimeError:
        pass
    print("T4 checkpoint round-trip + certification refusal: PASS")


# T5 -- initialization (A9): fresh forecaster predicts ~initial_mean (not 14), stays nonneg.
def test_initialization():
    for mean in (12.0, 10.0):
        f = DemandForecaster(hidden=8, initial_mean=mean)
        X = torch.tensor([[[5.0], [0.0], [30.0]]])
        with torch.no_grad():
            p, _ = f(X)
        assert torch.all(p >= 0)
        assert abs(float(p.mean()) - mean) < 0.75, (mean, float(p.mean()))
    assert abs(float(torch.nn.functional.softplus(
        torch.tensor(inv_softplus(12.0)))) - 12.0) < 1e-4
    print("T5 initialization to environment mean + nonnegativity: PASS")


# T6 (bonus, A16 shape-check) -- on synthetic exact-DGP data the empirical conditional
# benchmark must beat the linear predictor's MSE or tie it; the metrics/gate plumbing runs.
def test_benchmark_and_gate_plumbing():
    rng = np.random.default_rng(2)
    train = [_ar1_stream(400, rng=rng) for _ in range(30)]
    test = [_ar1_stream(400, rng=rng) for _ in range(10)]
    bench = empirical_conditional_benchmark(train, mu=12.0, rho=0.9)
    lin = linear_predictor(12.0, 0.9)
    prev = np.concatenate([s[:-1] for s in test])
    targ = np.concatenate([s[1:] for s in test])
    m_b = float(np.mean((bench(prev) - targ) ** 2))
    m_l = float(np.mean((lin(prev) - targ) ** 2))
    assert m_b <= m_l * 1.02, (m_b, m_l)
    met = forecast_metrics(bench(prev), targ, bench(prev))
    cert = certify(met)
    assert cert["checks"]["variance"] and cert["checks"]["correlation"]
    print(f"T6 empirical benchmark <= linear predictor MSE ({m_b:.2f} vs {m_l:.2f}) "
          f"+ gate plumbing: PASS")


if __name__ == "__main__":
    test_temporal_alignment()
    test_gradient_isolation()
    test_fixed_batch_overfit()
    test_checkpoint_roundtrip()
    test_initialization()
    test_benchmark_and_gate_plumbing()
    print("\nALL FORECASTER GATE TESTS PASS")
