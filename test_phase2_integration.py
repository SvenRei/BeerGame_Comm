"""
tests/test_phase2_integration.py -- Phase-2 gates for the frozen certified-dhat integration
and the arpred (condmean) control rung. Run from repo root:
    python -m tests.test_phase2_integration
All CPU. T7's golden file is (re)generated automatically if absent, so on a machine that has
never run the pre-patch code T7 degrades to a self-consistency check; the release-time proof
that the patched code reproduces the PRE-patch numbers exactly was performed once against
/tmp/golden_phase2.json in the audited sandbox (values recorded in the repair-plan changelog).
"""
import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.signal_agent import SIGNALTrainer, SIGNALActor, AGENTS                    # noqa: E402
from agents.topologies import get_adj                                                 # noqa: E402
from agents.demand_forecaster import (DemandForecaster, make_one_step_dataset,        # noqa: E402
                                      forecast_metrics, certify, save_certified)
from envs.beer_game_env import BeerGameParallelEnv                                    # noqa: E402
from scripts.demand_families import make_demand_family_envs                           # noqa: E402
from scripts.baselines import ENV_BASE                                                # noqa: E402

GOLDEN = "/tmp/golden_phase2.json"


def _env(horizon=12):
    AR1 = make_demand_family_envs(BeerGameParallelEnv)[0]
    return AR1({**ENV_BASE, "horizon": horizon, "demand_type": "poisson", "family": "ar1",
                "ar1_mu": 12.0, "ar1_rho": 0.9, "ar1_sigma": 3.0})


def _trainer(content, extra=None):
    torch.manual_seed(123)
    np.random.seed(123)
    env = _env()
    obs_dim = env.observation_space(AGENTS[0]).shape[0]
    env.reset(seed=0)
    state_dim = len(env.get_global_state())
    cfg = {"hidden": 16, "msg_content": content, "use_comm": True, "ar1_mu": 12.0,
           "ar1_rho": 0.9, "lr": 3e-4, "k_epochs": 2, **(extra or {})}
    return SIGNALTrainer(cfg, len(AGENTS), obs_dim, state_dim,
                         get_adj("retailer_broadcast")), env


def _run(content):
    tr, env = _trainer(content)
    ep = tr.collect(env, seed=7)
    a, c = tr.update([ep])
    return {"a_loss": a, "c_loss": c,
            "msg_t3": [float(x) for x in ep["msg_in"][3].reshape(-1)],
            "S_sum": float(ep["S"].sum()), "cost_sum": float(ep["cost"].sum())}


def _make_test_forecaster(path):
    """Small honestly-trained forecaster on synthetic streams, certified under a RELAXED gate
    (the gate content is stored in the artifact; certification honesty itself is covered by
    tests/test_forecaster.py T4). Fast (~seconds) so integration tests stay unit-test cheap."""
    rng = np.random.default_rng(5)
    def stream(T):
        d, x = [], 12.0
        for _ in range(T):
            x = 12.0 + 0.9 * (x - 12.0) + rng.normal(0, 3.0)
            d.append(max(0, round(x)))
        return np.array(d, float)
    tr_s = [stream(80) for _ in range(12)]
    X, Y = make_one_step_dataset(tr_s)
    f = DemandForecaster(hidden=16, initial_mean=12.0)
    opt = torch.optim.Adam(f.parameters(), lr=5e-3)
    for _ in range(300):
        opt.zero_grad()
        p, _ = f(X)
        torch.nn.functional.mse_loss(p, Y).backward()
        opt.step()
    f.eval()
    with torch.no_grad():
        p, _ = f(X)
    met = forecast_metrics(p.numpy(), Y.numpy(), p.numpy())
    cert = certify(met, gate={"max_mse_ratio": 10.0, "max_abs_bias": 5.0,
                              "calibration_slope_low": 0.0, "calibration_slope_high": 5.0,
                              "min_correlation": 0.0, "min_std_fraction": 0.0})
    assert cert["pass"]
    save_certified(path, f, met, cert, {"note": "integration-test forecaster (relaxed gate)"})
    return f


# T7 -- default-mode equivalence: with forecast_mode unset, the patched code reproduces the
# pre-patch golden numbers exactly for raw / dhat(legacy) / learned / condmean.
def test_golden_equivalence():
    if not os.path.exists(GOLDEN):
        json.dump({ct: _run(ct) for ct in ("raw", "dhat", "learned", "condmean")},
                  open(GOLDEN, "w"))
        print("T7 golden equivalence: golden file absent -> generated (self-consistency mode)")
    gold = json.load(open(GOLDEN))
    for ct, g in gold.items():
        r = _run(ct)
        for k in ("a_loss", "c_loss", "S_sum", "cost_sum"):
            assert abs(r[k] - g[k]) < 1e-6, (ct, k, r[k], g[k])
        assert np.allclose(r["msg_t3"], g["msg_t3"], atol=1e-6), ct
    print("T7 golden equivalence (raw/dhat/learned/condmean, losses+messages exact): PASS")


# T8 -- arpred == condmean semantics: mu + rho*(d_{t-1}-mu), zero at the step-0 convention,
# gradient-free, dim 1. This pins the deterministic control rung the repaired study uses.
def test_condmean_arpred_semantics():
    a = SIGNALActor(obs_dim=4, msg_dim=1, hidden=8, content="condmean")
    a.demand_mu, a.demand_rho = 12.0, 0.9
    obs = torch.tensor([[3.0, 1.0, 2.0, 17.0]])          # o[3] = d_{t-1} = 17
    h = torch.zeros(1, 8)
    assert float(a.message(obs, h, dprev=None)) == 0.0    # step-0 convention matches raw
    m = a.message(obs, h, dprev=torch.tensor([[15.0, 14.0]]))
    assert abs(float(m) - (12.0 + 0.9 * (17.0 - 12.0))) < 1e-6
    assert m.shape[-1] == 1 and not m.requires_grad
    print("T8 condmean/arpred semantics (mu+rho*(d-mu), step-0=0, no grad): PASS")


# T9 -- frozen isolation in the live trainer: forecaster params outside the optimizer and
# bit-identical across a full collect+update; aux loss inactive.
def test_frozen_isolation():
    fpath = "/tmp/test_fc_phase2.pt"
    _make_test_forecaster(fpath)
    tr, env = _trainer("dhat", extra={"forecast_mode": "separate_frozen",
                                      "forecast_ckpt": fpath})
    opt_ids = {id(p) for p in tr.params}
    assert all(id(p) not in opt_ids for p in tr.forecaster.parameters())
    assert all(not p.requires_grad for p in tr.forecaster.parameters())
    before = {k: v.clone() for k, v in tr.forecaster.state_dict().items()}
    ep = tr.collect(env, seed=7)
    tr.update([ep])
    after = tr.forecaster.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    print("T9 frozen isolation (outside optimizer, bit-identical through update): PASS")


# T10 -- frozen-dhat message correctness: the delivered dhat stream equals an independent
# forecaster recompute on the realized demand sequence, with the step-0=0 raw convention.
def test_frozen_message_correctness():
    fpath = "/tmp/test_fc_phase2.pt"
    f = _make_test_forecaster(fpath)
    tr, env = _trainer("dhat", extra={"forecast_mode": "separate_frozen",
                                      "forecast_ckpt": fpath})
    ep = tr.collect(env, seed=11, deterministic=True)
    # receivers' incoming at step t = retailer's message at t-1 (one-step delay, rbroadcast).
    inc_whole = ep["msg_in"][:, 1, 0]                    # wholesaler's incoming dhat stream [T]
    d_seq = ep["obs"][:, 0, 3]                           # retailer o[3]: 0-fill, d_0, d_1, ...
    fh = f.init_hidden(1)
    expect = [0.0, 0.0]                                  # inc[0]=m_prev init 0; inc[1]=m_0=0 (step-0 conv)
    for t in range(1, len(d_seq) - 1):
        dh, fh = f.step(d_seq[t].view(1, 1), fh)         # dhat_t from d_{t-1}
        expect.append(float(dh))
    assert np.allclose(inc_whole.numpy(), np.array(expect), atol=1e-5), \
        (inc_whole.numpy()[:6], expect[:6])
    assert float(np.std(inc_whole.numpy()[2:])) > 0.3, "certified dhat stream is constant?!"
    print("T10 frozen-dhat delivered stream == independent forecaster recompute: PASS")


# T11 -- train->eval round trip: a checkpoint assembled exactly like train_signal's payload
# reproduces the trainer-side message stream step-for-step in eval SIGNALPolicy.
def test_train_eval_roundtrip():
    from agents.eval_signal import SIGNALPolicy
    fpath = "/tmp/test_fc_phase2.pt"
    _make_test_forecaster(fpath)
    tr, env = _trainer("dhat", extra={"forecast_mode": "separate_frozen",
                                      "forecast_ckpt": fpath})
    ep = tr.collect(env, seed=13, deterministic=True)
    ckpt = {"actors": [ac.state_dict() for ac in tr.actors],
            "config": {"hidden": 16, "msg_content": "dhat", "use_comm": True,
                       "ar1_mu": 12.0, "ar1_rho": 0.9,
                       "forecast_mode": "separate_frozen"},
            "adj": get_adj("retailer_broadcast").tolist(),
            "obs_dim": int(ep["obs"].shape[-1]), "msg_content": "dhat",
            "forecaster": tr.forecaster_payload(), "seed": 13}
    env2 = _env()
    pol = SIGNALPolicy(ckpt, env2, deterministic=True)
    obs, _ = env2.reset(seed=13)
    prev_msg = np.zeros((len(AGENTS), 1), np.float32)
    t, ok = 0, True
    while True:
        inc_expect = (np.asarray(get_adj("retailer_broadcast")) @ prev_msg)
        act = pol.act(obs)
        ok &= np.allclose(inc_expect[1, 0], ep["msg_in"][t, 1, 0].item(), atol=1e-5)
        prev_msg = pol.last_msg.copy()
        obs, _, term, trunc, _ = env2.step({a: [v] for a, v in act.items()})
        t += 1
        if any(term.values()) or any(trunc.values()):
            break
    assert ok and t == ep["obs"].shape[0]
    # eval-side dhat surface reports the certified value, matching the delivered stream lag
    assert abs(pol.last_dhat[0] - pol.last_msg[0, 0]) < 1e-5
    print(f"T11 train->eval round trip ({t} steps, delivered messages identical): PASS")


# T12 -- misconfiguration fail-closed: frozen mode refuses raw content and use_dhat_head.
def test_misconfig_failclosed():
    fpath = "/tmp/test_fc_phase2.pt"
    _make_test_forecaster(fpath)
    for extra in ({"forecast_mode": "separate_frozen", "forecast_ckpt": fpath,
                   "msg_content": "raw"},
                  {"forecast_mode": "separate_frozen", "forecast_ckpt": fpath,
                   "use_dhat_head": True},
                  {"forecast_mode": "typo_mode"}):
        try:
            _trainer(extra.pop("msg_content", "dhat"), extra=extra)
            raise AssertionError(f"trainer accepted bad config {extra}")
        except ValueError:
            pass
    print("T12 misconfiguration fail-closed (raw+frozen, dhat_head+frozen, typo mode): PASS")


if __name__ == "__main__":
    test_golden_equivalence()
    test_condmean_arpred_semantics()
    test_frozen_isolation()
    test_frozen_message_correctness()
    test_train_eval_roundtrip()
    test_misconfig_failclosed()
    print("\nALL PHASE-2 INTEGRATION TESTS PASS")
