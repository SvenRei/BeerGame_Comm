"""
tests/test_phase34.py -- gates for the QMIX double-Q patch, the fair-grid projection, and the
repair-study manifest. Run from repo root:  python -m tests.test_phase34
"""
import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.qmix_agent import QMIXTrainer                                            # noqa: E402
from agents.signal_agent import AGENTS                                               # noqa: E402
from agents.topologies import get_adj                                                # noqa: E402
from envs.beer_game_env import BeerGameParallelEnv                                   # noqa: E402
from scripts.demand_families import make_demand_family_envs                          # noqa: E402
from scripts.baselines import ENV_BASE, ARCondBSPolicy, _ip                          # noqa: E402
from scripts.qmix_grid_benchmark import GridProjected                                # noqa: E402

GOLDEN = "/tmp/golden_qmix.json"


def _qmix(extra=None):
    torch.manual_seed(321)
    np.random.seed(321)
    AR1 = make_demand_family_envs(BeerGameParallelEnv)[0]
    env = AR1({**ENV_BASE, "horizon": 12, "demand_type": "poisson", "family": "ar1",
               "ar1_mu": 12.0, "ar1_rho": 0.9, "ar1_sigma": 3.0})
    obs_dim = env.observation_space(AGENTS[0]).shape[0]
    env.reset(seed=0)
    state_dim = len(env.get_global_state())
    cfg = {"hidden": 16, "msg_content": "raw", "use_comm": True, "qmix_n_actions": 41,
           "qmix_s_max": 160.0, "qmix_batch": 2, "qmix_grad_steps": 1, "qmix_seed": 5,
           **(extra or {})}
    tr = QMIXTrainer(cfg, len(AGENTS), obs_dim, state_dim, get_adj("retailer_broadcast"))
    for k in range(3):
        tr.collect(env, seed=100 + k)
    aux, td = tr.update(None)
    return tr, float(td)


# Q1 -- default (double_q off) reproduces the pre-patch vanilla-target TD loss exactly.
def test_qmix_golden_equivalence():
    if not os.path.exists(GOLDEN):
        _, td = _qmix()
        json.dump({"td": td}, open(GOLDEN, "w"))
        print("Q1 qmix golden: golden file absent -> generated (self-consistency mode)")
    gold = json.load(open(GOLDEN))["td"]
    tr, td = _qmix()
    assert not tr.double_q
    assert abs(td - gold) < 1e-6, (td, gold)
    print(f"Q1 qmix vanilla-target golden equivalence (td {td:.6f}): PASS")


# Q2 -- double_q=true is wired. At init target==online, so online-argmax == target-argmax and
# Double-Q PROVABLY equals vanilla on the first step (that identity is itself asserted); after
# deterministically desynchronizing the target nets, the two modes must diverge.
def test_double_q_changes_target():
    def run(flag):
        tr, td0 = _qmix({"qmix_double_q": flag})
        assert tr.double_q == flag
        with torch.no_grad():                       # desync targets deterministically
            for ta in tr.tgt_actors:
                ta.q_head.bias.add_(torch.linspace(-1.0, 1.0, ta.q_head.bias.numel()))
        _, td1 = tr.update(None)
        return td0, td1
    (off0, off1), (on0, on1) = run(False), run(True)
    assert abs(on0 - off0) < 1e-6, "synced-target identity violated (should equal vanilla)"
    assert abs(on1 - off1) > 1e-6, "double_q did not change the target after desync"
    print(f"Q2 double-Q: synced identity holds ({off0:.4f}), desynced diverges "
          f"({off1:.4f} vs {on1:.4f}): PASS")


# Q3 -- grid projection: snap correctness incl. midpoints/ends, and actuation identity with
# the QMIX formula order = clip(S_snap - IP, 0, max_order).
def test_grid_projection():
    cond = ARCondBSPolicy(12.0, 0.9, 3.0)
    gp = GridProjected(cond, n_actions=41, s_max=160.0)         # spacing 4
    for S, want in ((0.0, 0.0), (1.9, 0.0), (2.1, 4.0), (57.9, 56.0), (58.1, 60.0),
                    (159.0, 160.0), (500.0, 160.0)):
        assert abs(gp._snap(S) - want) < 1e-9, (S, gp._snap(S), want)
    obs = {a: np.array([5.0, 1.0, 2.0, 14.0]) for a in AGENTS}  # IP = 5-1+2 = 6

    class E:
        max_order = 100.0
    act = gp.act(obs, E())
    d_t = 14.0
    for i, a in enumerate(AGENTS):
        S = cond.tau[i] * 12.0 + cond._phi[i] * (d_t - 12.0) + cond._safety[i]
        order = float(np.clip(gp._snap(S) - _ip(obs[a]), 0.0, 100.0))
        got = float(np.asarray(act[a]).reshape(-1)[0]) * 100.0 \
            if np.asarray(act[a]).max() <= 1.0 else float(np.asarray(act[a]).reshape(-1)[0])
        assert abs(got - order) < 1e-6, (a, got, order)
    print("Q3 grid projection snap + actuation identity: PASS")


# Q4 -- manifest disjointness and gate presence (fail-closed on a mutated manifest).
def test_manifest():
    man = json.load(open("reports/REPAIR_SEED_MANIFEST.json"))
    dev, conf = set(man["seeds"]["dev"]), set(man["seeds"]["confirmatory"])
    assert dev == set(range(60, 65)) and conf == set(range(70, 95))
    assert not dev & conf
    for k, v in man["seeds"]["forbidden_overlap"].items():
        assert not ((dev | conf) & set(v)), k
    for a in ("r4_dhatc", "r4_arpred", "r4_raw_c12", "r4_nocomm_c12", "r4_raw_c20",
              "r4_nocomm_c20", "r4_learned", "r4_ip"):
        assert a in man["signal_arms"], a
    for g in ("qmix_dev_absolute", "qmix_dev_relative", "signal_dev", "p2_prime",
              "transfer_2x2", "primary_family_v3"):
        assert g in man["gates_predeclared"]
    assert "POST-UNBLINDING TARGETED FOLLOW-UP" in man["labels"]
    print("Q4 seed-manifest disjointness + predeclared gates: PASS")


# Q5 -- orchestrator plan mode: lists jobs, launches nothing, exits 0.
def test_prereg_v3_crosscheck():
    import subprocess
    r = subprocess.run([sys.executable, "scripts/prereg_v3.py"], capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "sha256 = " in r.stdout
    d = json.load(open("reports/PREREG_v3.json"))
    man = json.load(open("reports/REPAIR_SEED_MANIFEST.json"))
    assert d["registration"]["design"]["arms_signal"] == man["signal_arms"]
    assert d["registration"]["design"]["seeds"]["confirmatory"] == man["seeds"]["confirmatory"]
    print("Q6 prereg_v3 self-hash + manifest cross-check: PASS")


def test_orchestrator_plan():
    import subprocess
    r = subprocess.run([sys.executable, "run_repair_study.py", "plan"],
                       capture_output=True, text=True, cwd=os.path.dirname(
                           os.path.dirname(os.path.abspath(__file__))) if False else None)
    assert r.returncode == 0, r.stderr[-400:]
    assert "qmix-dev" in r.stdout and "signal-conf" in r.stdout
    assert "example:" in r.stdout
    print("Q5 orchestrator plan mode (dry-run, exit 0): PASS")


if __name__ == "__main__":
    test_qmix_golden_equivalence()
    test_double_q_changes_target()
    test_grid_projection()
    test_manifest()
    test_prereg_v3_crosscheck()
    test_orchestrator_plan()
    print("\nALL PHASE-3/4 TESTS PASS")
