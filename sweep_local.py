#!/usr/bin/env python3
"""
sweep_local.py -- dev-seed hyperparameter sweep for SIGNAL and QMIX.

  *** THIS IS TUNING, NOT CONFIRMATORY EVIDENCE. ***
  Runs on DEV seeds only (60, 61, ...). Confirmatory seeds 70-94 are never touched. Any setting
  adopted from this sweep is an INSTRUMENT CHANGE and requires a manifest amendment + a re-hashed
  prereg BEFORE the pod run. Legitimate today only because no confirmatory seed has produced data.

Three independent stages (run separately; each is resumable):

  learned : conditioning of the learned message head. v3.3 fixed the tanh saturation
            (|tanh|=1.000, gradient ~0.003 -> responsive) but the message went QUIET:
            zeroed delta fell +238 -> +14.8, V +227 -> +167. obs_scale trades gradient
            health against emitted amplitude; msg_gain sets the amplitude directly.
            Baseline included: obs_scale=1 reproduces the OLD saturated head.

  optim   : the noise-dominated-advantage problem. Critic explained variance ~= 0, so PPO
            degenerates toward REINFORCE-with-baseline: shallow gains then post-peak
            degradation. One-change-at-a-time from the registered baseline. Trains raw AND
            nocomm per config so V is measurable within each setting.

  qmix    : the FOUR ALREADY-REGISTERED variants (qr_base/doubleq/replay/eps) via the
            orchestrator. No new knobs -- these are in the manifest and prereg already.
            QMIX failed its competence gate by +93% at rho=0.9; run these before inventing more.

Usage (from the repo root, venv active):
    python sweep_local.py learned --jobs 2
    python sweep_local.py optim   --jobs 2
    python sweep_local.py qmix    --jobs 2
    python sweep_local.py table                 # collect + print everything measured so far
    python sweep_local.py learned --dry-run     # show the plan and the wall-clock estimate
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
OUT = os.path.join(ROOT, "repair_out", "sweep")
W = os.path.join(ROOT, "weights_signal")

# Registered baseline flags shared by every SIGNAL job in this sweep (the v3.2/A19 instrument).
BASE = ["agent=signal", "total_episodes=8000", "agent.heldout_every=200",
        "agent.heldout_episodes=8", "agent.patience=2000",
        "agent.budget_milestones=[1000,2000,4000,8000]",
        "env.penalty_at_retailer_only=false", "agent.train_env=ar1", "agent.ar1_rho=0.9",
        "agent.heldout_mode=ar1", "agent.comm_topology=retailer_broadcast",
        "agent.use_dhat_head=true"]

# ---------------------------------------------------------------------------------------------
# Stage definitions: (tag, extra hydra overrides, msg_content, use_comm)
# ---------------------------------------------------------------------------------------------
def learned_configs():
    cfgs = []
    for os_scale in (1, 10, 25, 100):
        for gain in (10, 30):
            if os_scale == 1 and gain == 30:
                continue                       # old head + bigger gain: saturates harder, skip
            cfgs.append((f"L_os{os_scale}_g{gain}",
                         [f"agent.obs_scale={os_scale}", f"agent.learned_msg_gain={gain}"],
                         "learned", True))
    return cfgs


def optim_configs():
    """One change at a time from the registered baseline. raw AND nocomm per config."""
    variants = [("O_base",     []),
                ("O_ent003",   ["agent.entropy_coef=0.003"]),
                ("O_ent01",    ["agent.entropy_coef=0.01"]),
                ("O_lr1e4",    ["agent.lr=1e-4"]),
                ("O_k2",       ["agent.k_epochs=2"]),
                ("O_rs100",    ["agent.reward_scale=100"]),
                ("O_gae90",    ["agent.gae_lambda=0.90"])]
    cfgs = []
    for tag, extra in variants:
        cfgs.append((f"{tag}_raw", extra, "raw", True))
        cfgs.append((f"{tag}_noc", extra, "raw", False))
    return cfgs


STAGES = {"learned": learned_configs, "optim": optim_configs}
HOURS_PER_JOB = 0.45          # measured locally: 8000-ep SIGNAL run with early stop


# ---------------------------------------------------------------------------------------------
def ck_for(tag, seed):
    pat = os.path.join(W, f"run_signal_*_{tag}_s{seed}", "signal_checkpoint_best.pt")
    c = [p for p in glob.glob(pat) if os.path.basename(os.path.dirname(p)).endswith(
        f"_{tag}_s{seed}")]
    return max(c, key=os.path.getmtime) if c else None


def train_one(job):
    tag, extra, content, use_comm, seed = job
    if ck_for(tag, seed):
        return tag, "skip"
    env = dict(os.environ, WANDB_MODE="disabled", SIGNAL_CSVLOG="1", PYTHONUNBUFFERED="1")
    cmd = [PY, "agents/train_signal.py", *BASE, f"seed={seed}",
           f"agent.msg_content={content}", f"agent.use_comm={str(use_comm).lower()}",
           *extra, f"agent.algorithm={tag}_s{seed}"]
    log = os.path.join(W, f".sweeplog_{tag}_s{seed}.txt")
    os.makedirs(W, exist_ok=True)
    with open(log, "w", buffering=1) as f:
        f.write("$ " + " ".join(cmd) + "\n")
        r = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    return tag, ("ok" if r.returncode == 0 else f"FAIL rc={r.returncode} (see {log})")


def evaluate(tag, seed, episodes):
    dest = os.path.join(OUT, tag)
    if os.path.exists(os.path.join(dest, f"seed{seed}.json")):
        return
    ck = ck_for(tag, seed)
    if not ck:
        return
    subprocess.run([PY, "agents/eval_signal.py", "--ckpt", ck, "--dump-comm", dest,
                    "--dump-ar1", "0.9", "--dump-episodes", str(episodes)],
                   cwd=ROOT, env=dict(os.environ, WANDB_MODE="disabled"),
                   stdout=subprocess.DEVNULL)


def cost_of(tag, seed):
    p = os.path.join(OUT, tag, f"seed{seed}.json")
    if not os.path.exists(p):
        return None
    v = json.load(open(p))
    return sum(float(x) for x in v.values()) / len(v)


def print_table(seeds):
    ref = None
    rp = os.path.join(ROOT, "results", "baselines_ar_v3.json")
    if os.path.exists(rp):
        ref = float(json.load(open(rp))["rungs"]["AR_BestBS"]["0.9"])
    print("\n" + "=" * 78)
    print("SWEEP RESULTS @ rho=0.9   (DEV SEEDS -- TUNING ONLY, NOT CONFIRMATORY)")
    if ref:
        print(f"reference AR_BestBS = {ref:.1f}   |   registered instrument: raw 3837.5, "
              f"nocomm 4462.4, V=+624.9")
    print("=" * 78)
    for seed in seeds:
        # ---- learned stage: V measured against the registered nocomm ---------------------
        rows = []
        for tag, _e, _c, _u in learned_configs():
            c = cost_of(tag, seed)
            if c is not None:
                rows.append((tag, c))
        if rows:
            noc = cost_of("O_base_noc", seed)
            base_noc = noc if noc is not None else 4462.4
            print(f"\n-- learned message head (seed {seed}; V vs nocomm {base_noc:.1f}) --")
            print(f"   {'config':16s} {'cost':>9s} {'V':>9s} {'gap':>8s}")
            for tag, c in sorted(rows, key=lambda r: r[1]):
                g = f"{100 * (c / ref - 1):+.1f}%" if ref else "  -"
                print(f"   {tag:16s} {c:9.1f} {base_noc - c:+9.1f} {g:>8s}")
        # ---- optim stage: V measured WITHIN each config ---------------------------------
        pairs = []
        for tag, _e in [(t.rsplit("_", 1)[0], None) for t, _x, _c, _u in optim_configs()]:
            if tag not in [p[0] for p in pairs]:
                pairs.append((tag, None))
        rows = []
        for tag, _ in pairs:
            cr, cn = cost_of(f"{tag}_raw", seed), cost_of(f"{tag}_noc", seed)
            if cr is not None and cn is not None:
                rows.append((tag, cr, cn, cn - cr))
        if rows:
            print(f"\n-- SIGNAL optimizer, one change at a time (seed {seed}) --")
            print(f"   {'config':12s} {'raw':>9s} {'nocomm':>9s} {'V':>9s} {'gap(raw)':>9s}")
            for tag, cr, cn, v in sorted(rows, key=lambda r: -r[3]):
                g = f"{100 * (cr / ref - 1):+.1f}%" if ref else "  -"
                print(f"   {tag:12s} {cr:9.1f} {cn:9.1f} {v:+9.1f} {g:>9s}")
    print("\n" + "=" * 78)
    print("Adopting ANY setting from this table requires: manifest amendment + "
          "`python scripts/prereg_v3.py` re-hash BEFORE the pod run.")
    print("=" * 78)


# ---------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["learned", "optim", "qmix", "table"])
    ap.add_argument("--seeds", default="60", help="comma list of DEV seeds (default 60)")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--episodes", type=int, default=200, help="eval episodes at rho=0.9")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    os.makedirs(OUT, exist_ok=True)

    if a.stage == "table":
        print_table(seeds)
        return

    if a.stage == "qmix":
        cmd = [PY, "run_repair_study.py", "qmix-dev", "--arms", "qr_base", "qr_doubleq",
               "qr_replay", "qr_eps", "--seeds-limit", str(len(seeds)), "--jobs", str(a.jobs)]
        print("  QMIX uses the REGISTERED variants via the orchestrator (no new knobs):")
        print("  $ " + " ".join(cmd))
        if a.dry_run:
            return
        subprocess.run(cmd, cwd=ROOT)
        subprocess.run([PY, "run_repair_study.py", "qmix-gate", "--seeds-limit",
                        str(len(seeds))], cwd=ROOT)
        return

    cfgs = STAGES[a.stage]()
    jobs = [(t, e, c, u, s) for (t, e, c, u) in cfgs for s in seeds]
    todo = [j for j in jobs if not ck_for(j[0], j[4])]
    print(f"== sweep '{a.stage}': {len(jobs)} jobs ({len(jobs) - len(todo)} already done) ==")
    print(f"   estimated wall: {len(todo) * HOURS_PER_JOB / max(a.jobs, 1):.1f} h "
          f"at --jobs {a.jobs}")
    for t, e, c, u, s in jobs:
        print(f"   {t}_s{s:<4d} content={c:<8s} comm={str(u):<5s} {' '.join(e)}")
    if a.dry_run:
        return

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for tag, status in ex.map(train_one, todo):
            done += 1
            el = (time.time() - t0) / 60
            print(f"   [{done}/{len(todo)}] {tag}: {status}   ({el:.1f} min elapsed)")
    for t, _e, _c, _u, s in jobs:
        evaluate(t, s, a.episodes)
    print_table(seeds)


if __name__ == "__main__":
    main()
