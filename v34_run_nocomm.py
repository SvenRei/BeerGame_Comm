#!/usr/bin/env python3
"""v34_run_nocomm.py -- learner-calibration runs: NOCOMM only, seeds 60+61, v3.4 parity ON.
Writes per-job trainlogs (so plot_train_curves.py works) and applies the completion invariant.
Run: python v34_run_nocomm.py            (full budget, ~50 min at 2 parallel)
     python v34_run_nocomm.py --ep 60 --he 20   (smoke)"""
import argparse, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
ROOT = os.path.dirname(os.path.abspath(__file__)); W = os.path.join(ROOT, "weights_signal")
ap = argparse.ArgumentParser(); ap.add_argument("--ep", type=int, default=8000)
ap.add_argument("--he", type=int, default=200); ap.add_argument("--seeds", default="61,60")
a = ap.parse_args(); os.makedirs(W, exist_ok=True)
os.environ.update(WANDB_MODE="disabled", SIGNAL_CSVLOG="1", PYTHONUNBUFFERED="1")
BASE = ["agent=signal", f"total_episodes={a.ep}", f"agent.heldout_every={a.he}",
        "agent.heldout_episodes=8", "agent.patience=2000",
        "agent.budget_milestones=[1000,2000,4000,8000]",
        "env.penalty_at_retailer_only=false", "agent.train_env=ar1", "agent.ar1_rho=0.9",
        "agent.heldout_mode=ar1", "agent.comm_topology=retailer_broadcast",
        "agent.msg_content=raw", "agent.use_comm=false", "agent.use_dhat_head=true",
        "agent.obs_norm=true", "agent.split_optimizer=true"]
def job(seed):
    tag = f"v34_nocomm_s{seed}"
    cmd = [sys.executable, "agents/train_signal.py", *BASE, f"seed={seed}",
           f"agent.algorithm={tag}"]
    log = os.path.join(W, f".trainlog_{tag}.txt")
    with open(log, "w", buffering=1) as f:
        f.write("$ " + " ".join(cmd) + "\n")
        r = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    txt = open(log, errors="ignore").read()
    done = ("EARLY STOP" in txt) or (f"budget milestone {a.ep}:" in txt) or a.ep < 1000
    return tag, ("ok" if (r.returncode == 0 and done) else
                 f"FAIL rc={r.returncode} complete={done} (see {log})")
seeds = [int(s) for s in a.seeds.split(",")]
print(f"== v3.4 NOCOMM calibration: seeds {seeds}, ep={a.ep} (2 parallel) ==")
with ThreadPoolExecutor(max_workers=2) as ex:
    for tag, st in ex.map(job, seeds):
        print(f"  {tag}: {st}")
print("\nnext: python v34_report.py")
