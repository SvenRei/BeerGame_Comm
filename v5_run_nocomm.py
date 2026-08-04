#!/usr/bin/env python3
"""v5_run_nocomm.py -- learner-calibration runs: NOCOMM only, seeds 60+61, v5 categorical head + parity ON (COLD START).
Writes per-job trainlogs (so plot_train_curves.py works) and applies the completion invariant.
Run: python v5_run_nocomm.py            (full budget, ~50 min at 2 parallel)
     python v5_run_nocomm.py --ep 60 --he 20   (smoke)"""
import argparse, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
ROOT = os.path.dirname(os.path.abspath(__file__)); W = os.path.join(ROOT, "weights_signal")
ap = argparse.ArgumentParser(); ap.add_argument("--ep", type=int, default=8000)
ap.add_argument("--he", type=int, default=200); ap.add_argument("--seeds", default="61,60")
ap.add_argument("--patience", type=int, default=2000)
ap.add_argument("--min-ep", type=int, default=0, dest="min_ep")
ap.add_argument("--suffix", default="", help="tag becomes v5<suffix>_nocomm_s<seed>")
ap.add_argument("--entropy-start", type=float, default=None, dest="es")
ap.add_argument("--anneal-frac", type=float, default=None, dest="fr")
ap.add_argument("--entropy-end", type=float, default=None, dest="ee", help="anneal floor (default 0.005); c5 entropy flatlined at 3.0 exactly at the floor")
ap.add_argument("--batch-eps", type=int, default=None, dest="be")
ap.add_argument("--grad-clip", type=float, default=None, dest="gc",
                help="actor/critic max_grad_norm (registered 0.5; QMIX uses 10)")
ap.add_argument("--lr", type=float, default=None, help="actor lr (registered 3e-4); under Adam the logit step is ~lr")
a = ap.parse_args(); os.makedirs(W, exist_ok=True)
os.environ.update(WANDB_MODE="disabled", SIGNAL_CSVLOG="1", PYTHONUNBUFFERED="1")
BASE = ["agent=signal", f"total_episodes={a.ep}", f"agent.heldout_every={a.he}",
        "agent.heldout_episodes=8", f"agent.patience={a.patience}",
        "agent.budget_milestones=[1000,2000,4000,8000]",
        "env.penalty_at_retailer_only=false", "agent.train_env=ar1", "agent.ar1_rho=0.9",
        "agent.heldout_mode=ar1", "agent.comm_topology=retailer_broadcast",
        "agent.msg_content=raw", "agent.use_comm=false", "agent.use_dhat_head=true",
        "agent.obs_norm=true", "agent.split_optimizer=true",
        "agent.head_type=categorical"]
if a.min_ep > 0:
    BASE.append(f"agent.min_train_episodes={a.min_ep}")
if a.es is not None:
    BASE.append(f"agent.entropy_start={a.es}")
if a.ee is not None:
    BASE.append(f"agent.entropy_end={a.ee}")
if a.fr is not None:
    BASE.append(f"agent.anneal ".replace(" ", "") + f"_frac={a.fr}" if False else f"agent.entropy_anneal_frac={a.fr}")
if a.be is not None:
    BASE.append(f"agent.batch_episodes={a.be}")
if a.gc is not None:
    BASE.append(f"agent.max_grad_norm={a.gc}")
if a.lr is not None:
    BASE.append(f"agent.lr={a.lr}")
def job(seed):
    tag = f"v5{a.suffix}_nocomm_s{seed}"
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
print(f"== v5 NOCOMM calibration (categorical, cold start): seeds {seeds}, ep={a.ep}, "
      f"patience={a.patience}, floor={a.min_ep}, tags v5{a.suffix}_nocomm_* (2 parallel) ==")
with ThreadPoolExecutor(max_workers=2) as ex:
    for tag, st in ex.map(job, seeds):
        print(f"  {tag}: {st}")
print("\nnext: python v5_report.py")
