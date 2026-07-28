import sys, subprocess, os
# per-job env (mirrors sweep run_one): wandb off everywhere; csvlog off for qmix only;
# sentinel is EP-STAMPED so a smoke run (--ep 40) can never satisfy a full run's resume check.
done = sys.argv[1]
cmd = sys.argv[2:]
os.environ["WANDB_MODE"] = "disabled"
os.environ["PYTHONUNBUFFERED"] = "1"
# v3 curve capture. SIGNAL: scalar CSV logger ON (pure observation, byte-neutral to training;
# without it metrics_heldout.csv / metrics_update.csv never exist and no training curve is
# recoverable post-run). QMIX: train_qmix REFUSES the csv logger by design (it reads
# MAPPO-specific internals) -> keep it unset and rely on the per-job training log below,
# which preserves the printed gate series (one line per heldout_every episodes) for every run.
if any("train_qmix.py" in c for c in cmd):
    os.environ.pop("SIGNAL_CSVLOG", None)
else:
    os.environ["SIGNAL_CSVLOG"] = "1"
ep = next((c.split("=", 1)[1] for c in cmd if c.startswith("total_episodes=")), "?")
tag = os.path.basename(done).replace(".done_", "")
tlog = os.path.join(os.path.dirname(done), f".trainlog_{tag}.txt")
os.makedirs(os.path.dirname(done), exist_ok=True)
with open(tlog, "a", buffering=1) as lf:
    lf.write("\n$ " + " ".join(cmd) + "\n")
    r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
if r.returncode != 0:
    try:
        print(open(tlog).read()[-2000:])       # surface the failure tail to the pool log
    except OSError:
        pass
if r.returncode == 0:
    open(done, "w").write(str(ep))
sys.exit(r.returncode)
