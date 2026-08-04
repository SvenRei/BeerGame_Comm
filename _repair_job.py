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
    # COMPLETION INVARIANT (added after a local incident: 4 jobs silently truncated yet
    # reported ok). rc==0 is NOT sufficient. The job's own trainlog must prove one of:
    #   (a) legitimate early stop  -- trainer prints "EARLY STOP at ep";
    #   (b) the final budget milestone was crossed -- "budget milestone <requested>" (SIGNAL
    #       prints milestones unconditionally; gate prints are improvement-only);
    #   (c) QMIX: last unconditional gate print "ep N:" reached requested - heldout_every;
    #   (d) smoke regime: requested below the first milestone (unverifiable, accepted).
    import re as _re
    txt = ""
    try:
        txt = open(tlog, errors="ignore").read()
    except OSError:
        pass
    req = int(ep) if str(ep).isdigit() else 0
    is_qmix = any("train_qmix.py" in c for c in cmd)
    okk = "EARLY STOP at ep" in txt
    if not okk and is_qmix:
        gates = [int(g) for g in _re.findall(r"ep (\d+): held-out", txt)]
        he = next((int(c.split("=", 1)[1]) for c in cmd
                   if c.startswith("agent.heldout_every=")), 200)
        okk = bool(gates) and max(gates) >= req - he
    if not okk and not is_qmix:
        okk = (f"budget milestone {req}:" in txt) or req < 1000
    if not okk:
        print(f"[wrapper] COMPLETION INVARIANT VIOLATED for {tag}: rc=0 but the trainlog "
              f"shows neither EARLY STOP nor the ep-{req} milestone. Truncated/killed run "
              f"-- NOT stamping the sentinel. Tail:\n" + txt[-1500:])
        sys.exit(3)
    open(done, "w").write(str(ep))
sys.exit(r.returncode)
