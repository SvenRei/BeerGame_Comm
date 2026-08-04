import glob, os, re, json
W, OUT = "weights_signal", "repair_out"
LADDER = ["r4_nocomm","r4_raw","r4_arpred","r4_learned","r4_dhatc","r4_ip"]
LEARN  = ["L_os1_g10","L_os10_g10","L_os10_g30","L_os25_g10","L_os25_g30","L_os100_g10","L_os100_g30"]
OPTIM  = [f"O_{v}_{s}" for v in ("base","ent003","ent01","lr1e4","k2","rs100","gae90") for s in ("raw","noc")]
QMIX   = ["qr_base","qr_doubleq","qr_replay","qr_eps"]
def rundir(tag, seed):
    c = [d for d in glob.glob(os.path.join(W, f"run_signal_*_{tag}_s{seed}"))
         if os.path.basename(d).endswith(f"_{tag}_s{seed}")]
    return max(c, key=os.path.getmtime) if c else None
def row(tag, seed, dumps):
    d = rundir(tag, seed)
    if not d: return f"  {tag:16s} s{seed}  --- NOT TRAINED ---"
    best = os.path.exists(os.path.join(d,"signal_checkpoint_best.pt"))
    mil  = sum(os.path.exists(os.path.join(d,f"signal_checkpoint_budget{m}.pt")) for m in (1000,2000,4000,8000))
    csv  = os.path.exists(os.path.join(d,"metrics_heldout.csv"))
    tl   = glob.glob(os.path.join(W, f".trainlog_{tag}_s{seed}.txt")) + glob.glob(os.path.join(W, f".sweeplog_{tag}_s{seed}.txt"))
    stop = "?"
    if tl:
        t = open(tl[0], errors="ignore").read()
        g = re.findall(r"ep (\d+): held-out mean cost", t)
        stop = ("EARLY@"+re.findall(r"EARLY STOP at ep (\d+)", t)[-1]) if "EARLY STOP at ep" in t else (f"gates={len(g)}")
    ev = [k for k in dumps if os.path.exists(k(tag, seed))]
    return (f"  {tag:16s} s{seed}  ckpt={'Y' if best else 'N'} milestones={mil}/4 "
            f"csv={'Y' if csv else 'N'} {stop:12s} evals={len(ev)}")
d200 = lambda t,s: f"{OUT}/devcheck200/{t}/seed{s}.json"
d200s= lambda t,s: f"{OUT}/devcheck200/{t}_s{s}/seed{s}.json"
swp  = lambda t,s: f"{OUT}/sweep/{t}/seed{s}.json"
print("="*84); print("LOCAL RUN INVENTORY"); print("="*84)
print("\n-- content ladder --")
for s in (60,61):
    for t in LADDER: print(row(t,s,[d200,d200s]))
print("\n-- learned-head sweep (seed 60) --")
for t in LEARN: print(row(t,60,[swp]))
print("\n-- optimizer sweep --")
for s in (60,61):
    for t in OPTIM:
        if rundir(t,s): print(row(t,s,[swp]))
print("\n-- QMIX variants (seed 60) --")
for t in QMIX: print(row(t,60,[d200,swp]))
n_mil = len(glob.glob(f"{OUT}/budget/*/seed*.json")); n_iv = len(glob.glob(f"{OUT}/iv/*/seed*_iv.json"))
print(f"\n-- derived artifacts --\n  milestone evals: {n_mil}   intervention dumps: {n_iv}   "
      f"figures: {len(glob.glob(f'{OUT}/figures/*.pdf'))}")
print("="*84)
