import csv, glob, os, sys
pats = sys.argv[1:] or ["repair_out/curves/signal/*/metrics_update.csv"]
files = sorted(f for p in pats for f in glob.glob(p))
for f in files:
    name = os.path.basename(os.path.dirname(f))
    r = list(csv.DictReader(open(f)))
    if len(r) < 4:
        continue
    num = lambda x, k: (float(x[k]) if x.get(k) not in (None, "", "nan") else float("nan"))
    print(f"\n== {name}  ({len(r)} updates) ==")
    print(f"   {'ep':>6} {'action_std':>11} {'approx_kl':>10} {'entropy':>9} {'grad_norm':>11} {'value_loss':>12}")
    idx = [int(i * (len(r) - 1) / 9) for i in range(10)]
    for i in idx:
        x = r[i]
        print(f"   {int(num(x,'episode')):6d} {num(x,'action_std'):11.4f} {num(x,'approx_kl'):10.4f} "
              f"{num(x,'entropy'):9.3f} {num(x,'grad_norm'):11.1f} {num(x,'value_loss'):12.1f}")
    a0, a1 = num(r[0], "action_std"), num(r[-1], "action_std")
    kl = [num(x, "approx_kl") for x in r]
    kmax = max(k for k in kl if k == k)
    flags = []
    if a1 < 0.5 * a0: flags.append(f"ACTION_STD COLLAPSE {a0:.3f}->{a1:.3f}")
    if kmax > 0.1:    flags.append(f"KL BLOWUP max={kmax:.2f} (PPO target ~0.02)")
    print("   " + ("  |  ".join(flags) if flags else "no std collapse, no KL blowup"))
