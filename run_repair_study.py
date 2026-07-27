#!/usr/bin/env python3
"""
run_repair_study.py -- staged local orchestrator for the post-unblinding repair study
(reports/REPAIR_SEED_MANIFEST.json is the frozen source of truth for arms, seeds, gates).

    python run_repair_study.py check          # suites + artifact + grid refs + torch (~3 min)
    python run_repair_study.py plan           # print every job + wall estimate, launch nothing
    python run_repair_study.py qmix-dev       # 4 variants x 5 dev seeds (one change each)
    python run_repair_study.py qmix-gate      # predeclared competence gates -> winner or NONE
    python run_repair_study.py signal-dev     # 4 repaired arms x 5 dev seeds
    python run_repair_study.py signal-gate
    python run_repair_study.py qmix-confirm signal-confirm     # confirmatory seeds (gated)
    python run_repair_study.py dump analyze   # CRN dumps + REPAIR_STUDY.md

Debug/smoke flags: --ep 40 --he 10 --arms r4_dhatc --seeds-limit 1 shrink any stage to a
minutes-long end-to-end check through the REAL training/eval CLIs.

Integrity: new arm identifiers only (r4_*, qr_*), sentinel/resume semantics identical to the
sweep's run_one, WANDB_MODE=disabled, original campaign untouched. Gates are read from the
manifest and evaluated mechanically; confirmatory stages refuse to run unless their gate
passed (override consciously with --force).
"""
import os
import sys
import json
import glob
import argparse
import subprocess
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from run_local import pool_run, run_logged, latest_ck                     # noqa: E402
import run_local as RL                                                    # noqa: E402

PY = sys.executable
MAN_PATH = os.path.join(ROOT, "reports", "REPAIR_SEED_MANIFEST.json")
OUT = os.path.join(ROOT, "repair_out")
GATES = os.path.join(OUT, "gates.json")
GRID_JSON = os.path.join(ROOT, "results", "qmix_grid_benchmark.json")
FC_PATH = os.path.join(ROOT, "results", "forecaster_ar1r9.pt")
SIG_H, QMIX_H = 2.0, 1.5                     # rough hours/job for the plan estimate (local core)


def say(msg=""):
    print(msg, flush=True)


def man():
    return json.load(open(MAN_PATH))


def expand(args_str, macros):
    for k, v in macros.items():
        args_str = args_str.replace(k, v)
    return args_str.split()


def train_cmd(arm, seed, args, is_qmix, ep, he, budget):
    entry = "agents/train_qmix.py" if is_qmix else "agents/train_signal.py"
    full = f"{arm}_s{seed}"
    return ([PY, entry, "agent=signal", f"seed={seed}", f"total_episodes={ep}",
             f"agent.heldout_every={he}",
             f"agent.heldout_episodes={budget['heldout_episodes']}",
             f"agent.patience={budget['patience']}",
             f"agent.budget_milestones={budget['budget_milestones']}",
             *args, f"agent.algorithm={full}"], full)


def build_jobs(m, table, seeds, ep, he, only_arms=None, seeds_limit=None):
    jobs, budget = [], m["training_budget"]
    seeds = list(seeds)[: (seeds_limit or len(seeds))]
    for arm, argstr in m[table].items():
        if only_arms and arm not in only_arms:
            continue
        is_qmix = table == "qmix_arms"
        args = expand(argstr, m["macros"])
        for s in seeds:
            cmd, full = train_cmd(arm, s, args, is_qmix, ep, he, budget)
            done = os.path.join(ROOT, "weights_signal", f".done_{full}")
            jobs.append((cmd, done, full, is_qmix))
    return jobs


WRAPPER = r"""import sys, subprocess, os
# per-job env (mirrors sweep run_one): wandb off everywhere; csvlog off for qmix only;
# sentinel is EP-STAMPED so a smoke run (--ep 40) can never satisfy a full run's resume check.
done = sys.argv[1]
cmd = sys.argv[2:]
os.environ["WANDB_MODE"] = "disabled"
os.environ["PYTHONUNBUFFERED"] = "1"
if any("train_qmix.py" in c for c in cmd):
    os.environ["SIGNAL_CSVLOG"] = "0"
ep = next((c.split("=", 1)[1] for c in cmd if c.startswith("total_episodes=")), "?")
r = subprocess.run(cmd)
if r.returncode == 0:
    open(done, "w").write(str(ep))
sys.exit(r.returncode)
"""


def _sentinel_matches(done, ep):
    if not os.path.exists(done):
        return False
    txt = open(done).read().strip()
    return txt in (str(ep), "ok")               # 'ok' = legacy sweep sentinel content


def run_train_stage(name, jobs, workers, logdir, ep):
    os.makedirs(logdir, exist_ok=True)
    wpath = os.path.join(ROOT, "_repair_job.py")
    with open(wpath, "w", encoding="utf-8") as f:
        f.write(WRAPPER)
    pool_jobs, skipped = [], 0
    for cmd, done, full, is_qmix in jobs:
        if _sentinel_matches(done, ep):
            skipped += 1
            continue
        if os.path.exists(done):
            say(f"  [restamp] {full}: sentinel from a different --ep -> retraining")
        pool_jobs.append(([PY, wpath, done] + cmd, None))
    say(f"== {name}: {len(jobs)} training jobs ({skipped} done at --ep {ep}) ==")
    ok = fail = 0
    if pool_jobs:
        ok, fail, _sk = pool_run(pool_jobs, workers, name,
                                 os.path.join(logdir, f"{name}.log"))
    say(f"  {name}: {ok} ok, {fail} failed, {skipped} previously done")
    return fail == 0


def dump_cells(cells, seeds, outroot, episodes, logp, workers):
    """cells: list of (arm, cell_name, is_qmix). Resumable per seed file."""
    jobs = []
    for arm, cell, is_qmix in cells:
        for s in seeds:
            ck = latest_ck(arm, s)
            if not ck:
                say(f"  MISSING ckpt {arm} s{s} -- cell will be short")
                continue
            done = os.path.join(outroot, cell, f"seed{s}.json")
            if is_qmix:
                cmd = [PY, "scripts/qmix_dump.py", "--ckpt", ck,
                       "--out", os.path.join(outroot, cell),
                       "--ar1-rho", "0.9", "--episodes", str(episodes)]
            else:
                cmd = [PY, "agents/eval_signal.py", "--ckpt", ck,
                       "--dump-comm", os.path.join(outroot, cell),
                       "--dump-ar1", "0.9", "--dump-episodes", str(episodes)]
            jobs.append((cmd, done))
    return pool_run(jobs, workers, "dumps", logp)


def cell_means(root, cell, seeds):
    vals = []
    for s in seeds:
        p = os.path.join(root, cell, f"seed{s}.json")
        if os.path.exists(p):
            d = json.load(open(p))
            vals.append(float(sum(map(float, d.values())) / len(d)))
    return vals


# ==============================================================================
def stage_check(a, m):
    ok = True
    for t in ("tests.test_forecaster", "tests.test_phase2_integration", "tests.test_phase34"):
        rc = subprocess.run([PY, "-m", t], cwd=ROOT, capture_output=True).returncode
        say(f"  suite {t}: {'PASS' if rc == 0 else 'FAIL'}")
        ok &= rc == 0
    if not os.path.exists(FC_PATH):
        say("  certified forecaster absent -> running scripts/forecast_pretrain.py ...")
        ok &= subprocess.run([PY, "scripts/forecast_pretrain.py", "--out", FC_PATH],
                             cwd=ROOT).returncode == 0
    else:
        d = json.load(open(FC_PATH.replace(".pt", "_metrics.json"))) \
            if os.path.exists(FC_PATH.replace(".pt", "_metrics.json")) else None
        say(f"  certified forecaster: present"
            + (f" (MSE ratio {d['metrics']['mse_ratio_vs_bench']:.3f}, "
               f"pass={d['certification']['pass']})" if d else ""))
    if not os.path.exists(GRID_JSON):
        say("  grid benchmark absent -> running scripts/qmix_grid_benchmark.py ...")
        ok &= subprocess.run([PY, "scripts/qmix_grid_benchmark.py"], cwd=ROOT).returncode == 0
    else:
        g = json.load(open(GRID_JSON))
        say(f"  grid benchmark: present (GridCondBS(41,160) = "
            f"{g['grids']['n41_smax160']['GridCondBS']:.1f})")
    say(f"  manifest: frozen {m['frozen_utc']}  dev {m['seeds']['dev']}  "
        f"confirmatory n={len(m['seeds']['confirmatory'])}"
        + ("  [amended A1]" if m.get("amendment_A1") else ""))
    # ---- provenance record (v3): git hash + prereg hash + manifest hash --------------------
    import hashlib
    meta = {"when": datetime.now().isoformat()}
    try:
        import subprocess as _sp
        meta["git_hash"] = _sp.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                   capture_output=True, text=True).stdout.strip() or None
    except Exception:
        meta["git_hash"] = None
    pr = os.path.join(ROOT, "reports", "PREREG_v3.json")
    if os.path.exists(pr):
        meta["prereg_v3_sha256"] = json.load(open(pr)).get("sha256")
        say(f"  prereg_v3: {meta['prereg_v3_sha256'][:16]}...")
    else:
        meta["prereg_v3_sha256"] = None
        say("  prereg_v3: MISSING -- run `python scripts/prereg_v3.py` before any dev job")
        ok = False
    meta["manifest_sha256"] = hashlib.sha256(open(MAN_PATH, "rb").read()).hexdigest()
    os.makedirs(OUT, exist_ok=True)
    json.dump(meta, open(os.path.join(OUT, "run_meta.json"), "w"), indent=2)
    return ok


def stage_plan(a, m):
    dev, conf = m["seeds"]["dev"], m["seeds"]["confirmatory"]
    q_dev = build_jobs(m, "qmix_arms", dev, a.ep, a.he, a.arms, a.seeds_limit)
    s_dev = build_jobs(m, "signal_arms", dev, a.ep, a.he, a.arms, a.seeds_limit)
    n_qc = 2 * len(conf)                 # winner pair
    n_sc = len(m["signal_arms"]) * len(conf)
    say(f"  qmix-dev    : {len(q_dev)} jobs (~{len(q_dev)*QMIX_H/a.jobs:.1f} h at -j{a.jobs})")
    say(f"  signal-dev  : {len(s_dev)} jobs (~{len(s_dev)*SIG_H/a.jobs:.1f} h)")
    say(f"  qmix-confirm: {n_qc} jobs (winner pair; ~{n_qc*QMIX_H/a.jobs:.1f} h) [gated]")
    say(f"  signal-conf : {n_sc} jobs (~{n_sc*SIG_H/a.jobs:.1f} h) [gated]")
    say(f"  dump+analyze: minutes")
    say(f"  TOTAL if all gates pass: ~"
        f"{(len(q_dev)*QMIX_H + len(s_dev)*SIG_H + n_qc*QMIX_H + n_sc*SIG_H)/a.jobs:.1f} h "
        f"wall at -j{a.jobs} (per-job hours are estimates; pools print live ETA)")
    for cmd, _d, full, _q in (q_dev + s_dev)[:2]:
        say(f"  example: {' '.join(cmd)}")
    return True


def stage_refs(a, m):
    """Regenerate the AR privileged references on the v3 eval streams (frontier diagnostic)."""
    out = os.path.join(ROOT, "results", "baselines_ar_v3.json")
    if os.path.exists(out) and not a.force:
        say(f"  refs present: {out} (--force regenerates)")
        return True
    from scripts.baselines import ar_benchmark
    ar_benchmark(rhos=(0.9,), eval_episodes=200, out_path=out)
    return os.path.exists(out)


def stage_transfer(a, m):
    """P2' transfer 2x2 evals: inf-trained raw under c12, c12-trained raw under inf.
    Uses make_env_override + eval_signal --env-json (the documented pilot machinery)."""
    wpath = os.path.join(ROOT, "_transfer_job.py")
    with open(wpath, "w", encoding="utf-8") as f:
        f.write("import sys, subprocess, os\n"
                "done, ck, setval, envjson, dumpdir, eps, py = sys.argv[1:8]\n"
                "os.environ['WANDB_MODE'] = 'disabled'\n"
                "r = subprocess.run([py, 'scripts/make_env_override.py', '--ckpt', ck,\n"
                "                    '--set', setval, '--out', envjson])\n"
                "if r.returncode == 0:\n"
                "    r = subprocess.run([py, 'agents/eval_signal.py', '--ckpt', ck,\n"
                "                        '--env-json', envjson, '--dump-comm', dumpdir,\n"
                "                        '--dump-ar1', '0.9', '--dump-episodes', eps])\n"
                "sys.exit(r.returncode)\n")
    specs = [("r4_raw", "obs_order_clip=12", "raw_tinf_ec12"),
             ("r4_raw_c12", "obs_order_clip=del", "raw_tc12_einf")]
    conf = m["seeds"]["confirmatory"][: (a.seeds_limit or None)]
    envdir = os.path.join(OUT, "transfer", "env")
    os.makedirs(envdir, exist_ok=True)
    jobs = []
    for arm, setval, cell in specs:
        for sd in conf:
            ck = latest_ck(arm, sd)
            if not ck:
                say(f"  MISSING ckpt {arm} s{sd} -- transfer cell will be short")
                continue
            done = os.path.join(OUT, "transfer", cell, f"seed{sd}.json")
            ej = os.path.join(envdir, f"{cell}_s{sd}.json")
            jobs.append(([PY, wpath, done, ck, setval, ej,
                          os.path.join(OUT, "transfer", cell), str(a.dump_episodes), PY],
                         done))
    ok, fail, _ = pool_run(jobs, a.jobs, "transfer",
                           os.path.join(OUT, "logs", "transfer.log"))
    return fail == 0


def stage_qmix_dev(a, m):
    jobs = build_jobs(m, "qmix_arms", m["seeds"]["dev"], a.ep, a.he, a.arms, a.seeds_limit)
    return run_train_stage("qmix-dev", jobs, a.jobs, os.path.join(OUT, "logs"), a.ep)


def stage_qmix_gate(a, m):
    dev = m["seeds"]["dev"][: (a.seeds_limit or None)]
    cells = [(arm, f"qmix_dev/{arm}", True) for arm in m["qmix_arms"]]
    dump_cells(cells, dev, OUT, a.gate_episodes, os.path.join(OUT, "logs", "qmix_gate.log"),
               a.jobs)
    g = json.load(open(GRID_JSON))
    ref = float(g["grids"]["n41_smax160"]["GridCondBS"])
    abs_thr = 1.20 * ref
    means = {arm: cell_means(OUT, f"qmix_dev/{arm}", dev) for arm in m["qmix_arms"]}
    base = means.get("qr_base", [])
    base_m = sum(base) / len(base) if base else float("nan")
    say(f"  gate refs: GridCondBS(41,160)={ref:.1f}  absolute<= {abs_thr:.1f}  "
        f"relative<= 0.85 x qr_base({base_m:.1f}) = {0.85*base_m:.1f}")
    verdicts, winner, best = {}, None, float("inf")
    for arm, vs in means.items():
        mmean = sum(vs) / len(vs) if vs else float("nan")
        p_abs = mmean <= abs_thr
        p_rel = (arm != "qr_base") and (mmean <= 0.85 * base_m)
        verdicts[arm] = {"n": len(vs), "mean": mmean, "abs_pass": p_abs, "rel_pass": p_rel}
        say(f"    {arm:<12} dev-mean {mmean:9.1f} (n={len(vs)})  "
            f"abs={'PASS' if p_abs else 'fail'}  rel={'PASS' if p_rel else 'fail'}")
        if p_abs and p_rel and mmean < best:
            winner, best = arm, mmean
    gates = json.load(open(GATES)) if os.path.exists(GATES) else {}
    gates["qmix"] = {"ref_GridCondBS": ref, "abs_threshold": abs_thr, "base_mean": base_m,
                     "verdicts": verdicts, "winner": winner,
                     "when": datetime.now().isoformat()}
    os.makedirs(OUT, exist_ok=True)
    json.dump(gates, open(GATES, "w"), indent=2)
    if winner:
        say(f"  -> WINNER: {winner} (earns the confirmatory pair)")
    else:
        say("  -> NO variant passed both predeclared gates. Per the manifest, V1 concordance "
            "is reported UNADJUDICABLE (comparison algorithm below competence); qmix-confirm "
            "will refuse without --force.")
    return True


def stage_qmix_confirm(a, m):
    gates = json.load(open(GATES)) if os.path.exists(GATES) else {}
    winner = a.force_arm or gates.get("qmix", {}).get("winner")
    if not winner:
        say("  qmix-confirm BLOCKED: no gate winner (see qmix-gate); use --force-arm to override")
        return not a.strict_gates
    if a.force_arm:                                  # paper trail: the report must name the arm
        gates.setdefault("qmix", {})["winner_forced"] = a.force_arm
        os.makedirs(OUT, exist_ok=True)
        json.dump(gates, open(GATES, "w"), indent=2)
    argstr = m["qmix_arms"][winner]
    raw_args = argstr.replace("agent.use_comm=false",
                              "$RB agent.msg_content=raw")
    conf = m["seeds"]["confirmatory"][: (a.seeds_limit or None)]
    m2 = dict(m)
    m2["qmix_arms"] = {"qrw_nocomm": argstr, "qrw_raw": raw_args}
    jobs = build_jobs(m2, "qmix_arms", conf, a.ep, a.he, None, None)
    say(f"  confirmatory pair from winner {winner}")
    return run_train_stage("qmix-confirm", jobs, a.jobs, os.path.join(OUT, "logs"), a.ep)


def stage_signal_dev(a, m):
    jobs = build_jobs(m, "signal_arms", m["seeds"]["dev"], a.ep, a.he, a.arms, a.seeds_limit)
    return run_train_stage("signal-dev", jobs, a.jobs, os.path.join(OUT, "logs"), a.ep)


def stage_signal_gate(a, m):
    dev = m["seeds"]["dev"][: (a.seeds_limit or None)]
    cells = [(arm, f"signal_dev/{arm}", False) for arm in m["signal_arms"]]
    dump_cells(cells, dev, OUT, a.gate_episodes, os.path.join(OUT, "logs", "signal_gate.log"),
               a.jobs)
    means = {arm: cell_means(OUT, f"signal_dev/{arm}", dev) for arm in m["signal_arms"]}
    complete = all(len(v) == len(dev) for v in means.values())
    mm = {k: (sum(v) / len(v) if v else float("nan")) for k, v in means.items()}
    direction = mm.get("r4_raw", 1e18) < mm.get("r4_nocomm", -1e18)
    for k, v in mm.items():
        say(f"    {k:<10} dev-mean {v:9.1f} (n={len(means[k])})")
    ok = complete and direction
    gates = json.load(open(GATES)) if os.path.exists(GATES) else {}
    gates["signal"] = {"dev_means": mm, "complete": complete, "raw_lt_nocomm": direction,
                       "pass": ok, "when": datetime.now().isoformat()}
    os.makedirs(OUT, exist_ok=True)
    json.dump(gates, open(GATES, "w"), indent=2)
    say(f"  -> signal dev gate: {'PASS' if ok else 'FAIL'} "
        f"(complete={complete}, raw<nocomm={direction})")
    return True


def stage_signal_confirm(a, m):
    gates = json.load(open(GATES)) if os.path.exists(GATES) else {}
    if not gates.get("signal", {}).get("pass") and not a.force:
        say("  signal-confirm BLOCKED: dev gate not passed (--force to override)")
        return not a.strict_gates
    conf = m["seeds"]["confirmatory"][: (a.seeds_limit or None)]
    jobs = build_jobs(m, "signal_arms", conf, a.ep, a.he, a.arms, None)
    return run_train_stage("signal-confirm", jobs, a.jobs, os.path.join(OUT, "logs"), a.ep)


def stage_dump(a, m):
    conf = m["seeds"]["confirmatory"][: (a.seeds_limit or None)]
    cells = [(arm, f"v1/{arm}", False) for arm in m["signal_arms"]]
    for arm in ("qrw_nocomm", "qrw_raw"):
        if glob.glob(os.path.join(ROOT, "weights_signal", f"run_signal_*_{arm}_s*")):
            cells.append((arm, f"v1/{arm}", True))
    ok, fail, _ = dump_cells(cells, conf, OUT, a.dump_episodes,
                             os.path.join(OUT, "logs", "dump.log"), a.jobs)
    return fail == 0


def _cell(name, seeds, root="v1"):
    """Load one cell's per-seed mean costs. Returns (vector|None, missing_list, dir_exists)."""
    import numpy as np
    cdir = os.path.join(OUT, root, name)
    if not os.path.isdir(cdir):
        return None, list(seeds), False
    vals, missing = [], []
    for sd in seeds:
        p = os.path.join(cdir, f"seed{sd}.json")
        if not os.path.exists(p):
            missing.append(sd)
            continue
        d = json.load(open(p))
        vals.append(float(np.mean(list(map(float, d.values())))))
    return (np.array(vals) if not missing else None), missing, True


def stage_analyze(a, m):
    import numpy as np
    from scipy import stats as sps
    from scripts.c1_stats import bootstrap_ci, paired, tost, compare_many
    conf = m["seeds"]["confirmatory"][: (a.seeds_limit or None)]
    n = len(conf)
    infer = n >= 3                    # inference off for micro smokes

    def load(name, root="v1"):
        v, miss, exists = _cell(name, conf, root)
        if exists and miss:
            sys.exit(f"FAIL-CLOSED: cell {root}/{name} missing seeds {miss}")
        return v                       # None <=> arm not trained (PENDING)

    meta = (json.load(open(os.path.join(OUT, "run_meta.json")))
            if os.path.exists(os.path.join(OUT, "run_meta.json")) else {})
    L = [f"# REPAIR STUDY v3 -- the journal run (fresh seeds {conf[0]}..{conf[-1]}, n={n})",
         "",
         f"manifest frozen {m['frozen_utc']}"
         + (" (amended A1 pre-execution)" if m.get("amendment_A1") else "")
         + f"; git {str(meta.get('git_hash'))[:12]}"
         + f"; prereg_v3 {str(meta.get('prereg_v3_sha256'))[:16]}",
         "POST-UNBLINDING TARGETED FOLLOW-UP lineage disclosed; pilot campaigns v1.1/v1.2/"
         "v2.1 cited by hash in scripts/prereg_v3.py.", ""]

    # ---------------- 1. inf-world value table --------------------------------------------
    noc = load("r4_nocomm")
    if noc is None:
        sys.exit("FAIL-CLOSED: r4_nocomm (inf) cell absent -- nothing to analyze")
    inf_arms = [x for x in m["signal_arms"] if x != "r4_nocomm" and "_c" not in x]
    L += ["## 1. Value vs r4_nocomm -- inf world (seed-paired, BCa CI, Wilcoxon)", "",
          "| arm | mean cost | V | 95% CI | Wilcoxon p | P(V>0) |", "|---|---|---|---|---|---|"]
    vals = {"r4_nocomm": noc}
    for arm in inf_arms:
        c = load(arm)
        if c is None:
            L.append(f"| {arm} | PENDING (not trained) | | | | |")
            continue
        vals[arm] = c
        v = noc - c
        lo, hi = bootstrap_ci(v)
        st = paired(noc, c)
        L.append(f"| {arm} | {np.mean(c):.1f} | {np.mean(v):+.1f} | [{lo:+.1f},{hi:+.1f}] | "
                 f"{st['wilcoxon_p']:.3g} | {float(np.mean(v > 0)):.2f} |")
    band = 0.02 * float(np.mean(noc))
    L += ["", f"v_distribution band note: TOST band = 2% of mean C(nocomm) = {band:.1f}.", ""]

    # ---------------- 2. frontier / optimality-gap diagnostic ------------------------------
    refp = os.path.join(ROOT, "results", "baselines_ar_v3.json")
    L.append("## 2. Optimality-gap diagnostic (registered tau grid; interpretation only)")
    L.append("")
    if os.path.exists(refp) and "r4_raw" in vals:
        rungs = json.load(open(refp))["rungs"]
        ref = float(min(rungs["AR_CondBS"]["0.9"], rungs["AR_StaticBS"]["0.9"],
                        rungs.get("AR_BestBS", {}).get("0.9", 1e18)))
        g_raw = float(np.mean(vals["r4_raw"])) / ref - 1.0
        g_noc = float(np.mean(noc)) / ref - 1.0
        taus = " ".join(f"tau={t:.2f}:{'PASS' if max(g_raw, g_noc) <= t else 'fail'}"
                        for t in (0.10, 0.20, 0.30))
        L.append(f"AR_BestBS(0.9) = {ref:.1f} (v3 eval streams). gap(raw) = {g_raw:+.1%}, "
                 f"gap(nocomm) = {g_noc:+.1%}. Information-value reading: {taus}.")
    else:
        L.append("PENDING: run `refs` stage and/or train r4_raw.")
    L.append("")

    # ---------------- 3. P2' worlds, Gamma, dose ------------------------------------------
    L += ["## 3. P2' -- censoring worlds (registered pilot-learned direction: Gamma < 0)", ""]
    worlds = m["gates_predeclared"]["p2_prime"]["worlds"]
    V = {}
    for w, (nc_a, rw_a) in worlds.items():
        ncw, rww = load(nc_a), load(rw_a)
        if ncw is None or rww is None:
            L.append(f"- world {w}: PENDING ({nc_a if ncw is None else rw_a} not trained)")
            continue
        V[w] = ncw - rww
        lo, hi = bootstrap_ci(V[w])
        L.append(f"- V_{w}(raw) = {np.mean(V[w]):+.1f} [{lo:+.1f},{hi:+.1f}]  "
                 f"(C_nocomm {np.mean(ncw):.1f}, C_raw {np.mean(rww):.1f})")
    p_p2 = None
    if "inf" in V and "c12" in V:
        G = V["c12"] - V["inf"]
        lo, hi = bootstrap_ci(G)
        p_p2 = float(sps.ttest_1samp(G, 0, alternative="less").pvalue) if infer else None
        L.append(f"- **Gamma = V_c12 - V_inf = {np.mean(G):+.1f} [{lo:+.1f},{hi:+.1f}]"
                 + (f"; one-sided p(<0) = {p_p2:.3g}**" if infer else " (n<3: no inference)**"))
        if "c20" in V:
            for lab, d in (("dose_a: V_inf-V_c20", V["inf"] - V["c20"]),
                           ("dose_b: V_c20-V_c12", V["c20"] - V["c12"])):
                pd_ = (float(sps.ttest_1samp(d, 0, alternative="greater").pvalue)
                       if infer else float("nan"))
                L.append(f"- {lab} = {np.mean(d):+.1f} (one-sided p = {pd_:.3g})")
    L.append("")

    # ---------------- 4. transfer 2x2 ------------------------------------------------------
    L += ["## 4. Transfer 2x2 (training-env dominance; registered secondaries)", ""]
    t_pa = t_pb = None
    te = load("raw_tinf_ec12", root="transfer")
    tn = load("r4_raw_c12")
    if te is not None and tn is not None:
        d = te - tn
        t_pa = float(sps.ttest_1samp(d, 0, alternative="greater").pvalue) if infer else None
        L.append(f"- eval@c12: C(t_inf) {np.mean(te):.1f} vs C(t_c12) {np.mean(tn):.1f}  "
                 f"diff {np.mean(d):+.1f}" + (f"  p(>0)={t_pa:.3g}" if infer else ""))
    else:
        L.append("- eval@c12: PENDING (transfer stage and/or r4_raw_c12)")
    ti = load("raw_tc12_einf", root="transfer")
    if ti is not None and "r4_raw" in vals:
        d = ti - vals["r4_raw"]
        t_pb = float(sps.ttest_1samp(d, 0, alternative="greater").pvalue) if infer else None
        L.append(f"- eval@inf: C(t_c12) {np.mean(ti):.1f} vs C(t_inf) "
                 f"{np.mean(vals['r4_raw']):.1f}  diff {np.mean(d):+.1f}"
                 + (f"  p(>0)={t_pb:.3g}" if infer else ""))
    else:
        L.append("- eval@inf: PENDING")
    L.append("")

    # ---------------- 5. REGISTERED CONFIRMATORY FAMILIES (prereg_v3) ---------------------
    L += ["## 5. Registered decisions (prereg_v3; Holm alpha=.05)", ""]
    if infer:
        fam = {}
        if "r4_raw" in vals:
            fam["P1prime"] = float(sps.ttest_1samp(noc - vals["r4_raw"], 0,
                                                   alternative="greater").pvalue)
        if p_p2 is not None:
            fam["P2prime"] = p_p2
        if "r4_dhatc" in vals:
            fam["CNULLprime"] = float(tost(noc - vals["r4_dhatc"], -band, band,
                                           alpha=0.05)["p_tost"])
        if len(fam) == 3:
            cm = compare_many(fam, method="holm")
            L.append("**Primary family** (COMPLETE):")
            for k in fam:
                L.append(f"- {k}: raw p = {cm[k]['raw']:.3g}, "
                         f"Holm-adj = {cm[k]['adjusted']:.3g} -> "
                         f"{'REJECT' if cm[k]['reject'] else 'retain'}")
        else:
            L.append(f"**Primary family INCOMPLETE** ({sorted(fam)} available) -- "
                     "confirmatory decisions deferred; no member-wise reading permitted.")
        sec = {}
        if "r4_arpred" in vals and "r4_raw" in vals:
            sec["HREPprime"] = float(tost(vals["r4_arpred"] - vals["r4_raw"], -band, band,
                                          alpha=0.05)["p_tost"])
        if "r4_dhatc" in vals and "r4_arpred" in vals:
            sec["D1prime_tax_tost"] = float(tost(vals["r4_dhatc"] - vals["r4_arpred"],
                                                 -band, band, alpha=0.05)["p_tost"])
        if "inf" in V and "c20" in V:
            sec["DOSEprime_a"] = float(sps.ttest_1samp(V["inf"] - V["c20"], 0,
                                                       alternative="greater").pvalue)
        if "c20" in V and "c12" in V:
            sec["DOSEprime_b"] = float(sps.ttest_1samp(V["c20"] - V["c12"], 0,
                                                       alternative="greater").pvalue)
        if t_pa is not None:
            sec["TRANSFER_a"] = t_pa
        if t_pb is not None:
            sec["TRANSFER_b"] = t_pb
        for lad in ("r4_learned", "r4_ip"):
            if lad in vals:
                sec[f"LADDER_{lad[3:]}"] = float(tost(noc - vals[lad], -band, band,
                                                      alpha=0.05)["p_tost"])
        if sec:
            cm2 = compare_many(sec, method="holm")
            L.append("")
            L.append(f"**Secondary family** ({len(sec)}/8 members available"
                     + ("" if len(sec) == 8 else " -- Holm over available; note in paper")
                     + "):")
            for k in sec:
                L.append(f"- {k}: raw p = {cm2[k]['raw']:.3g}, "
                         f"Holm-adj = {cm2[k]['adjusted']:.3g} -> "
                         f"{'REJECT/EQUIV' if cm2[k]['reject'] else 'retain/inconclusive'}")
    else:
        L.append(f"n = {n} < 3: registered inference disabled (smoke mode).")
    L.append("")

    # ---------------- 6. B5 + QMIX (unchanged semantics) -----------------------------------
    ck = latest_ck("r4_dhatc", conf[0])
    if ck:
        import torch
        d = torch.load(ck, map_location="cpu", weights_only=False)
        fc = d.get("forecaster") or {}
        met, cert = fc.get("metrics", {}), fc.get("certification", {})
        L += ["## 6. B5 forecast competence (r4_dhatc)", "",
              "One frozen certified artifact serves every seed BY DESIGN (removes "
              "forecaster-quality variance from the economic comparison):",
              f"- MSE ratio {met.get('mse_ratio_vs_bench', float('nan')):.3f}  "
              f"bias {met.get('bias', float('nan')):+.3f}  pred SD "
              f"{met.get('pred_sd', float('nan')):.3f} "
              f"(bench {met.get('bench_pred_sd', float('nan')):.3f})  "
              f"CERTIFIED={cert.get('pass')}",
              "- fail-closed loading: no economic reading exists for any uncertified "
              "forecaster.", ""]
    if os.path.isdir(os.path.join(OUT, "v1", "qrw_nocomm")):
        qn, _m1, _ = _cell("qrw_nocomm", conf)
        qr, _m2, _ = _cell("qrw_raw", conf)
        if _m1 or _m2:
            L += ["## 7. QMIX", "", f"- winner pair PARTIAL (missing seeds "
                  f"{sorted(set(_m1) | set(_m2))}): no confirmatory reading.", ""]
            qn = qr = None
        if qn is not None and qr is not None:
            v = qn - qr
            lo, hi = bootstrap_ci(v)
            gates = json.load(open(GATES)) if os.path.exists(GATES) else {}
            gw = gates.get("qmix", {}).get("winner")
            gwf = gates.get("qmix", {}).get("winner_forced")
            wname = gw or (f"{gwf} (FORCED past a failed gate -- exploratory only)"
                           if gwf else "?")
            sign = ("CONCORDANT" if np.mean(v) > 0 else "DISCORDANT")
            L += ["## 7. QMIX winner pair (gated; exploratory)", "",
                  f"- winner: {wname}  nocomm {np.mean(qn):.1f}  raw {np.mean(qr):.1f}  "
                  f"V = {np.mean(v):+.1f} [{lo:+.1f},{hi:+.1f}]  -> sign vs P1prime: {sign}"
                  + ("" if gw else "  [NOT confirmatory: forced]"), ""]
    else:
        L += ["## 7. QMIX", "", "- no winner pair trained: V1 status = UNADJUDICABLE by "
              "predeclared rule (or program not yet run).", ""]

    txt = "\n".join(L) + "\n"
    print(txt)
    with open(os.path.join(ROOT, "reports", "REPAIR_STUDY.md"), "w") as f:
        f.write(txt)
    say("-> wrote reports/REPAIR_STUDY.md")
    return True


STAGES = {"check": stage_check, "plan": stage_plan, "refs": stage_refs,
          "transfer": stage_transfer,
          "qmix-dev": stage_qmix_dev, "qmix-gate": stage_qmix_gate,
          "qmix-confirm": stage_qmix_confirm,
          "signal-dev": stage_signal_dev, "signal-gate": stage_signal_gate,
          "signal-confirm": stage_signal_confirm,
          "dump": stage_dump, "analyze": stage_analyze}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("stages", nargs="+", choices=list(STAGES))
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--ep", type=int, default=None, help="override total_episodes (smoke)")
    ap.add_argument("--he", type=int, default=None, help="override heldout_every (smoke)")
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--seeds-limit", type=int, default=None)
    ap.add_argument("--gate-episodes", type=int, default=100)
    ap.add_argument("--dump-episodes", type=int, default=200)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--force-arm", default=None)
    ap.add_argument("--strict-gates", action="store_true",
                    help="a BLOCKED confirm stage counts as failure instead of skip")
    a = ap.parse_args()
    m = man()
    a.ep = a.ep or m["training_budget"]["total_episodes"]
    a.he = a.he or m["training_budget"]["heldout_every"]
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
    os.environ["WANDB_MODE"] = "disabled"      # belt: wrapper sets it too
    RL.LOG = None
    ok = True
    for st in a.stages:
        say(f"\n=== STAGE {st} ===")
        ok &= bool(STAGES[st](a, m))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
