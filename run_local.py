#!/usr/bin/env python3
"""
run_local.py -- run the entire post-review correction + P2-identification session on a local
machine (Windows cmd / macOS / Linux), no pod, no bash. Place at the repo root and run:

    python run_local.py                        # all steps, auto-detecting what the data allows
    python run_local.py --steps check          # dry inventory: what CAN run on this machine
    python run_local.py --steps analyze,refs,p2
    python run_local.py --jobs 6               # parallel eval workers (default: cores-1)

DESIGN RULE -- ZERO REIMPLEMENTATION. The registered analysis logic lives inside
sweep_all_hypotheses.sh as two embedded Python programs (the H2 merge and the H1/H2/H3/family
block). This driver does NOT re-code them: it extracts those programs verbatim from the sweep
script at runtime (matched by unique content signatures) and executes them with the same
arguments, environment variables, and working directory the pod would have used. Local output
is therefore identical to a pod STAGE=analyze run. The curve/interventions calls invoke the
same scripts/comm_stats.py CLIs with the same flags.

STEPS (each prints READY/BLOCKED in `check`, and skips itself cleanly if inputs are missing):
  check     selftests + torch probe + data inventory + per-step capability verdicts (~1 min)
  merge     re-merge sweep_out/h2 per-rho dumps under the registered seed whitelist (seconds)
  analyze   the registered STAGE=analyze at n=25: H1/H2/H3, families, C3-exploratory, curves,
            interventions; the fresh log is scanned for any residual n=30 (minutes)
  refs      scripts/clipped_refs.py, 200 episodes (measured: ~17 s single-core)
  p2        scripts/p2_decompose.py -> reports/P2_DECOMPOSITION.md (seconds)
  transfer  100 cross-world evals (needs torch + weights_signal), resumable, then p2 re-run
  ops       175 operational dumps + 4 aggregate tables (needs torch + weights_signal), resumable

Everything is written under reports/ (per-step logs + one session log). transfer/ops skip any
job whose output file already exists, so a killed run resumes exactly where it stopped.
"""
import os
import re
import sys
import json
import glob
import time
import argparse
import subprocess
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
SWEEP = os.path.join(ROOT, "sweep_all_hypotheses.sh")

# ---- registered constants (mirrors of the sweep script's defaults; the sweep remains the
# ---- source of truth -- change there first, here second) ---------------------------------
SEEDS = list(range(25, 50))
SEEDS_STR = " ".join(str(s) for s in SEEDS)
SEEDS_SPEC = "25-49"
MLIST = [1000, 2000, 4000, 8000]
PROBE_ARMS = ["ar1r9_upstream", "ar1r9_rbroadcast", "ar1r9_rbroadcast_learned",
              "ar1r9_rbroadcast_raw", "ar1r9_beta0_upstream", "ar1r9_beta05_upstream"]
AR1_MU, AR1_SIGMA = "12.0", "3.0"
DUMP_EPISODES = 200                       # matches the pod's v13 producers (CRN-comparable)
OPS_EPISODES = 40

TRANSFER = [  # (arm_tag, env patch, output cell)
    ("ar1r9_rbroadcast_raw",        "obs_order_clip=12",  "raw_tinf_eclip12"),
    ("ar1r9_clip12_rbroadcast_raw", "obs_order_clip=del", "raw_tclip12_einf"),
    ("ar1r9_nocomm",                "obs_order_clip=12",  "noc_tinf_eclip12"),
    ("ar1r9_clip12_nocomm",         "obs_order_clip=del", "noc_tclip12_einf"),
]
OPS_ARMS = [  # (arm_tag, ops dir name, dp?)
    ("ar1r9_rbroadcast_raw", "ops_ar1r9_raw", False),
    ("ar1r9_nocomm", "ops_ar1r9_nocomm", False),
    ("ar1r9_rbroadcast", "ops_ar1r9_dhat", False),
    ("ar1r9_clip12_rbroadcast_raw", "ops_clip12_raw", False),
    ("ar1r9_clip12_nocomm", "ops_clip12_nocomm", False),
    ("dp_rbroadcast_raw", "ops_dp_raw", True),
    ("dp_nocomm", "ops_dp_nocomm", True),
]
OPS_AGG = [("ops_ar1r9_raw", "ops_ar1r9_nocomm", "OPS_ar1r9_raw.md"),
           ("ops_ar1r9_dhat", "ops_ar1r9_nocomm", "OPS_ar1r9_dhat.md"),
           ("ops_clip12_raw", "ops_clip12_nocomm", "OPS_clip12_raw.md"),
           ("ops_dp_raw", "ops_dp_nocomm", "OPS_dp_raw.md")]
MERGE_SIG = 'tags = {"0.0": "ar1r0"'
ANALYZE_SIG = "from scripts.comm_stats import load_cost_dir, value_of_sharing"
P2_CELLS = ["ar1r9_nocomm", "ar1r9_raw", "clip20_nocomm", "clip20_raw",
            "clip12_nocomm", "clip12_raw"]

# ---- delivery fingerprints: CRLF-normalized sha256[:16] of every audited file. The check
# ---- step verifies these so transport/sync corruption is named, not guessed. run_local.py
# ---- itself is excluded (self-reference); it self-verifies by running.
DELIVERY_SHA = {
    "sweep_all_hypotheses.sh": "ddc5cdee5c8d444f",
    "auto_campaign2.sh": "bc7bfc57998f53d0",
    "readme.md": "0a38b24fa5c2b573",
    "envs/beer_game_env.py": "ef726816c3198551",
    "scripts/comm_stats.py": "2f36dce358cc106a",
    "scripts/c1_stats.py": "ad8be2409bcd3083",
    "scripts/clipped_refs.py": "2fe1d4a72d3a22c1",
    "scripts/p2_decompose.py": "956d85787754cf99",
    "scripts/decompose_costs.py": "a457a70706561417",
    "scripts/make_env_override.py": "7f7af6d4a03145f1",
    "agents/signal_agent.py": "20cb5eaab3ba68c0",
    "agents/eval_signal.py": "fcbe040491e8bde7",
    "agents/train_signal.py": "08a8c078cade27c1",
    "agents/demand_forecaster.py": "073aaff543607713",
    "scripts/forecast_pretrain.py": "2c1dfb67e4e15e74",
    "tests/test_forecaster.py": "99a20bfc8aadbc9d",
    "tests/test_phase2_integration.py": "ab0ac0adb4b3f6c9",
    "conf/agent/signal.yaml": "59ba08faf34e44a2",
    "agents/qmix_agent.py": "ff621c114939382c",
    "scripts/qmix_grid_benchmark.py": "798313a7e88e9b58",
    "run_repair_study.py": "9b85524414cfab8e",
    "tests/test_phase34.py": "a456f8632cc7a499",
    "reports/REPAIR_SEED_MANIFEST.json": "dce0c876ba0dffd4",
    "reports/DHAT_QMIX_REPAIR_PLAN.md": "30f09997ff022d6a",
    "reports/PHASE34_IMPLEMENTATION_GUIDE.md": "805e344f5229a5e6",
    "results/qmix_grid_benchmark.json": "cd942e36639a396d",
    "results/forecaster_ar1r9.pt": "1b840cb2698405ce",
    "results/forecaster_ar1r9_metrics.json": "9ebd3a03683f4671",
    "scripts/prereg_v3.py": "de54f4a1b4ba30da",
    "reports/PREREG_v3.json": "2e900397293bad67",
    "results/baselines_ar_v3.json": "369c32e522fb1fd4",
    "reports/RUNPOD_TUTORIAL_v3.md": "fe48a1b3fadc4dbf",
    "reports/PAPER_SKELETON_v3.md": "14ca8055547c32e4",
    "reports/RESEARCH_SKELETON_v3.md": "ada25d37b91a0665",
    "reports/research_sceleton_v3.md": "fd018b8120242d4d",
    "reports/POD_SETUP_GUIDE_v3.md": "9b9464f47a98102d"
}
REQUIRED_API = {  # semantic layer: symbols siblings import (catches version skew after edits)
    "scripts/comm_stats.py": ["parse_seed_spec", "load_cost_dir", "value_of_sharing"],
    "scripts/c1_stats.py": ["bootstrap_ci", "paired", "tost"],
    "scripts/confirmatory_v2.py": ["load_cell"],
}

LOG = None                                 # session log handle, opened in main()


def say(msg=""):
    print(msg, flush=True)
    if LOG:
        LOG.write(msg + "\n")
        LOG.flush()


def hr(title):
    say("\n" + "=" * 78)
    say(f"== {title}")
    say("=" * 78)


def env_with_whitelist(strict):
    e = dict(os.environ)
    e["SIGNAL_SEEDS"] = SEEDS_STR
    e["SIGNAL_SEEDS_STRICT"] = "1" if strict else "0"
    e.setdefault("PYTHONIOENCODING", "utf-8")
    return e


def run_logged(cmd, log_path, env=None):
    """Run cmd (list, no shell), stream stdout+stderr to console AND log_path. Returns rc."""
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(f"\n$ {' '.join(cmd)}\n")
        p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                             errors="replace", bufsize=1)
        try:
            for line in p.stdout:
                sys.stdout.write(line)
                lf.write(line)
        finally:
            p.stdout.close()
        return p.wait()


def extract_embedded(signature, what):
    """Pull one embedded <<'PY' program verbatim out of the sweep script."""
    txt = open(SWEEP, encoding="utf-8", errors="replace").read()
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY\n", txt, re.S)
    hits = [b for b in blocks if signature in b]
    if len(hits) != 1:
        sys.exit(f"FATAL: expected exactly 1 embedded program matching the {what} signature, "
                 f"found {len(hits)} -- sweep_all_hypotheses.sh changed shape; refusing to "
                 f"guess which registered logic to run.")
    return hits[0]


def run_embedded(signature, what, argv, log_path, env):
    prog = extract_embedded(signature, what)
    tmp = os.path.join(ROOT, "reports", f"_embedded_{what}.py")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(prog + "\n")
    return run_logged([PY, tmp, *argv], log_path, env=env)


# Learner prefixes that share the run_signal_*_<arm>_s<seed> namespace. A bare MAPPO arm like
# "ar1r9_nocomm" is a SUFFIX of the QMIX dir "run_signal_<id>_qmix_ar1r9_nocomm_s<seed>", so the
# wildcard silently also matches the QMIX run -- and latest_ck's mtime pick hands back a QMIX
# checkpoint (q_head.* weights) that the MAPPO SIGNALActor loader rejects. Anchor on the
# algorithm field: train_signal names dirs run_signal_<runid>_<algorithm>[_s<seed>], so the
# token immediately before _s<seed> must equal the requested arm exactly.
_FOREIGN_LEARNER_PREFIXES = ("qmix_",)


def latest_ck(arm, seed, fname="signal_checkpoint_best.pt"):
    hits = glob.glob(os.path.join(ROOT, "weights_signal",
                                  f"run_signal_*_{arm}_s{seed}", fname))
    clean = []
    for h in hits:
        run_dir = os.path.basename(os.path.dirname(h))          # run_signal_<id>_<arm>_s<seed>
        m = re.match(rf"run_signal_.+_({re.escape(arm)})_s{seed}$", run_dir)
        if not m:
            continue                                            # arm token not exact -> skip
        head = run_dir[len("run_signal_"):-(len(arm) + len(f"_s{seed}") + 1)]
        # head is the runid (+ any foreign learner prefix). Reject if a foreign learner's tag
        # was absorbed by the wildcard and this arm itself is not that learner.
        if any(pre.rstrip("_") in head.split("_") for pre in _FOREIGN_LEARNER_PREFIXES) \
                and not arm.startswith(_FOREIGN_LEARNER_PREFIXES):
            continue
        clean.append(h)
    return max(clean, key=os.path.getmtime) if clean else None


def ckpt_learner(path):
    """Cheap peek at a checkpoint's learner type from its actor state_dict keys, WITHOUT building
    a policy. Returns 'mappo' | 'qmix' | 'unknown'. Lets transfer/ops reject a wrong-learner
    checkpoint with the arm name attached instead of a raw state_dict traceback."""
    try:
        import torch
        d = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return "unknown"
    # find any nested state_dict-like mapping of param-name -> tensor
    def keys_of(obj, depth=0):
        found = []
        if isinstance(obj, dict):
            strkeys = [k for k in obj if isinstance(k, str)]
            if any(k.endswith((".weight", ".bias")) or k in ("log_std",) for k in strkeys):
                found += strkeys
            if depth < 3:
                for v in obj.values():
                    found += keys_of(v, depth + 1)
        return found
    ks = set(keys_of(d))
    if any(k == "log_std" or k.startswith("head.") for k in ks):
        return "mappo"
    if any(k.startswith("q_head.") for k in ks):
        return "qmix"
    return "unknown"


def count_cell(root, cell):
    return sum(os.path.exists(os.path.join(root, cell, f"seed{s}.json")) for s in SEEDS)


def torch_probe():
    """Import torch + the eval module in a subprocess (guards against import hangs)."""
    t0 = time.time()
    try:
        r = subprocess.run([PY, "-c",
                            "import torch, sys; sys.path.insert(0,'.'); "
                            "from agents import eval_signal; print('torch', torch.__version__)"],
                           cwd=ROOT, capture_output=True, text=True, timeout=240)
        return r.returncode == 0, (r.stdout + r.stderr).strip(), time.time() - t0
    except subprocess.TimeoutExpired:
        return False, "import timed out (>240 s)", time.time() - t0


def pool_run(jobs, workers, desc, log_path):
    """jobs: list of (cmd_list, done_path). Skips done, streams ETA, returns (ok, fail, skip)."""
    todo = [(c, d) for c, d in jobs if not (d and os.path.exists(d))]
    skip = len(jobs) - len(todo)
    say(f"  {desc}: {len(jobs)} jobs ({skip} already done -> skipped, {len(todo)} to run, "
        f"{workers} parallel)")
    if not todo:
        return 0, 0, skip
    ok = fail = 0
    t0 = time.time()
    with open(log_path, "a", encoding="utf-8") as lf, \
         ThreadPoolExecutor(max_workers=workers) as ex:
        def one(cmd):
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
            return cmd, r
        futs = [ex.submit(one, c) for c, _ in todo]
        step = max(1, len(todo) // 10)
        for i, fu in enumerate(as_completed(futs), 1):
            cmd, r = fu.result()
            lf.write(f"\n$ {' '.join(cmd)}\n{r.stdout}{r.stderr}")
            if r.returncode == 0:
                ok += 1
            else:
                fail += 1
                tail = (r.stderr or r.stdout).strip().splitlines()[-3:]
                ckp = next((cmd[i + 1] for i, t in enumerate(cmd) if t == "--ckpt"), "")
                label = os.path.basename(os.path.dirname(ckp)) if ckp else ' '.join(cmd[-3:])
                say(f"    FAIL rc={r.returncode}: {label}")
                for t in tail:
                    say(f"      | {t}")
            if i % step == 0 or i == len(todo):
                rate = i / max(1e-9, time.time() - t0)
                eta = (len(todo) - i) / max(1e-9, rate)
                say(f"    [{i}/{len(todo)}] {rate*60:.1f} jobs/min, ETA {eta/60:.1f} min")
    return ok, fail, skip


# ==============================================================================
# steps
# ==============================================================================
def _integrity_scan():
    """Catch corrupted/stale sources BEFORE selftests: (a) delivery-hash mismatches (names the
    exact file that differs from the audited delivery), (b) missing API symbols siblings import
    (explains WHAT is old about it), (c) a module importing itself (wrong-window paste),
    (d) syntax errors, (e) shadow copies of core modules outside scripts/. Returns a list of
    human-actionable problem strings."""
    import hashlib
    import ast
    problems = []
    for rel, want in DELIVERY_SHA.items():
        f = os.path.join(ROOT, rel)
        if not os.path.exists(f):
            problems.append(f"{rel}: MISSING -- restore from the delivered bundle")
            continue
        got = hashlib.sha256(open(f, "rb").read().replace(b"\r\n", b"\n")).hexdigest()[:16]
        if got != want:
            problems.append(f"{rel}: differs from the audited delivery "
                            f"(sha {got} != {want}) -- replace with the delivered copy "
                            f"(OneDrive conflict-restores and partial pastes both look like this)")
    for rel, names in REQUIRED_API.items():
        f = os.path.join(ROOT, rel)
        if not os.path.exists(f):
            continue
        try:
            tree = ast.parse(open(f, encoding="utf-8", errors="replace").read())
            have = {n.name for n in tree.body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
            miss = [n for n in names if n not in have]
            if miss:
                problems.append(f"{rel}: missing {miss} -- an OLD version; other scripts "
                                f"import these names and will crash or silently mis-analyze")
        except SyntaxError:
            pass                                       # reported by the syntax scan below
    core = {os.path.basename(k) for k in DELIVERY_SHA if k.startswith("scripts/")}
    for f in glob.glob(os.path.join(ROOT, "**", "*.py"), recursive=True):
        rel = os.path.relpath(f, ROOT).replace(os.sep, "/")
        b = os.path.basename(f)
        if b in core and not rel.startswith(("scripts/", "reports/", "venv/", ".venv/")) \
                and "site-packages" not in rel:
            problems.append(f"{rel}: SHADOW copy of scripts/{b} -- can hijack imports; "
                            f"delete or rename it")
    for pat in ("scripts/*.py", "agents/*.py", "envs/*.py", "run_local.py"):
        for f in sorted(glob.glob(os.path.join(ROOT, pat))):
            mod = os.path.splitext(os.path.basename(f))[0]
            rel = os.path.relpath(f, ROOT)
            try:
                lines = open(f, encoding="utf-8", errors="replace").read().splitlines()
            except OSError as e:
                problems.append(f"{rel}: unreadable ({e})")
                continue
            for i, ln in enumerate(lines, 1):
                t = ln.strip()
                if re.match(rf"(from\s+(scripts|agents|envs)\.{mod}\s+import|import\s+(scripts|agents|envs)\.{mod}\b)", t):
                    problems.append(f"{rel}:{i}: file imports ITSELF ({t[:70]}...) -- a line "
                                    f"from another file was pasted in; delete this line or "
                                    f"restore the file from git / the delivered copy")
            try:
                compile("\n".join(lines) + "\n", f, "exec")
            except SyntaxError as e:
                problems.append(f"{rel}:{e.lineno}: syntax error -- {e.msg}")
    return problems


def step_check(a):
    hr("CHECK: selftests, torch, data inventory, capabilities")
    ok = True
    logp = os.path.join(a.reports, "check.log")
    probs = _integrity_scan()
    if probs:
        say("  SOURCE INTEGRITY: FAIL -- fix these before anything else:")
        for pr in probs:
            say(f"    {pr}")
        ok = False
    else:
        say("  source integrity (self-import + syntax scan): OK")
    for script in ("scripts/comm_stats.py", "scripts/p2_decompose.py",
                   "scripts/decompose_costs.py"):
        rc = run_logged([PY, script, "--selftest"], logp, env=env_with_whitelist(a.strict))
        say(f"  selftest {script}: {'PASS' if rc == 0 else 'FAIL'}")
        ok &= rc == 0
    for mod in ("numpy", "scipy", "statsmodels"):
        r = subprocess.run([PY, "-c", f"import {mod}"], capture_output=True)
        say(f"  dependency {mod}: {'OK' if r.returncode == 0 else 'MISSING'}")
        if mod != "statsmodels":
            ok &= r.returncode == 0
        elif r.returncode != 0:
            say("    -> statsmodels is REQUIRED by the F_GEOMETRY/F_CONTENT family block; "
                "analyze will FAIL until:  pip install statsmodels")
    t_ok, t_msg, t_dt = torch_probe()
    say(f"  torch + agents.eval_signal import: {'OK' if t_ok else 'UNAVAILABLE'} "
        f"({t_dt:.0f}s)  {t_msg.splitlines()[-1] if t_msg else ''}")
    out = a.outroot
    say(f"\n  data root: {out}")
    inv = {}
    for name, path in [("h2 per-rho dumps", os.path.join(out, "h2")),
                       ("families (fam)", os.path.join(out, "fam")),
                       ("incentive", os.path.join(out, "incentive")),
                       ("curve", os.path.join(out, "curve")),
                       ("probes", os.path.join(out, "probes")),
                       ("v13", os.path.join(out, "v13"))]:
        n = len(glob.glob(os.path.join(path, "**", "seed*.json"), recursive=True))
        inv[name] = n
        say(f"    {name:<18} {'MISSING' if not os.path.isdir(path) else f'{n} seed files'}")
    v13 = os.path.join(out, "v13")
    p2n = {c: count_cell(v13, c) for c in P2_CELLS}
    say("    p2 cells (need 25/25 each): "
        + "  ".join(f"{c}:{n}" for c, n in p2n.items()))
    cks = {arm: sum(1 for s in SEEDS if latest_ck(arm, s)) for arm, _, _ in TRANSFER}
    say("    checkpoints (transfer arms, path-resolved): "
        + "  ".join(f"{k}:{v}/25" for k, v in cks.items()))
    if t_ok:                                            # learner peek needs torch
        vck = {}
        for arm, _, _ in TRANSFER:
            good = sum(1 for s in SEEDS
                       if resolve_mappo_ck(arm, s, verify_learner=True)[1] == "ok")
            vck[arm] = good
        say("    checkpoints (MAPPO-verified): "
            + "  ".join(f"{k}:{v}/25" for k, v in vck.items()))
        if any(vck[a] < cks[a] for a in cks):
            say("      (fewer verified than resolved on some arm: the path glob saw a QMIX "
                "collision that the learner check filtered -- expected and now handled)")
        cks = vck                                       # gate readiness on the verified count
    caps = {
        "merge":    inv["h2 per-rho dumps"] > 0,
        "analyze":  inv["h2 per-rho dumps"] > 0 or inv["families (fam)"] > 0,
        "refs":     True,
        "p2":       all(n == len(SEEDS) for n in p2n.values()),
        "transfer": t_ok and all(v == 25 for v in cks.values()),
        "ops":      t_ok and any(v > 0 for v in cks.values()),
    }
    say("\n  capability verdicts:")
    for k, v in caps.items():
        say(f"    {k:<9} {'READY' if v else 'BLOCKED'}")
    if not caps["p2"]:
        say("    (p2 BLOCKED detail: a listed v13 cell is short of 25 registered seed files; "
            "p2_decompose is fail-closed by design -- check the v13 counts above)")
    if not t_ok:
        say("    (transfer/ops BLOCKED: torch or the eval module did not import in this "
            "Python env; the analyze/refs/p2 results are unaffected)")
    return ok, caps


def step_merge(a):
    hr("MERGE: h2 per-rho dumps -> whitelisted merged dirs (registered format)")
    logp = os.path.join(a.reports, "merge.log")
    rc = run_embedded(MERGE_SIG, "merge", [os.path.join(a.outroot, "h2")], logp,
                      env_with_whitelist(a.strict))
    say(f"  merge: {'OK' if rc == 0 else f'FAIL rc={rc}'}")
    return rc == 0


def step_analyze(a):
    hr("ANALYZE: registered n=25 statistics (verbatim sweep logic)")
    out = a.outroot
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    logp = os.path.join(a.reports, f"analyze_n25_{stamp}_local.txt")
    env = env_with_whitelist(a.strict)
    say(f"  seed whitelist: {SEEDS_STR}")
    say(f"  strict: {env['SIGNAL_SEEDS_STRICT']}   log -> {logp}")
    start_ofs = os.path.getsize(logp) if os.path.exists(logp) else 0   # scan only THIS run's
    rc = run_embedded(ANALYZE_SIG, "analyze",
                      [os.path.join(out, "h2"), os.path.join(out, "h1pois"),
                       os.path.join(out, "incentive"), os.path.join(out, "fam")],
                      logp, env)
    all_ok = rc == 0
    # ---- substitution curves (registered dp/ar1r9 pair + v1.3 raw pair, same guard) ------
    for reg in ["dp", "ar1r9", "dpraw", "ar1r9raw"]:
        cargs, complete = [], True
        for M in MLIST:
            c = os.path.join(out, "curve", f"{reg}_comm_b{M}")
            n = os.path.join(out, "curve", f"{reg}_nocomm_b{M}")
            if os.path.isdir(c) and os.path.isdir(n):
                cargs += ["--budget", str(M), c, n]
            else:
                complete = False
        if complete and cargs:
            lbl = "(v1.3 raw pair)" if reg.endswith("raw") else "(registered)"
            say(f"  -- SUBSTITUTION CURVE {reg} {lbl} --")
            all_ok &= run_logged([PY, "scripts/comm_stats.py", "curve",
                                  "--seeds", SEEDS_SPEC] + cargs, logp, env=env) == 0
        else:
            say(f"  (curve {reg}: milestone dumps incomplete -- skipped, matching the sweep)")
    # ---- content-attribution gate ---------------------------------------------------------
    for arm in PROBE_ARMS:
        d = os.path.join(out, "probes", f"iv_{arm}")
        if glob.glob(os.path.join(d, "seed*_iv.json")):
            say(f"  -- INTERVENTIONS: {arm} --")
            all_ok &= run_logged([PY, "scripts/comm_stats.py", "interventions",
                                  "--dir", d, "--seeds", SEEDS_SPEC], logp, env=env) == 0
    # ---- contamination scan on the fresh log ----------------------------------------------
    with open(logp, encoding="utf-8", errors="replace") as _f:
        _f.seek(start_ofs)
        txt = _f.read()
    n30 = len(re.findall(r"\bn\s*=\s*30\b", txt))
    excl = txt.count("EXCLUDED unregistered")
    audits = txt.count("[seeds]")
    imp_skips = re.findall(r"skipped \(No module named '([^']+)'\)", txt)
    say(f"\n  contamination scan: {audits} loader audit lines, {excl} with exclusions "
        f"(0/0 is normal when every loaded dir is already exactly the registered set); "
        + ("NO n=30 anywhere -- clean"
           if n30 == 0 else
           f"*** {n30} residual n=30 lines -- STOP, a loader is bypassing the whitelist ***"))
    if imp_skips:
        say(f"  *** REGISTERED SECTION SILENTLY SKIPPED on missing module(s) "
            f"{sorted(set(imp_skips))} -- pip install them and re-run analyze; the family "
            f"tables are the review-critical n=25 output and did NOT regenerate ***")
    return all_ok and n30 == 0 and not imp_skips


def step_refs(a):
    hr(f"REFS: clipped-world frontier + wedge ({a.refs_episodes} eps; ~17 s at 200)")
    logp = os.path.join(a.reports, "refs.log")
    rc = run_logged([PY, "scripts/clipped_refs.py", "--episodes", str(a.refs_episodes),
                     "--rhos", "0.9", "--clips", "inf,20,12",
                     "--out", "results/baselines_ar_clip_v2.json"], logp,
                    env=env_with_whitelist(a.strict))
    return rc == 0


def step_p2(a, tag=""):
    hr(f"P2 DECOMPOSITION{tag}: Gamma split, wedge capture, verdict, QMIX concordance")
    logp = os.path.join(a.reports, "p2.log")
    rc = run_logged([PY, "scripts/p2_decompose.py", "--root", os.path.join(a.outroot, "v13"),
                     "--seeds", SEEDS_SPEC, "--refs", "results/baselines_ar_clip_v2.json",
                     "--transfer-root", os.path.join(a.outroot, "p2_transfer"),
                     "--out", os.path.join("reports", "P2_DECOMPOSITION.md")], logp,
                    env=env_with_whitelist(a.strict))
    return rc == 0


def _batch_env_overrides(triples, log_path):
    """One subprocess, ONE torch import, all override JSONs. triples: (ckpt, patch, out)."""
    if not triples:
        return True
    prog = (
        "import sys, json, os, torch\n"
        "sys.path.insert(0, '.')\n"
        "from scripts.make_env_override import parse_val\n"
        "man = json.load(open(sys.argv[1]))\n"
        "bad = 0\n"
        "for ck, patch, out in man:\n"
        "    d = torch.load(ck, map_location='cpu', weights_only=False)\n"
        "    env = d.get('env') or d.get('config', {}).get('env')\n"
        "    if not isinstance(env, dict) or not env:\n"
        "        print('FAIL no saved env:', ck); bad += 1; continue\n"
        "    env = dict(env)\n"
        "    k, v = patch.split('=', 1)\n"
        "    if v.lower() == 'del':\n"
        "        env.pop(k, None)\n"
        "    else:\n"
        "        env[k] = parse_val(v)\n"
        "    os.makedirs(os.path.dirname(os.path.abspath(out)) or '.', exist_ok=True)\n"
        "    json.dump(env, open(out, 'w'), indent=2, sort_keys=True)\n"
        "print(f'overrides written: {len(man)-bad}/{len(man)}')\n"
        "sys.exit(1 if bad else 0)\n")
    man = os.path.join(ROOT, "reports", "_override_manifest.json")
    with open(man, "w", encoding="utf-8") as f:
        json.dump(triples, f)
    tmp = os.path.join(ROOT, "reports", "_batch_overrides.py")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(prog)
    return run_logged([PY, tmp, man], log_path) == 0


def resolve_mappo_ck(arm, seed, verify_learner):
    """(path, status): status in 'ok' | 'missing' | 'wrong_learner'. wrong_learner means a
    checkpoint dir resolved but its weights are not MAPPO -- distinct from no checkpoint at all,
    so a mis-resolution is never silently counted as 'missing'. verify_learner gates the (cheap
    but torch-loading) type peek so it can be skipped when torch is unavailable."""
    ck = latest_ck(arm, seed)
    if not ck:
        return None, "missing"
    if verify_learner and ckpt_learner(ck) == "qmix":
        return None, "wrong_learner"
    return ck, "ok"


def step_transfer(a):
    hr("TRANSFER: cross-world evals (train-time vs eval-time garbling)")
    logp = os.path.join(a.reports, "transfer.log")
    tdir = os.path.join(a.outroot, "p2_transfer")
    ejdir = os.path.join(a.reports, "_envjson")
    os.makedirs(ejdir, exist_ok=True)
    triples, jobs, missing, wrong = [], [], [], []
    for arm, patch, cell in TRANSFER:
        for s in SEEDS:
            ck, st = resolve_mappo_ck(arm, s, a.verify_learner)
            if st == "missing":
                missing.append(f"{arm} s{s}")
                continue
            if st == "wrong_learner":
                wrong.append(f"{arm} s{s}")
                continue
            ej = os.path.join(ejdir, f"{cell}_s{s}.json")
            done = os.path.join(tdir, cell, f"seed{s}.json")
            if not os.path.exists(done):               # only build overrides for pending jobs
                triples.append((ck, patch, ej))
            jobs.append(([PY, "agents/eval_signal.py", "--ckpt", ck, "--env-json", ej,
                          "--dump-comm", os.path.join(tdir, cell),
                          "--dump-ar1", "0.9", "--ar1-mu", AR1_MU, "--ar1-sigma", AR1_SIGMA,
                          "--dump-episodes", str(a.dump_episodes)], done))
    if wrong:
        say(f"  NOTE {len(wrong)} seed(s) had only a non-MAPPO (QMIX) checkpoint on the arm "
            f"path and were skipped (first: {wrong[0]}) -- the MAPPO run for that seed is "
            f"absent; not an error, but that cell will be short")
    if missing:
        say(f"  WARNING {len(missing)} checkpoints missing (first: {missing[0]}) -- those "
            f"cells will be short and p2's transfer table will show it")
    say(f"  building {len(triples)} env-override JSONs (single torch process)...")
    if not _batch_env_overrides(triples, logp):
        say("  override construction FAILED -- aborting transfer (see transfer.log)")
        return False
    ok, fail, skip = pool_run(jobs, a.jobs, "transfer evals", logp)
    say(f"  transfer: {ok} ok, {fail} failed, {skip} resumed-as-done")
    if ok or skip:
        return step_p2(a, tag=" (re-run with transfer table)") and fail == 0
    return fail == 0


def step_ops(a):
    hr("OPS: per-stage operational dumps + aggregate tables (the 10.6% answer)")
    logp = os.path.join(a.reports, "ops.log")
    odir = os.path.join(a.outroot, "ops")
    jobs, missing, wrong = [], [], []
    for arm, out, is_dp in OPS_ARMS:
        for s in SEEDS:
            ck, st = resolve_mappo_ck(arm, s, a.verify_learner)
            if st == "missing":
                missing.append(f"{arm} s{s}")
                continue
            if st == "wrong_learner":
                wrong.append(f"{arm} s{s}")
                continue
            done = os.path.join(odir, out, f"seed{s}_ops.json")
            cmd = [PY, "scripts/decompose_costs.py", "dump", "--ckpt", ck,
                   "--out", os.path.join(odir, out), "--episodes", str(OPS_EPISODES),
                   "--ar1-rho", "0.9"]
            if is_dp:
                cmd.append("--dp")
            jobs.append((cmd, done))
    if wrong:
        say(f"  NOTE {len(wrong)} seed(s) had only a non-MAPPO checkpoint and were skipped "
            f"(first: {wrong[0]})")
    if missing:
        say(f"  WARNING {len(missing)} checkpoints missing (first: {missing[0]})")
    ok, fail, skip = pool_run(jobs, a.jobs, "ops dumps", logp)
    agg_ok = True
    for comm, noc, rep in OPS_AGG:
        c, n = os.path.join(odir, comm), os.path.join(odir, noc)
        if all(os.path.exists(os.path.join(d, f"seed{s}_ops.json"))
               for d in (c, n) for s in SEEDS):
            say(f"  -- aggregate {rep} --")
            agg_ok &= run_logged([PY, "scripts/decompose_costs.py", "aggregate",
                                  "--comm", c, "--nocomm", n, "--seeds", SEEDS_SPEC,
                                  "--out", os.path.join("reports", rep)], logp,
                                 env=env_with_whitelist(a.strict)) == 0
        else:
            say(f"  (aggregate {rep}: dumps incomplete -- skipped)")
    return fail == 0 and agg_ok


# ==============================================================================
def main():
    global LOG
    ap = argparse.ArgumentParser(description="Local (pod-free) SIGNAL correction session.")
    ap.add_argument("--steps", default="check,merge,analyze,refs,p2,transfer,ops",
                    help="comma list from: check merge analyze refs p2 transfer ops")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 1),
                    help="parallel eval workers for transfer/ops (default: cores-1)")
    ap.add_argument("--outroot", default=os.path.join(ROOT, "sweep_out"))
    ap.add_argument("--reports", default=os.path.join(ROOT, "reports"))
    ap.add_argument("--refs-episodes", type=int, default=200)
    ap.add_argument("--dump-episodes", type=int, default=DUMP_EPISODES,
                    help="transfer eval episodes; keep 200 to match the native v13 cells")
    ap.add_argument("--strict", type=int, default=1, choices=(0, 1),
                    help="SIGNAL_SEEDS_STRICT (1 = fail-closed whitelist, registered default)")
    ap.add_argument("--verify-learner", type=int, default=1, choices=(0, 1),
                    help="peek each resolved checkpoint's learner type and skip non-MAPPO "
                         "(QMIX) collisions before eval (default on; set 0 only if torch-load "
                         "of checkpoints is the bottleneck and paths are known clean)")
    ap.add_argument("--stop-on-fail", action="store_true")
    a = ap.parse_args()
    if not os.path.exists(SWEEP):
        sys.exit("FATAL: run_local.py must sit at the repo root next to sweep_all_hypotheses.sh")
    os.makedirs(a.reports, exist_ok=True)
    LOG = open(os.path.join(a.reports,
                            f"local_session_{datetime.now():%Y-%m-%d_%H%M}.log"),
               "w", encoding="utf-8")
    say(f"run_local.py | {datetime.now():%Y-%m-%d %H:%M} | {sys.platform} | "
        f"python {sys.version.split()[0]} | jobs={a.jobs}")
    say(f"repo: {ROOT}")
    say(f"out:  {a.outroot}")
    steps = [s.strip() for s in a.steps.split(",") if s.strip()]
    caps = None
    results = {}
    t_all = time.time()
    for s in steps:
        t0 = time.time()
        if s == "check":
            ok, caps = step_check(a)
        elif s in ("transfer", "ops", "p2") and caps is not None and not caps.get(s, True):
            say(f"\n-- {s}: BLOCKED per capability check, skipping --")
            results[s] = "BLOCKED"
            continue
        elif s == "merge":
            ok = step_merge(a)
        elif s == "analyze":
            ok = step_analyze(a)
        elif s == "refs":
            ok = step_refs(a)
        elif s == "p2":
            ok = step_p2(a)
        elif s == "transfer":
            ok = step_transfer(a)
        elif s == "ops":
            ok = step_ops(a)
        else:
            say(f"unknown step {s!r}, skipping")
            continue
        results[s] = "OK" if ok else "FAIL"
        say(f"-- step {s}: {results[s]} ({time.time()-t0:.0f}s)")
        if not ok and a.stop_on_fail:
            break
    hr("SESSION SUMMARY")
    for s, r in results.items():
        say(f"  {s:<9} {r}")
    say(f"  total: {(time.time()-t_all)/60:.1f} min")
    say("\n  deliverables: reports/analyze_n25_*_local.txt   reports/P2_DECOMPOSITION.md")
    say("                reports/OPS_*.md   results/baselines_ar_clip_v2.json")
    sys.exit(0 if all(r in ("OK", "BLOCKED") for r in results.values()) else 1)


if __name__ == "__main__":
    main()
