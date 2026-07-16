#!/usr/bin/env bash
# ============================================================================
# auto_campaign.sh -- run the ENTIRE SIGNAL campaign unattended, gate-safe.
# ----------------------------------------------------------------------------
# WHAT THIS IS (student version):
#   The launch runbook as a machine. You start it once inside tmux, detach,
#   and come back to a finished, archived campaign -- UNLESS a registered
#   science gate fails, in which case the script STOPS ITSELF and writes a
#   GATE_*_STOP.txt file telling you exactly what it saw. That stopping
#   behavior is not a limitation: it IS the pre-registered protocol. A fully
#   automatic run that barrels through a failed positive control would be
#   automation of a scientific mistake.
#
# WHAT IT DOES, IN ORDER (each stage skips itself if already done):
#   S1  setup_pod.sh (CPU torch wheel -- NEVER pass GPU=1 on this project)
#       + matplotlib (needed for figures; commented out in requirements.txt)
#   S2  instrument freeze manifest (SHA256 of every measurement-touching file)
#   S3  behavioral benchmarks: union-lambda, --bar-per-echelon  [the ruler]
#   S4  DRYRUN assert (29x15=435) + NPROC calibration (cached)
#   S5  audibility pilot (1500-ep comm run + --messages report; advisory)
#   S6  TRAIN Phase A (165)                                   [40% of budget]
#   S7  GATES: C1 positive control (HARD STOP if Gap CI_lo <= 0)
#              futility at rho=0.9  (HARD STOP if TOST-equivalent to zero)
#   S8  TRAIN behavioral rest: B Bnull C E (+Bext)            [the other 60%]
#   S9  canonical block: refs + validate + json swap + D (+Dext) + RESTORE
#   S10 extraction: STAGE=dump (~660 jobs) + STAGE=probe (90)
#   S11 STAGE=analyze + STAGE=plot + confirmatory report (all tee'd to files)
#   S12 one archive tarball + SHA256  -> download via the RunPod GUI
#
# HOW TO LAUNCH (the only three lines you type):
#   tmux new -s signal
#   ./auto_campaign.sh 2>&1 | tee -a auto_campaign.log     # then Ctrl-b d
#   ./auto_campaign.sh status                              # check any time
#
# IF THE POD DIES: rent a new one, clone at the tag, run the SAME line.
#   Training resumes via the sweep's sentinels; finished stages skip via
#   auto_state/ markers; the benchmark/refs/gates re-verify themselves.
#
# OVERRIDABLE ENV (defaults in brackets):
#   NPROC[auto-calibrated]  INCLUDE_EXT[1 = run Bext+Dext]  SKIP_PILOT[0]
#   FORCE_CONTINUE[0 = respect gate stops]  FORCE_ARCHIVE[0]
# SUBCOMMANDS:  status | selftest
# ============================================================================
set -euo pipefail
trap 'echo; echo "!! auto_campaign aborted (see the last [HH:MM:SS] banner above). Fix the cause and re-run -- it resumes."' ERR
cd "$(dirname "$(readlink -f "$0" 2>/dev/null || echo "$0")")"
[[ -f agents/train_signal.py && -f sweep_all_hypotheses.sh ]] || {
  echo "ERROR: run from the BeerGame_Comm repo root."; exit 1; }
chmod +x setup_pod.sh sweep_all_hypotheses.sh 2>/dev/null || true   # self-heal: a clone may not carry the exec bit

INCLUDE_EXT="${INCLUDE_EXT:-1}"; SKIP_PILOT="${SKIP_PILOT:-0}"
FORCE_CONTINUE="${FORCE_CONTINUE:-0}"; FORCE_ARCHIVE="${FORCE_ARCHIVE:-0}"
ST=auto_state; mkdir -p "$ST" reports snapshots
PYBIN="python3"; [[ -x venv/bin/python ]] && PYBIN="venv/bin/python"

say()  { echo; echo "== [$(date +%H:%M:%S)] $*"; }
die()  { echo; echo "!! STOPPED: $*"; echo "!! Fix the cause, then re-run ./auto_campaign.sh -- it resumes."; exit 1; }
mark() { touch "$ST/$1.ok"; }
done_already() { [[ -f "$ST/$1.ok" ]]; }

# ---- helpers the gates depend on (each covered by `selftest`) --------------
runlog_counts() {  # runlog_counts <run.log> -> "done skip fail"
  local f="$1" d s x
  d=$(grep -ac '\[done\]' "$f" 2>/dev/null || true)
  s=$(grep -ac '\[skip\]' "$f" 2>/dev/null || true)
  x=$(grep -ac '\[FAIL\]' "$f" 2>/dev/null || true)
  echo "${d:-0} ${s:-0} ${x:-0}"
}
assert_train() {   # assert_train <expected done+skip> <label>
  read -r d s x <<< "$(runlog_counts sweep_out/run.log)"
  say "$2: done=$d skip=$s FAIL=$x (expected done+skip=$1)"
  (( x == 0 )) || die "$2 had $x [FAIL] jobs -- see sweep_out/logs/<arm>.log"
  (( d + s == $1 )) || die "$2 finished $((d+s))/$1 arms -- re-run to resume"
}
pick_nproc() {     # parse STAGE=calibrate output on stdin -> best NPROC (POSIX-safe: no gawk)
  grep -oE 'NPROC=[0-9]+ : [0-9]+ eps/s' \
    | sed -e 's/NPROC=//' -e 's/ : / /' -e 's| eps/s||' \
    | sort -k2,2n | tail -1 | cut -d' ' -f1 | grep -E '^[0-9]+$'
}
refs_ok() {        # refs_ok <json> : 9 lambdas AND per-echelon BAR levels
  "$PYBIN" - "$1" <<'PY'
import json, sys
try: d = json.load(open(sys.argv[1]))
except Exception: sys.exit(1)
lams = {float(k) for k in d.get("rungs", {}).get("BAR_static", {})}
bl = d.get("meta", {}).get("bar_levels") or []
sys.exit(0 if lams == {6.0,8.0,10.0,12.0,14.0,16.0,18.0,20.0,22.0} and len(set(bl)) >= 2 else 1)
PY
}
gate_c1() {        # gate_c1 <confirmatory_report.json> : PASS iff Gap CI_lo > 0
  "$PYBIN" - "$1" <<'PY'
import json, sys
c = json.load(open(sys.argv[1])).get("c1", {})
lo = None
for k, v in c.items():                      # tolerant: find the gap CI lower bound
    if "gap" in k.lower() and "ci" in k.lower() and isinstance(v, (list, tuple)) and len(v) == 2:
        lo = float(v[0]); break
if lo is None:
    g = c.get("gap_mean");  # fallback: mean only (weaker) -> warn via exit 2
    sys.exit(2 if (g is not None and float(g) > 0) else 1)
print(f"C1 Gap_Recovered CI_lo = {lo:+.3f}")
sys.exit(0 if lo > 0 else 1)
PY
}
gate_futility() {  # gate_futility <h2root> -> writes reports/gate_futility.txt; exit 1 iff TOST-equivalent zero
  "$PYBIN" - "$1" <<'PY'
import sys, io
sys.path.insert(0, ".")
from scripts.comm_stats import load_cost_dir, value_of_sharing
comm, noc = load_cost_dir(f"{sys.argv[1]}/comm"), load_cost_dir(f"{sys.argv[1]}/nocomm")
if not comm or not noc:
    print("futility gate: rho-0.9 dumps not found -> cannot evaluate"); sys.exit(3)
v = value_of_sharing(comm, noc, lambdas=[0.9])
line = (f"V(rho=0.9) = {v['v_cost_mean']:+.1f} ({v['v_cost_pct']:+.2f}%)  "
        f"CI=[{v['v_cost_ci'][0]:.1f},{v['v_cost_ci'][1]:.1f}]  n={v['n_seeds']}  "
        f"wilcoxon_p={v['wilcoxon_p']:.3g}  TOST_p={v['tost_p']:.3g}  equivalent={v['equivalent']}")
open("reports/gate_futility.txt", "w").write(line + "\n")
print(line)
sys.exit(1 if v["equivalent"] else 0)
PY
}
behavioral_json_in_place() {  # extraction/analysis must ALWAYS score against behavioral refs
  if [[ -f results/baselines_regime_v2.behavioral.json ]]; then
    cp results/baselines_regime_v2.behavioral.json results/baselines_regime_v2.json
  fi
}
snapshot() { tar czf "snapshots/partial_$(date +%m%d_%H%M).tgz" results/ sweep_out/ \
             weights_signal/run_signal_*/metrics_*.csv 2>/dev/null || true; }

# ============================ subcommand: status ============================
if [[ "${1:-}" == "status" ]]; then
  echo "== SIGNAL auto-campaign status =="
  for s in S1_setup S2_freeze S3_refs S4_calibrate S5_pilot S6_phaseA S7_gates \
           S8_behavioral S9_canonical S10_extract S11_analysis S12_archive; do
    printf "  %-14s %s\n" "$s" "$([[ -f $ST/$s.ok ]] && echo DONE || echo pending)"
  done
  echo "  sentinels (finished arms): $(ls weights_signal/.done_* 2>/dev/null | wc -l)"
  [[ -f sweep_out/run.log ]] && { read -r d s x <<< "$(runlog_counts sweep_out/run.log)";
    echo "  last train invocation: done=$d skip=$s FAIL=$x"; }
  [[ -f "$ST/nproc" ]] && echo "  NPROC (calibrated): $(cat "$ST/nproc")"
  for g in reports/gate_c1_verdict.txt reports/gate_futility.txt GATE_*_STOP.txt; do
    [[ -f "$g" ]] && echo "  gate: $g -> $(head -1 "$g")"; done
  exit 0
fi

# =========================== subcommand: selftest ===========================
if [[ "${1:-}" == "selftest" ]]; then
  echo "== auto_campaign selftest (decision functions on synthetic fixtures) =="
  T=$(mktemp -d)
  printf '[start] a\n[done] a\n[skip] b (done)\n[FAIL] c  (tail: boom)\n' > "$T/run.log"
  [[ "$(runlog_counts "$T/run.log")" == "1 1 1" ]] && echo "  runlog_counts: PASS" || { echo FAIL; exit 1; }
  printf 'NPROC=24 : 60 eps/s aggregate (10s wall)\nNPROC=36 : 71 eps/s aggregate (9s wall)\n' \
    | pick_nproc | grep -qx 36 && echo "  pick_nproc: PASS" || { echo FAIL; exit 1; }
  "$PYBIN" - "$T" <<'PY'
import json, sys, os
t = sys.argv[1]
json.dump({"rungs": {"BAR_static": {str(l): 1.0 for l in (6,8,10,12,14,16,18,20,22)}},
           "meta": {"bar_levels": [90, 95, 100, 105]}}, open(f"{t}/refs_good.json", "w"))
json.dump({"rungs": {"BAR_static": {"6.0": 1.0}}, "meta": {"bar_levels": [92,92,92,92]}},
          open(f"{t}/refs_bad.json", "w"))
json.dump({"c1": {"gap_mean": 0.42, "gap_ci": [0.31, 0.55]}}, open(f"{t}/conf_pass.json", "w"))
json.dump({"c1": {"gap_mean": 0.10, "gap_ci": [-0.05, 0.22]}}, open(f"{t}/conf_fail.json", "w"))
os.makedirs(f"{t}/h2/comm", exist_ok=True); os.makedirs(f"{t}/h2/nocomm", exist_ok=True)
for s in range(10, 25):
    json.dump({"0.9": 5000.0 + s}, open(f"{t}/h2/comm/seed{s}.json", "w"))
    json.dump({"0.9": 5220.0 + s}, open(f"{t}/h2/nocomm/seed{s}.json", "w"))   # V ~ +220 -> not equivalent
PY
  refs_ok "$T/refs_good.json" && ! refs_ok "$T/refs_bad.json" \
    && echo "  refs_ok: PASS" || { echo FAIL; exit 1; }
  gate_c1 "$T/conf_pass.json" >/dev/null && ! gate_c1 "$T/conf_fail.json" >/dev/null 2>&1 \
    && echo "  gate_c1: PASS" || { echo FAIL; exit 1; }
  gate_futility "$T/h2" >/dev/null && echo "  gate_futility (clear-signal case): PASS" || { echo FAIL; exit 1; }
  rm -f reports/gate_futility.txt          # selftest fixture verdict -- never show it as a real gate
  rm -rf "$T"; echo "selftest PASS"; exit 0
fi

# ================================ the campaign ==============================
[[ -n "${TMUX:-}" ]] || echo "!! WARNING: not inside tmux -- a dropped connection will kill this run. (tmux new -s signal)"
TAG="$(git describe --tags --exact-match 2>/dev/null || echo none)"
[[ "$TAG" == v1.2-launch ]] || echo "!! WARNING: HEAD is not at tag v1.2-launch (got: $TAG) -- you are not provably running the fingerprinted code."

# S1 -- environment ----------------------------------------------------------
if ! done_already S1_setup; then
  say "S1 setup: CPU torch wheel + full self-test battery (setup_pod.sh) + matplotlib"
  bash setup_pod.sh || die "setup_pod.sh failed -- read its output above"   # CPU wheel default. Do NOT set GPU=1.
  PYBIN="venv/bin/python"
  venv/bin/pip install -q matplotlib
  if ! "$PYBIN" - <<'PY'
import torch, sys
bad = torch.cuda.is_available() and "+cpu" not in torch.__version__
print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
sys.exit(1 if bad else 0)
PY
  then die "a CUDA torch wheel is active -- reinstall the CPU wheel (rerun setup_pod.sh without GPU=1)"; fi
  mark S1_setup
else say "S1 setup: already done"; fi
PYBIN="venv/bin/python"

# S2 -- freeze ----------------------------------------------------------------
if ! done_already S2_freeze; then
  say "S2 instrument freeze manifest"
  mkdir -p results
  sha256sum agents/signal_agent.py agents/train_signal.py agents/eval_signal.py \
    agents/signal_csvlog.py agents/topologies.py envs/beer_game_env.py envs/demand_randomization.py \
    scripts/demand_families.py scripts/comm_stats.py scripts/c1_stats.py scripts/prereg.py \
    scripts/baselines.py scripts/dp_optimum.py scripts/run_confirmatory_report.py \
    conf/config.yaml conf/agent/signal.yaml sweep_all_hypotheses.sh plot_curves.py \
    > results/FREEZE_MANIFEST_v1.2.txt
  "$PYBIN" scripts/prereg.py | grep -i sha256 >> results/FREEZE_MANIFEST_v1.2.txt || true
  mark S2_freeze
else say "S2 freeze: already done"; fi

# S3 -- behavioral benchmarks --------------------------------------------------
if ! done_already S3_refs; then
  if refs_ok results/baselines_regime_v2.json; then
    say "S3 refs: valid union-lambda per-echelon json already present"
  else
    say "S3 refs: generating (union lambdas, --bar-per-echelon; 1-3 h single-core)"
    "$PYBIN" scripts/baselines.py regime --lambdas 6 8 10 12 14 16 18 20 22 \
      --select-episodes 80 --eval-episodes 200 --bar-per-echelon 2>&1 | tee reports/refs_behavioral.log
    refs_ok results/baselines_regime_v2.json || die "refs json failed validation (9 lambdas + per-echelon BAR)"
  fi
  cp results/baselines_regime_v2.json results/baselines_regime_v2.behavioral.json
  mark S3_refs
else say "S3 refs: already done"; behavioral_json_in_place; fi

# S4 -- manifest assert + NPROC ------------------------------------------------
if ! done_already S4_calibrate; then
  say "S4 DRYRUN assert + NPROC calibration"
  dr_out="$(DRYRUN=1 STAGE=train PHASES=core ./sweep_all_hypotheses.sh 2>&1 || true)"
  grep -q "jobs=435" <<< "$dr_out" \
    || die "DRYRUN did not report the registered 29x15=435 manifest (did you override SEEDS?)"
  if [[ -n "${NPROC:-}" ]]; then echo "$NPROC" > "$ST/nproc"
  elif [[ ! -s "$ST/nproc" ]]; then
    STAGE=calibrate ./sweep_all_hypotheses.sh 2>&1 | tee reports/calibrate.log \
      | pick_nproc > "$ST/nproc" || die "could not parse calibrate output (reports/calibrate.log)"
  fi
  mark S4_calibrate
fi
NPROC="$(cat "$ST/nproc")"; export NPROC
say "workers: NPROC=$NPROC"

# S5 -- audibility pilot (advisory in auto mode) -------------------------------
if ! done_already S5_pilot; then
  if [[ "$SKIP_PILOT" == 1 ]]; then say "S5 pilot: skipped by env"
  else
    say "S5 audibility pilot (1500 eps comm arm + --messages; report saved, campaign continues)"
    "$PYBIN" agents/train_signal.py agent=signal agent.train_env=ar1 agent.ar1_rho=0.9 \
      agent.heldout_mode=ar1 agent.comm_topology=upstream_only agent.msg_content=dhat \
      seed=10 total_episodes=1500 agent.algorithm=pilot_aud_s10 > reports/pilot_train.log 2>&1 || true
    pc=$(ls -1dt weights_signal/run_signal_*_pilot_aud_s10/signal_checkpoint_best.pt 2>/dev/null | head -1 || true)
    if [[ -n "$pc" ]]; then
      "$PYBIN" agents/eval_signal.py --ckpt "$pc" --messages --episodes 40 \
        > reports/pilot_audibility.txt 2>&1 || true
      say "pilot report -> reports/pilot_audibility.txt (review when convenient; futility gate is the hard stop)"
    else echo "!! pilot training produced no checkpoint (reports/pilot_train.log) -- continuing; futility gate protects the budget"; fi
  fi
  mark S5_pilot
else say "S5 pilot: already done"; fi

# S6 -- Phase A ----------------------------------------------------------------
say "S6 TRAIN Phase A (165 arms) -- the gate inputs"
PHASES="A" ./sweep_all_hypotheses.sh 2>&1 | tee -a reports/launch_A.log >/dev/null \
  || die "Phase A invocation failed (reports/launch_A.log)"
assert_train 165 "Phase A"; mark S6_phaseA; snapshot

# S7 -- the registered gates ----------------------------------------------------
if ! done_already S7_gates; then
  say "S7 gates: dump A + C1 positive control + futility(rho=0.9)"
  behavioral_json_in_place
  STAGE=dump ./sweep_all_hypotheses.sh > reports/dumpA_stage.log 2>&1 || true
  mkdir -p results/signal_c1
  for s in $(seq 10 24); do
    [[ -f "results/signal_c1/seed${s}.json" ]] && continue
    ck=$(ls -1dt weights_signal/run_signal_*_dp_nocomm_s${s}/signal_checkpoint_best.pt 2>/dev/null | head -1 || true)
    [[ -n "$ck" ]] && "$PYBIN" agents/eval_signal.py --ckpt "$ck" --dump-c1 results/signal_c1 --episodes 200 \
      >> reports/c1_dump.log 2>&1 || true
  done
  "$PYBIN" scripts/run_confirmatory_report.py --signal-dir results/signal_c1 \
    --refs results/baselines_regime_v2.json | tee reports/gate_c1.txt \
    || die "confirmatory report failed (missing refs or C1 dumps?)"
  set +e; c1_out="$(gate_c1 results/confirmatory_report.json 2>&1)"; rc=$?; set -e
  printf '%s\n' "$c1_out" | tee reports/gate_c1_verdict.txt >/dev/null; echo "$c1_out"
  if   (( rc == 2 )); then echo "!! C1: CI key not found; gap_mean>0 only (weak pass) -- inspect reports/gate_c1.txt"
  elif (( rc != 0 )); then
    { echo "POSITIVE CONTROL FAILED: Gap_Recovered CI does not exclude 0."
      echo "The instrument did not demonstrably learn; no communication result on top of it is interpretable."
      echo "See reports/gate_c1.txt. Fix/diagnose, then FORCE_CONTINUE=1 only with written justification."; } \
      | tee GATE_C1_STOP.txt
    [[ "$FORCE_CONTINUE" == 1 ]] || die "registered gate C1 failed (GATE_C1_STOP.txt)"
  fi
  set +e; gate_futility sweep_out/h2; rc=$?; set -e
  if (( rc == 1 )); then
    { echo "FUTILITY: V(rho=0.9) is TOST-equivalent to zero in the theory-favorable cell."
      echo "Registered branch: stop, inspect audibility (reports/pilot_audibility.txt; eval --messages on an ar1r9 ckpt),"
      echo "document, and only then decide. FORCE_CONTINUE=1 overrides -- record why."; } | tee GATE_FUTILITY_STOP.txt
    [[ "$FORCE_CONTINUE" == 1 ]] || die "registered futility gate (GATE_FUTILITY_STOP.txt)"
  elif (( rc == 3 )); then
    die "futility gate could not evaluate: rho=0.9 dumps missing after Phase A + dump ran -- a registered gate may not be silently skipped (see reports/dumpA_stage.log)"
  fi
  STAGE=analyze ./sweep_all_hypotheses.sh > reports/analyze_A.txt 2>&1 || true
  mark S7_gates
else say "S7 gates: already passed"; fi

# S8 -- remaining behavioral ----------------------------------------------------
say "S8 TRAIN behavioral remainder (B Bnull C E = 225$( [[ $INCLUDE_EXT == 1 ]] && echo ' + Bext 60'))"
PHASES="B Bnull C E" ./sweep_all_hypotheses.sh 2>&1 | tee -a reports/launch_core.log >/dev/null \
  || die "B/Bnull/C/E invocation failed (reports/launch_core.log)"
assert_train 225 "B/Bnull/C/E"
if [[ "$INCLUDE_EXT" == 1 ]]; then
  PHASES="Bext" ./sweep_all_hypotheses.sh 2>&1 | tee -a reports/launch_core.log >/dev/null \
    || die "Bext invocation failed (reports/launch_core.log)"
  assert_train 60 "Bext"
fi
mark S8_behavioral; snapshot

# S9 -- canonical block ----------------------------------------------------------
if ! done_already S9_canonical; then
  say "S9 canonical: refs + validate + swap + D$( [[ $INCLUDE_EXT == 1 ]] && echo '+Dext') + RESTORE"
  if ! refs_ok results/baselines_regime_v2_canonical.json; then
    "$PYBIN" scripts/baselines.py regime --lambdas 6 8 10 12 14 16 18 20 22 \
      --select-episodes 80 --eval-episodes 200 --bar-per-echelon --penalty-at-retailer-only \
      2>&1 | tee reports/refs_canonical.log
  fi
  "$PYBIN" scripts/baselines.py validate-canonical --lambdas 6 10 14 18 22 \
    2>&1 | tee reports/validate_canonical.txt || die "validate-canonical failed"
  cp results/baselines_regime_v2_canonical.json results/baselines_regime_v2.json     # swap IN
  PHASES="D" ./sweep_all_hypotheses.sh 2>&1 | tee -a reports/launch_core.log >/dev/null \
    || die "Phase D invocation failed (reports/launch_core.log)"
  assert_train 45 "Phase D (canonical)"
  if [[ "$INCLUDE_EXT" == 1 ]]; then
    PHASES="Dext" ./sweep_all_hypotheses.sh 2>&1 | tee -a reports/launch_core.log >/dev/null \
      || die "Dext invocation failed (reports/launch_core.log)"
    assert_train 45 "Dext (canonical)"
  fi
  behavioral_json_in_place                                                            # RESTORE, always
  cmp -s results/baselines_regime_v2.json results/baselines_regime_v2.behavioral.json \
    || die "behavioral refs restore failed -- do not analyze until fixed"
  mark S9_canonical; snapshot
else say "S9 canonical: already done"; behavioral_json_in_place; fi

# S10 -- extraction ---------------------------------------------------------------
if ! done_already S10_extract; then
  say "S10 extraction: dump (~660 jobs) + probe (90 jobs) on CRN episodes"
  behavioral_json_in_place
  STAGE=dump  ./sweep_all_hypotheses.sh > reports/dump_stage.log 2>&1 || true
  STAGE=probe ./sweep_all_hypotheses.sh > reports/probe_stage.log 2>&1 || true
  for d in sweep_out/h2/comm sweep_out/h2/nocomm; do
    n=$(ls "$d"/seed*.json 2>/dev/null | wc -l)
    (( n == 15 )) || echo "!! WARN: $d has $n/15 seed files (grep Error reports/dump_stage.log)"
  done
  nf=$(ls -d sweep_out/fam/ar1r9_* 2>/dev/null | wc -l); (( nf == 13 )) || echo "!! WARN: fam dirs $nf/13"
  mark S10_extract
else say "S10 extraction: already done"; fi

# S11 -- analysis + figures --------------------------------------------------------
if ! done_already S11_analysis; then
  say "S11 registered statistics + figures + confirmatory report"
  behavioral_json_in_place
  STAGE=analyze ./sweep_all_hypotheses.sh 2>&1 | tee reports/analyze_FULL.txt
  STAGE=plot    ./sweep_all_hypotheses.sh 2>&1 | tee reports/plot_stage.log
  nfig=$(ls sweep_out/figs/*.pdf 2>/dev/null | wc -l)
  (( nfig >= 6 )) || echo "!! WARN: only $nfig figure PDFs (matplotlib missing? reports/plot_stage.log)"
  "$PYBIN" scripts/run_confirmatory_report.py --signal-dir results/signal_c1 \
    --refs results/baselines_regime_v2.json \
    --comm sweep_out/h1pois/nocomm sweep_out/h1pois/comm sweep_out/h1pois/rbroadcast \
    2>&1 | tee reports/confirmatory_FULL.txt || die "full confirmatory report failed"
  mark S11_analysis
else say "S11 analysis: already done"; fi

# S12 -- archive --------------------------------------------------------------------
if ! done_already S12_archive || [[ "$FORCE_ARCHIVE" == 1 ]]; then
  say "S12 archive (then download via the RunPod GUI and verify the hash on your laptop)"
  AR="SIGNAL_campaign_$(date +%F).tgz"
  tar czf "$AR" results/ sweep_out/ reports/ auto_campaign.log 2>/dev/null \
    weights_signal/run_signal_*/signal_checkpoint_best.pt \
    weights_signal/run_signal_*/signal_checkpoint_budget*.pt \
    weights_signal/run_signal_*/metrics_*.csv weights_signal/run_signal_*/run_meta.json \
    weights_signal/.done_* || true
  sha256sum "$AR" | tee ARCHIVE_HASH.txt
  mark S12_archive
  echo; echo "=================================================================="
  echo "  CAMPAIGN COMPLETE."
  echo "  1. Download via the RunPod GUI:  $(pwd)/$AR   and ARCHIVE_HASH.txt"
  echo "  2. On Windows:  certutil -hashfile $AR SHA256   -- must match."
  echo "  3. Only then stop the pod. Unblind in the registered order."
  echo "=================================================================="
fi