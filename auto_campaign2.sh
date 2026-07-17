#!/usr/bin/env bash
# =============================================================================
# auto_campaign2.sh -- SIGNAL campaign, UNATTENDED. Start it, sleep, come back
# to a finished tarball. It does not stop for anything you cannot change.
#
# WHAT CHANGED vs auto_campaign.sh (v1), and why:
#   1. NPROC is CONTAINER-AWARE. v1 trusted `lscpu`, which inside a RunPod
#      container reports the HOST topology (dual Xeon 6952P = 192 cores) -- so
#      v1 ran 48 vCPU at 4x oversubscription. Now: cgroup quota + nproc, then a
#      RAM clamp. Results were never affected (every run is seed-deterministic);
#      only wall time was.
#   2. GATES ARE ADVISORY BY DEFAULT. They still EVALUATE and are RECORDED to
#      reports/GATE_VERDICTS.md (append-only, so nothing overwrites them). They
#      no longer halt the campaign. Rationale, already adjudicated: the C1
#      positive control PASSED (+0.104) and the channel is demonstrably audible
#      (identical architecture+content buys +4.9% under DR-Poisson, +13% at
#      retailer_broadcast), so the AR(1) rho=0.9 zero is an ECONOMIC null
#      (Raghunathan recovered in the invertible limit), not instrument failure.
#      The gate's job was to protect budget before that was known. It is known.
#      GATES=strict restores v1 halting behavior.
#   3. REFS RUN IN PARALLEL. v1 generated behavioral refs, trained everything,
#      then generated canonical refs -- two separate 1-3h single-core blocks
#      with 47 cores idle for the second. Now both launch at once in the
#      background; canonical finishes long before Phase D needs it.
#   4. SOFT-FAIL ON ARMS. A handful of failed jobs no longer kills an 8-hour
#      overnight run. They are retried once, recorded in reports/FAILED_ARMS.txt,
#      and the campaign continues. Infrastructure failures (setup, refs) still
#      stop hard -- those poison everything downstream.
#   5. NO `set -e`. v1 inherited a class of bash traps (heredocs, pipelines,
#      command substitutions) where an incidental non-zero exit kills the run.
#      Critical steps are checked explicitly instead.
#
# USE:
#   tmux new -s signal
#   bash auto_campaign2.sh 2>&1 | tee -a auto_campaign.log      # Ctrl-b d, sleep
#   bash auto_campaign2.sh status                               # progress, anytime
#
# ENV: NPROC[auto] INCLUDE_EXT[1] GATES[advisory|strict] SKIP_PILOT[0]
#      JOB_MB[700] AUTO_STOP[0 = leave pod running when done]
# =============================================================================
set -uo pipefail
cd "$(dirname "$(readlink -f "$0" 2>/dev/null || echo "$0")")" || exit 1
[[ -f agents/train_signal.py && -f sweep_all_hypotheses.sh ]] || {
  echo "ERROR: run from the BeerGame_Comm repo root."; exit 1; }
chmod +x setup_pod.sh sweep_all_hypotheses.sh 2>/dev/null

INCLUDE_EXT="${INCLUDE_EXT:-1}"; GATES="${GATES:-advisory}"; SKIP_PILOT="${SKIP_PILOT:-0}"
JOB_MB="${JOB_MB:-700}"; AUTO_STOP="${AUTO_STOP:-0}"
ST=auto_state; mkdir -p "$ST" reports snapshots results
export SIGNAL_CSVLOG=1        # publication learning curves: forced here, not left to the sweep default
PYBIN="python3"; [[ -x venv/bin/python ]] && PYBIN="venv/bin/python"

say()  { echo; echo "== [$(date +%H:%M:%S)] $*"; }
note() { printf '%s | %s\n' "$(date +%FT%T)" "$*" >> reports/CAMPAIGN_LOG.md; echo "   -> $*"; }
fatal(){ echo; echo "!! FATAL: $*"; note "FATAL: $*";
         echo "!! Everything finished so far is on disk; re-running resumes."; exit 1; }
mark() { touch "$ST/$1.ok"; }
done_already() { [[ -f "$ST/$1.ok" ]]; }

# ---------------------------------------------------------------- helpers
runlog_counts() {
  local f="$1" d s x
  d=$(grep -ac '\[done\]' "$f" 2>/dev/null); s=$(grep -ac '\[skip\]' "$f" 2>/dev/null)
  x=$(grep -ac '\[FAIL\]' "$f" 2>/dev/null); echo "${d:-0} ${s:-0} ${x:-0}"
}
detect_nproc() {   # cgroup quota (the container's REAL limit) -> nproc -> fallback
  local n q p
  n="$(nproc 2>/dev/null || echo 4)"
  if [[ -r /sys/fs/cgroup/cpu.max ]]; then                       # cgroup v2
    read -r q p < /sys/fs/cgroup/cpu.max 2>/dev/null
    [[ "${q:-max}" != max ]] && [[ "${p:-0}" =~ ^[0-9]+$ ]] && (( p > 0 )) && n=$(( q / p ))
  elif [[ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]]; then        # cgroup v1
    q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null)
    p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null)
    [[ "${q:-0}" =~ ^[0-9]+$ ]] && [[ "${p:-0}" =~ ^[0-9]+$ ]] && (( q > 0 && p > 0 )) && n=$(( q / p ))
  fi
  (( n < 1 )) && n=1
  local mem_mb nmem                                              # RAM clamp
  mem_mb="$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo 2>/dev/null)"
  if [[ "${mem_mb:-0}" =~ ^[0-9]+$ ]] && (( mem_mb > 0 )); then
    nmem=$(( mem_mb * 85 / 100 / JOB_MB )); (( nmem < 1 )) && nmem=1
    (( n > nmem )) && n=$nmem
  fi
  echo "$n"
}
refs_ok() {
  "$PYBIN" - "$1" <<'PY' 2>/dev/null
import json, sys
try: d = json.load(open(sys.argv[1]))
except Exception: sys.exit(1)
lams = {float(k) for k in d.get("rungs", {}).get("BAR_static", {})}
bl = d.get("meta", {}).get("bar_levels") or []
sys.exit(0 if lams == {6.,8.,10.,12.,14.,16.,18.,20.,22.} and len(set(bl)) >= 2 else 1)
PY
}
gen_refs() {   # gen_refs <logfile> [extra flags...]
  local log="$1"; shift
  PYTHONUNBUFFERED=1 "$PYBIN" scripts/baselines.py regime \
    --lambdas 6 8 10 12 14 16 18 20 22 --select-episodes 80 --eval-episodes 200 \
    --bar-per-echelon "$@" > "$log" 2>&1
}
behavioral_refs_live() {
  [[ -f results/baselines_regime_v2.behavioral.json ]] && \
    cp results/baselines_regime_v2.behavioral.json results/baselines_regime_v2.json
}
train_phase() {   # train_phase <PHASES> <expected arms>  -- soft-fail, retry once
  local ph="$1" exp="$2" d s x
  say "TRAIN '$ph' (expect $exp arms, NPROC=$NPROC)"
  echo "   live progress:  tail -f sweep_out/run.log     summary:  bash auto_campaign2.sh status"
  PHASES="$ph" bash sweep_all_hypotheses.sh >> reports/launch.log 2>&1
  read -r d s x <<< "$(runlog_counts sweep_out/run.log)"
  # record failures BEFORE any retry: sweep_out/run.log is rewritten per invocation,
  # so a successful retry would otherwise erase all evidence that arms ever failed.
  (( x > 0 )) && grep -a '\[FAIL\]' sweep_out/run.log | sed "s|^|$ph attempt1: |" >> reports/FAILED_ARMS.txt 2>/dev/null
  if (( d + s < exp )) || (( x > 0 )); then
    note "phase '$ph': done=$d skip=$s FAIL=$x (expected $exp) -- retrying once"
    PHASES="$ph" bash sweep_all_hypotheses.sh >> reports/launch.log 2>&1
    read -r d s x <<< "$(runlog_counts sweep_out/run.log)"
    (( x > 0 )) && grep -a '\[FAIL\]' sweep_out/run.log | sed "s|^|$ph attempt2: |" >> reports/FAILED_ARMS.txt 2>/dev/null
  fi
  note "phase '$ph': done=$d skip=$s FAIL=$x (expected $exp)"
  if (( x > 0 )); then
    echo "!! $x arm(s) still failing in '$ph' -> reports/FAILED_ARMS.txt. CONTINUING (unattended)."
  fi
}
snapshot() {
  tar czf "snapshots/snap_$(date +%m%d_%H%M).tgz" results/ sweep_out/ reports/ auto_state/ \
    >/dev/null 2>&1
  ls -1t snapshots/*.tgz 2>/dev/null | tail -n +3 | xargs -r rm -f      # keep 2 newest
}

# ---------------------------------------------------------------- status
if [[ "${1:-}" == "status" ]]; then
  echo "== SIGNAL campaign status =="
  for s in S1_setup S2_freeze S3_refs S4_calibrate S5_pilot S6_phaseA S7_gates \
           S8_behavioral S9_canonical S10_extract S11_analysis S12_archive; do
    printf "  %-14s %s\n" "$s" "$([[ -f $ST/$s.ok ]] && echo DONE || echo pending)"
  done
  echo "  sentinels (finished arms): $(ls weights_signal/.done_* 2>/dev/null | wc -l)"
  [[ -f sweep_out/run.log ]] && { read -r d s x <<< "$(runlog_counts sweep_out/run.log)"
    echo "  last invocation: done=$d skip=$s FAIL=$x"; }
  [[ -f "$ST/nproc" ]] && echo "  NPROC: $(cat "$ST/nproc")"
  w="$(pgrep -fc 'train_signal.py' 2>/dev/null)"; echo "  training now: ${w:-0} workers"
  [[ -f reports/GATE_VERDICTS.md ]] && { echo "  -- gate verdicts --"; sed 's/^/  /' reports/GATE_VERDICTS.md; }
  [[ -f reports/FAILED_ARMS.txt ]] && echo "  FAILED arms: $(wc -l < reports/FAILED_ARMS.txt)"
  exit 0
fi

# ================================================================ CAMPAIGN
[[ -n "${TMUX:-}" ]] || echo "!! WARNING: not inside tmux -- a dropped connection kills this. (tmux new -s signal)"
note "campaign start (GATES=$GATES INCLUDE_EXT=$INCLUDE_EXT)"

# ---- S1 setup ------------------------------------------------------------
if ! done_already S1_setup || [[ ! -x venv/bin/python ]]; then
  # NOTE: the venv check is load-bearing. Restoring an archive onto a FRESH pod brings the
  # markers with it but not venv/ -- trusting the marker alone would skip setup and run the
  # whole campaign against a torch-less system python.
  say "S1 setup (CPU torch wheel + self-tests). Do NOT set GPU=1."
  bash setup_pod.sh || fatal "setup_pod.sh failed"
  PYBIN="venv/bin/python"
  venv/bin/pip install -q matplotlib 2>&1 | tail -1
  "$PYBIN" - <<'PY' || fatal "a CUDA torch wheel is active -- rerun setup without GPU=1"
import torch, sys
print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
sys.exit(1 if (torch.cuda.is_available() and "+cpu" not in torch.__version__) else 0)
PY
  mark S1_setup
else say "S1 setup: done"; fi
[[ -x venv/bin/python ]] && PYBIN="venv/bin/python"

# ---- S2 freeze -----------------------------------------------------------
if ! done_already S2_freeze; then
  say "S2 instrument freeze manifest"
  sha256sum agents/signal_agent.py agents/train_signal.py agents/eval_signal.py \
    agents/signal_csvlog.py agents/topologies.py envs/beer_game_env.py envs/demand_randomization.py \
    scripts/demand_families.py scripts/comm_stats.py scripts/c1_stats.py scripts/prereg.py \
    scripts/baselines.py scripts/dp_optimum.py scripts/run_confirmatory_report.py \
    conf/config.yaml conf/agent/signal.yaml sweep_all_hypotheses.sh plot_curves.py \
    > results/FREEZE_MANIFEST_v1.2.txt 2>/dev/null
  "$PYBIN" scripts/prereg.py 2>/dev/null | grep -i sha256 >> results/FREEZE_MANIFEST_v1.2.txt
  note "freeze manifest: $(wc -l < results/FREEZE_MANIFEST_v1.2.txt) lines"
  mark S2_freeze
else say "S2 freeze: done"; fi

# ---- S3 refs (BOTH cost models, in parallel background) -------------------
BEH_PID=""; CAN_PID=""
behavioral_refs_live                       # restore the live json from the backup if present
if ! done_already S3_refs || ! refs_ok results/baselines_regime_v2.json; then
  say "S3 refs: behavioral + canonical IN PARALLEL (single-core each, 30-120 min)"
  if refs_ok results/baselines_regime_v2.json && [[ ! -f results/baselines_regime_v2.behavioral.json ]]; then
    cp results/baselines_regime_v2.json results/baselines_regime_v2.behavioral.json
    note "behavioral refs: adopted valid hand-placed json (no regeneration)"
  fi
  echo "   progress: reports/refs_behavioral.log / reports/refs_canonical.log"
  if refs_ok results/baselines_regime_v2.behavioral.json; then
    behavioral_refs_live; note "behavioral refs: reused"
  else
    ( gen_refs reports/refs_behavioral.log && \
      cp results/baselines_regime_v2.json results/baselines_regime_v2.behavioral.json ) &
    BEH_PID=$!
  fi
  if refs_ok results/baselines_regime_v2_canonical.json; then note "canonical refs: reused"
  else gen_refs reports/refs_canonical.log --penalty-at-retailer-only & CAN_PID=$!; fi

  if [[ -n "$BEH_PID" ]]; then
    echo "   waiting on behavioral refs (training cannot start without the ruler)..."
    wait "$BEH_PID" || fatal "behavioral refs generation failed (reports/refs_behavioral.log)"
  fi
  refs_ok results/baselines_regime_v2.json || fatal "behavioral refs invalid (9 lambdas + per-echelon BAR)"
  [[ -f results/baselines_regime_v2.behavioral.json ]] || \
    cp results/baselines_regime_v2.json results/baselines_regime_v2.behavioral.json
  note "behavioral refs OK (backup ensured)"
  mark S3_refs
else say "S3 refs: reusing validated behavioral refs"
  [[ -f results/baselines_regime_v2.behavioral.json ]] || \
    cp results/baselines_regime_v2.json results/baselines_regime_v2.behavioral.json
  refs_ok results/baselines_regime_v2_canonical.json || { gen_refs reports/refs_canonical.log --penalty-at-retailer-only & CAN_PID=$!; }
fi

# ---- S4 NPROC (container-aware) ------------------------------------------
if [[ -n "${NPROC:-}" ]]; then echo "$NPROC" > "$ST/nproc"
elif [[ ! -s "$ST/nproc" ]]; then detect_nproc > "$ST/nproc"; fi
NPROC="$(cat "$ST/nproc")"; export NPROC
mark S4_calibrate
say "workers: NPROC=$NPROC  (cgroup/nproc-derived; lscpu would report the HOST's cores)"
if (( NPROC > 64 )); then
  echo "!! NPROC=$NPROC looks like HOST topology leaking through the container."
  echo "!! Strongly recommended: rerun as   NPROC=<your pod's vCPU count> bash auto_campaign2.sh"
  note "WARNING: suspicious NPROC=$NPROC (host leak?) -- set NPROC explicitly"
fi
note "NPROC=$NPROC"
dr_out="$(DRYRUN=1 STAGE=train PHASES=core bash sweep_all_hypotheses.sh 2>&1)"
grep -q "jobs=435" <<< "$dr_out" && note "manifest OK: 29 configs x 15 seeds = 435" \
  || note "WARNING: DRYRUN did not report jobs=435 -- check SEEDS/PHASES overrides"

# ---- S5 pilot (background, advisory) -------------------------------------
if ! done_already S5_pilot && [[ "$SKIP_PILOT" != 1 ]]; then
  say "S5 audibility pilot -> background (advisory only)"
  ( "$PYBIN" agents/train_signal.py agent=signal agent.train_env=ar1 agent.ar1_rho=0.9 \
      agent.heldout_mode=ar1 agent.comm_topology=upstream_only agent.msg_content=dhat \
      seed=10 total_episodes=1500 agent.algorithm=pilot_aud_s10 > reports/pilot_train.log 2>&1
    pc=$(ls -1dt weights_signal/run_signal_*_pilot_aud_s10/signal_checkpoint_best.pt 2>/dev/null | head -1)
    [[ -n "$pc" ]] && "$PYBIN" agents/eval_signal.py --ckpt "$pc" --messages --episodes 40 \
      > reports/pilot_audibility.txt 2>&1 ) &
fi
mark S5_pilot

# ---- S6-S8 training ------------------------------------------------------
train_phase "A" 165;                       mark S6_phaseA; snapshot

# ---- S7 gates: EVALUATE + RECORD, never halt (unless GATES=strict) --------
if ! done_already S7_gates; then
  say "S7 gates: dump A + C1 positive control + futility (mode: $GATES)"
  behavioral_refs_live
  STAGE=dump bash sweep_all_hypotheses.sh > reports/dumpA_stage.log 2>&1
  mkdir -p results/signal_c1
  for s in $(seq 10 24); do
    [[ -f "results/signal_c1/seed${s}.json" ]] && continue
    ck=$(ls -1dt weights_signal/run_signal_*_dp_nocomm_s${s}/signal_checkpoint_best.pt 2>/dev/null | head -1)
    [[ -n "$ck" ]] && "$PYBIN" agents/eval_signal.py --ckpt "$ck" --dump-c1 results/signal_c1 \
      --episodes 200 >> reports/c1_dump.log 2>&1
  done
  "$PYBIN" scripts/run_confirmatory_report.py --signal-dir results/signal_c1 \
    --refs results/baselines_regime_v2.json > reports/gate_c1.txt 2>&1
  {
    echo "### $(date +%F\ %T) gate read (mode=$GATES)"
    "$PYBIN" - results/confirmatory_report.json <<'PY' 2>&1
import json, sys
try: c = json.load(open(sys.argv[1])).get("c1", {})
except Exception as e: print(f"- C1: unreadable ({e})"); sys.exit(0)
ci = next((v for k, v in c.items() if "gap" in k.lower() and "ci" in k.lower()
           and isinstance(v, (list, tuple)) and len(v) == 2), None)
if ci: print(f"- C1 Gap_Recovered CI=[{ci[0]:+.3f},{ci[1]:+.3f}] -> {'PASS' if ci[0] > 0 else 'FAIL'}")
else:  print(f"- C1 gap_mean={c.get('gap_mean')} (CI key absent)")
PY
    "$PYBIN" - sweep_out/h2 <<'PY' 2>&1
import sys
sys.path.insert(0, ".")
try:
    from scripts.comm_stats import load_cost_dir, value_of_sharing
    comm, noc = load_cost_dir(f"{sys.argv[1]}/comm"), load_cost_dir(f"{sys.argv[1]}/nocomm")
    if not (comm and noc): print("- futility: rho=0.9 dumps missing (not evaluable)")
    else:
        v = value_of_sharing(comm, noc, lambdas=[0.9])
        print(f"- futility V(rho=0.9)={v['v_cost_mean']:+.1f} ({v['v_cost_pct']:+.2f}%) "
              f"CI=[{v['v_cost_ci'][0]:.1f},{v['v_cost_ci'][1]:.1f}] n={v['n_seeds']} "
              f"TOST_p={v['tost_p']:.3g} equivalent={v['equivalent']}")
except Exception as e: print(f"- futility: error ({e})")
PY
  } > "$ST/gate_read.txt" 2>&1
  cat "$ST/gate_read.txt" | tee -a reports/GATE_VERDICTS.md
  if [[ "$GATES" == strict ]]; then
    # judge ONLY this read: GATE_VERDICTS.md is append-only, so grepping the file would let a
    # stale verdict from an earlier invocation halt a campaign that just passed.
    grep -q "C1 .*-> FAIL" "$ST/gate_read.txt" && fatal "C1 positive control failed (GATES=strict)"
    grep -q "equivalent=True" "$ST/gate_read.txt" && fatal "futility gate (GATES=strict)"
  else
    echo "   GATES=advisory: verdicts recorded in reports/GATE_VERDICTS.md; campaign continues."
    note "gates advisory -- see reports/DECISION_LOG.md for the standing rationale"
  fi
  STAGE=analyze bash sweep_all_hypotheses.sh > reports/analyze_A.txt 2>&1
  mark S7_gates
else say "S7 gates: done"; fi

train_phase "B Bnull C E" 225
[[ "$INCLUDE_EXT" == 1 ]] && train_phase "Bext" 60
mark S8_behavioral; snapshot

# ---- S9 canonical --------------------------------------------------------
if ! done_already S9_canonical; then
  say "S9 canonical block (refs were generated in parallel during training)"
  [[ -n "$CAN_PID" ]] && { echo "   waiting on canonical refs..."; wait "$CAN_PID" 2>/dev/null; }
  if ! refs_ok results/baselines_regime_v2_canonical.json; then
    note "canonical refs missing/invalid -> regenerating synchronously (one retry)"
    gen_refs reports/refs_canonical.log --penalty-at-retailer-only
  fi
  if refs_ok results/baselines_regime_v2_canonical.json; then
    "$PYBIN" scripts/baselines.py validate-canonical --lambdas 6 10 14 18 22 \
      > reports/validate_canonical.txt 2>&1
    note "canonical refs OK; validate-canonical -> reports/validate_canonical.txt"
    cp results/baselines_regime_v2_canonical.json results/baselines_regime_v2.json
    train_phase "D" 45
    [[ "$INCLUDE_EXT" == 1 ]] && train_phase "Dext" 45
    behavioral_refs_live
    cmp -s results/baselines_regime_v2.json results/baselines_regime_v2.behavioral.json \
      && note "behavioral refs restored" || fatal "refs restore failed -- do not analyze"
  else
    note "WARNING: canonical refs unavailable -> phases D/Dext SKIPPED (reports/refs_canonical.log)"
  fi
  mark S9_canonical; snapshot
else say "S9 canonical: done"; behavioral_refs_live; fi

# ---- S10 extraction ------------------------------------------------------
if ! done_already S10_extract; then
  say "S10 extraction: dump (~660 jobs) + probe (90)"
  behavioral_refs_live
  STAGE=dump  bash sweep_all_hypotheses.sh > reports/dump_stage.log 2>&1
  STAGE=probe bash sweep_all_hypotheses.sh > reports/probe_stage.log 2>&1
  for d in sweep_out/h2/comm sweep_out/h2/nocomm; do
    n=$(ls "$d"/seed*.json 2>/dev/null | wc -l); (( n == 15 )) || note "WARN: $d has $n/15 seed files"
  done
  note "fam dirs: $(ls -d sweep_out/fam/ar1r9_* 2>/dev/null | wc -l)/13  probe dirs: $(ls -d sweep_out/probes/iv_* 2>/dev/null | wc -l)/6"
  mark S10_extract
else say "S10 extraction: done"; fi

# ---- S11 analysis --------------------------------------------------------
if ! done_already S11_analysis; then
  say "S11 statistics + figures (also regenerable offline from the archive)"
  behavioral_refs_live
  STAGE=analyze bash sweep_all_hypotheses.sh > reports/analyze_FULL.txt 2>&1
  STAGE=plot    bash sweep_all_hypotheses.sh > reports/plot_stage.log 2>&1
  note "figures: $(ls sweep_out/figs/*.pdf 2>/dev/null | wc -l) PDFs"
  "$PYBIN" scripts/run_confirmatory_report.py --signal-dir results/signal_c1 \
    --refs results/baselines_regime_v2.json \
    --comm sweep_out/h1pois/nocomm sweep_out/h1pois/comm sweep_out/h1pois/rbroadcast \
    > reports/confirmatory_FULL.txt 2>&1
  mark S11_analysis
else say "S11 analysis: done"; fi

# ---- S12 archive ---------------------------------------------------------
wait 2>/dev/null                              # reap any background child (pilot eval) before archiving
say "S12 archive"
df -h . | tail -1
AR="SIGNAL_campaign_$(date +%F_%H%M).tgz"
# build a file LIST rather than expanding ~2700 glob paths onto one command line at the very
# last step of a 10-hour run (ARG_MAX risk grows with the number of arms).
{ find results sweep_out reports auto_state -type f 2>/dev/null
  find weights_signal -type f \( -name 'signal_checkpoint_best.pt' -o -name 'signal_checkpoint_budget*.pt' \
       -o -name 'metrics_*.csv' -o -name 'run_meta.json' -o -name '.done_*' \) 2>/dev/null
  ls -1 ./*.log 2>/dev/null
} > "$ST/archive.list"
note "archiving $(wc -l < "$ST/archive.list") files"
tar czf "$AR" -T "$ST/archive.list" 2>/dev/null
[[ -s "$AR" ]] || { echo "!! archive empty -- retrying without compression"; tar cf "${AR%.tgz}.tar" -T "$ST/archive.list"; AR="${AR%.tgz}.tar"; }
sha256sum "$AR" > ARCHIVE_HASH.txt
mark S12_archive
[[ -d /workspace ]] && cp "$AR" ARCHIVE_HASH.txt /workspace/ 2>/dev/null && \
  echo "   copy on the persistent volume: /workspace/$AR"
note "archive: $AR ($(du -h "$AR" | cut -f1))"

echo
echo "======================================================================"
echo "  CAMPAIGN COMPLETE  $(date +%F\ %T)"
echo "  arms finished : $(ls weights_signal/.done_* 2>/dev/null | wc -l)"
echo "  failed arms   : $([[ -f reports/FAILED_ARMS.txt ]] && wc -l < reports/FAILED_ARMS.txt || echo 0)"
echo "  archive       : $(pwd)/$AR   ($(du -h "$AR" 2>/dev/null | cut -f1))"
echo "  sha256        : $(cut -d' ' -f1 ARCHIVE_HASH.txt)"
echo
echo "  1. Download the .tgz + ARCHIVE_HASH.txt (RunPod GUI file browser)."
echo "  2. Verify on Windows:  certutil -hashfile <file> SHA256"
echo "  3. THEN stop the pod. All analysis regenerates offline from this archive."
echo "  Gate verdicts: reports/GATE_VERDICTS.md   Log: reports/CAMPAIGN_LOG.md"
echo "======================================================================"

if [[ "$AUTO_STOP" == 1 ]] && command -v runpodctl >/dev/null 2>&1 && [[ -n "${RUNPOD_POD_ID:-}" ]]; then
  echo "AUTO_STOP=1 -> stopping pod in 10 min (archive is on /workspace)."; sleep 600
  runpodctl stop pod "$RUNPOD_POD_ID"
fi