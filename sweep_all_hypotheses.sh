#!/usr/bin/env bash
# ============================================================================
# sweep_all_hypotheses.sh
# FINAL multi-seed, parallel SIGNAL sweep for the value-of-communication study,
# engineered for a RunPod CPU pod (Linux). Phase-structured per the design review.
# ----------------------------------------------------------------------------
# PILOT FIRST: run  sweep_comm_value.bat 8000  (single-seed, exploratory) and confirm
#   (a) positive_listening / message_weight_audit show non-trivial slopes for EVERY comm
#       arm (incl. learned, now that the gain is a learnable parameter), and
#   (b) direction matches theory (comm helps at rho=0.9, not at rho=0).
#   Do NOT scale a broken arm to 15 seeds.
#
# DESIGN (fractional; one axis per hypothesis around a reference cell). SEEDS=15
#   (H1 is a TOST equivalence claim -> needs power above the 10-seed difference floor,
#    plus buffer for a couple of non-converging runs).
#
#   PHASE  QUESTION            FIXED                                   SWEPT                              runs(x15)
#   A      H1,H2,H3            content=dhat, topology=neighbor,        regime in {dr_poisson, ar1_rho0,   150
#                              behavioral cost                          .3,.6,.9} x {comm, nocomm}
#   B      H4 geometry         content=dhat, rho=0.9, behavioral        topo in {no_neighbor,             75
#                                                                        retailer_broadcast, upstream_only,
#                                                                        downstream_only, manufacturer_broadcast}
#   Bnull  H4 null-of-null     content=dhat, rho=0.0, behavioral        topo in {retailer_broadcast,      30
#                                                                        no_neighbor}
#   C      H5 content ladder   topology=retailer_broadcast, rho=0.9     content in {ip, dhat_ip, learned} 45
#   D      H7 strategic        content=dhat_ip, topology=neighbor,      (beta,tau) in {(1,0),(0,0),(0,t*)}45
#                              CANONICAL cost, dr_poisson
#   ---- core total = 345 ----
#   Bext   H4 wider geometry   as B                                     topo in {skip, full,             60
#                                                                        link_top_only, link_bottom_only}
#   Dext   H7 x autocorrelation as D but rho=0.9 (canonical)            (beta,tau) as D                   45
#   ---- extended total = +105 ----
#
#   REUSE (do NOT retrain): H4's neighbor point = Phase A's ar1r9_neighbor; H4's nocomm baseline =
#   A's ar1r9_nocomm; H5's dhat point = Phase B's ar1r9_rbroadcast. These are analysis-time reuses;
#   the per-phase run counts above already exclude them (all trained configs are distinct).
#
#   H6 (belief-capacity substitution) is EXCLUDED: it needs a GRU<->CRAFT encoder_type swap that does
#   not exist in signal_agent.py. Sweeping a knob with no code is a mistake; build it as its own phase.
#
#   H7 caveat: the empirical best-response / measured-PoA probe is NOT yet in eval_signal.py. These
#   arms TRAIN the {coop, selfish, contract} policies; measuring whether contract reaches the tau*-
#   coordinated equilibrium needs that probe (build in parallel, does not block training).
#
# COST MODELS (two, do not mix in one invocation):
#   A,B,Bnull,C,Bext  -> BEHAVIORAL  (env.penalty_at_retailer_only=false, the default).
#   D,Dext            -> CANONICAL   (env.penalty_at_retailer_only=true; Clark-Scarf retailer-only penalty).
#   The BAR/CEILING refs (results/baselines_regime_v2.json) must be regenerated per cost model:
#     python scripts/baselines.py regime      (once with each cost model set).
#   Refs affect only the LOGGED Gap_Recovered, not checkpoint SELECTION (raw held-out cost), so training
#   is valid regardless -- but the reported gap needs the matching refs. RUN BEHAVIORAL AND CANONICAL
#   PHASES AS SEPARATE INVOCATIONS with the json regenerated between them.
#
# tau* (canonical cost): tau* = p - b_private = backorder_cost(1.0) - 0.0 = 1.0. Under canonical cost
#   non-retailer stages carry NO backorder term (b_private=0), a more extreme boundary than the
#   textbook b>0; confirm coordination still holds via
#     python scripts/coordination_theory.py    # check_link(p=1.0, h=0.5, b_private=0.0)
#   (The 9.5 in that script's self-test is an ILLUSTRATIVE p=10 case, NOT this env.)
#
# KEY OM CITATIONS (corrected):
#   H1 null: Raghunathan (2001, MS 47(4):605-610) order-stream invertibility under a followed policy;
#            Cui, Allon, Bassamboo & Van Mieghem (2015, MS 61(11):2803-2824) generalization + empirics
#            (V=0 under ~75% of ARMA(1,1) params).  [NOT Axsater-Rosling 1993 -- that is installation-
#            vs echelon-stock policy dominance, the wrong result for this null.]
#   H2:      Lee, So & Tang (2000, MS 46(5):626-643) value rises with demand autocorrelation.
#   H3:      Lee, Padmanabhan & Whang (1997, MS 43(4):546-558); Chen et al. (2000, MS 46(3):436-443).
#   H4:      design-level operationalization of Lee et al. (1997) distance-amplifies-distortion (say so).
#   H5:      Lee-So-Tang (dhat channel); Cachon & Fisher (2000, MS 46(8):1032-1048) (ip/VMI channel).
#   H7:      Cachon & Zipkin (1999, MS 45(7):936-953) coordinating transfers; Cachon & Lariviere (2001),
#            Ren et al. (2010) contract-induced truthful sharing.  RL context: Oroojlooyjadid et al.
#            (2022, M&SOM); Liu et al. (2024, POM); Mao et al. (2024, MS); Gong & Simchi-Levi (2024, MS).
#
# PARALLELISM: SIGNAL is CPU-bound and ~single-threaded per run. Throughput = many runs at once, ONE
#   BLAS thread each. We pin OMP/MKL/OpenBLAS/NUMEXPR=1 (else each job grabs all cores and effective
#   parallelism -> ~1) and launch nproc jobs via xargs -P. Resumable: a job is skipped if its checkpoint
#   already exists (spot-pod safe -- just relaunch).
#
# USAGE (RunPod; run inside tmux):
#   chmod +x sweep_all_hypotheses.sh
#   DRYRUN=1 ./sweep_all_hypotheses.sh                       # print plan, run nothing
#   NPROC=48 ./sweep_all_hypotheses.sh                       # BEHAVIORAL core (A B Bnull C) + ...
#   PHASES="A B Bnull C" NPROC=48 ./sweep_all_hypotheses.sh  # behavioral phases (regenerate refs first)
#   # ... regenerate baselines_regime_v2.json for CANONICAL cost, then:
#   PHASES="D Dext" NPROC=48 ./sweep_all_hypotheses.sh       # canonical phases
#   PHASES=all ./sweep_all_hypotheses.sh                     # everything (warns about mixed cost models)
#
# OVERRIDABLE ENV (defaults in brackets):
#   SEEDS[10..24 =15] EP[8000] PATIENCE[2000] HELDOUT_EPISODES[8] THREADS_PER_JOB[1]
#   NPROC[cores/threads] TAU_STAR[1.0] PHASES[core] DRYRUN[0] PYTHON[python] OUTROOT[./sweep_out]
# ============================================================================
set -euo pipefail

# ------------------------------------------------------------ 0. locate repo root
cd "$(dirname "$(readlink -f "$0" 2>/dev/null || echo "$0")")"
ROOT="$(pwd)"
[[ -f agents/train_signal.py ]] || { echo "ERROR: run from repo root (agents/train_signal.py missing)"; exit 1; }

# ------------------------------------------------------------ 1. config
PYTHON="${PYTHON:-python}"
[[ -x venv/bin/python ]] && PYTHON="venv/bin/python"
SEEDS="${SEEDS:-10 11 12 13 14 15 16 17 18 19 20 21 22 23 24}"     # 15 seeds (H1 TOST power)
EP="${EP:-8000}"
PATIENCE="${PATIENCE:-2000}"
HELDOUT_EPISODES="${HELDOUT_EPISODES:-8}"
THREADS_PER_JOB="${THREADS_PER_JOB:-1}"
TAU_STAR="${TAU_STAR:-1.0}"                                        # CANONICAL-cost tau* = p - b_private = 1.0 (see header)
PHASES="${PHASES:-core}"
STAGE="${STAGE:-train}"                                            # train | dump | analyze | all
DUMP_EPISODES="${DUMP_EPISODES:-200}"                             # episodes/key for the per-seed producers
AR1_MU="${AR1_MU:-12.0}"; AR1_SIGMA="${AR1_SIGMA:-3.0}"          # AR(1) eval params (match training)
DRYRUN="${DRYRUN:-0}"
OUTROOT="${OUTROOT:-$ROOT/sweep_out}"
LOGDIR="$OUTROOT/logs"
mkdir -p "$OUTROOT" "$LOGDIR"

HE=$((EP / 4)); (( HE > 200 )) && HE=200; (( HE < 1 )) && HE=1
CORES="$(nproc 2>/dev/null || echo 4)"
NPROC="${NPROC:-$(( CORES / THREADS_PER_JOB ))}"; (( NPROC < 1 )) && NPROC=1

export OMP_NUM_THREADS="$THREADS_PER_JOB" MKL_NUM_THREADS="$THREADS_PER_JOB"
export OPENBLAS_NUM_THREADS="$THREADS_PER_JOB" NUMEXPR_NUM_THREADS="$THREADS_PER_JOB"
export WANDB_MODE=disabled PYTHONUNBUFFERED=1

# ------------------------------------------------------------ 2. phase selection
SEL=""
for p in $PHASES; do
  case "$p" in
    all)      SEL="$SEL A B Bnull C D Bext Dext" ;;
    core)     SEL="$SEL A B Bnull C D" ;;
    extended) SEL="$SEL Bext Dext" ;;
    *)        SEL="$SEL $p" ;;
  esac
done
want() { [[ " $SEL " == *" $1 "* ]]; }

# ------------------------------------------------------------ 3. arm manifest
JOBS="$OUTROOT/jobs.tsv"; : > "$JOBS"
declare -A SEEN
declare -A SHORT=( [neighbor]=neighbor [skip]=skip [full]=full [retailer_broadcast]=rbroadcast
                   [manufacturer_broadcast]=mbroadcast [upstream_only]=upstream
                   [downstream_only]=downstream [no_neighbor]=noneighbor
                   [link_top_only]=linktop [link_bottom_only]=linkbot )
declare -A RHOTAG=( [0.0]=ar1r0 [0.3]=ar1r3 [0.6]=ar1r6 [0.9]=ar1r9 )

DP="agent.train_env=dr_poisson agent.heldout_mode=poisson"
AR()   { echo "agent.train_env=ar1 agent.ar1_rho=$1 agent.heldout_mode=ar1"; }
BEHAV="env.penalty_at_retailer_only=false"
CANON="env.penalty_at_retailer_only=true"
COMM() { echo "agent.use_comm=true agent.comm_topology=$1"; }
NOCOMM="agent.use_comm=false"

emit() {                                  # emit <algo_base> <hydra args...>  (dedup; expand over SEEDS)
  local algo="$1"; shift
  [[ -n "${SEEN[$algo]:-}" ]] && return
  SEEN[$algo]=1
  local s
  for s in $SEEDS; do printf '%s\t%s\t%s\n' "$algo" "$s" "$*" >> "$JOBS"; done
}

# ---- Phase A: core gradient (H1, H2, H3) -- dhat, neighbor, behavioral ----
if want A; then
  emit "dp_neighbor" "$BEHAV $DP $(COMM neighbor)"
  emit "dp_nocomm"   "$BEHAV $DP $NOCOMM"
  for r in 0.0 0.3 0.6 0.9; do t="${RHOTAG[$r]}"
    emit "${t}_neighbor" "$BEHAV $(AR "$r") $(COMM neighbor)"
    emit "${t}_nocomm"   "$BEHAV $(AR "$r") $NOCOMM"
  done
fi

# ---- Phase B: geometry core (H4) -- rho0.9, dhat, behavioral ----
if want B; then
  for topo in no_neighbor retailer_broadcast upstream_only downstream_only manufacturer_broadcast; do
    emit "ar1r9_${SHORT[$topo]}" "$BEHAV $(AR 0.9) $(COMM "$topo")"
  done
fi

# ---- Phase Bnull: geometry under the stationarity null (H4 placebo-of-placebo) -- rho0.0 ----
if want Bnull; then
  for topo in retailer_broadcast no_neighbor; do
    emit "ar1r0_${SHORT[$topo]}" "$BEHAV $(AR 0.0) $(COMM "$topo")"
  done
fi

# ---- Phase Bext: extended geometry (H4) -- rho0.9 ----
if want Bext; then
  for topo in skip full link_top_only link_bottom_only; do
    emit "ar1r9_${SHORT[$topo]}" "$BEHAV $(AR 0.9) $(COMM "$topo")"
  done
fi

# ---- Phase C: content ladder (H5) -- rho0.9, retailer_broadcast, behavioral ----
if want C; then
  emit "ar1r9_rbroadcast_ip"      "$BEHAV $(AR 0.9) $(COMM retailer_broadcast) agent.msg_content=ip"
  emit "ar1r9_rbroadcast_dhatip"  "$BEHAV $(AR 0.9) $(COMM retailer_broadcast) agent.msg_content=dhat_ip"
  emit "ar1r9_rbroadcast_learned" "$BEHAV $(AR 0.9) $(COMM retailer_broadcast) agent.msg_content=learned"
fi

# ---- Phase D: strategic core (H7) -- CANONICAL cost, dr_poisson, dhat_ip, neighbor ----
if want D; then
  DFIX="$CANON $DP $(COMM neighbor) agent.msg_content=dhat_ip"
  emit "cn_dp_coop"     "$DFIX agent.srdqn_beta=1.0 agent.tau=0.0"
  emit "cn_dp_selfish"  "$DFIX agent.srdqn_beta=0.0 agent.tau=0.0"
  emit "cn_dp_contract" "$DFIX agent.srdqn_beta=0.0 agent.tau=$TAU_STAR"
fi

# ---- Phase Dext: strategic x autocorrelation (H7) -- CANONICAL cost, rho0.9 ----
if want Dext; then
  DXFIX="$CANON $(AR 0.9) $(COMM neighbor) agent.msg_content=dhat_ip"
  emit "cn_ar1r9_coop"     "$DXFIX agent.srdqn_beta=1.0 agent.tau=0.0"
  emit "cn_ar1r9_selfish"  "$DXFIX agent.srdqn_beta=0.0 agent.tau=0.0"
  emit "cn_ar1r9_contract" "$DXFIX agent.srdqn_beta=0.0 agent.tau=$TAU_STAR"
fi

NJOBS=$(wc -l < "$JOBS"); NCFG=$(cut -f1 "$JOBS" | sort -u | wc -l)

echo "============================================================"
echo " SIGNAL final sweep | stage=$STAGE | phases: [$SEL]"
echo " configs=$NCFG x seeds=$(echo $SEEDS | wc -w) => jobs=$NJOBS | EP=$EP gate=$HE patience=$PATIENCE"
echo " cores=$CORES NPROC=$NPROC threads/job=$THREADS_PER_JOB | tau*=$TAU_STAR | out=$OUTROOT"
echo "============================================================"

# cost-model guardrails
if want D || want Dext; then
  echo "!! CANONICAL-COST phases selected (D/Dext):"
  echo "   - regenerate results/baselines_regime_v2.json with penalty_at_retailer_only=True first"
  echo "     (python scripts/baselines.py regime); refs drive Gap_Recovered, not selection."
  echo "   - tau* used = $TAU_STAR (= p - b_private, canonical). Confirm: python scripts/coordination_theory.py"
fi
if { want D || want Dext; } && { want A || want B || want Bnull || want Bext || want C; }; then
  echo "WARNING: BEHAVIORAL and CANONICAL phases selected together. One baselines_regime_v2.json cannot"
  echo "  match both cost models -> the logged Gap_Recovered for one group will be off (selection is fine)."
  echo "  RECOMMENDED: run behavioral and canonical phases as SEPARATE invocations, regen refs between."
fi

if [[ "$DRYRUN" == "1" ]]; then
  echo "-- unique configs --"; cut -f1 "$JOBS" | sort -u
  echo "-- DRYRUN: nothing executed --"; exit 0
fi

# ------------------------------------------------------------ 4. TRAIN (parallel, resumable)
run_one() {
  local algo seed args full log
  IFS=$'\t' read -r algo seed args <<< "$1"
  full="${algo}_s${seed}"
  if compgen -G "weights_signal/run_signal_*_${full}/signal_checkpoint_best.pt" > /dev/null 2>&1; then
    echo "[skip] $full"; return 0
  fi
  log="$LOGDIR/${full}.log"
  echo "[start] $full"
  # shellcheck disable=SC2086  -- $args is intentionally word-split into hydra tokens
  if $PYTHON agents/train_signal.py agent=signal seed="$seed" total_episodes="$EP" \
        agent.heldout_every="$HE" agent.heldout_episodes="$HELDOUT_EPISODES" agent.patience="$PATIENCE" \
        $args agent.algorithm="$full" > "$log" 2>&1; then
    echo "[done] $full"
  else
    echo "[FAIL] $full  (tail: $(tail -n1 "$log" 2>/dev/null))"
  fi
}
export -f run_one
export PYTHON EP HE HELDOUT_EPISODES PATIENCE LOGDIR

if [[ "$STAGE" == train || "$STAGE" == all ]]; then
  echo "== TRAIN: $NJOBS jobs, $NPROC parallel =="
  xargs -P "$NPROC" -I LINE bash -c 'run_one "$1"' _ LINE < "$JOBS" | tee "$OUTROOT/run.log"
  echo "== TRAIN complete: done=$(grep -cF '[done]' "$OUTROOT/run.log" || true)" \
       "skip=$(grep -cF '[skip]' "$OUTROOT/run.log" || true)" \
       "FAIL=$(grep -cF '[FAIL]' "$OUTROOT/run.log" || true) =="
  grep -F '[FAIL]' "$OUTROOT/run.log" || true
fi

# ------------------------------------------------------------ 5. DUMP: per-seed producers (H1/H2/H3)
# H2 is IN-REGIME: each rho-trained Phase-A model (ar1r{0,3,6,9}_{neighbor,nocomm}) is scored at ITS
# OWN rho (--dump-ar1 "<r>") into a per-rho subdir, then MERGED into one rho-keyed file per seed --
# the {seed:{rho:cost}} format prereg.h2_slope + comm_stats consume. (Scoring one model across all
# rhos would be a generalization curve, NOT the value-of-sharing gradient.) The _ferr siblings the
# same dumps write give H3 (upstream forecast-error delta) at rho0.9 for free.
# H7 (Phases D/Dext) is NOT dumped here: its analysis needs the best-response/measured-PoA probe,
# which does not exist in eval_signal.py yet (train the arms now; analyze when the probe lands).
H2ROOT="$OUTROOT/h2"; H1ROOT="$OUTROOT/h1pois"
if [[ "$STAGE" == dump || "$STAGE" == all ]]; then
  echo "== DUMP: H1 Poisson + H2/H1 AR1 (in-regime) + H3 forecast error =="
  DJOBS="$OUTROOT/dump_jobs.sh"; : > "$DJOBS"
  for r in 0.0 0.3 0.6 0.9; do t="${RHOTAG[$r]}"
    for pair in "neighbor comm" "nocomm nocomm"; do
      suf="${pair% *}"; grp="${pair#* }"
      for ck in weights_signal/run_signal_*_"${t}_${suf}"_s*/signal_checkpoint_best.pt; do
        [[ -e "$ck" ]] || continue
        echo "$PYTHON agents/eval_signal.py --ckpt '$ck' --dump-comm '$H2ROOT/${grp}_${t}' \
--dump-ar1 '$r' --ar1-mu $AR1_MU --ar1-sigma $AR1_SIGMA --dump-episodes $DUMP_EPISODES" >> "$DJOBS"
      done
    done
  done
  for pair in "neighbor comm" "nocomm nocomm"; do            # H1 Poisson (dr_poisson, default lambdas)
    suf="${pair% *}"; grp="${pair#* }"
    for ck in weights_signal/run_signal_*_"dp_${suf}"_s*/signal_checkpoint_best.pt; do
      [[ -e "$ck" ]] || continue
      echo "$PYTHON agents/eval_signal.py --ckpt '$ck' --dump-comm '$H1ROOT/${grp}' \
--dump-episodes $DUMP_EPISODES" >> "$DJOBS"
    done
  done
  echo "  $(wc -l < "$DJOBS") dump jobs, $NPROC parallel"
  xargs -P "$NPROC" -I LINE bash -c 'eval "$1"' _ LINE < "$DJOBS" > "$OUTROOT/dump.log" 2>&1 || true
  # merge per-rho AR1 subdirs -> rho-keyed per-seed files (the h2_slope format)
  "$PYTHON" - "$H2ROOT" <<'PY'
import sys, os, glob, json
root = sys.argv[1]
tags = {"0.0": "ar1r0", "0.3": "ar1r3", "0.6": "ar1r6", "0.9": "ar1r9"}
for grp in ("comm", "nocomm"):
    merged = {}
    for r, t in tags.items():
        for p in glob.glob(os.path.join(root, f"{grp}_{t}", "seed*.json")):
            b = os.path.basename(p)
            if b.endswith("_ferr.json") or b.endswith("_bw.json"):
                continue
            s = b[4:-5]                                        # seed<N>.json -> N
            with open(p) as f:
                merged.setdefault(s, {}).update(json.load(f))
    out = os.path.join(root, grp); os.makedirs(out, exist_ok=True)
    for s, kv in merged.items():
        with open(os.path.join(out, f"seed{s}.json"), "w") as f:
            json.dump(kv, f, indent=2)
    print(f"  merged {grp}: {len(merged)} seeds -> {out}")
PY
fi

# ------------------------------------------------------------ 6. ANALYZE: registered H1/H2/H3 statistics
if [[ "$STAGE" == analyze || "$STAGE" == all ]]; then
  echo "== ANALYZE: H1 (TOST) + H2 (registered slope) + H3 (forecast delta) =="
  "$PYTHON" - "$H2ROOT" "$H1ROOT" <<'PY'
import sys, json
sys.path.insert(0, ".")
from scripts.comm_stats import load_cost_dir, value_of_sharing, load_ferr_dir, forecast_delta
from scripts.prereg import h2_slope, h1_decision
h2root, h1root = sys.argv[1], sys.argv[2]
comm, nocomm = load_cost_dir(h2root + "/comm"), load_cost_dir(h2root + "/nocomm")
if comm and nocomm:
    h2 = h2_slope(comm, nocomm)
    print("  H2 slope (registered): mean=%.2f CI95=%s -> H2 holds: %s"
          % (h2["mean_slope"], h2["ci95"], h2["h2_holds"]))
    v0 = value_of_sharing(comm, nocomm, lambdas=[0.0])
    print("  H1 AR1 rho=0 : V=%.1f CI=%s equiv=%s -> %s"
          % (v0["v_cost_mean"], v0["v_cost_ci"], v0["equivalent"], h1_decision(v0)))
    try:
        cf, nf = load_ferr_dir(h2root + "/comm_ar1r9"), load_ferr_dir(h2root + "/nocomm_ar1r9")
        fd = forecast_delta(cf, nf)
        print("  H3 upstream forecast-error delta (rho0.9; >0 => comm cuts error):")
        for a, st in fd["per_echelon"].items():
            print("      %-13s delta=%+.3f CI=%s n=%d" % (a, st["delta_mean"], st["delta_ci"], st["n"]))
    except Exception as e:
        print("  H3: skipped (%s)" % e)
else:
    print("  (no merged H2 dumps found -- run STAGE=dump first)")
try:
    cp, ncp = load_cost_dir(h1root + "/comm"), load_cost_dir(h1root + "/nocomm")
    if cp and ncp:
        vp = value_of_sharing(cp, ncp)
        print("  H1 Poisson  : V=%.1f CI=%s equiv=%s -> %s"
              % (vp["v_cost_mean"], vp["v_cost_ci"], vp["equivalent"], h1_decision(vp)))
except Exception as e:
    print("  H1 Poisson: skipped (%s)" % e)
print("  (H4 geometry / H5 content: same producers on the rho0.9 topology/content arms + "
      "comm_stats.value_of_sharing per arm with Holm across the family; H7 needs the PoA probe.)")
PY
fi

echo "ALL SELECTED STAGES COMPLETE ($(date))"
