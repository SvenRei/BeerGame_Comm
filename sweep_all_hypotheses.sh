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
#   A      H1,H2,H3 (+S1)      content=dhat, topology=upstream_only,   regime in {dr_poisson, ar1_rho0,   165
#                              behavioral cost                          .3,.6,.9} x {comm, nocomm}
#                              [upstream_only = the REGISTERED primary  + dp_rbroadcast (S1 sensitivity:
#                               P2 topology (prereg v1.1); neighbor       max-favorable geometry under the
#                               moved to Phase B as a geometry point]     Poisson null)
#   B      H4 geometry         content=dhat, rho=0.9, behavioral        topo in {no_neighbor,             75
#                                                                        retailer_broadcast, neighbor,
#                                                                        downstream_only, manufacturer_broadcast}
#   Bnull  H4 null-of-null     content=dhat, rho=0.0, behavioral        topo in {retailer_broadcast,      30
#                                                                        no_neighbor}
#   C      H5 content ladder   topology=retailer_broadcast, rho=0.9     content in {ip, dhat_ip,          60
#                              (+ D1 data-vs-forecast: raw vs dhat)      learned, raw}
#   D      H7 strategic        content=dhat_ip, topology=neighbor,      (beta,tau) in {(1,0),(0,0),(0,t*)}45
#                              CANONICAL cost, dr_poisson
#   E      F_INCENTIVE         content=dhat, topology=upstream_only,    beta in {0.0, 0.5} x               60
#          V(beta) cheap-talk  rho=0.9, BEHAVIORAL cost, tau=0          {comm, nocomm-at-MATCHED-beta}
#                              [beta=1.0 point REUSED from Phase A: ar1r9_upstream / ar1r9_nocomm]
#   ---- core total = 435 ----
#   Bext   H4 wider geometry   as B                                     topo in {skip, full,             60
#                                                                        link_top_only, link_bottom_only}
#   Dext   H7 x autocorrelation as D but rho=0.9 (canonical)            (beta,tau) as D                   45
#   ---- extended total = +105 (540) ----
#
#   REUSE (do NOT retrain): H4's upstream point = Phase A's ar1r9_upstream; H4's nocomm baseline =
#   A's ar1r9_nocomm; H5's dhat point = Phase B's ar1r9_rbroadcast; E's beta=1.0 pair = Phase A's
#   ar1r9_upstream / ar1r9_nocomm. Analysis-time reuses; per-phase counts exclude them.
#
#   SUBSTITUTION CURVE (registered exploratory): every train run also snapshots the best-so-far
#   checkpoint at agent.budget_milestones=$MILESTONES episodes (signal_checkpoint_budget{M}.pt =
#   deployable-at-budget-M). STAGE=dump scores the dp and ar1r9 primary pairs at each milestone;
#   STAGE=analyze fits the per-seed V-vs-log2(budget) slope (scripts/comm_stats.py curve).
#
#   INTERVENTION / CONTENT-ATTRIBUTION GATE (registered validity gate): STAGE=probe runs the do(m)
#   message-intervention probe (honest/shuffled/cross/zeroed) per seed on the V-claiming arms and
#   dumps seed{S}_iv.json; cross-seed gate: scripts/comm_stats.py interventions --dir DIR. A
#   positive V is attributed to CONTENT only if the cross-seed delta(shuffled) CI excludes 0.
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
#   STAGE=dump    ./sweep_all_hypotheses.sh                  # per-seed producers (needs checkpoints)
#   STAGE=probe   ./sweep_all_hypotheses.sh                  # do(m) intervention dumps (V-claiming arms)
#   STAGE=analyze ./sweep_all_hypotheses.sh                  # registered statistics
#   STAGE=plot    ./sweep_all_hypotheses.sh                  # seed-aggregated learning-curve PDFs -> sweep_out/figs
#   STAGE=calibrate ./sweep_all_hypotheses.sh                # 2-min NPROC throughput probe (optional)
#
# OVERRIDABLE ENV (defaults in brackets):
#   SEEDS[10..24 =15] EP[8000] PATIENCE[2000] HELDOUT_EPISODES[8] THREADS_PER_JOB[1]
#   NPROC[min(physical cores, RAM/JOB_MB)] JOB_MB[700 = measured per-job RSS high-water]
#   SHUFFLE[1 = randomize job order; result-invariant, CRN is by seed value] PIN[0 = taskset per slot]
#   TAU_STAR[1.0] PHASES[core] STAGE[train|dump|probe|analyze|plot|calibrate|all] DRYRUN[0] PYTHON[python]
#   SIGNAL_CSVLOG[1 = write per-run metrics_heldout.csv / metrics_update.csv / run_meta.json into
#                    each run_dir; the learning-curve artifact. Set 0 to skip. Pure observation.]
#   OUTROOT[./sweep_out] MILESTONES[[1000,2000,4000,8000]] DUMP_EPISODES[200]
#   NOTE: agent.budget_milestones now EXISTS in conf/agent/signal.yaml, so the plain (non-+)
#   override below is valid Hydra; RESUME is sentinel-based (weights_signal/.done_<arm>_s<seed>),
#   because a best-checkpoint alone does NOT mean the run completed (spot-pod preemption).
#   PROBE_EPISODES[40] PROBE_ARMS[ar1r9_upstream ar1r9_rbroadcast ar1r9_rbroadcast_learned
#                                 ar1r9_rbroadcast_raw ar1r9_beta0_upstream ar1r9_beta05_upstream]
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
STAGE="${STAGE:-train}"                                            # train | dump | probe | analyze | all
DUMP_EPISODES="${DUMP_EPISODES:-200}"                             # episodes/key for the per-seed producers
AR1_MU="${AR1_MU:-12.0}"; AR1_SIGMA="${AR1_SIGMA:-3.0}"          # AR(1) eval params (match training)
MILESTONES="${MILESTONES:-[1000,2000,4000,8000]}"                # REGISTERED substitution-curve grid
PROBE_EPISODES="${PROBE_EPISODES:-40}"                            # episodes/seed for the do(m) probe
PROBE_ARMS="${PROBE_ARMS:-ar1r9_upstream ar1r9_rbroadcast ar1r9_rbroadcast_learned ar1r9_rbroadcast_raw ar1r9_beta0_upstream ar1r9_beta05_upstream}"
SHUFFLE="${SHUFFLE:-1}"                                           # randomize job order (straggler hedge)
PIN="${PIN:-0}"                                                    # 1 = taskset each worker to a core slot
JOB_MB="${JOB_MB:-700}"                                            # measured per-job RSS high-water (~685 MB)
DRYRUN="${DRYRUN:-0}"
OUTROOT="${OUTROOT:-$ROOT/sweep_out}"
LOGDIR="$OUTROOT/logs"
mkdir -p "$OUTROOT" "$LOGDIR" weights_signal

# integer sanity (fail fast with ONE message instead of N identical hydra errors)
for _v in EP PATIENCE HELDOUT_EPISODES THREADS_PER_JOB DUMP_EPISODES PROBE_EPISODES JOB_MB; do
  [[ "${!_v}" =~ ^[0-9]+$ ]] || { echo "ERROR: $_v must be a non-negative integer (got '${!_v}')"; exit 1; }
done

HE=$((EP / 4)); (( HE > 200 )) && HE=200; (( HE < 1 )) && HE=1

# PHYSICAL cores, not SMT threads: single-threaded numeric jobs gain ~0-20% from HT at 2x RAM
# and scheduler churn -- start at physical, raise via NPROC/STAGE=calibrate if measured worthwhile.
# (|| true inside the substitutions: lscpu/grep failures must not trip set -e/pipefail.)
PHYS_CORES="$(lscpu -b -p=Core,Socket 2>/dev/null | grep -v '^#' | sort -u | wc -l || true)"
[[ "$PHYS_CORES" =~ ^[0-9]+$ ]] && (( PHYS_CORES >= 1 )) || PHYS_CORES="$(nproc 2>/dev/null || echo 4)"
CORES="$PHYS_CORES"
NPROC="${NPROC:-$(( CORES / THREADS_PER_JOB ))}"; (( NPROC < 1 )) && NPROC=1
# RAM law: never launch more workers than memory sustains (OOM-killed jobs would otherwise
# leave partial checkpoints; see sentinel-resume). Skipped when MemAvailable is unreadable.
MEM_MB="$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo 2>/dev/null || true)"
if [[ "$MEM_MB" =~ ^[0-9]+$ ]] && (( MEM_MB > 0 )); then
  NPROC_MEM=$(( MEM_MB * 85 / 100 / JOB_MB )); (( NPROC_MEM < 1 )) && NPROC_MEM=1
  if (( NPROC > NPROC_MEM )); then
    echo "NPROC clamped ${NPROC} -> ${NPROC_MEM} by RAM (${MEM_MB} MB avail @ ${JOB_MB} MB/job)"
    NPROC=$NPROC_MEM
  fi
fi
# does this xargs support per-slot vars (findutils >= 4.7)? needed only for PIN=1 core-pinning.
if xargs --process-slot-var=XSLOT true </dev/null >/dev/null 2>&1; then
  SLOTVAR=(--process-slot-var=XSLOT)
else
  SLOTVAR=(); [[ "$PIN" == 1 ]] && echo "WARN: xargs lacks --process-slot-var; PIN=1 ignored"
fi

export OMP_NUM_THREADS="$THREADS_PER_JOB" MKL_NUM_THREADS="$THREADS_PER_JOB"
export OPENBLAS_NUM_THREADS="$THREADS_PER_JOB" NUMEXPR_NUM_THREADS="$THREADS_PER_JOB"
export MALLOC_ARENA_MAX=2                                          # curb glibc arena bloat across N procs
export WANDB_MODE=disabled PYTHONUNBUFFERED=1
# SCIENTIFIC SCALAR LOGGER: on by default for the air-gapped sweep this was built for (W&B is
# disabled above, so the per-run metrics_*.csv IS the learning-curve artifact). Pure observation,
# byte-identical training (see agents/signal_csvlog.py). Override with SIGNAL_CSVLOG=0 to skip it.
export SIGNAL_CSVLOG="${SIGNAL_CSVLOG:-1}"

# ------------------------------------------------------------ 1b. CALIBRATE (optional): measure the
# NPROC knee ON THIS POD. Runs a fixed 60-episode arm at NPROC in {physical, 1.5x physical}
# (RAM-capped) and reports aggregate episodes/second; pick the winner, export NPROC, launch.
if [[ "$STAGE" == calibrate ]]; then
  _cands="$CORES $(( CORES * 3 / 2 ))"
  [[ -n "${NPROC_MEM:-}" ]] && _cands="$(for c in $_cands; do (( c > NPROC_MEM )) && c=$NPROC_MEM; echo "$c"; done)"
  _cands="$(echo "$_cands" | tr ' ' '\n' | sort -un | tr '\n' ' ')"
  echo "== CALIBRATE: candidates NPROC in {${_cands% }} (60 eps/job, gate suppressed) =="
  [[ "$DRYRUN" == "1" ]] && { echo "-- DRYRUN: nothing executed --"; exit 0; }
  for np in $_cands; do
    t0=$(date +%s)
    seq 1 "$np" | xargs -d '\n' -P "$np" -I S bash -c \
      "$PYTHON agents/train_signal.py agent=signal seed=\$((900+S)) total_episodes=60 \
       agent.heldout_every=60 agent.heldout_episodes=1 agent.patience=0 \
       agent.algorithm=cal_sS >/dev/null 2>&1 || true"
    dt=$(( $(date +%s) - t0 )); (( dt < 1 )) && dt=1
    echo "  NPROC=$np : $(( np * 60 / dt )) eps/s aggregate  (${dt}s wall)"
  done
  rm -rf weights_signal/run_signal_*_cal_s* 2>/dev/null || true
  exit 0
fi

# ------------------------------------------------------------ 2. phase selection
SEL=""
for p in $PHASES; do
  case "$p" in
    all)      SEL="$SEL A B Bnull C D E Bext Dext" ;;
    core)     SEL="$SEL A B Bnull C D E" ;;
    core2)    SEL="$SEL A2 B2 C2 D2 E2 F2" ;;                     # v1.3 content study (510 runs)
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

# ---- Phase A: core gradient (H1, H2, H3) -- dhat, upstream_only, behavioral ----
# upstream_only is the REGISTERED P1_C1/P2_H1 primary topology (prereg v1.1): the Lee-correct
# hop-by-hop VMI direction, the geometry the analytical mechanism speaks about. The dp_rbroadcast
# arm is the REGISTERED S1 sensitivity: if the Poisson null (P1) holds even under the maximally
# favorable clean-signal broadcast, the null is decisive, not a weak-instrument artifact.
if want A; then
  emit "dp_upstream"   "$BEHAV $DP $(COMM upstream_only)"
  emit "dp_nocomm"     "$BEHAV $DP $NOCOMM"
  emit "dp_rbroadcast" "$BEHAV $DP $(COMM retailer_broadcast)"
  for r in 0.0 0.3 0.6 0.9; do t="${RHOTAG[$r]}"
    emit "${t}_upstream" "$BEHAV $(AR "$r") $(COMM upstream_only)"
    emit "${t}_nocomm"   "$BEHAV $(AR "$r") $NOCOMM"
  done
fi

# ---- Phase B: geometry core (H4) -- rho0.9, dhat, behavioral ----
# (neighbor sits here now that upstream_only carries the Phase-A primary; the ar1r9 upstream
#  point and the nocomm baseline are REUSED from Phase A at analysis time.)
if want B; then
  for topo in no_neighbor retailer_broadcast neighbor downstream_only manufacturer_broadcast; do
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

# ---- Phase C: content ladder (H5 + D1) -- rho0.9, retailer_broadcast, behavioral ----
# raw = UNPROCESSED last demand (classical POS-data sharing, Cachon-Fisher 2000); dhat (reused
# from Phase B's ar1r9_rbroadcast) = the GRU FORECAST (Aviv 2001/2007). Their REGISTERED contrast
# is D1 (data vs forecast); prediction: dhat weakly cheaper (the forecast pre-filters noise).
if want C; then
  emit "ar1r9_rbroadcast_ip"      "$BEHAV $(AR 0.9) $(COMM retailer_broadcast) agent.msg_content=ip"
  emit "ar1r9_rbroadcast_dhatip"  "$BEHAV $(AR 0.9) $(COMM retailer_broadcast) agent.msg_content=dhat_ip"
  emit "ar1r9_rbroadcast_learned" "$BEHAV $(AR 0.9) $(COMM retailer_broadcast) agent.msg_content=learned"
  emit "ar1r9_rbroadcast_raw"     "$BEHAV $(AR 0.9) $(COMM retailer_broadcast) agent.msg_content=raw"
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

# ---- Phase E: incentive x communication (F_INCENTIVE) -- BEHAVIORAL, rho0.9, dhat, upstream ----
# The Crawford-Sobel question in RL form: does the VALUE of the dhat channel survive as sender
# incentives misalign (beta 1.0 -> 0.5 -> 0.0)? Design registered in prereg v1.1 F_INCENTIVE:
#   * V(beta) = cost(nocomm@beta) - cost(comm@beta), BOTH arms trained at the SAME beta -- the
#     matched-beta baseline isolates communication WITHIN each incentive regime (an unmatched
#     baseline would confound incentive effects with channel effects).
#   * V is ALWAYS measured in TEAM cost units (eval sums all stages' realized costs regardless of
#     the training beta), so V(beta) is comparable across the grid.
#   * dhat ONLY: dhat is self-verifying cheap talk (the sender's own S-head consumes its d_hat, so
#     misreporting is self-punishing; the strategic margin is receiver TRUST). The learned rung is
#     EXCLUDED by design -- under DIAL it is trained by the receivers' gradients (delegated
#     communication, not strategic sending), so a beta-grid on it would not test cheap talk.
#   * beta=1.0 point REUSED from Phase A (ar1r9_upstream / ar1r9_nocomm); tau=0 throughout
#     (the contract axis is Phase D's question, not this one).
# DISCLOSED LIMITATION: checkpoint selection gates on held-out TEAM cost in ALL arms (the shared
# instrument). Under beta<1 that criterion is not the agents' own objective, so absolute PoA
# levels are selected-optimistic; V(beta) contrasts two arms under the IDENTICAL rule, so the
# selection bias largely cancels in the registered quantity. State this in the paper.
if want E; then
  EFIX="$BEHAV $(AR 0.9) agent.msg_content=dhat agent.tau=0.0"
  emit "ar1r9_beta0_upstream"  "$EFIX $(COMM upstream_only) agent.srdqn_beta=0.0"
  emit "ar1r9_beta0_nocomm"    "$EFIX $NOCOMM agent.srdqn_beta=0.0"
  emit "ar1r9_beta05_upstream" "$EFIX $(COMM upstream_only) agent.srdqn_beta=0.5"
  emit "ar1r9_beta05_nocomm"   "$EFIX $NOCOMM agent.srdqn_beta=0.5"
fi

# ============================ v1.3 CONTENT STUDY (prereg v1.3; 34 arms x 15 = 510) ============
# Names deliberately REUSE Study-1 arms where the config is identical (dp_nocomm, dp_rbroadcast,
# ar1r9_nocomm, ar1r9_rbroadcast[+_raw,_learned], ar1r9_upstream, ar1r9_neighbor, ar1r{0,3,6}_
# nocomm, ar1r0_rbroadcast): those cells ARE the built-in replication of Study 1.
# ---- A2: DP crossover (P1'). oracle = per-episode true lambda, registered as a BOUND. ----
if want A2; then
  emit "dp_nocomm"            "$BEHAV $DP $NOCOMM"
  emit "dp_rbroadcast"        "$BEHAV $DP $(COMM retailer_broadcast)"
  emit "dp_rbroadcast_raw"    "$BEHAV $DP $(COMM retailer_broadcast) agent.msg_content=raw"
  emit "dp_upstream_dhat"     "$BEHAV $DP $(COMM upstream_only)"
  emit "dp_rbroadcast_true_lambda" "$BEHAV $DP $(COMM retailer_broadcast) agent.msg_content=true_lambda"
fi
# ---- B2: the rho0.9 content ladder @ retailer_broadcast (H-REP + ladder replication). ----
if want B2; then
  emit "ar1r9_nocomm"             "$BEHAV $(AR 0.9) $NOCOMM"
  emit "ar1r9_rbroadcast"         "$BEHAV $(AR 0.9) $(COMM retailer_broadcast)"
  emit "ar1r9_rbroadcast_raw"     "$BEHAV $(AR 0.9) $(COMM retailer_broadcast) agent.msg_content=raw"
  emit "ar1r9_rbroadcast_eps"     "$BEHAV $(AR 0.9) $(COMM retailer_broadcast) agent.msg_content=eps"
  emit "ar1r9_rbroadcast_condmean" "$BEHAV $(AR 0.9) $(COMM retailer_broadcast) agent.msg_content=condmean"
  emit "ar1r9_rbroadcast_learned" "$BEHAV $(AR 0.9) $(COMM retailer_broadcast) agent.msg_content=learned"
fi
# ---- C2: content x topology (H-TOP; per-link EDI vs source access) + the informative placebo. ----
if want C2; then   # H-SOURCE (secondary): source access vs per-link local sharing + informative placebo
  emit "ar1r9_upstream_raw"    "$BEHAV $(AR 0.9) $(COMM upstream_only) agent.msg_content=raw"
  emit "ar1r9_downstream_raw"  "$BEHAV $(AR 0.9) $(COMM downstream_only) agent.msg_content=raw"
fi
# ---- D2: the two-curve rho grid (P2' raw restoration + H2-dhat replication + D1(rho)). ----
if want D2; then
  for r in 0.0 0.3 0.6; do t="${RHOTAG[$r]}"
    emit "${t}_nocomm"         "$BEHAV $(AR "$r") $NOCOMM"
    emit "${t}_rbroadcast"     "$BEHAV $(AR "$r") $(COMM retailer_broadcast)"
    emit "${t}_rbroadcast_raw" "$BEHAV $(AR "$r") $(COMM retailer_broadcast) agent.msg_content=raw"
  done
fi
# ---- E2: the censoring manipulation (H-CENS). Each o_max level gets its OWN nocomm (MDP changes;
#      the paired V differences the physical-capacity channel out to first order). ----
if want E2; then   # P2: OBSERVATION-side order garbling (physics fixed; Blackwell-nested clips)
  for c in 12 20; do
    emit "ar1r9_clip${c}_nocomm"         "$BEHAV $(AR 0.9) $NOCOMM env.obs_order_clip=${c}"
    emit "ar1r9_clip${c}_rbroadcast_raw" "$BEHAV $(AR 0.9) $(COMM retailer_broadcast) agent.msg_content=raw env.obs_order_clip=${c}"
  done
fi
# ---- G (PENDING): recurrent-QMIX sign-concordance set (editor Q1(a)); emitted once the QMIX
#      trainer spike lands. Cells: dp {nocomm, rb_dhat, rb_raw}; ar1r9 {nocomm, rb_dhat, rb_raw};
#      clip12 {nocomm, rb_raw} (+clip20 pair if resources). ----
# ---- F2: the timeliness ladder (H-TIME): real-time feed vs batch reporting. ----
if want F2; then
  emit "ar1r9_rbroadcast_rawlag1" "$BEHAV $(AR 0.9) $(COMM retailer_broadcast) agent.msg_content=raw_lag1"
  emit "ar1r9_rbroadcast_rawlag2" "$BEHAV $(AR 0.9) $(COMM retailer_broadcast) agent.msg_content=raw_lag2"
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

# ------------------------------------------------------------ 3b. checkpoint resolver
# NEWEST checkpoint for (arm, seed): sentinel-based resume can legitimately leave one STALE
# run dir (incomplete, but with a best.pt) next to the completed retrain -- the dump/probe
# stages must then score the completed one, and never race two writers onto one seed file.
latest_ck() {   # latest_ck <arm_tag> <seed> <filename>  -> newest matching path (or nothing)
  ls -1dt weights_signal/run_signal_*_"$1"_s"$2"/"$3" 2>/dev/null | head -1 || true
}

# ------------------------------------------------------------ 4. TRAIN (parallel, resumable)
# RESUME SEMANTICS: a run counts as complete ONLY when its sentinel weights_signal/.done_<full>
# exists (written after the trainer's final line). A best-checkpoint alone means "passed >=1
# gate before dying" -- the old skip-on-checkpoint rule silently froze preempted arms at a
# 200-episode policy. To mark a legacy run complete by hand: touch weights_signal/.done_<full>.
run_one() {
  local algo seed args full log done_f pin
  IFS=$'\t' read -r algo seed args <<< "$1"
  full="${algo}_s${seed}"
  done_f="weights_signal/.done_${full}"
  if [[ -f "$done_f" ]]; then
    echo "[skip] $full (done)"; return 0
  fi
  if compgen -G "weights_signal/run_signal_*_${full}/signal_checkpoint_best.pt" > /dev/null 2>&1; then
    echo "[resume-warn] $full: checkpoint exists but no completion sentinel -> retraining"
    echo "              (fresh run dir; stale dir kept, dump/probe pick the NEWEST via latest_ck)"
  fi
  pin=""
  if [[ "${PIN:-0}" == 1 ]] && command -v taskset >/dev/null 2>&1; then
    pin="taskset -c $(( ${XSLOT:-0} % ${CORES:-1} ))"
  fi
  log="$LOGDIR/${full}.log"
  echo "[start] $full"
  # shellcheck disable=SC2086  -- $args / $pin are intentionally word-split
  if $pin $PYTHON agents/train_signal.py agent=signal seed="$seed" total_episodes="$EP" \
        agent.heldout_every="$HE" agent.heldout_episodes="$HELDOUT_EPISODES" agent.patience="$PATIENCE" \
        "agent.budget_milestones=$MILESTONES" \
        $args agent.algorithm="$full" > "$log" 2>&1; then
    touch "$done_f"
    echo "[done] $full"
  else
    echo "[FAIL] $full  (tail: $(tail -n1 "$log" 2>/dev/null))"
  fi
}
export -f run_one
export PYTHON EP HE HELDOUT_EPISODES PATIENCE LOGDIR MILESTONES CORES PIN

if [[ "$STAGE" == train || "$STAGE" == all ]]; then
  echo "== TRAIN: $NJOBS jobs, $NPROC parallel =="
  # shuffle hedges stragglers (durations vary ~2x via patience); provably result-invariant.
  [[ "$SHUFFLE" == 1 ]] && command -v shuf >/dev/null 2>&1 && shuf "$JOBS" -o "$JOBS"
  # -d '\n': disable xargs quote-parsing -- one malformed line then fails ALONE instead of
  # aborting the whole stage (verified: default xargs drops all remaining jobs on an unmatched quote).
  xargs -d '\n' "${SLOTVAR[@]}" -P "$NPROC" -I LINE bash -c 'run_one "$1"' _ LINE < "$JOBS" | tee "$OUTROOT/run.log"
  echo "== TRAIN complete: done=$(grep -cF '[done]' "$OUTROOT/run.log" || true)" \
       "skip=$(grep -cF '[skip]' "$OUTROOT/run.log" || true)" \
       "FAIL=$(grep -cF '[FAIL]' "$OUTROOT/run.log" || true) =="
  grep -F '[FAIL]' "$OUTROOT/run.log" || true
fi

# ------------------------------------------------------------ 5. DUMP: per-seed producers (H1/H2/H3)
# H2 is IN-REGIME: each rho-trained Phase-A model (ar1r{0,3,6,9}_{upstream,nocomm}) is scored at ITS
# OWN rho (--dump-ar1 "<r>") into a per-rho subdir, then MERGED into one rho-keyed file per seed --
# the {seed:{rho:cost}} format prereg.h2_slope + comm_stats consume. (Scoring one model across all
# rhos would be a generalization curve, NOT the value-of-sharing gradient.) The _ferr siblings the
# same dumps write give H3 (upstream forecast-error delta) at rho0.9 for free.
# H7 (Phases D/Dext) is NOT dumped here: its analysis needs the best-response/measured-PoA probe,
# which does not exist in eval_signal.py yet (train the arms now; analyze when the probe lands).
H2ROOT="$OUTROOT/h2"; H1ROOT="$OUTROOT/h1pois"; CURVROOT="$OUTROOT/curve"; EROOT="$OUTROOT/incentive"
FAMROOT="$OUTROOT/fam"          # F_GEOMETRY / F_CONTENT member dumps (registered Holm families)
MLIST="$(echo "$MILESTONES" | tr -d '[] ' | tr ',' ' ')"
if [[ "$STAGE" == dump || "$STAGE" == all ]]; then
  echo "== DUMP: H1 Poisson (+S1) + H2/H1 AR1 (in-regime) + H3 ferr + curve milestones + V(beta) =="
  DJOBS="$OUTROOT/dump_jobs.sh"; : > "$DJOBS"
  # -- H2 in-regime pairs (REGISTERED primary topology: upstream_only) -------------------------
  for r in 0.0 0.3 0.6 0.9; do t="${RHOTAG[$r]}"
    for pair in "upstream comm" "nocomm nocomm"; do
      suf="${pair% *}"; grp="${pair#* }"
      for s in $SEEDS; do
        ck="$(latest_ck "${t}_${suf}" "$s" signal_checkpoint_best.pt)"; [[ -n "$ck" ]] || continue
        echo "$PYTHON agents/eval_signal.py --ckpt '$ck' --dump-comm '$H2ROOT/${grp}_${t}' \
--dump-ar1 '$r' --ar1-mu $AR1_MU --ar1-sigma $AR1_SIGMA --dump-episodes $DUMP_EPISODES" >> "$DJOBS"
      done
    done
  done
  # -- H1 Poisson pair + the S1 sensitivity arm (max-favorable geometry under the null) --------
  for pair in "upstream comm" "nocomm nocomm" "rbroadcast rbroadcast"; do
    suf="${pair% *}"; grp="${pair#* }"
    for s in $SEEDS; do
      ck="$(latest_ck "dp_${suf}" "$s" signal_checkpoint_best.pt)"; [[ -n "$ck" ]] || continue
      echo "$PYTHON agents/eval_signal.py --ckpt '$ck' --dump-comm '$H1ROOT/${grp}' \
--dump-episodes $DUMP_EPISODES" >> "$DJOBS"
    done
  done
  # -- SUBSTITUTION CURVE: deployable-at-budget-M snapshots of the two PRIMARY pairs -----------
  # (dp pair on the Poisson lambdas; ar1r9 pair in-regime at rho 0.9; per-seed V(budget) is fit
  #  by scripts/comm_stats.py curve in STAGE=analyze.)
  for M in $MLIST; do
    for pair in "upstream comm" "nocomm nocomm"; do
      suf="${pair% *}"; grp="${pair#* }"
      for s in $SEEDS; do
        ck="$(latest_ck "dp_${suf}" "$s" "signal_checkpoint_budget${M}.pt")"; [[ -n "$ck" ]] || continue
        echo "$PYTHON agents/eval_signal.py --ckpt '$ck' --dump-comm '$CURVROOT/dp_${grp}_b${M}' \
--dump-episodes $DUMP_EPISODES" >> "$DJOBS"
      done
      for s in $SEEDS; do
        ck="$(latest_ck "ar1r9_${suf}" "$s" "signal_checkpoint_budget${M}.pt")"; [[ -n "$ck" ]] || continue
        echo "$PYTHON agents/eval_signal.py --ckpt '$ck' --dump-comm '$CURVROOT/ar1r9_${grp}_b${M}' \
--dump-ar1 0.9 --ar1-mu $AR1_MU --ar1-sigma $AR1_SIGMA --dump-episodes $DUMP_EPISODES" >> "$DJOBS"
      done
    done
  done
  # -- F_INCENTIVE (Phase E): matched-beta pairs, in-regime at rho 0.9, TEAM-cost units --------
  for b in beta0 beta05; do
    for pair in "upstream comm" "nocomm nocomm"; do
      suf="${pair% *}"; grp="${pair#* }"
      for s in $SEEDS; do
        ck="$(latest_ck "ar1r9_${b}_${suf}" "$s" signal_checkpoint_best.pt)"; [[ -n "$ck" ]] || continue
        echo "$PYTHON agents/eval_signal.py --ckpt '$ck' --dump-comm '$EROOT/${b}_${grp}' \
--dump-ar1 0.9 --ar1-mu $AR1_MU --ar1-sigma $AR1_SIGMA --dump-episodes $DUMP_EPISODES" >> "$DJOBS"
      done
    done
  done
  # -- F_GEOMETRY / F_CONTENT members (registered Holm families), in-regime at rho 0.9. --------
  # Baseline needs no extra runs: analyze pairs each member against the MERGED Phase-A nocomm
  # dumps restricted to the 0.9 key. The upstream geometry member is the Phase-A comm dump.
  for arm in rbroadcast neighbor noneighbor downstream mbroadcast skip full linktop linkbot \
             rbroadcast_ip rbroadcast_dhatip rbroadcast_learned rbroadcast_raw; do
    for s in $SEEDS; do
      ck="$(latest_ck "ar1r9_${arm}" "$s" signal_checkpoint_best.pt)"; [[ -n "$ck" ]] || continue
      echo "$PYTHON agents/eval_signal.py --ckpt '$ck' --dump-comm '$FAMROOT/ar1r9_${arm}' \
--dump-ar1 0.9 --ar1-mu $AR1_MU --ar1-sigma $AR1_SIGMA --dump-episodes $DUMP_EPISODES" >> "$DJOBS"
    done
  done
  # -- v1.3 cells -> $OUTROOT/v13/<cell> (costs + _ferr + _censor siblings ride along) --------
  V13="$OUTROOT/v13"
  for pair in "dp_nocomm dp_nocomm" "dp_rbroadcast dp_dhat" "dp_rbroadcast_raw dp_raw" \
              "dp_upstream_dhat dp_dhat_up" "dp_rbroadcast_true_lambda dp_true_lambda"; do
    arm="${pair% *}"; cell="${pair#* }"
    for s in $SEEDS; do
      ck="$(latest_ck "$arm" "$s" signal_checkpoint_best.pt)"; [[ -n "$ck" ]] || continue
      echo "$PYTHON agents/eval_signal.py --ckpt '$ck' --dump-comm '$V13/${cell}' \
--dump-episodes $DUMP_EPISODES" >> "$DJOBS"
    done
  done
  for pair in "ar1r9_nocomm ar1r9_nocomm" "ar1r9_rbroadcast ar1r9_dhat" \
              "ar1r9_rbroadcast_raw ar1r9_raw" "ar1r9_rbroadcast_eps ar1r9_eps" \
              "ar1r9_rbroadcast_condmean ar1r9_condmean" "ar1r9_rbroadcast_learned ar1r9_learned" \
              "ar1r9_upstream_raw top_up_raw" "ar1r9_downstream_raw top_down_raw" \
              "ar1r9_clip12_nocomm clip12_nocomm" "ar1r9_clip12_rbroadcast_raw clip12_raw" \
              "ar1r9_clip20_nocomm clip20_nocomm" "ar1r9_clip20_rbroadcast_raw clip20_raw" \
              "ar1r9_rbroadcast_rawlag1 lag1" "ar1r9_rbroadcast_rawlag2 lag2"; do
    arm="${pair% *}"; cell="${pair#* }"
    for s in $SEEDS; do
      ck="$(latest_ck "$arm" "$s" signal_checkpoint_best.pt)"; [[ -n "$ck" ]] || continue
      echo "$PYTHON agents/eval_signal.py --ckpt '$ck' --dump-comm '$V13/${cell}' \
--dump-ar1 0.9 --ar1-mu $AR1_MU --ar1-sigma $AR1_SIGMA --dump-episodes $DUMP_EPISODES" >> "$DJOBS"
    done
  done
  for r in 0.0 0.3 0.6; do t="${RHOTAG[$r]}"; rt="rho${r/0./}"
    for pair in "${t}_nocomm ${rt}_nocomm" "${t}_rbroadcast ${rt}_dhat" \
                "${t}_rbroadcast_raw ${rt}_raw"; do
      arm="${pair% *}"; cell="${pair#* }"
      for s in $SEEDS; do
        ck="$(latest_ck "$arm" "$s" signal_checkpoint_best.pt)"; [[ -n "$ck" ]] || continue
        echo "$PYTHON agents/eval_signal.py --ckpt '$ck' --dump-comm '$V13/${cell}' \
--dump-ar1 '$r' --ar1-mu $AR1_MU --ar1-sigma $AR1_SIGMA --dump-episodes $DUMP_EPISODES" >> "$DJOBS"
      done
    done
  done
  # -- v1.3 substitution curve on the INFORMATIVE signal (raw pairs, budget milestones) --------
  for M in $MLIST; do
    for pair in "dp_rbroadcast_raw dpraw_comm" "dp_nocomm dpraw_nocomm"; do
      arm="${pair% *}"; grp="${pair#* }"
      for s in $SEEDS; do
        ck="$(latest_ck "$arm" "$s" "signal_checkpoint_budget${M}.pt")"; [[ -n "$ck" ]] || continue
        echo "$PYTHON agents/eval_signal.py --ckpt '$ck' --dump-comm '$CURVROOT/${grp}_b${M}' \
--dump-episodes $DUMP_EPISODES" >> "$DJOBS"
      done
    done
    for pair in "ar1r9_rbroadcast_raw ar1r9raw_comm" "ar1r9_nocomm ar1r9raw_nocomm"; do
      arm="${pair% *}"; grp="${pair#* }"
      for s in $SEEDS; do
        ck="$(latest_ck "$arm" "$s" "signal_checkpoint_budget${M}.pt")"; [[ -n "$ck" ]] || continue
        echo "$PYTHON agents/eval_signal.py --ckpt '$ck' --dump-comm '$CURVROOT/${grp}_b${M}' \
--dump-ar1 0.9 --ar1-mu $AR1_MU --ar1-sigma $AR1_SIGMA --dump-episodes $DUMP_EPISODES" >> "$DJOBS"
      done
    done
  done
  echo "  $(wc -l < "$DJOBS") dump jobs, $NPROC parallel"
  xargs -d '\n' -P "$NPROC" -I LINE bash -c 'eval "$1"' _ LINE < "$DJOBS" > "$OUTROOT/dump.log" 2>&1 || true
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
            if b.endswith(("_ferr.json", "_bw.json", "_censor.json", "_iv.json")):
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

# ------------------------------------------------------------ 5b. PROBE: do(m) intervention dumps
# The REGISTERED content-attribution gate's producer: per seed on every V-claiming arm, replay the
# CRN eval episodes under do(m) in {honest, shuffled, cross, zeroed} and dump seed{S}_iv.json
# (episode-mean costs; the per-seed table also lands in the log, incl. the identity-replay check).
# Cross-seed inference (SEED = unit) happens in STAGE=analyze via `comm_stats.py interventions`.
# Arms default to every arm a positive V will be claimed for: the P2 primary (ar1r9_upstream), the
# H4 geometry star (ar1r9_rbroadcast), the C3/D1 content rungs (learned, raw), and the Phase-E
# comm arms (does the channel stay informative as beta falls?). Override with PROBE_ARMS.
PROBEROOT="$OUTROOT/probes"
if [[ "$STAGE" == probe || "$STAGE" == all ]]; then
  echo "== PROBE: message interventions (do(m)) on: $PROBE_ARMS =="
  mkdir -p "$PROBEROOT/logs"
  PJOBS="$OUTROOT/probe_jobs.sh"; : > "$PJOBS"
  for arm in $PROBE_ARMS; do
    for s in $SEEDS; do
      ck="$(latest_ck "${arm}" "$s" signal_checkpoint_best.pt)"; [[ -n "$ck" ]] || continue
      rid="$(basename "$(dirname "$ck")")"
      echo "$PYTHON agents/eval_signal.py --ckpt '$ck' --ar1 --ar1-rho 0.9 --ar1-mu $AR1_MU \
--ar1-sigma $AR1_SIGMA --episodes $PROBE_EPISODES --dump-iv '$PROBEROOT/iv_${arm}' \
> '$PROBEROOT/logs/${rid}.log' 2>&1" >> "$PJOBS"
    done
  done
  echo "  $(wc -l < "$PJOBS") probe jobs, $NPROC parallel"
  xargs -d '\n' -P "$NPROC" -I LINE bash -c 'eval "$1"' _ LINE < "$PJOBS" > "$OUTROOT/probe.log" 2>&1 || true
  for arm in $PROBE_ARMS; do
    d="$PROBEROOT/iv_${arm}"
    [[ -d "$d" ]] && echo "  $arm: $(ls "$d"/seed*_iv.json 2>/dev/null | wc -l) seed dumps -> $d"
  done
fi

# ------------------------------------------------------------ 6. ANALYZE: registered H1/H2/H3 statistics
if [[ "$STAGE" == analyze || "$STAGE" == all ]]; then
  echo "== ANALYZE: H1 (TOST) + S1 + H2 (registered slope) + H3 + V(beta) =="
  "$PYTHON" - "$H2ROOT" "$H1ROOT" "$EROOT" "$FAMROOT" <<'PY'
import sys, json
sys.path.insert(0, ".")
from scripts.comm_stats import load_cost_dir, value_of_sharing, load_ferr_dir, forecast_delta
from scripts.prereg import h2_slope, h1_decision
h2root, h1root, eroot = sys.argv[1], sys.argv[2], sys.argv[3]
famroot = sys.argv[4] if len(sys.argv) > 4 else ""
comm, nocomm = load_cost_dir(h2root + "/comm"), load_cost_dir(h2root + "/nocomm")
if comm and nocomm:
    h2 = h2_slope(comm, nocomm)
    print("  H2 slope (registered): mean=%.2f CI95=%s Wilcoxon p=%.3g -> H2 holds: %s"
          % (h2["mean_slope"], h2["ci95"], h2["wilcoxon_p"], h2["h2_holds"]))
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
ncp = {}
try:
    cp, ncp = load_cost_dir(h1root + "/comm"), load_cost_dir(h1root + "/nocomm")
    if cp and ncp:
        vp = value_of_sharing(cp, ncp)
        print("  H1 Poisson  : V=%.1f CI=%s equiv=%s -> %s"
              % (vp["v_cost_mean"], vp["v_cost_ci"], vp["equivalent"], h1_decision(vp)))
except Exception as e:
    print("  H1 Poisson: skipped (%s)" % e)
try:                                                   # S1: max-favorable geometry under the null
    rb = load_cost_dir(h1root + "/rbroadcast")
    if rb and ncp:
        vs1 = value_of_sharing(rb, ncp)
        print("  S1 Poisson retailer_broadcast (sensitivity): V=%.1f CI=%s equiv=%s -> %s"
              % (vs1["v_cost_mean"], vs1["v_cost_ci"], vs1["equivalent"], h1_decision(vs1)))
        print("      (P1 is DECISIVE only if the null also survives this max-favorable geometry.)")
except Exception as e:
    print("  S1: skipped (%s)" % e)
# ---- F_INCENTIVE: V(beta) at matched beta, TEAM-cost units, in-regime rho0.9 ----
try:
    rows = []
    for tag, blab in (("beta0", "0.0"), ("beta05", "0.5")):
        c = load_cost_dir(f"{eroot}/{tag}_comm"); n = load_cost_dir(f"{eroot}/{tag}_nocomm")
        if c and n:
            rows.append((blab, value_of_sharing(c, n)))
    if comm and nocomm:                                # beta=1.0 point REUSED from Phase A, rho0.9
        rows.append(("1.0", value_of_sharing(comm, nocomm, lambdas=[0.9])))
    if rows:
        print("  F_INCENTIVE V(beta) (matched-beta baselines; + => channel still buys team cost):")
        for blab, v in sorted(rows, key=lambda kv: float(kv[0])):
            print("      beta=%-4s V=%+8.1f (%+5.1f%%) CI=%s n=%d %s"
                  % (blab, v["v_cost_mean"], v["v_cost_pct"], v["v_cost_ci"], v["n_seeds"],
                     "sig" if v["v_cost_ci_excludes_0"] else "ns"))
        print("      (informative-communication survives misalignment if V stays positive as beta falls;")
        print("       babbling-equilibrium collapse shows as V -> 0/ns at low beta. Crawford-Sobel 1982.)")
        if not (comm and nocomm):
            print("      [warn] beta=1.0 point unavailable: Phase-A H2 dumps missing (it is REUSED, not retrained).")
except Exception as e:
    print("  F_INCENTIVE: skipped (%s)" % e)
# ---- F_GEOMETRY / F_CONTENT (registered Holm families) + D1 + C3, in-regime at rho 0.9 ----
try:
    if famroot and nocomm:
        from scripts.prereg import holm_family
        NULLS = {"no_neighbor", "downstream_only", "manufacturer_broadcast"}
        GEOM = {"upstream_only (reused A)": None, "retailer_broadcast": "rbroadcast",
                "neighbor": "neighbor", "skip": "skip", "full": "full",
                "link_top_only": "linktop", "link_bottom_only": "linkbot",
                "no_neighbor": "noneighbor", "downstream_only": "downstream",
                "manufacturer_broadcast": "mbroadcast"}
        CONT = {"ip": "rbroadcast_ip", "dhat_ip": "rbroadcast_dhatip",
                "learned": "rbroadcast_learned", "raw": "rbroadcast_raw"}
        def _arm(tag): return load_cost_dir("%s/ar1r9_%s" % (famroot, tag))
        for fam_name, members in (("F_GEOMETRY", GEOM), ("F_CONTENT", CONT)):
            res = {}
            for label, tag in members.items():
                d = comm if tag is None else _arm(tag)         # upstream member = Phase-A comm dumps
                if d:
                    try:
                        res[label] = value_of_sharing(d, nocomm, lambdas=[0.9])
                    except Exception:
                        pass
            if not res:
                continue
            hol = holm_family({k: v["wilcoxon_p"] for k, v in res.items()
                               if v["wilcoxon_p"] == v["wilcoxon_p"]})     # drop NaN p
            print("  %s @ rho0.9 (member vs merged Phase-A nocomm, seed-paired; Holm over %d):"
                  % (fam_name, len(hol)))
            for label, v in sorted(res.items(), key=lambda kv: -kv[1]["v_cost_mean"]):
                h = hol.get(label, {})
                nn = ("  TOST p=%.3g%s" % (v["tost_p"], " EQUIV" if v["equivalent"] else "")
                      ) if label in NULLS else ""
                print("      %-26s V=%+8.1f (%+5.1f%%) CI=[%.1f,%.1f] n=%d  p=%.3g adj=%.3g %s%s"
                      % (label, v["v_cost_mean"], v["v_cost_pct"], v["v_cost_ci"][0], v["v_cost_ci"][1],
                         v["n_seeds"], h.get("raw", float("nan")), h.get("adjusted", float("nan")),
                         "REJ" if h.get("reject") else "ns ", nn))
        dh, rw = _arm("rbroadcast"), _arm("rbroadcast_raw")    # D1: dhat vs raw (both rbroadcast)
        if dh and rw:
            d1 = value_of_sharing(dh, rw, lambdas=[0.9])       # V = cost(raw) - cost(dhat)
            print("  D1 forecast-vs-data (dhat vs raw @ rbroadcast): V=%+.1f CI=[%.1f,%.1f]"
                  " (+ => dhat cheaper; Aviv-consistent)"
                  % (d1["v_cost_mean"], d1["v_cost_ci"][0], d1["v_cost_ci"][1]))
        le, di = _arm("rbroadcast_learned"), _arm("rbroadcast_dhatip")     # C3 (executed design)
        if le and di:
            c3 = value_of_sharing(le, di, lambdas=[0.9])       # diff = cost(dhat_ip) - cost(learned)
            print("  C3 interpretability bound (learned vs dhat_ip @ rbroadcast): diff=%+.1f"
                  " CI=[%.1f,%.1f] TOST p=%.3g -> %s"
                  % (c3["v_cost_mean"], c3["v_cost_ci"][0], c3["v_cost_ci"][1], c3["tost_p"],
                     "EQUIVALENT" if c3["equivalent"] else "not equivalent"))
        print("  (NOTE the C3 wording amendment: prereg v1.1 says upstream_only; the executed Phase-C")
        print("   arms are retailer_broadcast -- reconcile by DATED amendment before unblinding.)")
except Exception as e:
    print("  F_GEOMETRY/F_CONTENT: skipped (%s)" % e)
print("  (H4 geometry / H5 content: same producers on the rho0.9 topology/content arms + "
      "comm_stats.value_of_sharing per arm with Holm across the family; H7 needs the PoA probe.)")
PY
  # ---- substitution curve (registered exploratory): per-seed V vs log2(budget) slope ----
  for reg in dp ar1r9; do
    cargs=(); ok=1
    for M in $MLIST; do
      c="$CURVROOT/${reg}_comm_b${M}"; n="$CURVROOT/${reg}_nocomm_b${M}"
      if [[ -d "$c" && -d "$n" ]]; then cargs+=(--budget "$M" "$c" "$n"); else ok=0; fi
    done
    if [[ "$ok" == 1 && "${#cargs[@]}" -gt 0 ]]; then
      echo "-- SUBSTITUTION CURVE ($reg pair): V(training budget) --"
      "$PYTHON" scripts/comm_stats.py curve "${cargs[@]}" || true
    else
      echo "  (curve $reg: milestone dumps incomplete -- STAGE=dump fills $CURVROOT/${reg}_*_b{$MILESTONES})"
    fi
  done
  # ---- registered content-attribution gate: cross-seed do(m) summaries ----
  for arm in $PROBE_ARMS; do
    d="$PROBEROOT/iv_${arm}"
    if compgen -G "$d/seed*_iv.json" > /dev/null 2>&1; then
      echo "-- INTERVENTIONS (cross-seed gate): $arm --"
      "$PYTHON" scripts/comm_stats.py interventions --dir "$d" || true
    fi
  done
fi

# ------------------------------------------------------------ 7. PLOT: seed-aggregated learning curves
# Builds the publishable comm-vs-no-comm held-out learning-curve PDFs directly from the per-run
# metrics_heldout.csv the trainer wrote under SIGNAL_CSVLOG (default on). One figure per registered
# comparison; arms with no CSVs yet are skipped, so this is safe to run at any point after TRAIN.
# The band is plot_curves.py's default (studentized bootstrap-95% CI over seeds).
PLOTROOT="$OUTROOT/figs"
_ck() { echo "weights_signal/run_signal_*_${1}_s*/metrics_heldout.csv"; }   # per-arm CSV glob (all seeds)
_have() { compgen -G "$1" >/dev/null 2>&1; }
plot_fig() {                                    # plot_fig OUT METRIC  LABEL GLOB [LABEL GLOB ...]
  local out="$1" metric="$2"; shift 2
  local args=() lbl glob
  while (( $# >= 2 )); do
    lbl="$1"; glob="$2"; shift 2
    _have "$glob" && args+=(--arm "$lbl" "$glob")
  done
  if (( ${#args[@]} == 0 )); then echo "  [plot] $out: no seed CSVs yet -- skipped"; return; fi
  "$PYTHON" plot_curves.py "${args[@]}" --metric "$metric" --out "$PLOTROOT/$out" || true
}

if [[ "$STAGE" == plot || "$STAGE" == all ]]; then
  echo "== PLOT: seed-aggregated learning-curve figures -> $PLOTROOT =="
  if ! "$PYTHON" -c "import matplotlib" >/dev/null 2>&1; then
    echo "  [plot] matplotlib not installed ('$PYTHON -m pip install matplotlib') -- PLOT stage skipped"
  else
    mkdir -p "$PLOTROOT"
    # H1 (Poisson): registered primary + the S1 max-favorable-geometry sensitivity, vs no-comm.
    plot_fig fig_h1_poisson.pdf       heldout_mean_cost  upstream "$(_ck dp_upstream)"   nocomm "$(_ck dp_nocomm)"
    plot_fig fig_h1_s1_rbroadcast.pdf heldout_mean_cost  rbroadcast "$(_ck dp_rbroadcast)" nocomm "$(_ck dp_nocomm)"
    # H2 (in-regime, per rho): upstream comm vs matched no-comm at each autocorrelation.
    for r in 0.0 0.3 0.6 0.9; do t="${RHOTAG[$r]}"
      plot_fig "fig_h2_${t}.pdf" heldout_mean_cost  comm "$(_ck ${t}_upstream)"  nocomm "$(_ck ${t}_nocomm)"
    done
    # H4 geometry @ rho0.9: every topology overlaid against the no-comm baseline.
    gargs=()
    for topo in upstream rbroadcast neighbor downstream mbroadcast noneighbor; do
      _have "$(_ck ar1r9_${topo})" && gargs+=("$topo" "$(_ck ar1r9_${topo})")
    done
    plot_fig fig_h4_geometry_ar1r9.pdf heldout_mean_cost "${gargs[@]}" nocomm "$(_ck ar1r9_nocomm)"
    # H5 content ladder @ rho0.9 (dhat point is Phase-B ar1r9_rbroadcast), vs no-comm.
    plot_fig fig_h5_content_ar1r9.pdf heldout_mean_cost \
      dhat "$(_ck ar1r9_rbroadcast)" ip "$(_ck ar1r9_rbroadcast_ip)" dhat_ip "$(_ck ar1r9_rbroadcast_dhatip)" \
      learned "$(_ck ar1r9_rbroadcast_learned)" raw "$(_ck ar1r9_rbroadcast_raw)" nocomm "$(_ck ar1r9_nocomm)"
    # E incentive: matched-beta comm/no-comm pairs.
    plot_fig fig_e_incentive_ar1r9.pdf heldout_mean_cost \
      b0_comm "$(_ck ar1r9_beta0_upstream)" b0_nocomm "$(_ck ar1r9_beta0_nocomm)" \
      b05_comm "$(_ck ar1r9_beta05_upstream)" b05_nocomm "$(_ck ar1r9_beta05_nocomm)"
    # D strategic (canonical): coop / selfish / contract.
    plot_fig fig_d_strategic_dp.pdf   heldout_mean_cost \
      coop "$(_ck cn_dp_coop)" selfish "$(_ck cn_dp_selfish)" contract "$(_ck cn_dp_contract)"
    plot_fig fig_d_strategic_ar1r9.pdf heldout_mean_cost \
      coop "$(_ck cn_ar1r9_coop)" selfish "$(_ck cn_ar1r9_selfish)" contract "$(_ck cn_ar1r9_contract)"
    # Mechanism curve (H3): upstream forecast error over training, comm vs no-comm at rho0.9.
    plot_fig fig_h3_ferr_ar1r9.pdf    forecast_error  comm "$(_ck ar1r9_upstream)" nocomm "$(_ck ar1r9_nocomm)"
    echo "  wrote $(ls "$PLOTROOT"/*.pdf 2>/dev/null | wc -l) figure(s) to $PLOTROOT"
  fi
fi

echo "ALL SELECTED STAGES COMPLETE ($(date))"