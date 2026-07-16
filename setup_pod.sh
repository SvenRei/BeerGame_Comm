#!/usr/bin/env bash
# ============================================================================
# setup_pod.sh -- one-shot environment bootstrap + verification for SIGNAL.
# ----------------------------------------------------------------------------
# WHAT THIS DOES (idempotent; safe to re-run):
#   1. detects the hardware (CPU vs NVIDIA GPU) and picks the matching torch wheel,
#   2. creates a project-local venv and installs EVERY runtime dependency,
#   3. VERIFIES the install (imports each package, prints versions, runs a torch op),
#   4. runs the component SMOKE TESTS (the self-tests that gate a trustworthy run),
#   5. runs an end-to-end TRAINING smoke and confirms the CSV logger writes its files,
#   6. prints a commented tutorial for generating all the paper's data.
#
# DESIGN: this is meant for a fresh RunPod/Linux CPU pod but works on any POSIX box
# with python3 + pip. It is CONSERVATIVE -- it never deletes your data, and every
# heavy/uncertain step is overridable via the env vars documented below.
#
# USAGE (run from the repo root, inside tmux on a spot pod):
#   chmod +x setup_pod.sh
#   ./setup_pod.sh                      # CPU torch wheel (default; portable to any box), full smoke
#   GPU=1 ./setup_pod.sh                # install a CUDA torch build instead (needs a matching driver)
#   GPU=1 TORCH_CUDA=cu124 ./setup_pod.sh   # CUDA build for a specific toolkit
#   SLOW_SMOKE=1 ./setup_pod.sh         # also run the slow baselines optimizer self-test
#   SKIP_INSTALL=1 ./setup_pod.sh       # skip pip, just re-verify + smoke an existing venv
#   SKIP_SMOKE=1 ./setup_pod.sh         # install only, no self-tests / training smoke
#
# OVERRIDABLE ENV (defaults in brackets):
#   PYTHON[python3]  VENV_DIR[venv]  GPU[0 = CPU wheel; 1 = CUDA wheel]
#   TORCH_CUDA[cu121]  (CUDA tag for the GPU wheel; used only when GPU=1)
#   SKIP_INSTALL[0]  SKIP_SMOKE[0]  SKIP_TRAIN_SMOKE[0]  SLOW_SMOKE[0]
# ============================================================================
set -euo pipefail

# ------------------------------------------------------------ 0. locate repo root
cd "$(dirname "$(readlink -f "$0" 2>/dev/null || echo "$0")")"
ROOT="$(pwd)"
[[ -f agents/train_signal.py ]] || { echo "ERROR: run from the repo root (agents/train_signal.py missing)"; exit 1; }
[[ -f requirements.txt ]]       || { echo "ERROR: requirements.txt missing -- wrong directory?"; exit 1; }

# ------------------------------------------------------------ 1. config
PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-venv}"
TORCH_CUDA="${TORCH_CUDA:-cu121}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"
SKIP_TRAIN_SMOKE="${SKIP_TRAIN_SMOKE:-0}"
SLOW_SMOKE="${SLOW_SMOKE:-0}"

command -v "$PYTHON" >/dev/null 2>&1 || { echo "ERROR: '$PYTHON' not found. Install Python 3.9+ or set PYTHON=."; exit 1; }

# Require Python >= 3.9 (hydra/omegaconf + f-strings used throughout). Fail fast with ONE message.
"$PYTHON" - <<'PY' || { echo "ERROR: Python 3.9+ required."; exit 1; }
import sys
raise SystemExit(0 if sys.version_info[:2] >= (3, 9) else 1)
PY
echo "== python: $("$PYTHON" --version 2>&1)  ($("$PYTHON" -c 'import sys;print(sys.executable)'))"

# ------------------------------------------------------------ 2. torch wheel selection
# This project is CPU-bound (the per-step belief encode is sequential; the sweep gets its throughput
# from many PARALLEL CPU jobs, not from a GPU). The CPU wheel is therefore the DEFAULT -- and it runs
# on literally ANY x86-64 Linux box regardless of driver/CUDA version, which is exactly what makes
# this script portable across hardware (your strong-CPU pod included). Opt into CUDA only if you have
# a specific reason: GPU=1 [TORCH_CUDA=cu124]. A CUDA wheel needs a matching NVIDIA driver on the box.
GPU="${GPU:-0}"
if [[ "$GPU" == 1 ]]; then
  TORCH_INDEX="https://download.pytorch.org/whl/${TORCH_CUDA}"
  echo "== torch wheel: CUDA ${TORCH_CUDA} (GPU=1 requested)"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/     GPU: /' \
    || echo "     [warn] GPU=1 but nvidia-smi did not respond -- the CUDA wheel needs a matching driver."
else
  TORCH_INDEX="https://download.pytorch.org/whl/cpu"
  echo "== torch wheel: CPU (default; training is CPU-bound -> optimal here, and portable everywhere)"
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    echo "     (note: a GPU is present but intentionally unused; pass GPU=1 to install a CUDA build.)"
  fi
fi

# ------------------------------------------------------------ 3. venv + install
VENV_PY="$VENV_DIR/bin/python"
[[ -x "$VENV_PY" ]] || VENV_PY="$VENV_DIR/Scripts/python.exe"   # Windows/Git-Bash fallback

if [[ "$SKIP_INSTALL" == 1 ]]; then
  echo "== SKIP_INSTALL=1: using existing venv at $VENV_DIR"
  [[ -x "$VENV_PY" ]] || { echo "ERROR: no venv at $VENV_DIR to reuse."; exit 1; }
else
  if [[ ! -x "$VENV_PY" ]]; then
    echo "== creating venv at $VENV_DIR"
    # A bare Ubuntu/RunPod base image can ship python3 WITHOUT the venv module -> give the exact fix
    # instead of a cryptic traceback. (Most ML pod images already have it.)
    "$PYTHON" -m venv "$VENV_DIR" || {
      echo "ERROR: 'python -m venv' failed. On a minimal image install it, then re-run:";
      echo "       apt-get update && apt-get install -y python3-venv python3-pip";
      exit 1; }
    VENV_PY="$VENV_DIR/bin/python"; [[ -x "$VENV_PY" ]] || VENV_PY="$VENV_DIR/Scripts/python.exe"
  else
    echo "== reusing existing venv at $VENV_DIR"
  fi
  "$VENV_PY" -m pip install --upgrade pip >/dev/null

  # torch FIRST from the hardware-specific index. Installing it here means the later
  # `-r requirements.txt` sees torch>=2.0 already satisfied and will NOT pull a different
  # wheel from PyPI (pip's default is upgrade-only-if-needed).
  echo "== installing torch from $TORCH_INDEX (this is the big download)"
  "$VENV_PY" -m pip install torch --index-url "$TORCH_INDEX"

  echo "== installing the rest of requirements.txt"
  "$VENV_PY" -m pip install -r requirements.txt

  # matplotlib is required by plot_curves.py (the learning-curve figure step); tensorboard is
  # only needed if you run with SIGNAL_TBLOG=1 (optional live-watch). Both are cheap.
  echo "== installing plotting deps (matplotlib; tensorboard for optional SIGNAL_TBLOG)"
  "$VENV_PY" -m pip install matplotlib tensorboard
fi

# ------------------------------------------------------------ 4. VERIFY the install
# Import every runtime package, print versions, and exercise a torch op. A missing/broken
# dependency fails HERE with a clear message instead of 300 jobs into the sweep.
echo ""
echo "== verifying requirements =="
"$VENV_PY" - <<'PY' || { echo "REQUIREMENT CHECK FAILED (see above)."; exit 1; }
import importlib, sys
mods = ["numpy", "scipy", "torch", "pettingzoo", "gymnasium",
        "hydra", "omegaconf", "wandb", "matplotlib"]
missing = []
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"   {m:<12} {getattr(mod, '__version__', '?')}")
    except Exception as e:                                  # noqa: BLE001
        missing.append(m); print(f"   {m:<12} MISSING ({type(e).__name__})")
import torch
x = torch.ones(3); assert float((x + x).sum()) == 6.0, "torch arithmetic broken"
print(f"   torch device check: cuda_available={torch.cuda.is_available()}  threads={torch.get_num_threads()}")
sys.exit(1 if missing else 0)
PY
echo "   OK: all runtime requirements import and torch works."

# ------------------------------------------------------------ 5. SMOKE TESTS (component self-tests)
# These are numpy/scipy/torch-only and need no W&B login, no data, no network. A green board
# here means the instrument itself is sound. `py_compile` catches syntax breaks in the entry
# points that have no standalone self-test.
if [[ "$SKIP_SMOKE" != 1 ]]; then
  echo ""
  echo "== component smoke tests =="
  fails=0

  echo "   -- py_compile entry points --"
  "$VENV_PY" -m py_compile agents/train_signal.py agents/eval_signal.py agents/signal_agent.py \
             agents/signal_csvlog.py plot_curves.py \
    && echo "      py_compile OK" || { echo "      py_compile FAIL"; fails=$((fails+1)); }

  # self-test command : human label   (the fast set -- each < ~15s)
  SMOKE=(
    "agents/signal_agent.py|agent shape + DIAL + policy-gradient self-test (torch)"
    "scripts/c1_stats.py|C1 inference primitives (bootstrap/TOST/Wilcoxon)"
    "scripts/comm_stats.py|value-of-sharing aggregator"
    "scripts/demand_families.py|AR(1)/NegBin/family samplers"
    "scripts/coordination_theory.py|tau* regime-invariance theorem"
    "scripts/dp_optimum.py|single-stage exact-DP optimum"
    "scripts/run_confirmatory_report.py|confirmatory analyzer (synthetic dumps)"
    "scripts/distill_symbolic.py|symbolic distillation (linear fallback, no PySR)"
    "scripts/prereg.py self-test|pre-registration integrity + hash"
  )
  for entry in "${SMOKE[@]}"; do
    cmd="${entry%%|*}"; label="${entry#*|}"
    if "$VENV_PY" $cmd >/tmp/smoke.out 2>&1; then
      echo "      PASS  $label"
    else
      echo "      FAIL  $label  ($cmd)  :: $(tail -1 /tmp/smoke.out)"; fails=$((fails+1))
    fi
  done

  # The baselines optimizer self-test is CORRECT but slow (~1-2 min); off by default.
  if [[ "$SLOW_SMOKE" == 1 ]]; then
    echo "   -- SLOW_SMOKE: baselines serial base-stock optimizer --"
    "$VENV_PY" scripts/baselines.py selftest >/tmp/smoke.out 2>&1 \
      && echo "      PASS  baselines base-stock optimizer" \
      || { echo "      FAIL  baselines selftest :: $(tail -1 /tmp/smoke.out)"; fails=$((fails+1)); }
  fi

  # ---------------------------------------------------------- 6. end-to-end TRAINING smoke
  # A 30-episode run that exercises the WHOLE pipeline (env -> agent -> update -> held-out gate
  # -> checkpoint) AND the scalar logger, then asserts the CSV artifact was actually written.
  if [[ "$SKIP_TRAIN_SMOKE" != 1 ]]; then
    echo "   -- end-to-end training smoke (30 eps, CSV logger ON) --"
    SMOKE_DIR="weights_signal"
    if SIGNAL_CSVLOG=1 WANDB_MODE=disabled "$VENV_PY" agents/train_signal.py agent=signal \
         seed=0 total_episodes=30 agent.heldout_every=10 agent.heldout_episodes=1 \
         agent.batch_episodes=8 agent.log_every=1000 "agent.budget_milestones=[]" \
         agent.algorithm=setup_smoke_s0 >/tmp/train_smoke.out 2>&1; then
      csv="$(find "$SMOKE_DIR" -path '*setup_smoke_s0*' -name metrics_heldout.csv 2>/dev/null | head -1)"
      if [[ -n "$csv" ]] && [[ "$(wc -l < "$csv")" -ge 2 ]]; then
        echo "      PASS  training + CSV logger (wrote $(basename "$(dirname "$csv")")/metrics_heldout.csv)"
        # plotter ingest check (no display; just parse + aggregate the smoke CSV)
        "$VENV_PY" - "$csv" <<'PY' && echo "      PASS  plot_curves ingest" || { echo "      FAIL plot_curves ingest"; fails=$((fails+1)); }
import sys, plot_curves as pc
x, y = pc.load_series(sys.argv[1], "episode", "heldout_mean_cost")
assert len(x) >= 1 and len(x) == len(y)
PY
      else
        echo "      FAIL  training smoke produced no metrics_heldout.csv :: $(tail -1 /tmp/train_smoke.out)"; fails=$((fails+1))
      fi
      rm -rf "$SMOKE_DIR"/run_signal_*setup_smoke_s0 2>/dev/null || true   # clean the smoke artifact
    else
      echo "      FAIL  training smoke crashed :: $(tail -3 /tmp/train_smoke.out)"; fails=$((fails+1))
    fi
  fi

  echo ""
  if [[ "$fails" -eq 0 ]]; then
    echo "== SMOKE: ALL GREEN =="
  else
    echo "== SMOKE: $fails FAILURE(S) -- fix before launching the sweep (logs in /tmp/*.out) =="
    exit 1
  fi
fi

# ------------------------------------------------------------ 7. DATA-GENERATION TUTORIAL
# Printed as guidance (nothing below runs). Every command uses the venv python you just built.
cat <<TUT

============================================================================
 SETUP COMPLETE. How to generate all the data (commented tutorial)
============================================================================
 All commands assume the repo root and the venv you just built. On the pod,
 either 'source $VENV_DIR/bin/activate' first, or prefix with '$VENV_PY'.
 Keep W&B off for the air-gapped sweep: 'export WANDB_MODE=disabled'.

 -- STEP 0: activate + sanity ------------------------------------------------
   source $VENV_DIR/bin/activate            # then 'python' is the venv python
   python agents/signal_agent.py            # re-run the agent self-test any time

 -- STEP 1: BENCHMARK once (BAR / Oracle refs; per cost model) ---------------
   # Behavioral cost model (the default for phases A/B/Bnull/C/E):
   python scripts/baselines.py regime --lambdas 6 10 14 18 22 --select-episodes 80 --eval-episodes 200
   # -> writes results/baselines_regime_v2.json. REGENERATE this whenever you flip
   #    env.penalty_at_retailer_only (canonical phases D/Dext use a different cost scale).

 -- STEP 2a: a SINGLE manual run (understand the moving parts) ---------------
   SIGNAL_CSVLOG=1 WANDB_MODE=disabled python agents/train_signal.py agent=signal \\
       seed=10 total_episodes=8000 agent.comm_topology=full agent.algorithm=dp_full_s10
   #   -> weights_signal/run_signal_<id>_dp_full_s10/{signal_checkpoint_best.pt,
   #      metrics_heldout.csv, metrics_update.csv, run_meta.json}
   #   The no-comm control (identical architecture, ADJ zeroed):
   SIGNAL_CSVLOG=1 WANDB_MODE=disabled python agents/train_signal.py agent=signal \\
       seed=10 total_episodes=8000 agent.use_comm=false agent.algorithm=dp_nocomm_s10

 -- STEP 2b: the FULL registered sweep (all 435 arms, parallel, resumable) ---
   chmod +x sweep_all_hypotheses.sh
   DRYRUN=1 ./sweep_all_hypotheses.sh                        # preview the plan, run nothing
   NPROC=48 PHASES="A B Bnull C E" ./sweep_all_hypotheses.sh # BEHAVIORAL core (SIGNAL_CSVLOG on by default)
   # ... regenerate refs for the CANONICAL cost model, THEN (separate invocation): ...
   NPROC=48 PHASES="D Dext" ./sweep_all_hypotheses.sh        # STRATEGIC (canonical) phases
   # (STAGE=calibrate ./sweep_all_hypotheses.sh first to measure the best NPROC on this pod.)

 -- STEP 3: score the trained checkpoints (per-seed producers) ---------------
   STAGE=dump  NPROC=48 PHASES="A B Bnull C E" ./sweep_all_hypotheses.sh   # {lambda|rho: cost} + ferr
   STAGE=probe NPROC=48 ./sweep_all_hypotheses.sh                          # do(m) intervention dumps

 -- STEP 4: the registered statistics (H1 TOST, H2 slope, H3, V(beta), ...) --
   STAGE=analyze ./sweep_all_hypotheses.sh
   # Or by hand for one comm/no-comm pair:
   #   python agents/eval_signal.py --ckpt <comm ckpt>   --dump-comm results/comm_on
   #   python agents/eval_signal.py --ckpt <nocomm ckpt> --dump-comm results/comm_off
   #   python scripts/comm_stats.py report --comm-dir results/comm_on --nocomm-dir results/comm_off

 -- STEP 5: the publishable learning-curve figures (from the CSVs) -----------
   python plot_curves.py \\
     --arm full   "weights_signal/run_signal_*_dp_full_s*/metrics_heldout.csv" \\
     --arm nocomm "weights_signal/run_signal_*_dp_nocomm_s*/metrics_heldout.csv" \\
     --out fig_full_vs_nocomm.pdf
   # any logged column via --metric (gap_recovered, forecast_error, positive_listening, msg_std, ...)

 -- STEP 6: pull artifacts off the pod (CSVs are tiny; checkpoints are GB) ----
   tar czf metrics.tgz weights_signal/run_signal_*/metrics_*.csv \\
       weights_signal/run_signal_*/run_meta.json sweep_out
============================================================================
TUT
