# SIGNAL — Strategic Information-sharing for Game-theoretic, Networked, Adaptive Logistics

A reinforcement-learning testbed for the **value of information sharing** in the 4-echelon serial
supply chain ("beer game"). The agent, **SIGNAL**, is a deliberately minimal communicating MAPPO
policy: a small per-agent recurrent belief, a linear base-stock head, and a configurable **message
ladder**. It is an *instrument*, not the contribution — kept simple so that any measured effect is
attributable to the message channel, not to architectural cleverness. The name points at the object
of measurement: the Lee–So–Tang **demand signal** shared between echelons.

The project answers two questions:

1. **Cooperative (the primary study).** How much is sharing demand information worth to a chain that
   must *infer its demand regime online*, and through what mechanism? We measure the value of sharing
   across communication topologies and demand autocorrelation, and locate it (does it cut *upstream*
   forecast error — the Lee–So–Tang see-through-bullwhip channel?).
2. **Strategic (the extension).** When echelons are self-interested, what is the *price of anarchy*,
   and can a single **coordinating transfer τ\*** align incentives? The accompanying theory shows τ\*
   is *regime-invariant* (`scripts/coordination_theory.py`).

> New to the code? Read the module docstring at the top of [`agents/signal_agent.py`](agents/signal_agent.py)
> (the agent + trainer), then the per-step phase map in [`envs/beer_game_env.py`](envs/beer_game_env.py),
> then [`agents/train_signal.py`](agents/train_signal.py) (the training loop).

---

## Repository layout

```
envs/
  beer_game_env.py          PettingZoo ParallelEnv: 4-echelon beer game, 4-scalar partial obs,
                            configurable lead times, CTDE global state. Cost = h*inventory + b*backlog
                            (every stage, or retailer-only via penalty_at_retailer_only).
  demand_randomization.py   DemandRandomizedBeerGame: per-episode Poisson-rate randomization (the
                            regime-inference training env). Self-contained subclass of the env.
agents/
  signal_agent.py           The agent: SIGNALActor (GRU belief + linear base-stock head + demand
                            readout + message ladder), SIGNALCritic (CTDE), SIGNALTrainer (MAPPO +
                            the economics layer). Self-contained: torch/numpy only.
  train_signal.py           Hydra/W&B training entry point (env construction, loop, held-out gate,
                            checkpointing). Folds the held-out-lambda gate in directly.
  eval_signal.py            Checkpoint loader (SIGNALPolicy) + measurement layer: standard benchmark,
                            per-stage dashboard, C1 regime-uncertainty, comm-value decomposition
                            (forecast-error delta = the Lee mechanism), honesty probe, positive-
                            listening + message-intervention (do(m)) probes, and the --dump-c1 /
                            --dump-comm / --dump-iv producers for the stats scripts.
  signal_csvlog.py          Scientific scalar logger (opt-in via SIGNAL_CSVLOG=1). PURE OBSERVATION:
                            reads tensors the update/eval paths already compute and writes one
                            metrics_heldout.csv (the true per-gate learning curve), one
                            metrics_update.csv (PPO/convergence diagnostics), and a run_meta.json
                            provenance manifest per run_dir. Stdlib csv only; default-off is
                            byte-identical to the frozen trainer. Replaces W&B for the sweep.
  topologies.py             Communication topologies (ADJ matrices): neighbor / skip / full /
                            retailer_broadcast / no_neighbor (placebo) + the geometry variants.
conf/
  config.yaml               Master Hydra config (env block + globals). defaults: agent: signal.
  agent/signal.yaml         Agent hyperparameters + the study knobs (msg_content, topology, beta, tau).
scripts/
  baselines.py              Classical comparators (static base-stock, adaptive, Bayes, per-lambda
                            oracle) + the regime benchmark. Writes results/baselines_regime_v2.json.
  demand_families.py        AR(1) / NegBin / family-randomized demand (subclasses the env; no env edit).
  c1_stats.py               C1 inference: load refs, bootstrap CI, paired Wilcoxon, TOST, IQM,
                            performance profile. `report` CLI + numpy-only self-test.
  comm_stats.py             Cross-seed value-of-sharing aggregator: CRN-pairs comm vs no-comm arms,
                            reports V_cost (bootstrap CI + paired + TOST equivalence) and the
                            per-echelon forecast-error delta (the mechanism). Consumes eval --dump-comm.
  coordination_theory.py    Numerical proof of the regime-invariance theorem for tau* (the strategic
                            study). scipy newsvendor, no torch.
  dp_optimum.py             Exact single-stage finite-horizon DP: the critical-fractile base-stock IS
                            the optimum (the "gap to optimal" anchor).
  run_confirmatory_report.py  Locked analyzer: runs the C1 + bullwhip + communication tables end-to-end
                            from the per-seed dumps. Torch-free.
  prereg.py                 Pre-registration of the confirmatory analysis (the single source of truth
                            for the primary/secondary contrasts + multiplicity). `python scripts/prereg.py`
                            prints the registration and its SHA256 (logged into run_meta.json). Torch-free.
  distill_symbolic.py       C3 symbolic distillation (PySR + DAgger) of the SIGNAL policy into a
                            closed-form base-stock equation. Falls back to a linear fit without PySR.
test/
  test_signal.py            [TODO] agent component/integration tests (needs torch).
  test_beer_game_env.py     [TODO] env unit tests (numpy + pettingzoo; no torch).
requirements.txt            CPU torch + numpy/scipy/pettingzoo/gymnasium/hydra/omegaconf/wandb/pytest.
                            (matplotlib is needed for plot_curves.py; seaborn/pysr optional; see file.)
setup_pod.sh                One-shot environment bootstrap for a fresh (RunPod/Linux) box: detects
                            CPU vs GPU, builds the venv, installs everything, VERIFIES the install,
                            runs the component + end-to-end smoke tests, and prints the data-gen
                            tutorial. Idempotent; see `OVERRIDABLE ENV` in its header.
sweep_all_hypotheses.sh     The full registered multi-seed sweep orchestrator (train | dump | probe |
                            analyze) for a parallel CPU pod. xargs -P job pool, BLAS pinned to 1
                            thread/job, sentinel-based resume. Enables SIGNAL_CSVLOG by default.
plot_curves.py              Standalone learning-curve plotter: ingests the per-seed metrics_heldout.csv
                            for one/many arms, aligns on episode, plots the seed-aggregated mean of any
                            column with a bootstrap-95%-CI (default) or IQR band, to a vector PDF. No
                            tracker; matplotlib + numpy only.
```

Generated at runtime (git-ignored): `weights_signal/` (checkpoints + per-run metrics_*.csv /
run_meta.json), `results/` (refs + dumps), `sweep_out/` (sweep job lists + logs), `outputs/`
(Hydra run dirs), `wandb/`.

---

## Architecture (one screen)

- **Environment** — serial chain retailer → wholesaler → distributor → manufacturer. Each agent
  observes only 4 local scalars `[inventory, backlog, on_order, last_incoming]`; a centralized critic
  sees the full pipeline during training only (CTDE, `get_global_state()` is 133-d). The action is a
  **fraction in [0,1]**; the env reconstructs `order = round(fraction × max_order)`. Per-period cost
  is `h·inventory + b·backlog` at every echelon by default, or retailer-only with
  `env.penalty_at_retailer_only=true` (the canonical Clark–Scarf cost the contract theory uses).
- **Belief** — one small **GRU per agent** over `[local obs, received message]`. Its hidden state *is*
  the demand-regime belief, trained end-to-end by the policy gradient (no ELBO/KL/reconstruction).
- **Head** — ONE linear layer maps `[obs, belief, message] → S` (order-up-to level); softplus is only
  a positivity link. The env order is `clip(S − inventory_position)`.
- **Message ladder** (the object of study) — the message is one of:
  `raw` (the sender's last realized incoming demand — classical POS-data sharing, Cachon–Fisher) ·
  `dhat` (the demand belief, the Lee signal) · `ip` (inventory position, the VMI signal) ·
  `dhat_ip` (both named signals) · `learned` (a learned vector — the interpretability control).
  Routed along the topology with a one-step delay. The **named** rungs (`raw`/`dhat`/`ip`/`dhat_ip`)
  are stored detached, i.e. as fixed extended observations (plain MAPPO), which keeps `dhat`
  grounded by the aux loss and the honesty probe well-posed. The **`learned`** rung is instead
  recomputed in-graph in the update (**DIAL**, Foerster et al.), so the channel actually trains —
  a self-test asserts its params move (`dial_dW`/`dial_dgain`).
- **Economics layer** (rides on the reward, independent of the agent) —
  `rᵢ = −(own_costᵢ + β·others_cost) − τ·backlogᵢ (+ credit to customer)`.
  `β = 1` cooperative (team cost) ↔ `β = 0` self-interested (price of anarchy);
  `τ = 0` no contract ↔ `τ = τ\* = p − b` the coordinating transfer.

---

## Install

**One-shot (recommended, esp. on a fresh RunPod/Linux pod)** — detects CPU vs GPU, builds the
venv, installs + **verifies** everything, and runs the smoke tests:
```bash
chmod +x setup_pod.sh && ./setup_pod.sh          # GPU=0 to force CPU wheels; see the header for env knobs
```

**Manual (CPU torch — training is CPU-bound; the per-step encode is sequential):**
```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install matplotlib                 # for plot_curves.py (the learning-curve figures)
# On a GPU box, install the CUDA torch build for your toolkit instead of the CPU wheel.
```

**Lightweight (env + scripts + theory, no torch):**
```bash
python -m pip install numpy scipy pettingzoo gymnasium hydra-core omegaconf wandb pytest
```

---

## Smoke test

Each component runs standalone with a self-test (no W&B login needed for these):

```bash
python agents/signal_agent.py              # agent shape self-test (all 4 message rungs build/act/update; needs torch)
python scripts/c1_stats.py                 # C1 stats primitives self-test
python scripts/comm_stats.py               # value-of-sharing aggregator self-test
python scripts/demand_families.py          # AR(1)/NegBin sampler self-test
python scripts/coordination_theory.py      # regime-invariance theorem numerical check
python scripts/dp_optimum.py               # single-stage exact-DP optimum check
python scripts/baselines.py selftest       # serial base-stock optimizer self-test
python scripts/run_confirmatory_report.py  # confirmatory analyzer self-test (synthetic dumps)
python scripts/distill_symbolic.py         # symbolic-distillation self-test (linear backend, no PySR)
```

Before trusting any number, run a short training smoke and confirm the agent clears the static
base-stock at the held-out λ (competence is the floor, not the contribution — see Caveats).

---

## The research pipeline

The benchmark is generated **once** and read by every training run (do not regenerate per run).
Set `WANDB_MODE=disabled` to skip W&B, and `SIGNAL_CSVLOG=1` to write the per-run learning-curve
CSVs (`metrics_heldout.csv` = the true per-gate curve, `metrics_update.csv` = PPO diagnostics,
`run_meta.json` = provenance) into each `run_dir`. The CSV logger is pure observation — with the
flag unset, training is byte-identical to the frozen instrument. For the parallel sweep it is on by
default (see `sweep_all_hypotheses.sh`); W&B is the interactive-single-run convenience.

```bash
# 1) BENCHMARK (once): static-base-stock BAR / per-lambda Oracle -> results/baselines_regime_v2.json
#    Regenerate this whenever you flip env.penalty_at_retailer_only — the cost scale changes.
python scripts/baselines.py regime --lambdas 6 10 14 18 22 --select-episodes 80 --eval-episodes 200

# 2) TRAIN. Held-out gate = VALIDATION lambdas [8,12,16,20] (leakage-safe; refs auto-loaded).
#    Checkpoints to weights_signal/run_signal_<id>_<algorithm>/signal_checkpoint_best.pt
#    --- cooperative, neighbour topology, share the demand belief:
python agents/train_signal.py agent=signal agent.msg_content=dhat agent.comm_topology=neighbor \
       seed=10 agent.algorithm=signal_dhat_s10 total_episodes=15000
#    --- no-comm control (IDENTICAL architecture, ADJ zeroed -> comm vs no-comm differ only in flow):
python agents/train_signal.py agent=signal agent.use_comm=false \
       seed=10 agent.algorithm=signal_nocomm_s10 total_episodes=15000

# 3) EVAL a checkpoint (standard benchmark + dashboard + C1 + comm decomposition + honesty probe)
python agents/eval_signal.py --ckpt weights_signal/run_signal_<id>_<algorithm>/signal_checkpoint_best.pt --full

# 4a) C1 across seeds: dump per-seed rows, then aggregate (Gap_Recovered CI, IQM, vs-Bayes).
python agents/eval_signal.py --ckpt <comm ckpt> --dump-c1 results/signal_c1
python scripts/c1_stats.py report --signal-dir results/signal_c1 --refs results/baselines_regime_v2.json

# 4b) VALUE OF SHARING across seeds: dump comm and no-comm arms, then aggregate (V_cost + TOST).
python agents/eval_signal.py --ckpt <comm ckpt>    --dump-comm results/comm_on
python agents/eval_signal.py --ckpt <nocomm ckpt>  --dump-comm results/comm_off
python scripts/comm_stats.py report --comm-dir results/comm_on --nocomm-dir results/comm_off

# 5) CONFIRMATORY report (C1 + bullwhip + communication tables in one locked script):
python scripts/run_confirmatory_report.py --signal-dir results/signal_c1 \
       --refs results/baselines_regime_v2.json --comm results/comm_off results/comm_on

# 6) LEARNING-CURVE figures from the per-run CSVs (needs SIGNAL_CSVLOG=1 at train time).
#    Aggregates all matching seeds; default band = studentized bootstrap-95% CI.
python plot_curves.py \
       --arm comm   "weights_signal/run_signal_*_<comm_arm>_s*/metrics_heldout.csv" \
       --arm nocomm "weights_signal/run_signal_*_<nocomm_arm>_s*/metrics_heldout.csv" \
       --out fig.pdf                     # --metric {gap_recovered,forecast_error,positive_listening,...}
```

**Full registered sweep.** The steps above are the manual path; `sweep_all_hypotheses.sh` runs all
~435 arms across 15 seeds on a parallel CPU pod and drives the same producers/stats through its
`STAGE={train,dump,probe,analyze}` phases. See `setup_pod.sh` (which ends with a commented
data-gen tutorial) and the header of `sweep_all_hypotheses.sh`. Quick path on a pod:
```bash
./setup_pod.sh                                             # install + verify + smoke
python scripts/baselines.py regime                        # refs (behavioral cost model)
DRYRUN=1 ./sweep_all_hypotheses.sh                        # preview the plan
NPROC=48 PHASES="A B Bnull C E" ./sweep_all_hypotheses.sh # train the behavioral core
STAGE=dump ./sweep_all_hypotheses.sh && STAGE=analyze ./sweep_all_hypotheses.sh
```

Train **≥10 seeds** per arm for the statistical headline (5 is a pilot — the nonparametric p is
floored ≈0.06 at n=5).

### Experiment map (knob → question)

| Question | Knob(s) | Confirming metric |
|---|---|---|
| Does sharing help at all? (H1 null) | `use_comm` on/off | `comm_stats`: V_cost + **TOST** equivalence band |
| Value vs autocorrelation (H2) | `agent.train_env=ar1 agent.ar1_rho=…` | V_cost rising with ρ |
| Mechanism — *where* it helps (H3) | comm on/off | per-echelon forecast-error delta, concentrated upstream |
| Topology geometry (H4) | `agent.comm_topology` (incl. `no_neighbor` placebo) | retailer_broadcast ≥ neighbor > placebo ≈ 0 |
| What to share (H5) | `agent.msg_content={dhat,ip,dhat_ip,learned}` | dhat vs ip vs both; learned as the control |
| Price of anarchy / contract (strategic) | `agent.srdqn_beta`, `agent.tau` | cost at β=0,τ=0 vs β=0,τ=τ\*; `coordination_theory` |

---

## Configuration

Hydra composes `conf/config.yaml` + `conf/agent/signal.yaml`; override anything on the CLI
(`key=value`). Forward slashes in paths work on Windows.

| Override | Values | Meaning |
|---|---|---|
| `agent.use_comm` | `true` \| `false` | enable messages (false zeroes the ADJ — no-comm control) |
| `agent.msg_content` | `raw` \| `dhat` \| `ip` \| `dhat_ip` \| `learned` | the semantic message ladder (`raw` = last incoming demand) |
| `agent.comm_topology` | `neighbor` \| `skip` \| `full` \| `retailer_broadcast` \| `no_neighbor` \| … | message routing (+ geometry variants) |
| `agent.srdqn_beta` | `0.0`..`1.0` | reward sharing: 1 cooperative, 0 self-interested |
| `agent.tau` | float | coordinating transfer on upstream backorders (τ\* = p − b) |
| `agent.train_env` | `dr_poisson` \| `ar1` \| `family` | training demand process |
| `agent.ar1_rho` | float | AR(1) autocorrelation for the H2 sweep |
| `env.penalty_at_retailer_only` | `false` \| `true` | beer-game cost vs canonical Clark–Scarf cost |
| `agent.heldout_mode` | `poisson` \| `ar1` | gate on held-out λ or held-out ρ |
| `total_episodes`, `seed` | int | run length / RNG seed (top-level, not under `agent.`) |

---

## Testing

```bash
./setup_pod.sh                        # full board: verify install + all self-tests + end-to-end smoke
python agents/signal_agent.py         # agent shape self-test (needs torch)
# python test/test_beer_game_env.py   # env unit tests [TODO — test/ not yet written]
# python test/test_signal.py          # agent integration tests [TODO]
```

---

## Status

- **Done & self-tested:** env, agent (`signal_agent.py`, shape + DIAL + policy-gradient self-test),
  trainer (`train_signal.py`, py-compiled + end-to-end smoke in `setup_pod.sh`), `eval_signal.py`
  (py-compiled), `signal_csvlog.py` (byte-identical-when-off verified; smoke-tested), `plot_curves.py`
  (ingest/aggregate tested), `demand_randomization.py`, `baselines.py`, `demand_families.py`,
  `c1_stats.py`, `comm_stats.py`, `coordination_theory.py`, `dp_optimum.py`, `prereg.py`,
  `run_confirmatory_report.py`, `distill_symbolic.py`. Orchestration: `setup_pod.sh`,
  `sweep_all_hypotheses.sh` (bash-linted).
- **TODO:** `test/` package — `test_beer_game_env.py` (env unit tests) and `test_signal.py` (agent
  integration tests). The directory does not exist yet.

### Caveats
- **Reference refs vs. gate λ.** `baselines.py regime` defaults to the TEST λ `[6,10,14,18,22]`. The
  training gate scores the VALIDATION λ `[8,12,16,20]`, which are not in that JSON, so the in-training
  Gap_Recovered falls back to the config scalars. That is fine for checkpoint selection (a monotone
  gate signal); the headline numbers come from `eval_signal.py` + `c1_stats.py` on the TEST λ. If you
  want the gate to read the JSON, also run `baselines.py regime --lambdas 8 12 16 20`.
- **Truncation vs termination.** Episodes end by time-limit truncation, but the GAE in
  `SIGNALTrainer` treats the final step as terminal (no value bootstrap). With a fixed horizon this is
  a small constant bias; see the inline note in `signal_agent.py`.
