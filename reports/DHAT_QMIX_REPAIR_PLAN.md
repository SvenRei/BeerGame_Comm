# DHAT + QMIX REPAIR PLAN (post-unblinding; original campaign immutable)

Status: Phase 1 COMPLETE and validated. Phase 2 awaits approval (it edits train_signal.py,
which the working raw arms share). Everything here follows the external repair spec's
integrity constraints: new files, new dirs, fresh seeds, explicit post-unblinding labels,
no rerun of the original sweep, no edits to original checkpoints or results.

## 1. SIGNAL dhat data-flow map (defect confirmed at source)

* `agents/signal_agent.py:83`  `demand_estimate(h)` -- dhat is a linear+softplus readout of
  the POLICY belief-GRU hidden state `h` (the same `h` that feeds `base_stock`).
* `:57-62`  with `use_dhat_head`, dhat is appended to the base-stock head input with init
  weight `dhat_coef=5.0` -- PPO's preferred order-up-to level back-propagates INTO dhat.
  This is the mechanism for the measured +1.68 bias toward ~13.5 (~S/5), not just drift.
* `:396-409`  aux MSE `F.mse_loss(dhat, dtgt)` is ADDED to the PPO loss
  (`ploss = ploss + aux_coef*aux`, `aux_coef=0.1`).
* `:191-192`  ONE Adam over actors+critic -- no separate forecast optimizer, no isolation.
* `:260`  target `dtgt = infos[a]["training_targets"]["demand"]` (env-side; retailer target
  is the realized customer demand; upstream targets follow obs_order_clip/aux_target_clip).
* Checkpoint: forecaster has no independent identity; it lives inside actor state_dicts.

Diagnostic (seeds 25-49): pred SD 0.50 vs benchmark 5.52; median MSE ratio 4.29; bias +1.68;
25/25 fail the 1.25 screen. Conclusion adopted: dhat is a degenerate latent, and original
raw-vs-dhat contrasts are labeled "raw vs learned-forecast-head (failed certification)".

## 2. QMIX training data-flow map (channel alive; competence is the problem)

* `agents/qmix_agent.py:219`  reward = -team_cost / qmix_reward_scale(100) -- SIGN CORRECT.
* `:183-232`  full-episode replay entries; `:75-80` `q_sequence` recomputes hidden state over
  the WHOLE episode during updates -- no burn-in/truncation bug class (C4 largely moot).
* `:137-141,152`  hard target sync every 20 grad steps.
* `:131-133`  **action grid: `qmix_n_actions=21` over `[0, qmix_s_max=200]` -> 10-unit
  spacing on the order-up-to level.** Under AR(1) mu=12 the conditional base stock moves a
  few units per period; a 10-unit actuator cannot track it. This single number plausibly
  explains BOTH findings: the competence gap (nocomm 6603 vs privileged 3747) and the
  maladaptive-but-alive channel (flip rate 0.604 = raw messages flipping decisions across
  10-unit boundaries, injecting order variance instead of precision).

Adopted verdicts: channel is alive (msg SD 6.6, weight ratio 0.50, zero collapse
classifications) -- do NOT redesign the channel; fix competence first.

## 3. Evaluation of the external spec (adopted, with four sharpenings)

1. Part A architecture (separate forecaster, detach, pretrain->certify->freeze) is correct
   and now IMPLEMENTED; the certification thresholds (A14) are demonstrably achievable
   (measured ratio 1.047 <= 1.10 on held-out exact-DGP streams), so no renegotiation.
2. Elevate B3 cell 4: the deterministic AR linear-predictor broadcast ("arpred") is the
   load-bearing control. It is immune to every certification question and answers "was the
   forecast any good?" with an analytic yes. If certified-dhat AND arpred both deliver ~0
   while raw delivers +462, the content result upgrades to: even a (near-)optimal one-step
   forecast is not what upstream needs -- the realized trajectory is. That is a stronger
   paper than a successful repair would be.
3. Part C: skip straight to C6/C11/C16-step-3. C5 (reward sign) and the C4 bug class are
   already clean at source (map above). The grid experiment + a fair 21-point discrete
   benchmark (C11: same grid, scripted conditional base stock projected onto the grid) come
   before any of C7-C14; if the grid explains the gap, most of C dissolves.
4. Thesis-scope cuts: defer A17 (Poisson forecast benchmark), A19 (continuity mode),
   C7-C10/C12-C14 (full diagnostic program), and all Poisson B3 cells until the AR spine is
   done. Minimum viable follow-up study = 4 AR cells + 1 QMIX grid pair.

## 4. Phases

### Phase 1 -- standalone forecaster + gates  [DONE, all tests pass, artifact certified]
* `agents/demand_forecaster.py`  DemandForecaster (own GRU/head; init-to-mean via inverse
  softplus; no policy inputs), empirical conditional benchmark of the exact rounded/
  truncated DGP, metrics + certification gate, certified save/load (loader refuses
  uncertified artifacts).
* `tests/test_forecaster.py`  T1 temporal alignment; T2 gradient isolation (policy loss
  leaks nothing, forecast loss trains); T3 fixed-batch overfit (MSE 4.97 vs const 36.6,
  pred SD 5.60 -- proves the defect was the coupling, not the task); T4 checkpoint round
  trip + certification refusal; T5 init-to-mean + nonnegativity; T6 benchmark plumbing.
* `scripts/forecast_pretrain.py`  samples streams from the real env (demand exogenous ->
  any policy yields the true DGP), trains with early stopping, certifies on held-out
  streams, saves artifact; exits nonzero on gate failure (fail-closed for campaigns).
  Seed bases 700000/710000/720000/730000 -- disjoint from 25-54 and all eval spaces.
* Measured certification (rho=0.9): RMSE 3.027 vs bench 2.958, ratio 1.047, bias -0.018,
  SD 5.616/5.625, corr 0.879, slope 0.994 -> CERTIFIED.

### Phase 2 -- train_signal integration  [NEXT; the reviewed patch -- touches shared file]
One patch, config-gated, default-off so every existing arm is byte-identical:
* `forecast_mode: none|separate_frozen` (+ `forecast_ckpt: path`). In separate_frozen the
  trainer loads a CERTIFIED artifact (loader enforces the gate), keeps a per-agent forecast
  hidden state beside the policy hidden state, and dhat := frozen forecaster on the agent's
  observed demand stream. Frozen => no optimizer, no gradients (A5/A6/A7 satisfied
  structurally); dhat_used stored in the rollout buffer and replayed in updates (A10).
* Message content `dhat` in this mode emits the frozen certified forecast (detached by
  construction); content `arpred` (NEW rung) emits mu + rho*(d_obs - mu) clipped at 0 --
  deterministic, retailer-sourced, msg_dim 1.
* A18 clean mode: `use_dhat_head:false` everywhere in the repaired study; the internal-dhat
  original path stays untouched for continuity.
* Checkpoints save forecaster identity + certification metrics (A12).
* Tests added with the patch: config-off equivalence (bitwise same losses on a fixed batch
  vs current code), frozen-params invariance across an update, arpred determinism.

### Phase 3 -- QMIX grid experiment  [after Phase 2 lands; independent of it]
* New config ids only: `qmix_n_actions: 81` over `[0, 120]` (1.5-unit spacing) as variant 1;
  keep 21/[0,200] as the original. 3-5 dev seeds first (spec C17), competence gate = within
  an approved % of the fair discrete benchmark.
* Fair benchmark (C11): scripted AR conditional base stock PROJECTED onto each grid --
  separates "QMIX can't learn" from "the grid can't express the policy".
* Only if the fine-grid nocomm approaches the fair benchmark do we interpret QMIX comm
  arms; then rerun the raw pair on fresh seeds and revisit V1 concordance.

### Phase 4 -- repaired mini-study (B3 minimum, AR rho=0.9, fresh seeds)
Cells: nocomm / raw / certified-dhat / arpred (+ recommended: dhat_ip, raw lag-1).
Seed manifest (proposal, to be frozen in a dated file before launch): dev 60-64;
confirmatory 70-89 (n=20). ~4-6 cells x 20-25 seeds x 8k episodes -- one pod night or ~a
weekend locally; NOT the original sweep.
Required table per forecast cell (B5): certification metrics + economic value per seed; no
economic reading from any uncertified seed.

### Phase 5 -- paper integration
Original dhat arm reported as "learned forecast-head (failed certification)" -- itself a
methodological finding (aux heads inside policy optimizers degenerate while the policy still
benefits); repaired study reported as post-unblinding targeted follow-up with its own seed
manifest; V1 concordance revisited only on competent QMIX.

## 5. Definition-of-done mapping
Spec Part H "SIGNAL dhat fixed" items 1-9: DONE (Phase 1). Items "policy receives exact
certified forecast / grounded outgoing message / fresh-seed comparison": Phase 2 + 4.
Spec "QMIX repaired enough": Phase 3 gates.

## 6. Changelog
* 2026-07-26  Phase 3+4 infrastructure delivered and CHAIN-VALIDATED end-to-end in the audited
  sandbox (hydra installed; every stage exercised through the REAL CLIs):
  - scripts/qmix_grid_benchmark.py RUN: expressiveness gap at the EXECUTED (41,160) grid is
    +18.8 of the +2855 total qmix-nocomm gap (0.7%); grids down to 0.75-unit spacing are
    cost-neutral -> GRID HYPOTHESIS DEAD; 99.3% of the gap is LEARNING. Phase-3 pivots to
    one-change variants at the fixed executed grid: qr_base / qr_doubleq / qr_replay(200) /
    qr_eps(2500).
  - agents/qmix_agent.py: config-gated Double-Q (qmix_double_q, default false) -- vanilla
    golden td 21.738173 EXACT with flag off; synced-target identity + desync divergence proven
    (tests Q1/Q2).
  - reports/REPAIR_SEED_MANIFEST.json frozen 2026-07-21T16:47:56Z (dev 60-64, confirmatory
    70-89, predeclared gates incl. the V1-UNADJUDICABLE branch).
  - run_repair_study.py: staged resumable orchestrator; smoke matrix ALL PASS: check, plan,
    signal-dev (r4_dhatc ckpt embeds certified forecaster: pass=True ratio 1.047,
    use_dhat_head=False), qmix-dev (all 4 variants), signal-gate + qmix-gate (formulas fire;
    30-ep policies correctly FAIL both gates; blocking semantics verified), signal-confirm
    (resume verified mid-run), qmix-confirm --force-arm (winner_forced recorded for the paper
    trail), dump (12/12 incl. qrw pair; seed{S}.json + _ferr/_censor sidecars), analyze
    (REPAIR_STUDY.md: V table with BCa n=2 fallback, contrasts, B5 block, qrw section).
  - tests/test_phase34.py Q1-Q5 PASS (incl. orchestrator plan-mode dry run).
  - Handoff: reports/PHASE34_IMPLEMENTATION_GUIDE.md (implementation + verification +
    production runbook for the executing agent).
* 2026-07-21  Phase 2 delivered: forecast_mode=separate_frozen wired through signal_agent
  (message dhat_ext, frozen block outside the optimizer, aux gate), eval_signal (embedded-
  forecaster rebuild + mirrored step-0/lag pipeline), train_signal (forecaster payload in the
  best checkpoint), conf/agent/signal.yaml (keys registered: forecast_mode none default,
  forecast_ckpt). ARPRED RESOLUTION: the deterministic predictor already exists as the v1.3
  `condmean` rung (mu+rho*(d_{t-1}-mu), raw-matched step-0 convention) -- zero new agent code;
  T8 pins its semantics. Root-cause closure: conf default dhat_init: 14 explains the measured
  prediction mean 13.49 (the head barely left its init). Verification: golden equivalence vs
  pre-patch capture EXACT for raw/dhat/learned/condmean (a_loss 64.1857/65.1864/54.6990/
  63.4751); integration suite T7-T12 all PASS; env suite 11/11; forecaster suite all PASS.
  Phase-4 launch grammar (sweep emit style, all four arms with agent.use_dhat_head=false):
    nocomm   : $NOCOMM
    raw      : $(COMM retailer_broadcast) agent.msg_content=raw
    cert-dhat: $(COMM retailer_broadcast) agent.msg_content=dhat \
               agent.forecast_mode=separate_frozen agent.forecast_ckpt=results/forecaster_ar1r9.pt
    arpred   : $(COMM retailer_broadcast) agent.msg_content=condmean
  Precondition: python scripts/forecast_pretrain.py (exits nonzero unless CERTIFIED).
* 2026-07-21  Phase 1 delivered: demand_forecaster.py, tests/test_forecaster.py,
  forecast_pretrain.py; all six gates pass; first certified artifact produced
  (ratio 1.047). No existing training or analysis file modified.
