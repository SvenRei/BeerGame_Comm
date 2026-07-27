# PHASE 3+4 IMPLEMENTATION GUIDE
### Post-unblinding repair study: QMIX competence program + repaired SIGNAL content study
**Audience:** the implementing agent (Claude Code / Opus-class) working in the local repo on
Windows. **Authority order:** this guide > reports/REPAIR_SEED_MANIFEST.json > code comments.
If reality contradicts this guide (a file is missing, a golden value mismatches), STOP and
report; do not improvise around a mismatch.

---
## 0. Mission and ground rules

**Phase 3 (QMIX):** determine whether any one-change training variant lifts QMIX to the
predeclared competence gate; only then re-adjudicate the V1 sign-concordance question on a
fresh confirmatory pair. The action grid is already exonerated (§3.1) — do not touch it.

**Phase 4 (SIGNAL):** train the 4-cell repaired content study — nocomm / raw / CERTIFIED
frozen dhat / analytic AR predictor — on fresh seeds, and produce reports/REPAIR_STUDY.md
(value table, registered contrasts, B5 forecast-competence block).

**Integrity constraints (non-negotiable):**
1. The original v2.x campaign is IMMUTABLE: never retrain, rename, edit, or delete any
   original arm, checkpoint, dump, or sentinel. Repair work uses NEW identifiers only:
   `r4_*`, `qr_*`, `qrw_*`.
2. Fresh seeds only, from the frozen manifest: dev {60..64}, confirmatory {70..89}.
   Forbidden: 25–54 (registered+quarantined), 500000+ (eval), 700000–730000 (forecaster).
3. Every output of this study carries the POST-UNBLINDING TARGETED FOLLOW-UP label
   (the manifest and analyze stage do this automatically — keep it that way).
4. Fail-closed everywhere: an uncertified forecaster must refuse to load; a missing seed file
   must abort analysis; a failed gate must block its confirm stage (only `--force` /
   `--force-arm` may override, and the override is recorded in the report).
5. Working style: verify before editing; run the test gates before any training; never invent
   Hydra keys (every key used below is registered in `conf/agent/signal.yaml` or top-level
   `conf/config.yaml`); complete files over fragments; stop and report on ambiguity.

---
## 1. Deliverable inventory (all PROVIDED and sandbox-verified)

| File | Role | Verifying gate |
|---|---|---|
| agents/demand_forecaster.py | standalone certified forecaster (GRU, init-to-mean, gate, fail-closed loader) | tests/test_forecaster.py T1–T6 |
| scripts/forecast_pretrain.py | pretrain+certify on the exact env DGP; exits nonzero unless CERTIFIED | its own exit code; §5 V0 |
| results/forecaster_ar1r9.pt (+_metrics.json) | **canonical certified artifact** (ratio 1.047, bias −0.018, SD 5.616/5.625, corr 0.879, slope 0.994). Use THIS file on every machine for bit-identical frozen dhat; regenerate only if absent | check stage prints its metrics |
| agents/signal_agent.py | Phase-2: `forecast_mode=separate_frozen`, `dhat_ext` routing, aux gate, structural optimizer exclusion, `forecaster_payload()` | tests/test_phase2_integration.py T7–T12 |
| agents/eval_signal.py | mirrored frozen-dhat pipeline; rebuilds forecaster from ckpt payload; refuses frozen ckpt without payload | T11 |
| agents/train_signal.py | embeds `forecaster` payload in `signal_checkpoint_best.pt` | T11 + §5 V3 ckpt assertion |
| conf/agent/signal.yaml | keys registered: `forecast_mode: none`, `forecast_ckpt`, `qmix_double_q: false` | hydra accepts overrides (V3) |
| agents/qmix_agent.py | config-gated Double-Q target (`qmix_double_q`) | tests/test_phase34.py Q1–Q2 |
| scripts/qmix_grid_benchmark.py (+results/qmix_grid_benchmark.json) | C11 fair grid benchmark; gate reference | Q3 + §3.1 table |
| reports/REPAIR_SEED_MANIFEST.json | frozen 2026-07-21T16:47:56Z: arms, macros, seeds, predeclared gates, budget | Q4 |
| run_repair_study.py | staged resumable orchestrator (all stages chain-validated) | Q5 + §5 V3 smoke matrix |
| tests/test_phase34.py | Q1–Q5 | self |
| reports/DHAT_QMIX_REPAIR_PLAN.md | design history + changelog | read it |

If any file is missing, STOP: request redelivery of `signal_delivery_phase3.zip`. Do not
re-derive from memory; goldens below will not match a re-derivation.

---
## 2. Environment contract

- Repo root: `C:\Users\sven\OneDrive\Desktop\Beer_Game_Project9` (venv active). All commands
  run **from repo root**. Python = the venv interpreter.
- Required packages (all already used by prior campaigns): torch (CPU), hydra-core, omegaconf,
  numpy, scipy, statsmodels, pettingzoo, gymnasium, wandb (imported even when disabled),
  pyyaml. Quick probe: `python -c "import torch, hydra, wandb, scipy, statsmodels"`.
- Env vars: the orchestrator and its per-job wrapper set `WANDB_MODE=disabled` and
  `PYTHONUNBUFFERED=1` for every job, and `SIGNAL_CSVLOG=0` for QMIX jobs only. Never launch
  training outside the orchestrator unless you replicate these.
- **Launch grammar** (what the orchestrator emits; shown for audit, not for hand-typing):
  `python <entry> agent=signal seed=<S> total_episodes=<EP> agent.heldout_every=<HE>
  agent.heldout_episodes=8 agent.patience=2000 agent.budget_milestones=[1000,2000,4000,8000]
  <arm args> agent.algorithm=<arm>_s<S>` — entry is `agents/train_qmix.py` for arms from the
  manifest's `qmix_arms` table, else `agents/train_signal.py`. (If you ever hand-type in
  cmd.exe, quote the milestones token: `"agent.budget_milestones=[1000,2000,4000,8000]"`.)
- Run dirs: `weights_signal/run_signal_<wandbid>_<arm>_s<S>/signal_checkpoint_best.pt`.
  Checkpoint resolution is `run_local.latest_ck(arm, seed)` — regex-anchored on the arm token
  and the exact seed (s250 ≠ s25; a foreign `qmix_` prefix is rejected). Never glob by hand.
- Sentinels: `weights_signal/.done_<arm>_s<S>`, **ep-stamped** (file content = the
  total_episodes of the completed run). A stage skips a job only if the stamp matches its
  `--ep`; a smoke sentinel (`30`) can never satisfy the production run (`8000`) — you will
  see `[restamp] ... retraining`.
- `forecast_ckpt` is resolved relative to the CWD; running from repo root is therefore part
  of the contract (verified working with hydra 1.3, which does not chdir here).

---
## 3. Phase 3 — QMIX competence program

### 3.1 Fair grid benchmark (DONE — reference data; do not re-litigate the grid)
`python scripts/qmix_grid_benchmark.py --episodes 200` reproduces (CRN seed base 500000;
values are deterministic — expect equality to ±0.1):

| config | GridCondBS | gap vs continuous 3747.6 |
|---|---|---|
| continuous CondBS | 3747.6 | — |
| n=21, s_max=200 (code default, 10.0 spacing) | 3784.2 | +36.5 |
| **n=41, s_max=160 (EXECUTED G2, 4.0 spacing)** | **3766.4** | **+18.8** |
| n=81, s_max=120 (1.5) | 3721.1 | −26.5 |
| n=161, s_max=120 (0.75) | 3718.8 | −28.9 |

Split at the executed grid: measured qmix-nocomm 6603 → total gap +2855.4 =
**expressiveness +18.8 (0.7%) + learning +2836.6 (99.3%)**. Consequence: all variants keep
`agent.qmix_n_actions=41 agent.qmix_s_max=160`; the intervention surface is training
dynamics only.

### 3.2 The four one-change variants (manifest `qmix_arms`; nocomm; dev seeds first)
| arm | single change | targeted pathology |
|---|---|---|
| qr_base | none (exact G2 config) | matched dev baseline |
| qr_doubleq | `agent.qmix_double_q=true` | max-operator overestimation (vanilla target confirmed at source) |
| qr_replay | `agent.qmix_buffer=200` (yaml default 1500) | replay staleness over 8k episodes |
| qr_eps | `agent.qmix_eps_anneal=2500` (yaml 5000) | half the run spent semi-random |

Double-Q semantics (already implemented, `agents/qmix_agent.py`): with the flag on, the
online net's `argmax` selects the action and the target net evaluates it; with the flag off
the code is byte-identical vanilla (golden td **21.738173** on the seeded micro-batch, and
the synced-target identity — Double-Q ≡ vanilla when target==online — is asserted by Q2).

### 3.3 Predeclared gates (manifest `gates_predeclared`; evaluated by `qmix-gate`)
- absolute: variant dev-mean nocomm cost ≤ 1.20 × GridCondBS(41,160) = **4519.7**
- relative: ≤ 0.85 × qr_base dev-mean (variants only)
- winner = cheapest variant passing BOTH → earns the confirmatory pair
  `qrw_nocomm`/`qrw_raw` (raw pair = winner args with `agent.use_comm=false` swapped for
  `agent.use_comm=true agent.comm_topology=retailer_broadcast agent.msg_content=raw`).
- **If no variant passes: V1 sign-concordance is reported UNADJUDICABLE (comparison
  algorithm below its predeclared competence gate).** That is a legitimate, registered
  outcome — write it, do not chase it with ad-hoc tuning. `--force-arm` exists for
  exploratory continuation only and is stamped into the report as FORCED.

---
## 4. Phase 4 — repaired SIGNAL content study

### 4.1 Arms (manifest `signal_arms`; all use `agent.use_dhat_head=false` — A18 clean mode)
| arm | message content | expanded args (macros resolved) |
|---|---|---|
| r4_nocomm | — | `env.penalty_at_retailer_only=false agent.train_env=ar1 agent.ar1_rho=0.9 agent.heldout_mode=ar1 agent.use_comm=false agent.use_dhat_head=false` |
| r4_raw | last realized demand d_{t−1} | nocomm args with `agent.use_comm=true agent.comm_topology=retailer_broadcast agent.msg_content=raw` |
| r4_dhatc | **frozen CERTIFIED forecast** | raw args with `agent.msg_content=dhat agent.forecast_mode=separate_frozen agent.forecast_ckpt=results/forecaster_ar1r9.pt` |
| r4_arpred | analytic predictor μ+ρ(d−μ) | raw args with `agent.msg_content=condmean` |

Budget (registered): total_episodes 8000, patience 2000, heldout_every 200,
heldout_episodes 8, milestones [1000,2000,4000,8000]. Signal dev gate: 20/20 dev runs
complete AND r4_raw dev-mean < r4_nocomm dev-mean (direction sanity only — dev tunes
nothing).

### 4.2 Stage contracts (`run_repair_study.py <stages...>`)
| stage | trains/produces | blocks on | resume unit |
|---|---|---|---|
| check | runs 3 test suites; ensures artifact + grid json (auto-generates if absent) | — | — |
| plan | dry-run job list + wall estimate; exits 0 | — | — |
| refs | regenerates AR privileged references on the v3 eval streams → `results/baselines_ar_v3.json` (frontier/τ-grid diagnostic) | — | file exists |
| transfer | P2′ transfer 2×2 evals (make_env_override + `--env-json`): inf-trained raw @c12, c12-trained raw @inf → `repair_out/transfer/<cell>/seed<S>.json` | ckpts exist | per seed file |
| qmix-dev / signal-dev | 4×5 dev jobs each | — | ep-stamped sentinel |
| qmix-gate / signal-gate | dev eval dumps (`--gate-episodes`) → `repair_out/gates.json` + verdict lines | — | `repair_out/{qmix,signal}_dev/<arm>/seed<S>.json` |
| qmix-confirm | winner pair × confirmatory seeds | gate winner (or `--force-arm`, recorded) | sentinel |
| signal-confirm | 4 arms × confirmatory seeds | signal gate pass (or `--force`) | sentinel |
| dump | CRN eval of every confirmatory cell → `repair_out/v1/<cell>/seed<S>.json` (+`_ferr`,`_censor` sidecars; loader reads only `seed<S>.json` = `{"0.9": mean_team_cost}`) | ckpts exist | per seed file |
| analyze | `reports/REPAIR_STUDY.md` | every `seed<S>.json` present (fail-closed) | idempotent |

Flags: `--jobs N` (default cores−1), `--ep/--he` (budget override → smoke), `--arms ...`,
`--seeds-limit K`, `--gate-episodes` (default 100), `--dump-episodes` (default 200),
`--force`, `--force-arm ARM`, `--strict-gates`.

### 4.3 REPAIR_STUDY.md contract (produced by analyze)
1. **Value table** per arm vs r4_nocomm: mean cost, V, BCa 95% CI (scipy bootstrap; falls
   back to percentile on degenerate resamples), Wilcoxon p (seed-paired), P(V>0).
2. **Registered contrasts**: raw↔dhatc, raw↔arpred, dhatc↔arpred (the "learning tax"),
   each with paired BCa CI.
3. **B5 block**: certification metrics read from the r4_dhatc checkpoint's embedded
   payload. One frozen artifact serves every seed BY DESIGN → certification columns are
   seed-invariant; this is a feature (removes forecaster-quality variance), state it, don't
   "fix" it. No economic reading from any uncertified forecaster.
4. **QMIX pair** (only if qrw cells exist): winner name (or `FORCED past a failed gate`),
   nocomm/raw means, V with CI, and the exploratory-only caveat.

---
## 5. Verification protocol — run IN ORDER; every expected output was observed in the audited sandbox

**V0 — unit gates (≈2 min, no training):**
```
python -m tests.test_forecaster            -> "ALL FORECASTER GATE TESTS PASS"
python -m tests.test_phase2_integration    -> "ALL PHASE-2 INTEGRATION TESTS PASS"
python -m tests.test_phase34               -> "ALL PHASE-3/4 TESTS PASS"
```
Golden values that MUST hold exactly: T7 losses raw 64.1857 / dhat 65.1864 / learned
54.6990 / condmean 63.4751 (golden file auto-regenerates on a fresh machine → T7 then runs
in self-consistency mode: acceptable); Q1 td 21.738173; forecaster certification ratio
1.047 (from the shipped artifact — check stage prints it).

**V1 —** `python run_repair_study.py check` → three PASS lines, artifact
`(MSE ratio 1.047, pass=True)`, `GridCondBS(41,160) = 3766.4`, manifest echo.

**V2 —** `python run_repair_study.py plan` → qmix-dev 20 / signal-dev 50 /
qmix-confirm 50 / signal-conf 250 jobs + example CLIs matching §2's grammar. (Counts
reflect manifest amendment A1: 10 SIGNAL arms, confirmatory n=25, seeds 70–94.)

**V3 — micro smoke matrix (≈20 min total at `--jobs 8`; validates the live chain):**
```
python run_repair_study.py signal-dev  --ep 30 --he 10 --seeds-limit 1 --jobs 4
python run_repair_study.py qmix-dev    --ep 30 --he 10 --seeds-limit 1 --jobs 4
python run_repair_study.py signal-gate qmix-gate --gate-episodes 5 --seeds-limit 1
python run_repair_study.py signal-confirm --force --ep 30 --he 10 --seeds-limit 2 --jobs 8
python run_repair_study.py qmix-confirm --force-arm qr_doubleq --ep 30 --he 10 --seeds-limit 2
python run_repair_study.py dump --seeds-limit 2 --dump-episodes 5
python run_repair_study.py analyze --seeds-limit 2
```
Expected: every train stage `N ok, 0 failed`; ckpt payload check
`python -c "import torch,glob; ck=torch.load(glob.glob('weights_signal/run_signal_*_r4_dhatc_s60/signal_checkpoint_best.pt')[0],map_location='cpu',weights_only=False); print(ck['config']['forecast_mode'], ck['forecaster']['certification']['pass'], ck['config']['use_dhat_head'])"`
→ `separate_frozen True False`; both gates FAIL (30-episode policies are garbage — the
fail is the gates working); confirm stages proceed only via the force flags; analyze prints
all four report sections with the QMIX winner line reading
`qr_doubleq (FORCED past a failed gate -- exploratory only)`.

**V4 — smoke cleanup (MANDATORY before production; touch ONLY repair-study prefixes):**
```
del /q weights_signal\.done_r4_* weights_signal\.done_qr_* weights_signal\.done_qrw_*
for /d %d in (weights_signal\run_signal_*_r4_*)  do rmdir /s /q "%d"
for /d %d in (weights_signal\run_signal_*_qr_*)  do rmdir /s /q "%d"
for /d %d in (weights_signal\run_signal_*_qrw_*) do rmdir /s /q "%d"
rmdir /s /q repair_out
```
(The ep-stamp would force retraining anyway; cleanup keeps gate dumps from ever mixing
smoke and production evals.) Never touch any other `weights_signal` content.

---
## 6. Production runbook

- **Night 1:** `python run_repair_study.py signal-dev qmix-dev --jobs 11` (40 jobs ≈ 6–7 h
  wall at −j11 on the 12-thread box; pools print live ETA — trust it over these estimates).
- **Morning:** `python run_repair_study.py signal-gate qmix-gate` (minutes). Decision
  matrix: signal gate PASS → signal-confirm tonight; FAIL → inspect
  `repair_out/logs/*.log` + dev dumps before anything else (do NOT force past an
  incomplete/inverted dev without a diagnosed cause). QMIX winner → qmix-confirm tonight;
  no winner → **skip qmix-confirm**, record UNADJUDICABLE, proceed with SIGNAL alone.
- **Night 2:** `python run_repair_study.py signal-confirm --jobs 11` (80 jobs ≈ 14–15 h;
  splits cleanly across two nights if thermals demand — sentinels resume) and, if earned,
  `qmix-confirm` (+40 jobs ≈ 5–6 h; run it the following night if both won).
- **After:** `python run_repair_study.py dump transfer analyze` (≈30–45 min for dumps, seconds for
  analyze) → `reports/REPAIR_STUDY.md` + archive `repair_out/` and `gates.json` alongside.

---
## 7. Failure triage

| symptom | cause | fix |
|---|---|---|
| every job fails instantly, log ends at `import hydra` / `import wandb` | venv missing dep | `pip install hydra-core wandb` in the venv |
| `Could not override 'agent.<key>'` | running an old `conf/agent/signal.yaml` | restore delivered yaml (keys §1) |
| `forecaster ... FAILED certification; refusing to load` | wrong/corrupt artifact | restore shipped `results/forecaster_ar1r9.pt`; only regenerate via `forecast_pretrain.py` (exit 0 required) |
| `checkpoint ... forecast_mode=separate_frozen but no forecaster payload` | ckpt from pre-Phase-2 code | retrain that arm with delivered `train_signal.py` |
| `[restamp] ... retraining` on production launch | leftover smoke sentinel | expected — or run §5 V4 cleanup |
| gate stage: `MISSING ckpt <arm> s<S>` | that dev job failed or is still running | check its sentinel + `repair_out/logs/` |
| analyze: `FAIL-CLOSED: <cell> missing seed<S>.json` | dump incomplete | rerun `dump` (resumable per file) |
| Wilcoxon p = nan at small n | ties/degenerate diffs | expected; BCa CI is primary at small n |
| qrw arms absent from dump/analyze | no gate winner and no force | expected under UNADJUDICABLE branch |

---
## 8. Definition of done
- [ ] V0–V3 all pass on the target machine; V4 cleanup executed.
- [ ] `python scripts/prereg_v3.py` printed VERIFIED with the committed hash BEFORE dev.
- [ ] Dev: 50 SIGNAL + 20 QMIX jobs sentinel-complete at ep 8000; both gate verdicts in
      `repair_out/gates.json`.
- [ ] Confirmatory: 250 SIGNAL jobs (+50 QMIX iff winner) complete; `dump` + `transfer`
      wrote every `repair_out/{v1,transfer}/<cell>/seed{70..94}.json`.
- [ ] `reports/REPAIR_STUDY.md` contains all sections of §4.3 with n=20, and names the QMIX
      winner or states UNADJUDICABLE.
- [ ] No file outside `r4_*/qr_*/qrw_*` namespaces, `repair_out/`, and `reports/` was
      created or modified.

## 9. Interpretation contract (for the write-up; do not exceed it)
- r4_raw vs r4_nocomm replicating +V on fresh seeds = the load-bearing replication.
- r4_dhatc ≈ r4_raw → the original dhat null was purely the defect. r4_dhatc ≈ 0 while raw
  > 0 → the null SURVIVES certification: even a near-optimal one-step forecast is not what
  upstream needs — the stronger finding; r4_arpred (analytic, learning-free) is the control
  that makes it airtight either way, and dhatc−arpred is the learning tax.
- QMIX: winner passing gates + qrw pair → re-read V1 concordance on competent policies;
  otherwise V1 is UNADJUDICABLE by predeclared rule — report it as such, full stop.
