# RUNPOD v3 FINAL-RUN TUTORIAL
### For the student executing the run. Written to the standard the resulting paper will be reviewed at: every step names the validity threat it controls.

**What this run is.** One clean, tagged, preregistered execution of the repair study on a
RunPod instance. It delivers, under a single code hash: (1) a fresh-seed replication of the
core claim (raw-POS broadcast value vs no-comm), (2) the clean content triangle (raw vs
CERTIFIED learned forecast vs analytic predictor), (3) the gated QMIX competence program.
Nothing needs to be *built* — Phases 1–4 are implemented and chain-validated
(reports/PHASE34_IMPLEMENTATION_GUIDE.md is the executing agent's contract; this tutorial is
the human-level procedure around it).

**The three commitments you make by launching** (a reviewer will hold you to all three):
1. *Replication ethics:* seeds 70–89 retest the headline claim from scratch. You report the
   outcome whatever it is. A wobbling replication goes in the paper next to the original.
2. *Gates are binding:* the QMIX competence gate and the SIGNAL dev gate were predeclared in
   the frozen manifest. "No winner → V1 UNADJUDICABLE" is a reportable scientific outcome,
   not a bug to tune away. `--force*` flags exist for exploration only and are stamped into
   the report as FORCED.
3. *No post-tag edits:* once v3.0 is tagged and the campaign starts, code does not change.
   If something breaks, you stop, diagnose, fix, re-tag v3.1, and restart the affected
   stage — you never patch mid-campaign and resume.

---
## Step 1 — Freeze the scope (local, ~30 min, YOUR decision)
Default scope = the manifest as delivered: 4 SIGNAL arms (r4_nocomm, r4_raw, r4_dhatc,
r4_arpred) × seeds {dev 60–64, confirmatory 70–89} + 4 QMIX variants (qr_base, qr_doubleq,
qr_replay, qr_eps) + winner pair.

**The journal-run scope is now IMPLEMENTED and REGISTERED** (manifest amendment A1 +
scripts/prereg_v3.py, content-hash 2ac8be23…88ee45): 10 SIGNAL arms — the r4 core four, the
ladder (r4_learned, r4_ip), and the P2′ worlds (r4_{nocomm,raw}_{c20,c12}) — plus the 4-variant
QMIX program; confirmatory n=25 (seeds 70–94, honoring the v2.1 registered power fallback).
Your Step-1 job is now only: read prereg_v3, agree, run `python scripts/prereg_v3.py` (must
print VERIFIED), commit `reports/PREREG_v3.json`. Any further extension (D-DP, D-RHO, D-GEO,
D-BETA) is a pre-run AMENDMENT: edit manifest + registration together, re-hash, document —
never after data exists. **Do not create new seeds; all arms share 60–64 / 70–94 (pairing).**
*Threat controlled: garden of forking paths; broken pairing; silent scope drift.*

## Step 2 — Clean and tag: the "new hash" (local, Claude Code executes, ~1 h)
1. `git checkout -b v3-clean`
2. Apply the review dispositions (comment-only review, now enforced):
   - `mkdir scripts/legacy_v2` and move: `dp_optimum.py, coordination_theory.py,
     distill_symbolic.py, plot_curves.py` plus root `check_frozen.py`. **KEEP in place:**
     `clipped_refs.py, p2_decompose.py, decompose_costs.py, make_env_override.py` — they
     are in the v3 run path (P2′/transfer/ops; the transfer stage shells make_env_override). Move `prereg.py`
     (the v1 registration) there too — it is a historical record: never edit it, never
     delete it.
   - Delete every `__pycache__/` and `_repair_job.py` (runtime-generated); add both plus
     `repair_out/` and `weights_signal/` to `.gitignore`.
   - Keep in place (they are the run + science path): env + demand_families; all of
     `agents/`; conf; `scripts/{baselines, qmix_dump, forecast_pretrain,
     qmix_grid_benchmark, c1_stats, comm_stats, confirmatory_v2, prereg_v2,
     verify_manifest, run_confirmatory_report}.py`; `run_repair_study.py` + `run_local.py`
     (the orchestrator imports pool_run/latest_ck from it — leave the coupling alone until
     after the run); `setup_pod.sh`; all tests; readme; decision log; reports.
3. Commit the frozen scientific inputs INTO the repo (small, hash-pinned):
   `results/forecaster_ar1r9.pt`, `results/forecaster_ar1r9_metrics.json`,
   `results/qmix_grid_benchmark.json`. The shipped forecaster is canonical — every machine
   must run the same bytes of frozen dhat.
4. Full local test battery (all must pass):
   `python -m tests.test_forecaster` · `python -m tests.test_phase2_integration` ·
   `python -m tests.test_phase34` (incl. Q6 prereg cross-check) · `python test_obs_clip.py` ·
   `python test_new_rungs.py` · `python scripts/prereg_v3.py` (→ VERIFIED)
5. `git add -A && git commit -m "v3.0 clean final-run codebase"` → `git tag v3.0` →
   record `git rev-parse HEAD` in a new `reports/V3_RUN_RECORD.md` (hash, date, scope
   decision from Step 1, manifest unchanged/extended).
*Threat controlled: "which code produced these numbers"; dead-code ambiguity.*

## Step 3 — Provision the pod (~30 min)
- Instance: CPU-optimized, ≥60 vCPU (campaign heritage: NPROC=64), ≥64 GB RAM, ≥50 GB disk.
- `git clone --branch v3.0 <your-repo> && cd <repo>` → `bash setup_pod.sh` →
  `pip install hydra-core` (verify the probe passes:
  `python -c "import torch,hydra,wandb,scipy,statsmodels,pettingzoo,gymnasium"`).
- Freeze the environment record: `pip freeze > reports/pod_env_v3.txt` (commit later with
  results).
*Threat controlled: environment nondeterminism; unrecorded dependency drift.*

## Step 4 — Verify before you burn hours (pod, ~30–40 min total)
Run the guide's protocol IN ORDER (§5 of PHASE34_IMPLEMENTATION_GUIDE.md):
- **V0** — the five test commands above. Note: T7 and Q1 goldens self-generate on a fresh
  machine (self-consistency mode) — acceptable and expected; the artifact certification
  line must read `MSE ratio 1.047, pass=True`.
- **V1** — `python run_repair_study.py check` (three PASS lines, artifact metrics,
  `GridCondBS(41,160) = 3766.4`, manifest echo).
- **V2** — `python run_repair_study.py plan` (job counts 20 / 20 / 40 / 80 for the default
  scope; +40 signal-dev/conf split if you extended).
- **V3** — the seven-command micro-smoke matrix. Expected: every train stage `N ok, 0
  failed`; **both gates FAIL** — 30-episode policies are garbage and the fail is the gates
  working; analyze renders all report sections with the forced-winner wording.
- **V4** — smoke cleanup, bash form (repair namespaces ONLY — never touch anything else):
```
rm -f  weights_signal/.done_r4_* weights_signal/.done_qr_* weights_signal/.done_qrw_*
rm -rf weights_signal/run_signal_*_r4_* weights_signal/run_signal_*_qr_* \
       weights_signal/run_signal_*_qrw_* repair_out
```
*Threat controlled: burning a pod night on a broken chain; smoke evals contaminating
production dumps (the ep-stamped sentinels are the mechanical backstop; V4 is hygiene).*

## Step 5 — Launch
- **Before launch:** `python run_repair_study.py check refs` (provenance record + fresh
  frontier references — check must show the prereg hash).
- **Dev (both programs, ~2.5 h at −j60; 70 jobs):**
  `nohup python run_repair_study.py signal-dev qmix-dev --jobs 60 > repair_out/night1.log 2>&1 &`
  (tmux or nohup; the pool prints live ETA — trust it over estimates).
- **Gates (minutes):** `python run_repair_study.py signal-gate qmix-gate`, then the
  decision matrix:

| outcome | action |
|---|---|
| signal gate PASS | run signal-confirm |
| signal gate FAIL | STOP. Read `repair_out/logs/*.log` + dev dumps; diagnose before any force. An incomplete or inverted dev at 8000 episodes is a finding to understand, not to override. |
| QMIX winner found | run qmix-confirm (winner pair auto-built) |
| no QMIX winner | skip qmix-confirm; record UNADJUDICABLE in V3_RUN_RECORD.md; proceed with SIGNAL alone |

- **Confirms (~9–10 h at −j60; 250 SIGNAL + 50 QMIX-if-winner):**
  `python run_repair_study.py signal-confirm --jobs 60` and, only if earned,
  `python run_repair_study.py qmix-confirm --jobs 60`.
- **Harvest stages:** `python run_repair_study.py dump transfer analyze`.
Everything is sentinel-resumable; a pod hiccup costs only the in-flight jobs.
*Threat controlled: mid-run meddling; untracked forced continuations.*

## Step 6 — Harvest and archive (~30 min)
- Completeness: every expected sentinel; every `repair_out/v1/<cell>/seed{70..89}.json`
  (analyze fail-closes on gaps — if it printed the report, the set is complete).
- Pull to durable storage: `reports/REPAIR_STUDY.md`, `repair_out/` (incl. `gates.json`
  and logs), all `r4_*/qr_*/qrw_*` best checkpoints, `pod_env_v3.txt`, night logs.
- Finish `reports/V3_RUN_RECORD.md`: hash, scope, gate verdicts, wall clock, anomalies.
  Commit record + results metadata to the repo (results branch is fine).
*Threat controlled: unarchivable claims; "trust me" numbers.*

## Step 7 — Anti-patterns (the reviewer's checklist of ways to fail)
Re-rolling seeds after the manifest froze · running arms on non-identical seed sets ·
editing code after the tag and resuming · reporting a FORCED continuation as if gated ·
suppressing an unfavorable replication · comparing arms on different CRN eval episodes ·
deleting legacy analysis scripts before the final analysis is written.

## Step 8 — Who does what
- **You:** Step 1 scope decision; Step 5 gate decisions; Step 6 archive sign-off.
- **Claude Code (Opus):** executes Steps 2–6 mechanically against
  PHASE34_IMPLEMENTATION_GUIDE.md (§5 verification, §6 runbook) and this tutorial.
- **Claude (chat):** interprets `REPAIR_STUDY.md` when it comes back (guide §9 is the
  interpretation contract) and integrates it into the paper.

## Step 9 — Clock (default scope, −j60)
Scope-verify+clean+tag ~1.5 h · pod setup ~0.5 h · verify ~0.7 h · dev ~2.5 h · gates ~0.2 h ·
confirms ~9–10 h · dump+transfer+analyze ~1 h · harvest ~0.5 h ⇒ **one working day plus one
long pod night (confirms split cleanly across two nights if thermals demand — sentinels resume).**
