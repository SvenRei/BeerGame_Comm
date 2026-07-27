# RESEARCH SKELETON v3 — the full program registry
### This is the RESEARCH skeleton (every registered object, its decision rule, its status, its v3 disposition). PAPER_SKELETON_v3.md is the journal-facing projection of this document; §9 maps between them.
Status legend: **SUPPORTED** (registered test passed) · **NULL-R** (registered null/equivalence
confirmed) · **FALSIFIED** (registered direction reversed — reported as such) ·
**DEAD-CHANNEL** (ran on the defective dhat; uninterpretable as registered) ·
**PENDING-v3** (decided by the repair/replication run) · **VERIFY** (outcome exists in the
v2 tables; re-read `analyze_n25` output before citing).
Authoritative sources: `scripts/prereg_v2.py` (v2.1 registration, self-hashed),
`reports/REPAIR_SEED_MANIFEST.json` (post-unblinding layer), v2 confirmatory tables.

---
## 0. The object and its axes
V(channel; regime, geometry, budget, β, algorithm) = E_seed[C(nocomm) − C(channel)] under
CRN pairing. One object, five axes:

| axis | design cells | v2 status | v3 disposition |
|---|---|---|---|
| content | raw / dhat / ip / dhat_ip / learned / condmean / eps / lags | raw SUPPORTED (+462, frontier); derived rungs ≈0; dhat = DEAD-CHANNEL cause discovered | r4 triangle (raw / certified-dhat / arpred) PENDING-v3 |
| timeliness (ρ, lags) | ρ-grid; raw_lag1/2; H-TIME | H2 ρ-slope ran on dhat → DEAD-CHANNEL; H-TIME lags VERIFY | optional raw-channel ρ-grid (menu §8) |
| geometry | topology suite TOSTs | dhat channel → DEAD-CHANNEL as registered | appendix as registered nulls; optional raw-channel rerun (§8) |
| incentive (β) | β ∈ {0, .5, 1} matched pairs | dhat channel → DEAD-CHANNEL | same as geometry |
| learnability/budget | milestones 1k/2k/4k/8k; substitution curves | complements (slopes +101 dp-raw, +131 ar-raw); truncated milestones excluded per registration | keep as v2 exploratory; v3 milestones logged automatically |
| algorithm | QMIX spine | registered as exploratory w/ convergence gate; raw cells failed competence (V=−485) | gated variant program PENDING-v3 |

---
## 1. Registered PRIMARY family {P1, P2, C-NULL} — joint Holm, α=.05
**P1 — the crossover (IU over four one-sided paired t-tests):**
[V_DP(dhat) − V_DP(raw) > 0] ∧ [V_AR.9(raw) − V_AR.9(dhat) > 0] ∧ [V_DP(dhat) > 0] ∧
[V_AR.9(raw) > 0]; p_P1 = max component p.
- AR legs: **SUPPORTED** — V_AR.9(raw)=+462, adj p=7.9e-06; raw−dhat contrast +430.5.
- DP legs: **DEAD-CHANNEL** — they require a *functioning* forecast under regime
  uncertainty; the head was a biased constant, so the registered crossover was never
  actually tested on its DP side. Consequence for claims: the paper's supported statement
  is the AR-side content result, NOT the registered two-regime crossover, unless resurrected
  (§8, decision D-DP). The v2 P1 verdict is reported exactly as the registration demands:
  IU fails ⇒ P1 not supported as registered; component evidence reported per leg.
**P2 — garbling:** registered Γ = V_AR.9^raw(clip12) − V^raw(inf) > 0. Measured Γ(12) =
−462 [−577, −331], 25/25 seeds ⇒ **FALSIFIED** (registered direction reversed). Mechanism
decomposition (exploratory, preregistered decomposition machinery): M2 verdict —
learning failure under censoring (capture(12) = −0.0%), not information recovery; the
ΔC_noc=+171 common own-obs penalty vs ΔC_raw=+633 split localizes the loss to the comm arm.
**C-NULL — dhat redundancy:** registered Schuirmann TOST |V_AR.9(dhat)| within ±2% of
nocomm cost. v2: equivalence held — but the *registered rationale* (information redundancy
given own history) is confounded by the defect (content was degenerate, not redundant).
Honest status: **NULL-R with contaminated interpretation → PENDING-v3**: r4_dhatc retests
C-NULL with certified content. Branch A (still ≈0) = C-NULL survives certification, the
redundancy reading strengthens massively; Branch B (value appears) = C-NULL was a
defect artifact; registered honesty requires reporting the reversal.
**Registered claim scope (keep verbatim in the paper):** P1/P2 are claims about EXPECTED V;
AR9-raw CoV ≈ 0.9, so a nontrivial fraction of individual runs show V ≤ 0 — the supported
statement is "sharing lowers expected cost," never "information always helps," with the
registered v_distribution panel (P(V>0), deciles, min/max) alongside.

## 2. Registered SECONDARIES (one Holm family of six)
| ID | registered rule | status |
|---|---|---|
| H-REP-a | raw ~ eps TOST ±2% (same-information equivalence) | VERIFY (v2 tables) |
| H-REP-b | raw ~ ar1_linear_predictor (=condmean) TOST ±2% | VERIFY; fresh-seed retest = r4_arpred vs r4_raw (PENDING-v3) |
| H-TIME | raw > lag1 > lag2, one-sided | VERIFY |
| H-SOURCE | upstream_raw > downstream_raw, one-sided | VERIFY |
| P2-dose | Γ(12) ≥ Γ(20), one-sided | direction inherits the P2 reversal: measured V(20)=+88.7, V(12)=−0.2 ⇒ dose-monotone *degradation*, i.e., informative under the falsified sign — report descriptively with the reversal |
Registration note the planner already made: H-REP planning sd (261) is strongly
conservative (affine transforms of one stream) — inconclusive H-REP is reportable as such.

## 3. v1-frozen secondary analyzers (re-run on fresh data, rules frozen at v1.2)
Geometry positives + placebo TOSTs; F_INCENTIVE matched-β pairs; C1 positive control
(per-echelon BAR). All three executed on the **dhat channel** ⇒ **DEAD-CHANNEL** as
registered evidence; disposition = report as registered outcomes with the channel-defect
caveat (Appendix G) + optional raw-channel resurrection (§8, D-GEO/D-BETA).

## 4. Registered interpretation diagnostics
**Optimality gap** (registered, DIAGNOSTIC ONLY, τ grid {.10, .20, .30}; symmetric refs:
DP per-λ Oracle; AR = AR_BestBS = min(CondBS, StaticBS), privileged): AR side — SIGNAL-raw
3745.8 vs CondBS 3747.6 ⇒ **information-value reading passes at every τ, including a gap
of ≈0** (the frontier claim, C1 of the paper). DP side — arms ~65% above Oracle ⇒ V_DP is a
LOWER BOUND contaminated by learning gap at every τ; the paper must say so.
**v_distribution:** registered panel; keep in every primary table.

## 5. Exploratory registry (registered demotions + preregistered probes)
- QMIX: registered as exploratory with a per-run convergence gate (improve ≥ once at
  ep≥2000 AND best < 1.25× matched MAPPO run minimum). v2 raw cells: channel alive
  (msg SD 6.6, flip 0.604), value negative (−485), competence gap 99.3% learning (fair
  grid benchmark: expressiveness +18.8 of +2855). v3: one-change variant program
  {base, double-Q, replay-200, ε-2500} behind the manifest's competence gate; V1
  sign-concordance re-read ONLY on a gate-passing winner; else **UNADJUDICABLE by rule**.
- Registered exploratory demotions honored: dhat_ip & learned rungs (no
  architecture-matched nocomm), QMIX dhat cells, H3 forecast-error mechanism, truncated-
  milestone substitution points, learned-channel H-SEM scope.
- Preregistered-machinery exploratory results (report as exploratory, they carry the
  narrative): intervention content gates (reliance≠value: learned passes attribution
  p<1e-6 with V≈0; do(m=0) on raw costs +2154); transfer 2×2 (training env determines
  everything: raw_tinf→eclip12 +91 vs train-native; eval-env ≈ nothing); ops decomposition
  (raw cuts backorder ~64% AND holding ~36% at every echelon; upstream zero-order-frac
  +0.06–0.11 ties to censoring; DP manufacturer-dominated, Lee-consistent).

## 6. Post-unblinding layer (its own frozen manifest; labels mandatory)
- Defect audit: pred SD 0.50 vs benchmark 5.52; bias +1.68; 25/25 fail — mechanism:
  aux-in-policy-optimizer + d̂→S-head coupling (init 5.0) + dhat_init 14 anchor.
- Certification method (contribution C3): pretrain→certify→freeze; gate
  {ratio ≤1.10, |bias| ≤.5, slope .8–1.2, corr ≥.8, SD ≥50% bench}; achieved 1.047 / −0.018
  / .994 / .879 / 5.616. Fail-closed loading; checkpoint-embedded identity.
- Repair cells → registry mapping: r4_raw & r4_nocomm ⇒ fresh-seed replication of the P1
  AR leg; r4_dhatc ⇒ C-NULL under certification; r4_arpred ⇒ H-REP-b fresh; dhatc−arpred ⇒
  the learning-tax decomposition (new, exploratory); qr_*/qrw_* ⇒ V1 program.
- Seeds: dev 60–64, confirmatory 70–89; disjoint from 25–49 (registered), 50–54
  (quarantined), 100000+/500000+ (gate/eval), 700000–730000 (forecaster). Frozen
  2026-07-21.

## 7. Theory spine
Proposition 1 (scoped analytically): null recovered in the invertible limit; strict
information gap under order censoring (Blackwell/σ-algebra argument); gap increasing in ρ.
Ties to: P2 mechanism section (6.3 of the paper) and the trajectory-vs-statistic reading of
the content triangle. Status: drafted argument — [finalize proofs for Appendix].

## 8. Open scope decisions (each = manifest rows + prereg addendum; NO new code)
| ID | what | why | cost @ −j60 |
|---|---|---|---|
| D-LADDER | + r4_learned, r4_ip on seeds 70–89 | full content ladder under one hash | +40 jobs ≈ 1 h |
| D-DP | DP-regime pair with a DP-certified forecaster (deferred A17: Poisson/regime benchmark + certification) + dp_raw/dp_dhatc/dp_nocomm | resurrects P1's crossover legs — the ONLY path back to the registered two-regime headline | forecaster work ~half day + ~60 jobs ≈ 1.5 h |
| D-RHO | raw-channel ρ-grid {0, .3, .6} + matched nocomm | V(ρ) on the live channel (H2 successor) | +120 jobs ≈ 3 h |
| D-GEO | 2–3 topologies on raw | V(geometry) resurrection | +80–120 jobs |
| D-BETA | β ∈ {0, .5, 1} on raw + matched | V(β) resurrection | +120 jobs |
Recommendation: D-LADDER now; D-DP decide by whether the paper leads with "crossover" or
with "raw beats derived encodings" (current skeleton assumes the latter); D-RHO/GEO/BETA
only against the thesis clock.

## 9. Registry → paper mapping
P1-AR + optimality gap → §6.1 · content ladder + C-NULL branches + repair triangle → §6.2 ·
P2 + M-verdict + transfer → §6.3 · intervention gates → §6.4 · H-TIME/H-REP → §6.5 (or
merged into 6.2) · QMIX/V1 → §6.6 · ops → §6.7 · budget → §6.8 · geometry/incentive
DEAD-CHANNEL outcomes → Appendix G · certification + defect audit → §5.2–5.3 + Appendix E ·
Prop 1 → §7 + Appendix · registration verbatims (claim_scope, v_distribution, τ-grid rule)
→ quoted in §5.4/§6.1.
