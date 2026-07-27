# PAPER SKELETON v3 — "SIGNAL"
### Target: M&SOM / Management Science. Updated for: frontier result, certification-failure mechanism, repaired content triangle, gated QMIX program, v3 fresh-seed replication.
Placeholders in [BRACKETS] are filled by the v3 run; where the outcome branches, every
branch's text is pre-written so no result forces improvisation.

**What changed vs the v2 skeleton:** (1) the dhat story is now a two-stage narrative —
certification failure as a *methodological finding*, then the repaired triangle; (2) the
transfer 2×2 and P2 falsification are promoted into the results spine; (3) F_GEOMETRY /
F_INCENTIVE are demoted to registered dhat-channel nulls (appendix) unless rerun on the raw
channel later; (4) a Reproducibility statement and the v3 replication are first-class.

---
## 0. Title candidates
1. *What Makes Shared Demand Information Usable? Raw Signals Beat Derived Forecasts in
   Multi-Agent Supply Chains*
2. *The Value of Usable Information: Content, Timeliness, and Architecture in Learned
   Supply-Chain Communication*
3. *Share the Data, Not the Forecast: A Preregistered Study of Demand-Information Sharing
   with Learning Agents*

## 1. Abstract (template, ~200 words)
Sixty years of supply-chain theory recommends sharing demand information to tame the
bullwhip effect, usually as forecasts. We ask a sharper question: *which encodings of the
same information can learning agents actually use?* In a preregistered study of a
four-echelon beer game with typed communication (CTDE-MAPPO; n=25 seeds; BCa/TOST/Holm),
broadcasting the retailer's raw point-of-sale datum recovers +11% of team cost
(V=+462, adj. p=8×10⁻⁶) and reaches the cost frontier of a privileged conditional
base-stock oracle — while every derived encoding (a learned forecast head, inventory
position, a learned differentiable code) delivers ≈0. A post-unblinding audit traced the
forecast head's null to optimizer coupling that collapsed it to a near-constant — itself a
caution for auxiliary heads in policy-gradient training — motivating a certified, frozen
forecaster and an analytic-predictor control. In the v3 replication (fresh seeds, single
code hash), [BRANCH-A: the certified forecast remains ≈0 while raw replicates at +[x] —
even a near-optimal one-step forecast is not what upstream agents need; the realized
trajectory is / BRANCH-B: the certified forecast recovers +[x], attributing the original
null to the defect]. Mechanism probes preregistered under order-censoring falsify the
information-recovery account (capture 0%); training-environment transfer, not evaluation
environment, determines behavior. Value is algorithm-conditional: [QMIX branch]. We
conclude that *usability* — content, timeliness, routing, learnability, architecture — not
information content per se, governs the value of sharing.

## 2. Introduction (~4 pp)
- Hook: the info-sharing canon (Lee et al. 1997; Cachon–Fisher 2000; Chen et al.) tells
  firms to share; modern practice increasingly delegates ordering to learning agents. What
  should be *transmitted*?
- The object of study: V(·), the value of a message channel, examined along five axes —
  content, timeliness (ρ), routing/geometry, learnability, architecture.
- Preview of results in strategic order: frontier-reaching raw value; derived-encoding
  nulls incl. the certification story; falsified mechanism + transfer causality;
  reliance≠value; algorithm conditionality.
- Contributions (C1–C6):
  C1 Raw-POS broadcast attains the privileged conditional-base-stock frontier (+11%).
  C2 The content ladder: derived encodings of the same information deliver ≈0 —
     [strength depends on triangle branch].
  C3 A certification methodology for auxiliary forecast heads + the failure mechanism
     (policy-optimizer capture of aux heads; init-anchored constant collapse).
  C4 Preregistered mechanism falsification under order censoring (P2) + transfer 2×2
     identifying training-environment causality.
  C5 Reliance ≠ value: a channel can be load-bearing for behavior and worthless for cost.
  C6 Algorithm conditionality with a predeclared competence gate ([concordance /
     unadjudicable-by-rule]).
- Transparency paragraph: two-stage design (registered v2 campaign; post-unblinding
  repair study with its own frozen manifest; v3 single-hash replication).

## 3. Related work (~2.5 pp)
(a) Information sharing in serial supply chains — bullwhip, VMI, forecast sharing.
(b) MARL communication — DIAL, CommNet, NDQ; grounding and positive listening.
(c) RL for inventory control — recent deep-RL beer-game and newsvendor lines.
(d) Preregistration & replication practice in ML — what this paper adopts.

## 4. Environment and the value object V (~3 pp)
- Four-echelon beer game: costs (h=0.5, b=1.0; behavioral vs canonical Clark–Scarf
  penalty), lead times, horizon 50; DR-Poisson and AR(1) (μ=12, σ=3, ρ-grid) demand.
- Observation/action; order-up-to actuation; the order-censoring treatment
  (obs_order_clip) as an experimental instrument.
- Typed channel: contents {raw, dhat, ip, dhat_ip, learned, condmean(=analytic AR
  predictor), lag rungs}; topologies incl. retailer_broadcast; one-step delay.
- Formal definition of V(channel) = E[cost(nocomm) − cost(channel)] under CRN pairing;
  the five axes as sections of one object.

## 5. Methods (~4 pp)
- 5.1 Learners: CTDE-MAPPO (primary); QMIX (exploratory replication spine) with the
  discrete order-up-to grid (41 points, s_max 160) — grid fairness established in §6.6.
- 5.2 The certified forecaster (repair): standalone GRU on demand history only;
  pretrain→certify→freeze; certification gate (MSE ratio ≤1.10 vs empirical conditional
  benchmark of the exact rounded/truncated DGP; bias, calibration, correlation,
  variance-floor); fail-closed loading; checkpoint-embedded identity. Achieved: ratio
  1.047, bias −0.018, SD 5.616/5.625, corr 0.879, slope 0.994.
- 5.3 The failure mechanism of the original head (for C3): aux MSE added to the PPO loss
  under one optimizer; d̂→base-stock coupling (init weight 5; dhat_init 14); measured
  collapse (pred SD 0.50 vs 5.52; bias +1.68; 25/25 fail).
- 5.4 Design & inference: three disjoint seed spaces; registered n=25 (power analysis);
  v3 manifest (dev 60–64; confirmatory 70–89); CRN eval (seed base 500000); BCa bootstrap,
  Wilcoxon, TOST equivalence, Holm across the family; all inference via
  scipy/statsmodels.
- 5.5 Baselines & frontier: analytic AR conditional/static base stock (privileged);
  grid-projected variants for QMIX fairness.
- 5.6 Registrations: v2.1 prereg (hash 81d4178d…052d), repair manifest (frozen
  2026-07-21), v3 tag + [git hash], addendum if scope extended.

## 6. Results (~8 pp)
- **6.1 Raw information reaches the privileged frontier.** V(raw)=+462 (+11.0%),
  adj. p=7.9×10⁻⁶; SIGNAL-raw 3745.8 vs privileged AR-CondBS 3747.6 → the learning-slack
  objection is closed for the money arm. v3 replication: V(raw)=[x] [CI]; frontier gap
  [x]. [If replication weaker: report both, discuss seed sensitivity honestly.]
- **6.2 The content ladder and the certification story.** Original ladder: learned, ip,
  dhat_ip, dhat all ≈0 (registered nulls). The dhat null decomposes: certification audit →
  defect (C3). Repaired triangle on fresh seeds:
  [BRANCH-A: certified dhat ≈0 AND arpred ≈0 while raw>0 ⇒ *even optimal one-step
   forecasts are not what upstream needs* — the trajectory-vs-statistic thesis, with
   dhatc−arpred as the (≈0) learning tax.]
  [BRANCH-B: certified dhat ≈ raw ⇒ the null was purely the defect; forecast sharing works
   when the forecast is real — reframe C2 accordingly; arpred locates how much is content
   vs learning.]
  [BRANCH-C: arpred>0 but dhatc≈0 ⇒ learning tax is the story; certification necessary but
   not sufficient.]
- **6.3 Mechanism: preregistered falsification and transfer causality.** P2 under
  censoring: V(12)=−0.2; Γ(12)=−462 [−577,−331]; capture 0% ⇒ registered M2 verdict
  (learning failure under censoring, not information recovery). Transfer 2×2: training
  environment determines everything (raw_tinf→eclip12: +91 vs train-native; eval env
  ~nothing). Frame as the causal identification of the paper.
- **6.4 Reliance ≠ value.** Learned channel passes the content-attribution gate (share
  0.31, p<10⁻⁶) with V≈0; do(m=0) on raw costs +2154. A methodological warning for
  emergent-communication evaluation.
- **6.5 Timeliness.** raw vs condmean vs [lag rungs if run]: the value of the *datum*
  degrades with [x] per period of delay [or: appendix if not run at scale].
- **6.6 Algorithm conditionality (exploratory, gated).** Grid exonerated (expressiveness
  +18.8 of +2855 = 0.7%); channel alive but maladaptive on the incompetent learner
  (V_qmix=−485; 17/25 nocomm-better). Variant program: [winner + qrw pair concordance
  read / UNADJUDICABLE by predeclared gate — reported as such].
- **6.7 Operational decomposition.** Raw shifts the holding–backorder frontier at every
  echelon (backorder −64%, holding −36%); upstream zero-order fraction +0.06–0.11 links
  to the censoring account; DP regime manufacturer-dominated (Lee-consistent).
- **6.8 Complements, not substitutes.** V(budget) slopes positive (+101 dp-raw, +131
  ar1-raw): information and training complement.

## 7. Discussion (~3 pp)
- Why trajectories beat statistics upstream: Blackwell/σ-algebra argument under order
  censoring (Prop. 1 spine) tied to 6.2–6.3.
- Design implications: share the data feed, not the forecast API [branch-conditional
  softening]; architecture matters as much as information (6.6).
- For MARL: certification of aux heads; reliance metrics are not value metrics.
- Limitations: single cost structure; CPU-scale learners; two-stage disclosure; QMIX
  program bounded by its gate.

## 8. Reproducibility statement (~0.5 pp)
Registrations + hashes (v2.1 prereg SHA, repair manifest timestamp, v3 git tag/hash);
seed-space table; certified-forecaster artifact + metrics; code release; environment
freezes (pod_env_v3.txt); one-command reproduction (`run_repair_study.py check plan …`).

## Appendices
A v2.1 preregistration + amendments · B power analysis (n*=25) · C robustness:
obs_order_clip treatment; heldout regimes · D full tables (all arms × axes; TOST panels) ·
E forecaster certification report + failure-mode audit of the original head ·
F QMIX: fair-grid benchmark table, gate verdicts, variant details · G geometry & incentive
axes: registered dhat-channel nulls (and rationale for demotion) · H prompts/decision log
excerpts for the two-stage disclosure.

## Figures & tables plan
F1 environment + channel schematic · F2 headline bar: nocomm/raw/frontier (with v3
replication overlay) · F3 content ladder incl. triangle [branch styling] · F4 P2 capture +
transfer 2×2 heat · F5 intervention (reliance vs value scatter) · F6 ops frontier shift ·
F7 substitution curves · T1 registered hypotheses↔outcomes ledger · T2 certification
metrics · T3 QMIX gates.

## Reviewer-objection ledger (pre-answered)
| Objection | Answered in |
|---|---|
| "Your agents are just undertrained; +11% is slack" | 6.1 frontier equality |
| "Was your forecast any good?" | 5.2 certification + arpred control (6.2) |
| "Post-hoc storytelling / p-hacking" | 5.6 registrations; T1 ledger; v3 fresh-seed replication |
| "One algorithm anecdote" | 6.6 gated program; UNADJUDICABLE branch is pre-registered honesty |
| "Censoring confound" | obs_order_clip treatment + P2 falsification (6.3) |
| "Seed leakage / selection" | three disjoint seed spaces; CRN; manifest freezes |
| "Small n" | n=25/n=20 with BCa+TOST; nonparametric floors acknowledged |
| "Why did the learned head fail?" | 5.3 mechanism + C3 framing |
| "Emergent-comm metrics say the channel matters" | 6.4 reliance≠value |
