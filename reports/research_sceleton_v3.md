# SIGNAL — Research Skeleton & Defense Dossier, v2.0 (combined campaign)
### When is demand-information sharing redundant? Signal-level evidence from supply-chain echelons that must *learn* to infer
*Supersedes the v1.x skeleton. One registered campaign regenerates ALL confirmatory data (Studies 1+2 unified); the v1.1/v1.2 registrations (hashes cfae5dee…58b8, b9e9cf6e…cdc59) and the completed 540-run v1.2 campaign are cited as **prior evidence and variance priors**, never as confirmatory numbers. Canonical-cost Phase D/Dext (H7 τ*-contract) is **out of the story** (2026-07-19 decision). Target: M&SOM; MS conditional on the Proposition-1 appendix.*

> **[v3 STATUS BLOCK — 2026-07-26.]** The v2.0 campaign described below WAS EXECUTED
> (as registration v2.1, hash 81d4178d…052d, n*=25 fallback fired, seeds 25–49; seeds 50–54
> quarantined by the analysis-side whitelist fix). This v3 revision annotates every section
> with its executed outcome, the post-unblinding dhat defect + certification layer (§13), and
> the v3 single-run strategy (§14): **the journal receives only the v3 campaign**, with the
> v1.x/v2.1 runs disclosed as the registered pilot lineage (their hashes cited as history —
> exactly the convention this dossier already uses for v1.x). Legitimacy condition:
> pilot → preregister(v3, informed predictions disclosed) → one confirmatory run on FRESH
> seeds. Same-seed regeneration is excluded by design (re-observing draws that informed the
> predictions is not confirmation). Annotations are quoted blocks like this one; the
> original v2.0 text is unmodified.

---

## 0. Title candidates and abstract template
- *Redundancy Is a Property of Signals, Not Supply Chains: Learned Inference and the Value of Demand Sharing*
- *What to Share, Not Whether: Content-Level Value of Demand Information under Learned Inference*
- *The Crossover: When Raw Data Beats Forecasts in Multi-Echelon Ordering*

> **[v3]** Title decision is downstream of ONE scope decision (D-DP, §14): if the DP
> crossover leg is resurrected with a certified DP forecaster, keep *The Crossover*; if not,
> the supportable headline is content-asymmetry at the frontier — closest existing candidate:
> *What to Share, Not Whether*, with the abstract's crossover sentence replaced by: raw POS
> attains a privileged conditional-base-stock frontier (+11%) while every derived encoding
> — including a CERTIFIED one-step forecast and the analytic AR predictor — is null/pending
> per the r4 triangle branches. P2's sentence must flip to its falsified direction (§5
> annotation).

**Abstract template.** Classical results say demand sharing is redundant when upstream firms can invert the order stream (Raghunathan 2001; Gaur–Giloni–Seshadri 2005) and valuable when demand is persistent (Lee–So–Tang 2000). Both presume known-model inference. We re-pose the question for echelons that must *learn* to infer, using a frozen multi-agent RL instrument on the four-echelon beer game with a typed, audited communication channel. Prior campaign evidence (540 registered runs) found the classical *aggregate* predictions break while a sharper *signal-level* regularity emerges: a grounded demand forecast (d̂) carries value under regime uncertainty where raw data does not, and raw point-of-sale data carries value under persistence where the forecast is redundant. The registered campaign here tests that crossover as a conjunctive primary (P1), and tests its proposed mechanism — order-stream informativeness — by Blackwell-garbling the *observed* order signal while leaving physics untouched (P2). [Result sentences.] Robustness: sign concordance under a second learner class (recurrent QMIX). Implications: "should we share?" is ill-posed; the operational question is *which signal*, and the answer flips with the demand regime.

---

## 1. Introduction skeleton
1. Hook: the oldest quantified question in supply-chain theory — is sharing worth anything? — has answers that disagree by assumption: redundant (Raghunathan), valuable and increasing in ρ (LST), ~2% with good policies (Cachon–Fisher), zero for most ARMA parameters (Cui et al.).
2. Common thread: every answer fixes *how* the receiver turns information into orders (known model, Bayesian updating, optimal policy). Empirics (Croson–Donohue) say human receivers don't work that way; practice increasingly delegates ordering to learned software agents — inference quality becomes a *design variable*.
3. The move: hold the physical chain fixed, replace known-model inference with end-to-end learned inference, and *type the channel* — nothing ("nocomm"), the raw realized demand ("raw"), a grounded forecast ("d̂"), engineered summaries (IP, d̂+IP), innovations/benchmark encodings (ε, cond-mean, true-λ), and a fully learned code (DIAL) — so value is measured **per signal**, not per "sharing."
4. Prior-campaign headline (three sentences, clearly labeled as prior evidence): dhat-sharing worth +5.2% under Poisson regime uncertainty (redundancy null rejected where the classical setup least expects it); at ρ=.9, dhat exactly null but raw worth +293/seed — value that a known-model receiver would not need. Signal-level, not chain-level, redundancy.
5. This paper: one fresh registered campaign (all cells regenerated, fresh seeds) with two primaries — the **crossover** (P1) and its **mechanism** (P2, observation-garbling) — plus the re-registered supporting families, an analytical spine (Prop. 1), and a second-learner robustness arm.
6. Contributions: (i) signal-level reframing of the value-of-information question with a crossover finding; (ii) a causal, Blackwell-ordered manipulation of order-stream informativeness inside a fixed physical system; (iii) a frozen, audited MARL instrument with registered audibility/attribution gates (do(m) interventions); (iv) the operational-representation result (what matters is the data, not its parameterization); (v) full preregistration lineage with public hashes.

---

## 2. Literature and the gap (four pillars; every entry peer-reviewed) — *carried over intact from v1.x; annotations updated*

### Pillar A — Analytical value of information sharing (the theory under stress-test)
1. **Clark & Scarf (1960), MS 6(4)** — serial-system echelon optimality. *v2.0 note: cited for lineage; the canonical-cost phases built on it are removed from the design.*
2. **Lee, Padmanabhan & Whang (1997), MS 43(4)** — bullwhip's four causes; demand-signal processing is the one SIGNAL manipulates; design-level source for F_GEOMETRY.
3. **Chen (1998), MS 44(12-2)** — value of centralized demand information (~0–9%) under optimal policies.
4. **Chen, Drezner, Ryan & Simchi-Levi (2000), MS 46(3)** — bullwhip under forecasting; BW_cum convention.
5. **Gavirneni, Kapuscinski & Tayur (1999), MS 45(1)** — information value is condition-dependent.
6. **Lee, So & Tang (2000), MS 46(5)** — value of sharing under AR(1), increasing in ρ. *Now the comparative-static backdrop for the ρ-gradient secondary and the ρ=.9 site of the crossover.*
7. **Cachon & Fisher (2000), MS 46(8)** — ~2% with good policies; the "modest value" precedent that makes clean nulls publishable; economic ancestry of the ip/raw rungs **and of the ±2% equivalence band**.
8. **Raghunathan (2001), MS 47(4):605–610** — **the redundancy null**: with AR(1) and known parameters the supplier extracts demand from order history. *The known-model assumption is exactly what SIGNAL removes.*
9. **Gaur, Giloni & Seshadri (2005), MS 51(6)** — the ARMA invertibility boundary: when order history suffices. With #8, the classical boundary re-tested under learned inference — **and the boundary P2 manipulates**: garbling the observed order stream pushes the system away from invertibility by construction.
10. **Cui, Allon, Bassamboo & Van Mieghem (2015), MS 61(11)** — V = 0 for most ARMA(1,1) parameters; value hinges on receiver *use* — SIGNAL designs the use.
11. **Aviv (2001), MS 47(10); Aviv (2007), MS 53(5)** — collaborative forecasting; value depends on relative forecasting capability. **D1's forecast-vs-data contrast — now central: the crossover is D1 resolved in both directions.**
12. **Zipkin (2000), Foundations of Inventory Management, Ch. 8–9** — echelon DP background for the benchmark anchors.
13. **Blackwell (1953), Ann. Math. Stat. 24(2)** — comparison of experiments/garbling. **New load-bearing citation: the clip treatment min(o,12)=min(min(o,20),12) is a literal Blackwell garbling chain over a fixed physical process (P2's identification), and Prop. 1(b)'s σ-algebra argument lives here.**

> ⚠️ Keep removed: Axsäter & Rosling (1993) is *not* the redundancy-null lineage.

### Pillar B — Behavioral and empirical evidence
14. **Sterman (1989), MS 35(3)** — the beer game as science; supply-line under-weighting.
15. **Steckel, Gupta & Banerji (2004), MS 50(4)** — POS sharing can fail or hurt depending on demand pattern. *Direct ancestor of the signal-level framing.*
16. **Croson & Donohue (2006), MS 52(3)** — bullwhip persists with shared data; inference limits bind — un-manipulable in humans, precisely manipulable in learners.
17. **Bray & Mendelson (2012), MS 58(5)** — field evidence of demand-signal processing.
18. **Özer, Zheng & Chen (2011), MS 57(6)** — trust in forecast sharing; ancestry of the honesty probe (here trained and measurable).

### Pillar C — Strategic information sharing (scopes the β axis)
19. **Cachon & Zipkin (1999), MS 45(7)** — decentralized incentives distort. *v2.0: retained for the F_INCENTIVE (β) axis framing only; the τ*-contract phase is out of the design.*
20. **Li (2002), MS 48(9); Ha & Tong (2008), MS 54(4)** — voluntary sharing under competition; the F_INCENTIVE framing (never "cheap talk"; DIAL excluded from the β-grid by design — delegated, not strategic, communication).
21. **Ren, Cohen, Ho & Terwiesch (2010), OR** — truthful sharing as reputation. *Discussion-only in v2.0 (no contract axis).*

### Pillar D — Learning agents and learned communication
22. **Oroojlooyjadid, Nazari, Snyder & Takáč (2022), M&SOM 24(1)** — DQN beer game; single learner, no channel. The testbed precedent.
23. **Gijsbrechts, Boute, Van Mieghem & Zhang (2022), M&SOM 24(3)** — DRL-as-instrument legitimacy in this journal family.
24. **Foerster, Assael, de Freitas & Whiteson (2016), NeurIPS** — RIAL/DIAL (the learned rung's trainer); **Sukhbaatar, Szlam & Fergus (2016), NeurIPS**.
25. **Lowe, Foerster, Boureau, Pineau & Dauphin (2019), AAMAS** — positive signaling vs positive listening; license for the audibility gates. **Eccles et al. (2019), NeurIPS**.
26. **Yu et al. (2022), NeurIPS** — MAPPO (the instrument). **Rashid et al. (2018), ICML** — QMIX (**now the registered second learner: recurrent DRQN agents + monotone mixer, standalone certified implementation**).
27. **Recent RL-in-OM wave (one sentence of context)**: Liu et al. (2024, POM); Mao et al. (2024, MS); Gong & Simchi-Levi (2024, MS) — none poses the value-of-sharing estimand.

> **[v3] Pillar D additions (new, for the repair layer and QMIX program):**
> 28. **van Hasselt, Guez & Silver (2016), AAAI** — Double Q-learning / Double DQN: the
>     max-operator overestimation fix behind the registered qr_doubleq variant.
> 29. **Jaderberg et al. (2017), ICLR (UNREAL)** — auxiliary tasks in deep RL: the lineage
>     the dhat head belonged to; §13's capture mechanism is the cautionary counterpart
>     (auxiliary heads under a shared policy optimizer can degenerate while the policy
>     still benefits from the shared representation).
> 30. **Henderson et al. (2018), AAAI** — RL reproducibility: motivates the certification
>     gates, frozen manifests, and tensor-level certificates as method, not hygiene.

### The differentiation table — update the v1.x table with one new row
| Axis | Classical | Behavioral | RL-in-OM | **SIGNAL v2.0** |
|---|---|---|---|---|
| Inference | known model | human, unmeasurable | learned, unexamined | learned, **typed & audited per signal** |
| Manipulation of informativeness | assumed (invertibility) | none | none | **Blackwell-garbled observation, physics fixed** |
| Value estimand | chain-level V | qualitative | task reward | **V per signal, CRN seed-paired, registered** |

---

## 3. Model and environment
Four-echelon beer game (retailer → wholesaler → distributor → manufacturer), PettingZoo-parallel; order/shipping/production leads; behavioral cost throughout (holding 0.5, backorder 1.0 **at every stage**; the Clark–Scarf retailer-only variant is out of the design). Demand regimes: DR-Poisson (λ ~ U[4,24] per episode; regime uncertainty) and AR(1) (μ=12, σ=3, ρ ∈ {0,.3,.6,.9}; persistence). Observations: 4 local scalars [inventory, backlog, on-order, last incoming order]. **P2 treatment (env feature, default off):** non-retailer agents observe min(order, c), c ∈ {12, 20, ∞} — a deterministic, CRN-safe garbling; costs proven byte-identical across c under fixed actions; min(o,12)=min(min(o,20),12) gives the Blackwell chain. Channel: typed message, one-step delay, topology-routed (ADJ matrix); nocomm = identical architecture, zero ADJ.

## 4. The instrument (frozen and hashed)
MAPPO (CTDE): per-agent GRU belief over [obs, incoming message]; base-stock head S with order = clip(S − IP, 0, max); grounded d̂ head (aux-MSE to own realized demand); message ladder {raw, dhat, ip, dhat_ip, eps, condmean, true_lambda, raw_lag1, raw_lag2, learned(DIAL)} — every named rung is a deterministic function of local information (proven message==f(obs) end-to-end; 18/18 suite). **Second learner (robustness): recurrent QMIX** — DRQN per agent over the same inputs, discrete order-up-to grid (41 pts on [0,160]), monotone hypernet mixer on the 133-d global state; scope {nocomm, raw, dhat}; standalone `train_qmix.py`, certified tensor-identical to the shared-harness path at freeze. Selection: held-out CRN gate, best-checkpoint, patience — identical machinery across learners. Certificates are **tensor-level** (torch.equal), not file-sha (serialization order is hash-unstable; documented).

---

## 5. Hypotheses and pre-registration (v2.0 registry; all confirmatory numbers from the fresh campaign)

### Confirmatory primaries (joint Holm, familywise α = .05; one-sided bootstrap-t CIs; CRN seed-paired throughout)
- **P1 — the crossover (conjunctive, intersection–union).** Δ_DP = V_DP(dhat) − V_DP(raw) > 0 **and** Δ_AR = V_AR,ρ=.9(raw) − V_AR,ρ=.9(dhat) > 0. Both one-sided CIs must exclude 0; the conjunction is the claim ("redundancy is signal-specific"). *Prior evidence: Study 1 found V_DP(dhat)=+203 and V_AR(raw)−V_AR(dhat)≈+294 with dhat null — but had no dp_raw arm; v2.0 completes the 2×2.*
- **P2 — mechanism by garbling (Γ).** Γ = V_AR,ρ=.9^raw(clip c=12) − V_AR,ρ=.9^raw(no clip) > 0: degrading the *observed* order stream raises the value of direct demand sharing. Secondary dose-response: Γ(20) ≤ Γ(12) direction reported. Clip levels validated by the registered **clip-rate pilot** (predeclared frequency windows, outcome-blind, pilot seeds ≥ 50).

> **[v3 STATUS — P2.]** Registered direction FALSIFIED, 25/25 seeds: Γ(12) = −462
> [−577, −331]; V(inf)=+462 → V(20)=+88.7 → V(12)=−0.2 (dose-monotone DEGRADATION, so
> P2-dose is informative under the reversed sign). Preregistered decomposition machinery
> delivered the mechanism verdict M2 — learning failure under censoring (capture(12) =
> −0.0%), not information recovery; the comm-specific loss is exactly the +462 (ΔC_raw
> +633.4 vs common own-obs penalty ΔC_noc +171.3). Companion causal identification
> (exploratory, promoted): transfer 2×2 — training environment determines behavior
> (raw_tinf→eclip12 +91 vs train-native; eval env ≈ nothing). v3 registration: register
> the LEARNED direction (Γ<0 with M2 capture prediction) as the confirmatory P2′ on fresh
> seeds — falsification in the pilot, confirmation of the revised mechanism in the run the
> journal sees. The Blackwell design (§3, ref 13) is untouched and remains the paper's
> identification core.

> **[v3 STATUS — P1.]** Executed as the four-leg IU of prereg_v2.1. AR legs:
> SUPPORTED — V_AR.9(raw) = +462 (+11.0%), adj p = 7.9e-06; raw−dhat = +430.5; SIGNAL-raw
> 3745.8 vs privileged AR-CondBS 3747.6 ⇒ the registered optimality-gap rule reads the AR
> leg as INFORMATION-VALUE at every τ (gap ≈ 0). DP legs: DEFECT-CONTAMINATED — the
> executed dhat was a policy-captured near-constant (§13), so the crossover's DP side was
> never actually tested; v2 DP-leg numbers are reported per registration but interpreted
> under the defect disclosure [exact v2 DP values: VERIFY from the confirmatory tables
> before citing]. Resurrection path: D-DP in §14 (DP-certified forecaster + dp pair). The
> v1.x prior (V_DP(dhat)=+203) is therefore also under the defect cloud unless D-DP
> replicates it with certified content.

### Holm-corrected registered families (each family Holm-internal)
- **F_CONTENT (ladder @ ρ=.9, retailer_broadcast):** {raw, ip, dhat_ip, learned, eps, condmean} vs dhat and vs nocomm. **H-REP (operational representation):** raw ≈ eps ≈ condmean, pairwise TOST at ±2% of nocomm cost — value rides on the data, not its parameterization. Benchmark rungs (renamed, frozen aliases documented): ar1_condmean, dp_true_lambda.
- **F_GEOMETRY:** positives {upstream_only, retailer_broadcast}; predicted-null placebos {no_neighbor, downstream_only, manufacturer_broadcast} TOST-tested; exploratory {neighbor, skip, full, link_top_only, link_bottom_only}. Direction, not adjacency.
- **H-SOURCE (raw, ρ=.9):** upstream_only(raw) vs downstream_only(raw) — per-link EDI vs source access; contrast registered directional.
- **H-TIME:** raw_lag1/raw_lag2 vs raw — timeliness gradient, monotone direction registered.
- **F_INCENTIVE:** V(β), β ∈ {0, .5, 1} with matched-β nocomm baselines; dhat only; disclosed selection-bias cancellation argument unchanged.
- **H2 (ρ-gradient, now on the informative signal):** per-seed OLS slope of V_raw(ρ) over {0,.3,.6,.9}; CI > 0. (The v1.x dhat-slope result — flat — is reported as the contrast.)
- **C1 (positive control):** Gap_Recovered vs per-echelon base-stock BAR > 0 on DP test λs; instrument-validity gate.
- **QMIX concordance (registered decision rule):** sign agreement with MAPPO on both P1 conjuncts and Γ across the 8 QMIX cells; discordance triggers the algorithm-conditionality discussion, not post-hoc cell hunting.

> **[v3 STATUS — QMIX.]** The registered convergence gate did its job: raw cells
> FAILED competence (V_qmix = −485 [−778, −167], 17/25 nocomm-better; channel ALIVE — msg
> SD 6.6, flip 0.604, zero collapse). Post-unblinding fair-grid benchmark (grid-projected
> CondBS): expressiveness explains +18.8 of the +2855 gap (0.7%) at the executed (41,160)
> grid ⇒ 99.3% learning. v3 replaces cell-hunting with a PREDECLARED one-change variant
> program {qr_base, qr_doubleq, qr_replay-200, qr_eps-2500} at the fixed executed grid,
> behind a competence gate (dev-mean ≤ 1.20×GridCondBS = 4519.7 AND ≤ 0.85×qr_base);
> winner earns the qrw_nocomm/qrw_raw concordance pair; **no winner ⇒ V1 UNADJUDICABLE by
> rule** — reported as such, never tuned past. Concordance scope stays raw+nocomm (dhat
> cells remain excluded per the registered demotion).

> **[v3 STATUS — F_GEOMETRY / H-SOURCE / F_INCENTIVE / H2.]** The executed geometry
> and incentive families returned nulls on their registered channel; per the §13 defect
> they are DEAD-CHANNEL as evidence wherever that channel was dhat [per-arm content:
> VERIFY against the sweep manifest before the paper commits language]. H2 as written here
> (ρ-gradient ON RAW) — executed outcome: VERIFY from the v2 D2 cells; if the executed
> slope family ran on dhat, it is dead-channel and D-RHO (§14) is its v3 resurrection.
> Disposition options are costed in §14 (D-GEO, D-BETA, D-RHO); default = report executed
> outcomes with the defect caveat in the appendix, resurrect only what the thesis clock
> allows.

> **[v3 STATUS — F_CONTENT / H-REP / C-NULL.]** raw: SUPPORTED (above). learned, ip,
> dhat_ip: registered nulls at ρ=.9 (audibility/attribution gates: learned PASSES content
> attribution, share 0.31, p<1e-6, with V≈0 — the reliance≠value methodological finding;
> do(m=0) on raw costs +2154, so listening is proven). dhat and the executed C-NULL
> equivalence: held numerically but interpretation contaminated by §13 (equivalence of a
> degenerate signal is vacuous). The v3 r4 triangle retests with certified content:
> r4_dhatc → C-NULL under certification (Branch A: survives ⇒ redundancy reading
> strengthens; Branch B: value appears ⇒ report the reversal); r4_arpred → H-REP-b fresh
> (raw ~ linear predictor); dhatc−arpred → the learning tax. H-REP-a (raw~eps) and H-TIME
> lag results: VERIFY from v2 tables; both re-registerable in v3 at ladder cost.

### Supporting / exploratory / analytical
- **H3 mechanism** (forecast-error deltas, conservative-bias disclosure) — unchanged.
- **Substitution curves** on both primary pairs via budget milestones {1k,2k,4k,8k}; slope CI sign decides substitutes/complements; *prior: DP complements (+74.65/log₂), AR flat.*
- **do(m) attribution gates** (honest/shuffled/cross/zeroed; identity-replay self-check): a positive V is content-attributed only if Δ(shuffled−honest) CI < 0 excludes 0; *prior shares: raw .15, learned .51, dhat channel-inert.*
- **H-SEM (decoder protocol):** linear + shallow decoders from learned-channel activity to (d̂, IP, innovation) with seed-split train/test, permutation nulls, projection controls; effect sizes reported, **no 50% threshold**.
- **Censoring fingerprint:** per-stage zero-order/max-order fractions on the CRN cost episodes; P(censor) predicted increasing in ρ and predictive of V across seeds.
- **Proposition 1 (analytical spine, four parts, unchanged):** (a) null recovered in the invertible limit; (b) strict information gap under order censoring (Blackwell/σ-algebra); (c) newsvendor cost translation (h+b)·φ(z*)·(σ_O−σ_D); (d) gap increasing in ρ.
- **Bullwhip descriptives** (BW_cum; prior: SIGNAL 10.9 vs Bayes-known-model 57.6).

> **[v3 STATUS — fingerprint & ops.]** Direction confirmed in v2: upstream zero-order
> fraction rises +0.06–0.11 under raw at ρ=.9 (max-order fraction 0 everywhere — upper
> censoring inactive, correctly). Ops decomposition (exploratory): raw shifts the
> holding–backorder frontier at EVERY echelon (backorder ≈ −64%, holding ≈ −36%); DP-raw
> manufacturer-dominated (Lee-consistent). Substitution curves on the value channels:
> COMPLEMENTS (slopes +101 dp-raw, +131 ar-raw), truncated milestones excluded per
> registration.

### Validity gates (registered, unchanged in spirit)
Audibility gates for any economic-null claim; content-attribution rule via do(m); gates advisory in the driver, strict mode available; every gate verdict logged before unblinding.

---

## 6. Experimental design (the one campaign)
- **Sweep:** PHASES=full → **56 arms × n seeds** (840 jobs at n=15; 1,120 at n=20). Phase map: A/B/Bnull/C/E/Bext (Study-1 families re-run fresh) + A2/B2/C2/D2/E2/F2 (crossover completion, ladder extensions, source, ρ-grid on raw, clip, lags) + G2 (QMIX, 8 arms). Dedup automatic on shared arm names. Canonical D/Dext deleted from the sweep (git tag v1.2 preserves history).
- **Seeds:** confirmatory 30–(30+n−1), disjoint from Study 1 (10–24) and dev/pilot (≥50). **n from the registered power protocol** (SIGNAL_v13_power_analysis.md): partial results in — DP conjunct powered ≥.94 at n=15 (Holm, 50% effect); AR9_dhat is the null cell (superiority power meaningless by design; TOST power ≈1 given sd 26 vs band 82). n* waits on the AR9_raw pair. **Fallback clause (add to registration before hashing): if no n ∈ {15,20,25} meets every target, n* = 25 with the shortfall reported and the observed-effect (100%) row as the primary planning scenario.**
- **Refs:** one behavioral baselines_regime_v2.json, regenerated on-pod (S3), per-echelon BAR (bar_per_echelon=true).
- **Compute:** ~7.5–8.5 h at 54 vCPU for 840 (≈10–11 h for 1,120); QMIX jobs csvlog-stripped by the runner.
- **Registration:** one standalone v2.0 document; prior hashes cited as history; freeze manifest FREEZE_MANIFEST_v1.3.txt covering agents/envs/scripts/conf/sweep/tests.

> **[v3 DESIGN DELTA.]** The v3 campaign (the run the journal sees) is NOT the 56-arm
> sweep rerun. Core = the frozen repair manifest (reports/REPAIR_SEED_MANIFEST.json): r4
> {nocomm, raw, dhatc, arpred} + the QMIX variant program; extensions per the §14 scope
> menu (P2′ clip arms, ladder completion, D-DP/RHO/GEO/BETA). Seeds: dev 60–64;
> confirmatory 70–89 (n=20) — **OPEN ITEM: the v2.1 registration's power fallback fixed
> n*=25; before prereg_v3 hashes, either extend confirmatory to 70–94 (n=25, still
> disjoint; recommended, +25% compute) or register an updated power justification from
> observed effects.** Infrastructure: Phases 1–4 delivered and chain-validated (orchestrator
> run_repair_study.py: dev→gates→confirm→dump→analyze, ep-stamped sentinels, fail-closed
> analysis); execution contract = reports/PHASE34_IMPLEMENTATION_GUIDE.md + 
> reports/RUNPOD_TUTORIAL_v3.md. Compute at −j60: core ≈ one short night; full menu ≈ two.

## 7. Statistical methodology
Per-seed CRN pairing as the unit; studentized bootstrap-t CIs (10k, seed-resampled); Holm within families and jointly over the two primaries; TOST at ±2% of nocomm cost with the Cachon–Fisher materiality justification; intersection–union logic for P1 (no multiplicity inflation: both must reject); OLS-per-seed slopes for gradients; joint seed-vector bootstrap for cross-family statements; decision tree per cell: TOST → CI⁺ → CI⁻ → inconclusive. Power analysis determines n only (protocol doc; approximations disclosed there).

## 8. Results plan (statistic → figure/table)
R1 crossover 2×2 (V by signal × regime) — the paper's Figure 1; R2 Γ dose-response (V_raw vs clip level, with clip-rate pilot inset); R3 content ladder @ ρ=.9 incl. eps/condmean (H-REP brackets); R4 geometry forest plot with placebo TOST bands; R5 ρ-gradient on raw vs dhat (two slopes, one flat); R6 substitution curves both regimes; R7 do(m) intervention shares; R8 β-grid; R9 QMIX concordance table (signs only); R10 censoring fingerprint vs V scatter; T1 full verdict table with Holm-adjusted CIs; A1 Prop-1 proofs; A2 instrument audits (18/18, 11/11, certificates).

> **[v3 RESULTS-PLAN DELTA.]** R1 (crossover 2×2) exists only under D-DP; otherwise
> Figure 1 = frontier bar (nocomm/raw/AR-CondBS) + content triangle. R2 becomes the
> falsification+mechanism figure: V vs clip level WITH the capture decomposition and the
> transfer 2×2 inset. R3 gains the certified-dhat and arpred rungs with branch styling.
> R5 per H2 status (§5 annotation). R9 becomes gate-conditional (winner concordance table
> or the UNADJUDICABLE statement). NEW: R11 certification panel (defect audit vs certified
> metrics — the §13 story in one figure); R12 reliance≠value scatter (attribution share vs
> V). T1 gains a registered-vs-outcome ledger column covering the pilot lineage.

## 9. Threats to validity and pre-empted referee attacks (updated)
- *"The manipulation changes the game, not the information."* — No: P2 garbles the observation map only; transition kernel and costs proven byte-identical across c; Blackwell-nested levels. (The v1.x max_order idea was abandoned for exactly this confound; concession documented.)
- *"One optimizer's artifact."* — Registered QMIX concordance arm; different objective/factorization, same env/channel/selection machinery; standalone file certified tensor-identical to the shared harness at freeze.
- *"Selection rules differ across arms/learners."* — Single harness code path (MAPPO) + certified merge (QMIX); gate seeds disjoint from test λs/ρ.
- *"Nulls are just deaf channels."* — Audibility gates precede any economic-null claim; do(m) separates channel-helps from content-helps.
- *"Effects are noise-mined."* — Full preregistration lineage with public hashes; fresh seeds; every threshold numeric before unblinding; Study-1 numbers quarantined as prior evidence.
- *"Reproducibility."* — Tensor-level behavioral certificates (byte-sha shown unstable to serialization order — incident documented); CRN; frozen manifest; archived campaign tarball with SHA.
- *"External validity of RL agents."* — Claims scoped to learned-inference receivers (the growing deployed case); behavioral pillar motivates, never licenses, generalization to humans.

> **[v3 — four new pre-empted attacks.]**
> - *"Your forecast arm was broken; why trust the repair?"* — Certification gates
>   (ratio ≤1.10 achieved 1.047; bias, calibration, correlation, variance floor) against an
>   empirical conditional benchmark of the exact rounded/truncated DGP; fail-closed loading;
>   checkpoint-embedded identity; and the analytic arpred control that no learning defect
>   can touch. Full defect audit disclosed (§13).
> - *"You reran until it worked."* — One pilot lineage (hashed registrations v1.1/v1.2/
>   v2.1), one v3 preregistration written before the single confirmatory run, fresh seeds,
>   frozen manifest, ep-stamped resume trail, FORCED-flag stamping in reports. The journal
>   run is the FIRST execution under prereg_v3.
> - *"n=20 is below your own registered power protocol."* — resolved by the §6 open item
>   (extend to n=25 or register the updated power analysis) BEFORE hashing prereg_v3.
> - *"The falsified P2 becomes a garden of forking paths."* — P2′ registers the pilot-
>   learned direction and mechanism (M2 capture) as a risky prediction on fresh seeds; the
>   pilot falsification is reported in full alongside.

## 10. Novelty statement
Author: first causal, Blackwell-ordered manipulation of order-stream informativeness inside a fixed physical supply chain, joined to a signal-typed value estimand under learned inference — producing and then mechanically explaining a crossover the classical chain-level question cannot express. Referee-proof form: every prior treats informativeness as an assumption (invertibility) or leaves it unmeasured; here it is a treatment with a dose.

## 11. Thesis chapter mapping
Ch.1 intro = §1; Ch.2 lit = §2; Ch.3 model+Prop.1 = §3+A1; Ch.4 instrument+audits = §4+A2; Ch.5 registered design = §5–7; Ch.6 results = §8; Ch.7 robustness (QMIX, gates, censoring) = R7–R10; Ch.8 managerial implications + limitations = §9–10 material.

## 12. Pre-launch operations checklist (current, 2026-07-19)
Done: sweep final (56/840, D/Dext excised, verified), standalone train_qmix certified, obs-clip env + config struct proven, QMIX dump path proven end-to-end, power protocol + script (fixture-tested), clip-rate pilot coded into S5. Open, in order: (1) fam/ dir listing → AR9_raw pair → n*; (2) local gauntlet (18/18, 11/11, qmix selftest); (3) driver retarget to PHASES=full + counts + mock-harness re-run; (4) registration v2.0 text (this skeleton §5 is its outline) → prereg.py → hash; (5) DECISION_LOG entries; (6) git tag v2.0-launch; (7) pod tutorial. Post-campaign offline: timeseries dumps for H-SEM, selective-ε probe, full registered analyze with joint bootstrap.

> **[v3 §13 — POST-UNBLINDING DEFECT AND CERTIFICATION LAYER.]**
> Diagnostic (seeds 25–49): the grounded d̂ head was a biased near-constant — pred SD 0.50
> vs benchmark 5.52, median MSE ratio 4.29, bias +1.68 toward ≈13.5, 25/25 seeds fail.
> Mechanism at source: aux MSE ADDED to the PPO loss under one Adam (policy captures the
> shared GRU); d̂ feeds the base-stock head at init weight 5.0 with dhat_init 14 — PPO's
> preferred order level back-propagates INTO the forecast (d̂ → ≈ S/5). Repair (contribution
> C3): standalone GRU forecaster on demand history only; pretrain → certify → FREEZE;
> achieved ratio 1.047, bias −0.018, pred SD 5.616/5.625, corr .879, slope .994; structural
> optimizer exclusion; golden-equivalence tests prove every legacy arm byte-identical;
> train↔eval message identity proven. Labels: every repair/v3 artifact carries
> POST-UNBLINDING TARGETED FOLLOW-UP; the original campaign is immutable.
>
> **[v3 §14 — WHAT IS MISSING FOR THE POD RUN (gap register; everything else is delivered
> and chain-validated).]**
> G1 SCOPE SIGN-OFF (author decision): D-LADDER (+learned,+ip; ≈1h), D-DP (DP-certified
>     forecaster = deferred A17 + dp pair; ≈half-day + 1.5h — the ONLY path back to R1),
>     D-RHO (≈3h), D-GEO, D-BETA; P2′ clip arms {r4_nocomm,r4_raw}×{clip12,clip20}
>     (+4 arms — required if P2′/R2 is in the journal run).
> G2 prereg_v3.py: self-hashed registration for the chosen scope; informed-prediction
>     disclosure; P2′ direction; QMIX gate + UNADJUDICABLE branch; n decision (G3).
> G3 Seed decision: extend confirmatory 70–89 → 70–94 (n=25) or register updated power.
> G4 Manifest rows for G1 choices (JSON only; the orchestrator iterates the tables).
> G5 Analysis reach: transfer-2×2 and clipped-reference machinery return to the run path —
>     UN-ARCHIVE scripts/{clipped_refs, make_env_override, p2_decompose, decompose_costs}.py
>     (they were legacy-listed before the only-new-run strategy); add a refs-regeneration
>     stage (baselines ar + regime + clipped refs on the v3 eval streams) and a
>     confirmatory cell-map pointing the frozen analyzers at repair_out/v1.
> G6 Tag v3.0 AFTER G1–G5; stamp the git hash into checkpoints; then RUNPOD_TUTORIAL_v3
>     Steps 3–6 verbatim.
> Bounded effort: G2–G5 ≈ one focused implementing-agent day; G7(D-DP) adds ≈ half a day.
>
> **[v3 §15 — OPS CHECKLIST (supersedes §12's list for the v3 run, 2026-07-26).]**
> (1) G1 scope matrix signed in the DECISION_LOG; (2) G3 seed decision; (3) prereg_v3
> written → hashed → committed; (4) manifest updated + Q4 disjointness test green;
> (5) un-archive per G5 + full local gauntlet (T1–T12, Q1–Q5, 11/11, 18/18-equivalents);
> (6) git tag v3.0; (7) pod: tutorial Steps 3–4 (env probe, V0–V4 smoke incl. the
> by-design gate failures); (8) launch dev → gates → confirm → dump → analyze; (9) harvest
> + V3_RUN_RECORD; (10) interpretation strictly per the branch texts already written
> (PAPER_SKELETON_v3 §6.2/6.6; PHASE34 guide §9).
