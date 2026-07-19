#!/usr/bin/env python3
"""prereg_v2.py -- the SIGNAL v2.0 registration (review 3.0 problem 3). Prints the canonical
registry and its SHA-256; the hash printed by this file IS the preregistration anchor.
Supersedes scripts/prereg.py (v1.x; retained for history). Frozen analysis = confirmatory_v2.py.
"""
import hashlib, json

REGISTRY = {
 "version": "2.0", "date": "2026-07-19",
 "inference_stack": "ALL confirmatory inference delegates to community-validated libraries "
   "(scipy>=1.10 stats; statsmodels>=0.14 multitest) -- NO hand-rolled estimators. One-sided "
   "decisions = scipy.stats.ttest_1samp (paired-difference t-test, exact under normality, "
   "CLT-robust at n=25, deterministic); equivalence = Schuirmann TOST via scipy.stats.ttest_1samp; "
   "effect-size 95% CIs = scipy.stats.bootstrap BCa; multiplicity = statsmodels multipletests "
   "(Holm); nonparametric robustness = scipy.stats.wilcoxon. The registered power analysis "
   "(reports/power_v13.txt) already computes the one-sided t-test rejection rate, so the "
   "confirmatory decision and the power calculation are the SAME test.",
 "history_hashes": {"v1.1": "cfae5dee...58b8", "v1.2": "b9e9cf6e...cdc59"},
 "replication_framing": "HONEST STATUS: the P1 crossover DIRECTION (dhat wins under DR-Poisson, raw "
   "wins under AR(1) rho=.9) was DISCOVERED in Study 1 (v1.2) on the SAME environment, algorithm, "
   "and data-generating process. v2.0 is therefore a PRE-REGISTERED REPLICATION WITH TIGHTENED "
   "INFERENCE at registered power on FRESH, DISJOINT seeds (final-eval 500000+, seeds 30..54; Study "
   "1 used other streams), NOT an independent discovery -- and is reported as such. Genuine novelty "
   "over Study 1: (i) the P2 information-degradation (garbling) experiment; (ii) Proposition 1's "
   "analytical VOI result; (iii) registered power, joint Holm, absolute-value anchors, and the "
   "library-based decision; (iv) the optimality-gap and V-distribution honesty diagnostics. The "
   "abstract must not imply the crossover is a new finding.",
 "campaign": {"phases": "full", "arms": 56, "jobs_at_n15": 840,
   "seeds": "30..54 FINAL: n*=25 by the registered FALLBACK CLAUSE -- the 2026-07-19 power run "
            "(reports/power_v13.txt; AR9_raw sd=268.6) met no target at any menu n under the 50%-effect "
            "sensitivity (P1-conj 0.47/0.63/0.74, Gamma x1.5 0.26/0.34/0.42, H-REP proxy ~0); at the "
            "OBSERVED effects n=25 gives P1-conj 1.00 and Gamma(x1.5) 0.94. H-REP planning proxy "
            "(raw-vs-dhat sd=261) disclosed as strongly conservative: eps/linpred are affine "
            "transforms of the SAME raw stream, so true contrast sd is far smaller; inconclusive "
            "H-REP outcomes are reportable as such. Pilot/dev seeds >=50 excluded.",
   "manifest": "reports/FROZEN_CAMPAIGN_MANIFEST.tsv written once; scripts/verify_manifest.py "
               "fail-closed over every registered cell before any analysis"},
 "primaries": {
  "P1_crossover": "IU over FOUR one-sided PAIRED-DIFFERENCE t-tests (scipy.stats.ttest_1samp, "
    "alternative=greater): [V_DP(dhat)-V_DP(raw)>0] AND [V_AR.9(raw)-V_AR.9(dhat)>0] AND "
    "[V_DP(dhat)>0] AND [V_AR.9(raw)>0]; p_P1=max(component p). BCa bootstrap 95% CIs reported "
    "for every V; Wilcoxon signed-rank reported as nonparametric robustness. Frozen in "
    "scripts/confirmatory_v2.py (7-scenario fixture self-test).",
  "P2_garbling": "Gamma=V_AR.9^raw(obs_order_clip=12)-V_AR.9^raw(inf)>0 by one-sided paired "
    "t-test (scipy), observation-consistent "
    "training (clipped aux targets); CTDE critic global, disclosed; clip levels validated by the "
    "outcome-blind clip-rate pilot (windows: >12 in [15,95]%, >20 in [3,80]%)",
  "correction": "joint Holm over {P1,P2}, familywise alpha=.05, via statsmodels multipletests",
  "claim_scope": "P1/P2 are claims about the EXPECTED (seed-mean) value of sharing; V has high "
    "run-to-run dispersion (AR9_raw coefficient of variation ~0.9, so a nontrivial fraction of "
    "individual runs show V<=0). The supported statement is 'sharing lowers EXPECTED cost' with the "
    "dispersion reported (see v_distribution), NOT 'information always helps'."},
 "companion": {"C-NULL": "Schuirmann TOST (scipy) |V_AR.9(dhat)| within +/-2% of AR nocomm "
   "cost (Cachon-Fisher band); dhat-null is the best-powered claim (sd~26 vs band~82)"},
 "optimality_gap": "REGISTERED interpretation diagnostic (scripts/confirmatory_v2.optimality_gap): "
   "V=C(nocomm)-C(comm) is a difference of ACHIEVED costs and identifies information VALUE only "
   "where both learned policies are near-optimal; otherwise it also absorbs a learning-gap term. "
   "The diagnostic reports each key arm's relative gap to the best available benchmark (DR-Poisson: "
   "the per-lambda Oracle rung, a STRONG benchmark; AR(1) rho=.9: the static base-stock, a WEAK "
   "reference in an autocorrelated regime where the optimal order-up-to level is state-dependent -- "
   "disclosed). Interpretation rule, tau=0.30: a primary contrast is read as INFORMATION-VALUE only "
   "if BOTH arms sit within tau of the near-optimal reference; otherwise V is reported as a LOWER "
   "BOUND on information value contaminated by the learning gap. DIAGNOSTIC ONLY -- it never alters "
   "the t-test decision, it bounds the decision's interpretation.",
 "v_distribution": "REGISTERED reporting: the seed-level distribution of every V (P(V>0), deciles, "
   "min/max) accompanies each primary, surfacing the run-to-run dispersion that a mean+CI hides. "
   "Reported honestly whether or not P(V>0)=1.",
 "secondaries_frozen_in_confirmatory_v2": [
   "H-REP: raw~eps and raw~ar1_linear_predictor TOST at +/-2% (renamed per review: signals are the "
   "AR(1) LINEAR PREDICTOR and OBSERVED ONE-STEP RESIDUAL, not true conditional mean/innovation)",
   "H-TIME: raw>lag1 and lag1>lag2 one-sided", "H-SOURCE: upstream_raw>downstream_raw one-sided",
   "P2-dose: Gamma(12)>=Gamma(20) one-sided"],
 "secondaries_frozen_in_v1_analyzers": "geometry positives+placebo TOSTs, F_INCENTIVE matched-beta "
   "pairs, C1 positive control (per-echelon BAR) -- the v1.2-frozen analyzers re-run on fresh data",
 "benchmarks": "dp_true_lambda / ar1_condmean are PRIVILEGED PARAMETER BENCHMARKS, not perfect "
   "disclosure and not attainable messages; reported as reference rungs only",
 "exploratory_demotions": ["dhat_ip and learned rungs (no architecture-matched nocomm; option B)",
   "QMIX dhat cells (endogenous message in replay)", "H3 forecast-error mechanism (target mismatch)",
   "substitution curves (truncated-milestone contamination); truncated milestones excluded from "
   "any descriptive plot", "learned-channel H-SEM (aux-regularized objective disclosed; "
   "learned_aux_detach=true is the registered configuration, false only as the coupling ablation)"],
 "qmix": {"status": "DEMOTED TO EXPLORATORY / ILLUSTRATIVE (review): the 3000-ep certification "
   "pilot did NOT converge to the MAPPO level (best held-out ~1.7x matched MAPPO; eps_anneal=5000 "
   "means exploration is still ~0.43 at pilot end), so QMIX does NOT support a confirmatory "
   "ALGORITHM-INDEPENDENCE claim. It is reported as a secondary robustness PROBE only. The abstract "
   "and contributions must NOT claim algorithm-agnostic results; at most 'the crossover sign "
   "reproduces under a value-based learner in the cells where that learner converges'.",
   "reported_scope": "descriptive sign concordance with MAPPO on P1 components and Gamma over "
   "raw+nocomm cells, shown ONLY for cells passing the per-run convergence gate below; "
   "non-converged cells reported as non-converged, never as evidence either way",
   "hypers": "qmix_lr=2.5e-4 qmix_buffer=1500 qmix_eps_anneal=5000 qmix_target_update=40 "
   "(adjusted after the 3000-ep pilot diverged post-ep2300; the re-run is stable but still short of "
   "MAPPO within 3000 ep)",
   "convergence_gate": "PER RUN, predeclared: a cell is shown only if its best held-out gate "
   "improved at least once at episode>=2000 AND final best gate cost < 1.25x the run minimum of its "
   "matched MAPPO cell; non-converged cells are reported as such, never silently dropped"},
 "seed_spaces": {"train": "per-run RNG", "gate": "100000+", "final_eval": "500000+ (baselines "
   "regenerated on the same streams)"},
 "eval_terminology": "zeroed-message deltas are MESSAGE RELIANCE, never communication value; "
   "economic value is always C(pi_nocomm)-C(pi_comm), seed-paired"}

blob = json.dumps(REGISTRY, sort_keys=True, separators=(",", ":")).encode()
print(json.dumps(REGISTRY, indent=1, sort_keys=True))
print("\nSHA256:", hashlib.sha256(blob).hexdigest())