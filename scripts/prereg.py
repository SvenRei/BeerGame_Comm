"""
prereg.py -- pre-registration of the SIGNAL confirmatory analysis.
================================================================================
Frozen 2026-07-02, before the multi-seed sweep is launched. This module is the
single source of truth for which contrasts are confirmatory, what statistic and
decision rule each uses, and how multiplicity is corrected. Everything not named
here is exploratory and is reported as such. This controls the multiple-comparison
(forking-paths) risk: with 10 topologies x 4 contents x several regimes the design
admits dozens of contrasts, and an unregistered analysis of that grid leaves any
'significant' cell uninterpretable. One primary contrast, one designated
sensitivity, two Holm-corrected secondary families, one slope test.

Integrity: `python scripts/prereg.py` prints the registration and its SHA256 over
the canonical JSON. Commit that hash (thesis appendix / README). Any later edit to
the registered fields changes the hash; legitimate changes go through
REGISTRATION['amendments'] with a date and reason, never by silent edit.

BUDGET ARITHMETIC (training runs, 15 seeds/arm; == sweep_all_hypotheses.sh after v1.1):
  Phase A (H1/H2/H3 gradient, upstream_only): 10 configs x 15 = 150   (+ dp_rbroadcast S1: 15)
  Phase B/Bnull (F_GEOMETRY @ rho0.9 + rho0 placebos): 7 configs x 15                        = 105
  Phase C (F_CONTENT @ rbroadcast rho0.9): {raw, ip, dhat_ip, learned} x 15                  = 60
  Phase D (H7 strategic, canonical): 3 x 15                                                  = 45
  Phase E (F_INCENTIVE @ rho0.9): {b0, b0.5} x {comm, nocomm} x 15                           = 60
  CORE ~ 435 (+ extended Bext/Dext 105). Only P1/P2/S1/S2 arms are binding; secondary
  arms may be trimmed (fewer seeds) via a dated amendment BEFORE unblinding, not after.

Consumers: scripts/comm_stats.py produces value_of_sharing dicts; h1_decision()
applies the registered decision rule to one. holm_family() wraps
c1_stats.compare_many for the secondary families. h2_slope() computes the
registered H2 statistic from per-rho dumps (eval_signal --dump-comm --dump-ar1).

Self-test: python scripts/prereg.py self-test   (numpy only; c1_stats for Holm/CI)
Print + hash: python scripts/prereg.py
================================================================================
"""
import os
import sys
import json
import hashlib
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.c1_stats import bootstrap_ci, compare_many                     # noqa: E402

# ============================================================================ #
# THE REGISTRATION (frozen fields; edit only via 'amendments')                  #
# ============================================================================ #
REGISTRATION = {
    "study": "SIGNAL: the value of demand-information sharing in decentralized MARL (4-echelon beer game)",
    "version": "1.1",
    "frozen": "2026-07-02",     # v1.1 same-day PRE-LAUNCH amendment (see 'amendments'); no data observed

    # ---------------------------------------------------------------- design
    "design": {
        "seeds_per_arm": 15,
        "seeds": list(range(10, 25)),            # == sweep_all_hypotheses.sh SEEDS; CRN pairing is BY SEED
        "test_lambdas": [6.0, 10.0, 14.0, 18.0, 22.0],     # scoring only; NEVER used for selection
        "validation_lambdas": [8.0, 12.0, 16.0, 20.0],     # checkpoint gate only (train_signal heldout)
        "eval_episodes_per_lambda": 200,                    # eval_signal --dump-episodes
        "crn_eval_seed_base": 100000,                       # == baselines/eval HELDOUT_SEED_BASE
        "cost_model": "team (holding+backorder charged at every stage) for H1-H4 headline; "
                      "the canonical (penalty_at_retailer_only) variant is the C1-gap chapter only, "
                      "with baselines regenerated on the SAME cost model",
        "primary_inference": "95% STUDENTIZED bootstrap-t CI (10k resamples) over seed-paired "
                             "means; Wilcoxon signed-rank as the p-value companion (n=15, exact). "
                             "Method choice is calibration-driven: at n=15, percentile coverage "
                             "measured .910 (normal) / .887 (skewed), BCa .912/.890 (no repair), "
                             "bootstrap-t .953/.938 (near-nominal) -- appendix reports the sims",
        "budget_milestones": [1000, 2000, 4000, 8000],   # deployable-at-budget snapshots (substitution curve)
        "training_defaults": "EP=8000, patience=2000, gate every 200 eps on validation keys only",
        "tost_band_frac": 0.02,                             # practical-equivalence band: +-2% of no-comm cost
        "alpha": 0.05,
    },

    # ------------------------------------------------------------ primary
    # P1 and P2 are co-primary claims of DIFFERENT chapters (regime inference; value of
    # sharing) -- no multiplicity correction across them, each carries its own decision rule.
    "primary": {
        "P1_C1_regime_inference": {
            "claim": "the no-comm SIGNAL agent beats the deployable fixed bar under regime uncertainty",
            "arm": "nocomm",
            "statistic": "Gap_Recovered = (BAR - cost)/(BAR - CEILING), per seed, mean over TEST lambdas",
            "decision": "95% bootstrap CI over seeds excludes 0 from below -> C1 holds",
        },
        "P2_H1_value_of_sharing": {
            "claim": "sharing the retailer's demand belief upstream changes team cost",
            "arms": {"comm": "topology=upstream_only, content=dhat", "baseline": "nocomm"},
            "statistic": "V_cost = mean_over_test_lambdas(nocomm_s - comm_s), seed-paired (CRN)",
            "decision_rule_ordered": [
                "1. TOST within +-2% of nocomm cost passes  -> NULL (practically negligible value)",
                "2. elif CI_lo(V) > 0                        -> POSITIVE (sharing has economic value)",
                "3. elif CI_hi(V) < 0                        -> NEGATIVE (sharing hurts)",
                "4. else                                     -> INCONCLUSIVE (underpowered)",
            ],
            "rationale": "the research question is ECONOMICALLY MEANINGFUL value; a statistically "
                         "significant effect inside the +-2% band is registered as negligible, "
                         "with the CI reported alongside",
        },
    },

    # --------------------------------------------------------- sensitivity
    "sensitivity": {
        "S1_max_favorable_geometry": {
            "arms": {"comm": "topology=retailer_broadcast, content=dhat (arm dp_rbroadcast)",
                     "baseline": "nocomm"},
            "regime": "dr_poisson, TEST lambdas (SAME regime as P2 -- the comparison it interprets)",
            "rule": "same decision rule as P2. Interprets P2: if even the maximally favorable "
                    "geometry (clean customer signal to every stage, one hop) is NULL, the serial "
                    "null is decisive; if S1 is POSITIVE while P2 is NULL, hop-by-hop relay is the "
                    "bottleneck (a mechanism finding, not a contradiction).",
        },
        "S2_stationarity_null_family_robustness": {
            "arms": {"comm": "topology=upstream_only, content=dhat", "baseline": "nocomm"},
            "regime": "AR(1) rho=0 (white noise), scored in-regime at rho=0",
            "rule": "same decision rule as P2; checks the stationarity null is not a Poisson-family "
                    "artifact (the rho=0 endpoint of the H2 gradient, reused).",
        },
    },

    # ---------------------------------------------- secondary (Holm families)
    # Within each family: Wilcoxon signed-rank per member vs its baseline, HOLM-corrected
    # ACROSS the family (c1_stats.compare_many); bootstrap CI reported per member.
    # Members with prediction 'null' are placebos: claiming the null ADDITIONALLY requires a
    # TOST pass, with the TOST p-values Holm-corrected within the predicted-null subfamily.
    "secondary_families": {
        "F_GEOMETRY": {
            "content": "dhat", "baseline": "nocomm at the SAME regime",
            "regime": "AR(1) rho=0.9, scored IN-REGIME (rho-keyed dumps) -- geometry is tested where "
                      "value exists to route; at the stationarity null every geometry is trivially null",
            "members": {
                "neighbor":              {"prediction": "positive", "why": "contains the Lee upstream link"},
                "skip":                  {"prediction": "exploratory", "why": "2-hop shortcuts"},
                "full":                  {"prediction": "exploratory", "why": "upper bound on connectivity"},
                "link_top_only":         {"prediction": "exploratory", "why": "where distortion is worst"},
                "link_bottom_only":      {"prediction": "exploratory", "why": "cleanest single signal"},
                "no_neighbor":           {"prediction": "null", "why": "placebo: hears only non-adjacent"},
                "downstream_only":       {"prediction": "null", "why": "direction placebo (wrong way)"},
                "manufacturer_broadcast": {"prediction": "null", "why": "dirty-signal placebo"},
            },
        },
        "F_CONTENT": {
            "topology": "retailer_broadcast", "baseline": "nocomm at the SAME regime",
            "regime": "AR(1) rho=0.9, scored in-regime (max-favorable geometry isolates CONTENT with "
                      "the strongest channel; matches sweep Phase C)",
            "members": {
                "raw":      {"prediction": "exploratory", "why": "POS DATA-sharing rung; the classical "
                             "CPFR contrast forecast-vs-data (Aviv 2001/2007; Cachon-Fisher 2000)"},
                "ip":       {"prediction": "exploratory", "why": "VMI channel (Cachon-Fisher)"},
                "dhat_ip":  {"prediction": "positive", "why": "superset of the primary content"},
                "learned":  {"prediction": "exploratory", "why": "end-to-end optimized channel (DIAL)"},
            },
            "D1_forecast_vs_data": {
                "contrast": "dhat vs raw (both retailer_broadcast, rho=0.9), seed-paired",
                "prediction": "dhat weakly cheaper (a smoothed grounded forecast dominates one noisy "
                              "observation -- Aviv); directional, exploratory",
            },
            "C3_interpretability_bound": {
                "contrast": "learned vs dhat_ip (both upstream_only), seed-paired",
                "prediction": "equivalent (TOST +-2%)",
                "claim_if_pass": "the value of communication is EXACTLY the two named signals; "
                                 "an unconstrained optimized channel adds nothing beyond them",
                "claim_if_fail": "the learned channel encodes value beyond {dhat, ip} "
                                 "(exploratory follow-up: what does it encode? honesty probe)",
            },
        },
    },

    # ----------------------------------------------- incentives (Tier-2 axis)
    "F_INCENTIVE": {
        "question": "does the value of sharing survive incentive misalignment?",
        "content_and_topology": "dhat x upstream_only (the P2 channel)",
        "regime": "AR(1) rho=0.9, scored in-regime (V must exist for its incentive-modulation to be "
                  "measurable; at a null V there is nothing to modulate)",
        "betas": {"0.0": "self-interested", "0.5": "half-aligned", "1.0": "cooperative (REUSED from "
                  "the H2 rho=0.9 pair; not retrained)"},
        "baseline": "nocomm at the MATCHED beta -- V(beta) = cost(nocomm,beta) - cost(comm,beta) "
                    "isolates communication within a fixed incentive regime",
        "estimand_note": "V is always TEAM cost (system efficiency), even when training is "
                         "self-interested; per-stage costs reported descriptively",
        "why_dhat_only": "dhat is SELF-VERIFYING cheap talk: the sender's own S-head consumes its "
                         "d_hat, so misreporting is self-punishing regardless of beta -- the "
                         "strategic margin under misalignment is receiver-side TRUST/USE (listening), "
                         "not sender-side lying. The 'learned' content is EXCLUDED from this axis: "
                         "under DIAL its channel is trained by the RECEIVERS' gradients, so it is "
                         "delegated communication, not strategic sending -- a beta-grid on it would "
                         "not test cheap talk (Crawford-Sobel) and would mislabel the mechanism.",
        "tests": "Holm over {beta0, beta05} member superiority tests vs matched-beta nocomm; "
                 "exploratory per-seed slope of V(beta) over {0,.5,1}; mechanism gates "
                 "(honesty corr, listening slope) reported PER beta",
        "predictions": {"honesty(beta)": "flat (self-verifying signal)",
                        "listening/V(beta)": "open -- degradation = incentive-sensitive trust; "
                                             "flat = information is information"},
    },

    # ---------------------------------------- substitution curve (Tier-1 attachment)
    "SUBSTITUTION_CURVE": {
        "status": "exploratory-registered (named statistic, no confirmatory alpha spent)",
        "claim_frame": "inference capacity and information sharing are SUBSTITUTES; the classical "
                       "redundancy null (Raghunathan 2001) is the infinite-inference limit",
        "producers": "signal_checkpoint_budget{1000,2000,4000,8000}.pt per seed on the P2 pair "
                     "(dr_poisson) AND the rho=0.9 pair (AR1), dumped per budget",
        "statistic": "per-seed OLS slope of V_s over log2(budget), seeds with all budgets; "
                     "bootstrap-t CI; tie-naive Spearman companion (comm_stats.py curve)",
        "readings": {"slope CI < 0": "substitutes (V vanishes as inference is learned)",
                     "slope CI > 0": "complements (exploiting a channel must itself be learned)",
                     "spans 0": "no budget trend at these milestones"},
    },

    # ------------------------------------------------------------------ H2
    "H2_autocorrelation": {
        "claim": "the value of sharing rises with demand autocorrelation (Lee-So-Tang 2000)",
        "rho_grid_test": [0.0, 0.3, 0.6, 0.9],
        "rho_grid_validation_gate": [0.15, 0.45, 0.75],     # train_signal heldout_ar1_rhos; disjoint
        "arms_per_rho": {"comm": "topology=upstream_only, content=dhat", "baseline": "nocomm"},
        "producer": "eval_signal --dump-comm DIR --dump-ar1 \"0,0.3,0.6,0.9\" per checkpoint",
        "statistic": "per-seed OLS slope of V_s(rho) over the rho grid (V_s = nocomm_s - comm_s "
                     "at matched rho, CRN-paired); one slope per seed",
        "decision": "95% bootstrap CI of the mean slope excludes 0 from below -> H2 holds",
        "caution": "the rho-AVERAGED V (value_of_sharing run on the rho-keyed dumps) is NOT the H2 "
                   "statistic: averaging over the grid mechanically dilutes a strong high-rho effect "
                   "below the +-2% band (e.g. V rising 0->3.6% of cost averages to 1.8% -> "
                   "'equivalent'). A TOST-equivalent averaged V therefore does NOT contradict a "
                   "positive slope; H2 is decided by the slope CI alone.",
    },

    # ------------------------------------------------- validity gates (H3)
    "validity_gates": {
        "rule": "a cost NULL on any comm arm is claimable as an ECONOMIC null only if the channel "
                "is demonstrably audible on that arm: honesty corr(dhat component, sender demand) "
                "high AND (listening slope dS/dTold materially > 0 at >=1 receiver OR message-weight "
                "ratios materially > 0). Otherwise the verdict is 'instrument failure (deaf/pruned "
                "channel)' and the cell is excluded from economic claims.",
        "instruments": ["honesty_probe", "positive_listening (per component)", "message_weight_audit",
                        "message_intervention (honest vs shuffled vs cross vs zeroed, CRN-paired)"],
        "content_attribution_rule": "a POSITIVE V is attributed to message CONTENT (not channel "
                                    "presence/scale) only if the intervention probe shows "
                                    "delta(shuffled - honest) CI > 0 on that arm; otherwise the "
                                    "claim is 'the channel helps' without a content mechanism",
    },

    "exploratory": "everything else: per-lambda breakdowns, bullwhip decompositions, forecast-error "
                   "deltas, jitter/service, black_swan/extreme_chaos regimes, symbolic distillation, "
                   "the beta/tau economics grid, per-stage dashboards.",

    "amendments": [
        # append dicts {"date": ..., "change": ..., "reason": ...} BEFORE unblinding; never edit above
        {"date": "2026-07-02", "change": "v1.0 -> v1.1 (pre-launch, no data observed): (1) primary CI "
         "percentile -> studentized bootstrap-t after calibration sims (coverage .91 -> .95); "
         "(2) seeds aligned to the sweep (10..24); (3) S1 pinned to dr_poisson via new dp_rbroadcast "
         "arm; S2 (rho=0 family-robustness) added; (4) F_GEOMETRY/F_CONTENT regimes pinned to the "
         "executed design (rho=0.9 in-regime; F_CONTENT topology=retailer_broadcast per sweep Phase C); "
         "(5) raw POS-data rung + D1 forecast-vs-data contrast added; (6) F_INCENTIVE (Tier-2 axis) "
         "added; (7) SUBSTITUTION_CURVE exploratory-registered; (8) message-intervention instrument + "
         "content-attribution rule added",
         "reason": "reconcile registration with sweep_all_hypotheses.sh before launch; statistics "
         "calibration; Tier-1/Tier-2 scope additions"},
    ],
}


# ============================================================================ #
# Canonicalization + integrity hash                                             #
# ============================================================================ #
def canonical_json(reg=None):
    """Deterministic JSON of the registration (sorted keys, fixed separators)."""
    return json.dumps(REGISTRATION if reg is None else reg, sort_keys=True, separators=(",", ":"))


def registration_hash(reg=None):
    """SHA256 over the canonical JSON. Commit this; any edit to registered fields changes it."""
    return hashlib.sha256(canonical_json(reg).encode("utf-8")).hexdigest()


def print_registration():
    print(json.dumps(REGISTRATION, indent=2))
    print(f"\nREGISTRATION SHA256: {registration_hash()}")
    print("(commit this hash; amendments append to REGISTRATION['amendments'] and change it)")


# ============================================================================ #
# Registered decision rules as code                                             #
# ============================================================================ #
def h1_decision(vs):
    """Apply the registered P2/S1 decision rule to a comm_stats.value_of_sharing dict
    (keys used: 'equivalent' [TOST +-band], 'v_cost_ci' [lo, hi]). Ordered exactly as
    registered: equivalence (practical negligibility) is checked FIRST."""
    lo, hi = float(vs["v_cost_ci"][0]), float(vs["v_cost_ci"][1])
    if bool(vs.get("equivalent", False)):
        return "NULL (practically negligible; TOST within band)"
    if lo > 0:
        return "POSITIVE (sharing has economic value; CI excludes 0)"
    if hi < 0:
        return "NEGATIVE (sharing hurts; CI excludes 0)"
    return "INCONCLUSIVE (underpowered; CI spans 0, TOST fails)"


def holm_family(named_pvals):
    """Holm-correct a *pre-registered family* of p-values: {member: raw_p} ->
    {member: {'raw', 'adjusted', 'reject'}} (reject at the registered alpha=0.05, baked into
    c1_stats.adjust_pvalues). Thin wrapper over c1_stats.compare_many so the correction is one
    shared implementation."""
    return compare_many(named_pvals, method="holm")                 # c1_stats: Holm step-down


def h2_slope(comm, nocomm, n_boot=10000, seed=0):
    """Registered H2 statistic. Inputs: per-arm {seed: {rho: cost}} (comm_stats loader format
    on the AR(1)-keyed dumps). For each shared seed: V_s(rho) = nocomm - comm at matched rho
    (CRN), then the OLS slope of V_s against rho. Returns (slopes per seed, mean, 95% bootstrap
    CI of the mean slope). Decision (registered): CI_lo > 0 -> H2 holds."""
    seeds = sorted(set(comm) & set(nocomm))
    if not seeds:
        raise ValueError("no shared seeds between arms")
    slopes = []
    for s in seeds:
        rhos = sorted(set(map(float, comm[s])) & set(map(float, nocomm[s])))
        if len(rhos) < 2:
            raise ValueError(f"seed {s}: <2 shared rhos")
        v = np.array([float(nocomm[s][r]) - float(comm[s][r]) for r in rhos])
        slopes.append(float(np.polyfit(np.array(rhos), v, 1)[0]))
    slopes = np.asarray(slopes, float)
    lo, hi = bootstrap_ci(slopes, n_boot=n_boot, seed=seed)
    from scripts.c1_stats import paired
    pr = paired(slopes, np.zeros_like(slopes))          # Wilcoxon companion on the per-seed slopes
    return {"seeds": seeds, "slopes": slopes.tolist(), "mean_slope": float(slopes.mean()),
            "ci95": [float(lo), float(hi)], "wilcoxon_p": pr["wilcoxon_p"],
            "h2_holds": bool(lo > 0)}


# ============================================================================ #
# Self-test                                                                     #
# ============================================================================ #
def _self_test():
    # 1. hash: deterministic, and sensitive to any registered-field mutation
    h1, h2 = registration_hash(), registration_hash()
    assert h1 == h2 and len(h1) == 64
    mutated = json.loads(canonical_json())
    mutated["design"]["seeds_per_arm"] = 14
    assert registration_hash(mutated) != h1, "hash must change when a registered field changes"

    # 2. decision rule: all four branches, in registered order (equivalence first)
    assert h1_decision({"equivalent": True, "v_cost_ci": [5.0, 40.0]}).startswith("NULL")
    assert h1_decision({"equivalent": False, "v_cost_ci": [12.0, 90.0]}).startswith("POSITIVE")
    assert h1_decision({"equivalent": False, "v_cost_ci": [-80.0, -6.0]}).startswith("NEGATIVE")
    assert h1_decision({"equivalent": False, "v_cost_ci": [-30.0, 55.0]}).startswith("INCONCLUSIVE")

    # 3. Holm passthrough on a fabricated family (one clear signal, two noise)
    res = holm_family({"neighbor": 0.001, "no_neighbor": 0.60, "downstream_only": 0.45})
    assert res["neighbor"]["reject"] and not res["no_neighbor"]["reject"]
    assert res["neighbor"]["adjusted"] >= res["neighbor"]["raw"]    # Holm never shrinks p

    # 4. H2 slope on synthetic monotone data: V(rho) = 100*rho + noise -> slope ~ 100, CI > 0
    rng = np.random.default_rng(0)
    rhos = [0.0, 0.3, 0.6, 0.9]
    comm, nocomm = {}, {}
    for s in range(101, 116):
        comm[s] = {r: 3000.0 - 100.0 * r + rng.normal(0, 5) for r in rhos}
        nocomm[s] = {r: 3000.0 + rng.normal(0, 5) for r in rhos}
    out = h2_slope(comm, nocomm, n_boot=2000, seed=1)
    assert 80.0 < out["mean_slope"] < 120.0 and out["h2_holds"], out
    # and a flat (null) world must NOT trigger H2 -- deterministic zero-effect case
    flat = {s: {r: 3000.0 for r in rhos} for s in range(101, 116)}
    out0 = h2_slope(flat, dict(flat), n_boot=2000, seed=2)
    assert out0["mean_slope"] == 0.0 and not out0["h2_holds"], out0

    print("prereg self-test PASS (hash integrity, 4-branch decision rule, Holm family, H2 slope)")
    print(f"REGISTRATION SHA256: {h1}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "self-test":
        _self_test()
    else:
        print_registration()