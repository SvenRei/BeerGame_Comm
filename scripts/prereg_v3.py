"""
prereg_v3.py -- the v3 (journal-run) preregistration. SELF-HASHING, fail-closed
cross-checked against reports/REPAIR_SEED_MANIFEST.json.

Pipeline convention (identical to prereg_v2): the registration is a dict; its canonical
JSON is hashed (sha256); the hash is printed and written with the document to
reports/PREREG_v3.json. Any edit changes the hash. Run BEFORE the first dev job:
    python scripts/prereg_v3.py
Exit nonzero if the manifest and this registration disagree (arms or seeds).
"""
import os
import sys
import json
import hashlib
import datetime

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAN = os.path.join(_ROOT, "reports", "REPAIR_SEED_MANIFEST.json")

REG = {
 "title": "SIGNAL v3 -- single confirmatory campaign (the journal run)",
 "date_utc": None,   # stamped at hash time
 "lineage_disclosure": (
   "PILOT -> PREREGISTER -> CONFIRM. This registration is written AFTER an extensive "
   "registered pilot program: v1.1 (cfae5dee...58b8), v1.2 (b9e9cf6e...cdc59), and the "
   "executed v2.1 campaign (81d4178d...052d, seeds 25-49, n*=25 fallback). Every prediction "
   "below is therefore INFORMED by pilot outcomes and is stated as a risky directional "
   "prediction on FRESH seeds; the pilot results (including the falsified v2.1 P2 direction "
   "and the dhat forecast-head defect) are disclosed in full in the paper. No cell of this "
   "campaign had produced dev or confirmatory data at registration time (chain-validation "
   "smokes at total_episodes=30 excluded mechanically by ep-stamped sentinels)."),
 "instrument": (
   "Frozen v3 codebase (git tag v3.0; hash recorded in reports/V3_RUN_RECORD.md and "
   "repair_out/run_meta.json). Post-unblinding repair layer: dhat messages come from a "
   "CERTIFIED frozen forecaster (results/forecaster_ar1r9.pt: MSE ratio 1.047 vs empirical "
   "conditional benchmark, bias -0.018, pred SD 5.616/5.625, corr .879, slope .994; gate "
   "thresholds ratio<=1.10, |bias|<=0.5, slope in [0.8,1.2], corr>=0.8, SD>=50% bench); "
   "loader fail-closed on certification; gradient isolation structural (outside the "
   "optimizer); legacy arms golden-equivalence tested byte-identical."),
 "design": {
   "seeds": {"dev": [60, 61, 62, 63, 64], "confirmatory": list(range(70, 95)),
             "n_confirmatory": 25,
             "power_note": ("n=25 honors the v2.1 registered fallback n*=25. Observed pilot "
                            "effects (V_AR9(raw)=+462, sd~330; C-NULL sd~26 vs band~82) give "
                            "power ~1.0 for P1' and TOST members; P2' Gamma sd~[pilot 25/25 "
                            "unanimous]; the binding members are the triangle TOSTs, for "
                            "which n=25 preserves the v2.1 planning margins.")},
   "eval": "CRN final-eval seed base 500000 (shared with all baselines/references)",
   "arms_signal": None,   # copied from the manifest at hash time (single source of truth)
   "arms_qmix": None,
   "budget": {"total_episodes": 8000, "patience": 2000, "heldout_every": 200,
              "heldout_episodes": 8, "milestones": [1000, 2000, 4000, 8000]},
   "worlds": {"inf": ["r4_nocomm", "r4_raw", "r4_dhatc", "r4_arpred", "r4_learned", "r4_ip"],
              "c20": ["r4_nocomm_c20", "r4_raw_c20"], "c12": ["r4_nocomm_c12", "r4_raw_c12"]}},
 "primaries_joint_holm_alpha_05": {
   "P1prime": ("V_inf(raw) = C(r4_nocomm) - C(r4_raw) > 0, one-sided paired t "
               "(scipy.stats.ttest_1samp, alternative='greater'); BCa CI and Wilcoxon "
               "reported. Fresh-seed replication of the pilot's load-bearing AR-leg."),
   "P2prime": ("Gamma = V_c12(raw) - V_inf(raw) < 0, one-sided paired t (alternative="
               "'less'). DIRECTION IS THE PILOT-LEARNED REVERSAL of the v2.1 registration "
               "(pilot: Gamma(12)=-462 [-577,-331], 25/25). Registered mechanism "
               "diagnostic: censoring-capture(c12) ~= 0 (M2, learning failure under "
               "censoring), computed by the frozen decomposition machinery; diagnostic "
               "bounds interpretation, never the decision."),
   "CNULLprime": ("Schuirmann TOST: |V_inf(dhatc)| within +/-2% of mean C(r4_nocomm) "
                  "(scripts.c1_stats.tost). Tests whether the pilot's dhat-redundancy "
                  "equivalence SURVIVES certification -- the registered reading of "
                  "Branch A. If instead V_inf(dhatc) rejects >0 one-sided (reported "
                  "two-sided sensitivity), the reversal is reported as the finding "
                  "(Branch B) with the equivalence claim withdrawn."),
   "correction": "joint Holm over {P1prime, P2prime, CNULLprime} via "
                 "scripts.c1_stats.compare_many(method='holm'), familywise alpha=.05"},
 "secondaries_one_family_holm": {
   "HREPprime": "cost(r4_arpred) - cost(r4_raw) TOST within +/-2% band (raw ~ analytic "
                "linear predictor: operational-representation equivalence, fresh seeds)",
   "D1prime_learning_tax": "cost(r4_dhatc) - cost(r4_arpred): decision tree TOST -> CI+ -> "
                           "CI- -> inconclusive (certified-learned vs analytic content)",
   "DOSEprime_a": "V_inf(raw) - V_c20(raw) > 0 one-sided", 
   "DOSEprime_b": "V_c20(raw) - V_c12(raw) > 0 one-sided",
   "TRANSFER_a": "C(raw trained inf, eval c12) - C(raw trained c12, eval c12) > 0 one-sided",
   "TRANSFER_b": "C(raw trained c12, eval inf) - C(raw trained inf, eval inf) > 0 one-sided",
   "LADDER_learned": "V_inf(learned): decision tree (pilot: null with attribution-gate pass)",
   "LADDER_ip": "V_inf(ip): decision tree (pilot: null)",
   "correction": "Holm within this eight-member family; raw and adjusted p both reported"},
 "diagnostics_registered_interpretation_only": {
   "optimality_gap": ("tau grid {.10,.20,.30} vs AR_BestBS regenerated on the v3 eval "
                      "streams (scripts/baselines.py ar -> results/baselines_ar_v3.json); "
                      "pilot: raw at gap ~0 -- information-value reading expected to "
                      "replicate; rule never alters test decisions"),
   "v_distribution": "P(V>0), deciles, min/max accompany every primary",
   "audibility_and_attribution": "do(m) gates precede any economic-null language for a "
                                  "channel (frozen v2 machinery)"},
 "qmix_exploratory_program": {
   "variants_one_change_each": ["qr_base", "qr_doubleq", "qr_replay(buffer=200)",
                                 "qr_eps(anneal=2500)"],
   "grid": "FIXED at executed (n=41, s_max=160); exonerated by the fair grid benchmark "
           "(expressiveness +18.8 of +2855, results/qmix_grid_benchmark.json)",
   "competence_gate": "dev-mean nocomm <= 1.20 x GridCondBS(41,160)=4519.7 AND <= 0.85 x "
                      "qr_base dev-mean; cheapest passer wins the qrw_{nocomm,raw} pair",
   "concordance_rule": "sign(V_qmix(raw)) on the winner pair vs sign of P1prime; "
                       "NO WINNER => V1 reported UNADJUDICABLE (predeclared outcome; "
                       "no post-hoc tuning); --force* continuations stamped FORCED and "
                       "excluded from confirmatory language"},
 "analysis_binding": (
   "All decisions computed by run_repair_study.py analyze from repair_out/v1 and "
   "repair_out/transfer dumps: scipy.stats.ttest_1samp (one-sided), scripts.c1_stats.tost "
   "(Schuirmann), scripts.c1_stats.bootstrap_ci (BCa, percentile fallback), "
   "scripts.c1_stats.compare_many (Holm). Missing cells fail closed. The report "
   "(reports/REPAIR_STUDY.md) embeds git hash + this registration's hash."),
 "amendment_policy": (
   "Scope extensions (D-DP, D-RHO, D-GEO, D-BETA) require a written amendment to this "
   "document, re-hashed, BEFORE any such cell trains. Post-unblinding changes of any "
   "registered rule are prohibited; deviations are reported as deviations."),
}


def main():
    man = json.load(open(MAN))
    REG["date_utc"] = datetime.datetime.now(datetime.UTC).isoformat()
    REG["design"]["arms_signal"] = man["signal_arms"]
    REG["design"]["arms_qmix"] = man["qmix_arms"]
    # ---- fail-closed cross-checks against the manifest ------------------------------------
    errs = []
    if man["seeds"]["confirmatory"] != REG["design"]["seeds"]["confirmatory"]:
        errs.append("confirmatory seeds disagree with manifest")
    if man["seeds"]["dev"] != REG["design"]["seeds"]["dev"]:
        errs.append("dev seeds disagree with manifest")
    for w, arms in REG["design"]["worlds"].items():
        for a in arms:
            if a not in man["signal_arms"]:
                errs.append(f"registered world arm missing from manifest: {a}")
    if errs:
        print("PREREG_V3 CROSS-CHECK FAILED:\n  " + "\n  ".join(errs))
        sys.exit(1)
    # CONTENT hash excludes the timestamp: re-running this script must be a VERIFY, not a
    # re-registration -- otherwise "the hash binds the design" dies on its own re-execution.
    content = {k: v for k, v in REG.items() if k != "date_utc"}
    blob = json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    sha = hashlib.sha256(blob).hexdigest()
    path = os.path.join(_ROOT, "reports", "PREREG_v3.json")
    if os.path.exists(path):
        prev = json.load(open(path))
        if prev.get("sha256") == sha:
            print(f"PREREG v3 VERIFIED (unchanged). sha256 = {sha}")
            print(f"  registered {prev['registration'].get('date_utc')}")
            return
        print("!! REGISTRATION CONTENT CHANGED since the stored hash.")
        print(f"   stored {prev.get('sha256')}\n   new    {sha}")
        print("   If no data has been produced under the old hash, delete "
              "reports/PREREG_v3.json and re-run to re-register. If data exists, this is a "
              "post-registration amendment: document it, keep both hashes.")
        sys.exit(1)
    out = {"registration": REG, "sha256": sha,
           "hash_scope": "sha256 over the registration dict EXCLUDING date_utc"}
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"PREREG v3 registered. sha256 = {sha}")
    print("-> reports/PREREG_v3.json  (commit this file; the hash binds the design)")


if __name__ == "__main__":
    main()
