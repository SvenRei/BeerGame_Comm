# SIGNAL Decision Log
(Reconstructed 2026-07-18 from CAMPAIGN_LOG timestamps, gate records, and session notes;
 maintained contemporaneously from this date forward.)

2026-07-16 | Futility gate at rho=0.9 fired (V=-1.1, TOST p=4.7e-9) and was OVERRIDDEN
(FORCE_CONTINUE). Rationale: a PRECISE null on a demonstrably audible instrument -- the
identical architecture/content earns +4.9% (DP) and +13% (S1). Vindicated 2026-07-17 by
F_CONTENT: the cell was favorable, the registered signal (dhat) was degenerate.

2026-07-17 06:46 | Gates set to ADVISORY for the unattended rerun; verdicts recorded to
GATE_VERDICTS.md. The budget-protection question the gates served was already answered
(C1 PASS; audibility established). GATES=strict preserved in the driver.

2026-07-17 | Frozen-agent hypothesis TESTED and REJECTED locally (check_frozen.py):
max|KL|=3.8e-2 vs 5.8e-7 for a validated lr=0 control; best@ep200; milestones==best 4/4.
The AR(1) flatness is CONVERGED-EARLY redundancy, not instrument failure.

2026-07-17 | Analysis-side fix: c1_stats.load_signal_dir crashed on sibling dumps
(_ferr/_censor) when --comm first met a full campaign. Fixed via exact-filename match;
regression-tested (selftest H); sibling-laden fixture added to the confirmatory selftest.
Measurement dumps untouched. Freeze-manifest addendum recorded.

2026-07-17 | Analysis-side fix: substitution-curve verdict guarded against degenerate
series (per-seed slopes = fp residue ~1e-17 when budget checkpoints are byte-identical;
the label was decided by the SIGN OF ROUNDING NOISE -- machine-dependent). Correct
reading: FLAT.

2026-07-18 | Reporting decision: dhat-rho0.9 cost nulls are labeled channel-inert /
signal-degenerate per the registered validity gate (zeroed-delta CIs include 0), not
generic economic nulls.