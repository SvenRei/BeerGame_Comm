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

D-2026-07-19-1 — Combined-campaign scope (v2.0)

Decision. One registered campaign (PHASES=full, 56 arms) regenerates every confirmatory number for both studies. The completed v1.2 campaign (540 runs, archive SIGNAL_campaign_2026-07-17_1115.tgz) is reclassified as prior evidence and variance priors for planning; no old data enters any registered statistic.
Rationale. Single registration, no cross-campaign splicing, uniform seeds/instrument/refs; removes every "were the two studies comparable?" attack.
Evidence. Sweep DRYRUN: configs=56 × seeds=15 ⇒ jobs=840 (dedup verified). Prior hashes cited as history: cfae5dee…58b8 (v1.1), b9e9cf6e…cdc59 (v1.2).

D-2026-07-19-2 — Canonical H7 (Phases D/Dext) removed

Decision. The τ*-contract axis and both canonical-cost phases are deleted from the sweep and driver; H7 leaves the registered story. v1.2 git tag preserves the historical design.
Rationale. Descriptive-only claim; editor parsimony ruling; deleting the dual-refs machinery (background canonical generation, swap/validate/restore) removes a whole failure surface and ~1 h compute.
Evidence. Post-excision: bash -n clean; zero residual tokens (cn_*, TAU_STAR, want D/Dext); manifests full=56/840 unchanged, core honestly re-reports 26/390 with a tag pointer.

D-2026-07-19-3 — Standalone train_qmix.py (merge of the shared harness)

Decision. agents/train_qmix.py is a self-contained entry (harness inlined by mechanical merge from train_common.py, MAPPO branch removed). train_common.py + train_signal.py remain the MAPPO instrument pair.
Rationale. Maintainer preference for one auditable file per learner; acceptable because the harness freezes at the v2.0 hash and equivalence is proven, not trusted.
Evidence. Same-seed certificate: old common-harness path vs standalone ⇒ torch.equal on every actor/critic tensor and identical gate score. Mirror-and-recertify rule embedded in the module docstring.

D-2026-07-19-4 — Checkpoint learner-stamp REJECTED; dump routing by arm name

Decision. No learner marker is written into checkpoint payloads. Phase-G dumps route to scripts/qmix_dump.py by arm-name prefix (qmix_*) in the sweep.
Rationale. Stamping mutates frozen-instrument checkpoint bytes for a convenience the sweep already possesses structurally.
Evidence. Stamp variant was implemented, shown to alter serialized payloads, and reverted; routing proven in the Phase-G micro end-to-end (8/8 arms trained + dumped through the runner).

D-2026-07-19-5 — Behavioral certificates are TENSOR-LEVEL, not file-sha

Decision. All instrument-inertness claims use torch.equal over checkpoint tensors (plus gate-score identity). Byte-sha comparison is retired for checkpoints.
Rationale. Incident: four distinct file hashes including a same-code self-repeat; probe exonerated the trainer (init/collect/update bit-reproduce); root cause is serialization-order instability of the config payload (hash-seed-dependent set iteration in config composition). Tensor equality is strictly stronger evidence and immune to byte-layout noise.
Evidence. Probe transcript 2026-07-19; split certificate re-passed at tensor level (split1 ≡ split2 ≡ pre-dispatch).

D-2026-07-19-6 — Observation-side order garbling (env.obs_order_clip)

Decision. P2's treatment is an observation-map clip: non-retailer agents observe min(order, c), c ∈ {12, 20}; physics untouched; default null. Config struct key added (conf/config.yaml) so hydra accepts the override.
Rationale. Replaces the abandoned max_order manipulation (physics confound, conceded); min-clipping yields Blackwell-nested information sets over one fixed process — the causal informativeness dose.
Evidence. 11/11 suite: cost byte-invariance across c under fixed actions; nesting min(o,12)=min(min(o,20),12); no-op default; legacy MAPPO training tensor-identical post-patch. Hydra acceptance proven (the missing struct key would have killed every clip arm — caught by micro-run).

D-2026-07-19-7 — SIGNAL_CSVLOG handling under QMIX

Decision. The campaign exports SIGNAL_CSVLOG=1 globally; the sweep runner strips it per qmix_* job; manual train_qmix invocation with the flag still hard-fails.
Rationale. The scalar logger reads MAPPO internals; silent garbage is worse than refusal, but a global flag must not kill a registered arm.
Evidence. First micro-run: 6/8 QMIX arms killed by the guard; post-fix: 8/8 done, FAIL=0, with the flag deliberately ON.

D-2026-07-19-8 — Power-analysis fallback clause (pre-hash amendment)

Decision. If no n ∈ {15, 20, 25} meets every registered power target, n* = 25 with the shortfall reported, and the observed-effect (100%) row is the primary planning scenario. Margins remain substantive (±2%, Cachon–Fisher materiality); the analysis determines n only.
Rationale. The AR9_raw seed-sd may make 0.90 conjunction power unreachable at 50%-effect on the menu; the rule must not be improvised after seeing it.
Evidence. Partial run 2026-07-19: DP conjunct 0.94@n=15 (Holm, 50% eff); AR9_dhat null cell as designed (TOST power ≈1 given sd 26 vs band 81.8); AR9_raw pending directory identification.

D-2026-07-19-9 — Clip-rate pilot probe corrected (mock-harness catch)

Decision. The S5 pilot's fixed-action range is 0.05–0.35 of max_order (orders ~5–35), replacing 0.10–0.20.
Rationale. The original range mechanically zeroed the >20 clipping rate (orders could never exceed 20): a probe-design artifact, not evidence about clip levels.
Evidence. Harness T1 verdict "RECONSIDER" traced to the range; corrected probe on the real env: rate(>12)=70.5% ∈ [15,95], rate(>20)=45.0% ∈ [3,80] ⇒ clip levels {12,20} bind as registered. Windows unchanged (predeclared before any cost is computed; outcome-blind)