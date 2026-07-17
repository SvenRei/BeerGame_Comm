# SIGNAL Decision Log

2026-07-16: Futility gate at rho=0.9 OVERRIDDEN (FORCE_CONTINUE). V(0.9)=-1.1,
TOST p=4.7e-9 = a PRECISE economic null, not instrument failure: the identical
channel buys +4.9% (CI [110,251]) under DR-Poisson and +13% at retailer_broadcast.
Raghunathan recovered in the invertible limit; Phases B/C/E proceed as robustness
of the null.

2026-07-16: Gates switched to ADVISORY (auto_campaign2) for the unattended rerun.
Rationale: the budget-protection question the gates existed for is already
answered (C1 passed +0.104; audibility established). GATES=strict preserved.

2026-07-17: Frozen-agent hypothesis TESTED and REJECTED locally (check_frozen.py,
fresh 1500-ep ar1r9 run): max|KL|=3.8e-2 vs 5.8e-7 for a validated lr=0 frozen
control; best@ep200, later gates worse, milestones==best 4/4. The AR(1) null is
CONVERGED-EARLY redundancy (economic null), not instrument failure.