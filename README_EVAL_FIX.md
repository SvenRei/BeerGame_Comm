# v5 eval fix + stop-rule controls

## What broke
Both rho=0.9 evaluations crashed with `Unexpected key(s): s_grid, s_logits_head.*` -- my bug:
the eval-side actor was constructed WITHOUT the head configuration, so categorical checkpoints
could not load. Fixed: the constructor now reads head_type / act_bins / act_smax / obs_scale
from the checkpoint config, and also sets input-parity (`in_norm`) from `obs_norm` so evaluation
replicates the training-time input scaling (for near-init gaussian ckpts the old mismatch was
numerically mild; for a trained categorical head it would be fatal). Backward compatible:
gaussian checkpoints load and evaluate exactly as before (regression-checked).

## Step 1 -- FREE: evaluate the runs you already have (no retraining, ~5 min)
```bat
python v5_report.py
```
The failed evals left no JSON, so the report re-runs them automatically off the existing
best checkpoints and prints the two decision numbers vs the frozen v3 refs (4462.4 / 4304.7).

## Step 2 -- stop-rule rerun for the cold start (~1.5-2 h, only if Step 1 warrants)
Your fig14/log evidence: s60 was killed at ep 5800 with its TRAIN minimum at 5600 (still
improving at the deployment regime); s61 was killed at 3000 mid-recovery. patience=2000 was
designed for the warm start and is now the binding constraint -- the first time a stop-rule
change is justified by data (the A3 probe on the warm gaussian proved +0; that result was
instrument-specific).
```bat
python v5_run_nocomm.py --patience 4000 --min-ep 4000 --suffix f
python v5_report.py --prefix v5f_nocomm
python plot_train_curves.py --seeds 60,61 --arms v5f_nocomm
```
Read: new bests past the old stop points (5800 / 3000) => the stop rule was binding; same stall
points => it wasn't, and the next (held) lever is the exploration schedule, not more patience.

## Held deliberately (do NOT stack changes)
entropy_start 0.05 -> 0.02 / anneal_frac 0.5 -> 0.35 stay untouched until the stop-rule answer
is in; bins / lr / k_epochs / critic are not on the table at all yet.
