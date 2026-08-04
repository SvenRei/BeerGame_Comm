# v5 action-head redesign -- benchmark, rationale, deployment

## Design principle (yours, adopted as the criterion)
For a communication study the learner must be RELIABLE, not maximal: V is a paired difference on
shared machinery, so the simplest head that learns every seed is the best instrument.

## The benchmark
| head design | start | seed-61 nocomm evidence | verdict |
|---|---|---|---|
| A. Normal-over-S, warm (v3 registered) | warm S~74 | frozen best@ep200, eval 4304.7, train flat (full budget, your machine) | reliable but INERT |
| B. A + v3.4 parity (obs/100, split opt) | warm | mild drift to ~4150 mid-run, gate-blind, decays; eval 4187 (full budget, your machine) | insufficient |
| C. your reference MAPPO (BeerGame repo) | cold, categorical bins | learns smoothly for 8k eps (your report) | existence proof: THIS env is learnable |
| D. **v5: categorical over the S-grid + parity + entropy anneal** | cold uniform | sandbox, 700 eps: gate 7395 -> 4325 within 100 eps (fast sharpening -- something the Gaussian never did from anywhere), then exploration plateau at entropy~0.046; full budget pending on your machine | **the candidate** |

Rejected variants: categorical over ORDER quantities (discards the base-stock inductive bias and
breaks comparability with AR_CondBS/GridCondBS); squashed-Gaussian / Beta (complexity without the
executed-action log-prob guarantee).

## Why D is "the simplest best"
1. log-prob is of the EXACTLY EXECUTED decision (the reference's own retired-Normal lesson).
2. Base-stock semantics preserved: the policy is still a state-dependent order-up-to level, so
   every frontier comparison (AR_CondBS 3747.6) and the paper's OR framing survive.
3. The grid IS QMIX's G2 grid (41 x s_max 160): one shared action set across both learners --
   kills a reviewer confound and may restore V1 comparability.
4. Proven family: orthogonal init g=0.01 (near-uniform cold start) + entropy 0.05 -> 0.005
   annealed + split optimizer + obs/100 -- the recipe from the implementation you KNOW learns
   this game for 8k episodes.
5. Communication machinery is BYTE-IDENTICAL v3 -> v5 (message mux, topology, frozen forecaster,
   probes): the treatment is untouched; only the actuation policy's distribution family changed.

## Deployment (your machine)
```bat
python v5_check.py
python v5_run_nocomm.py            :: nocomm, seeds 61+60, cold start, ~50-70 min
python v5_report.py
python plot_train_curves.py --seeds 60,61 --arms v5_nocomm
```
Expected curve SHAPE (finally the textbook one): very high initial cost (uniform S over 0..160),
steep early descent, long exploration middle, sharpening as entropy anneals past ~ep 4000.
PASS per seed = still improving at ep >= 1000 AND deterministic eval beats the frozen v3
instrument (4462.4 / 4304.7). Both PASS -> comm pairs next (I build the v5 pair scripts), then
fresh dev validation of the full ladder, amendment + re-registration, all arms retrained.
This is a NEW instrument: nothing carries over silently; the continuity claim in the paper
becomes "identical communication machinery, actuation family changed for learner reliability,
disclosed and benchmarked."

## Watchpoints
- a_loss magnitude runs ~200 on the cold start (vs ~30 warm-Gaussian) -- scale shift from the
  categorical log-probs; monitor, not yet a defect.
- The gate regime (rho .15/.45/.75) stays leakage-safe; final-weights checkpoint saves on every
  run for the gate-blindness diagnostic.
