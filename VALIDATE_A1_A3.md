# A1 (CRN control-variate baseline) + A3 (early-stop floor) — build record & decision protocol

## What was built
**A1** `agent.baseline_mode=condbs` (default `none` = registered legacy). For every training
episode, a FROZEN analytic AR conditional base-stock policy is rolled on a CRN **twin env reset to
the identical seed** (same demand path), and its shaped per-step cost is added back into the
learner's reward: `r'_t = r_t − r^CondBS_t` in reward terms. The subtracted term depends only on
the exogenous demand path, never on the learner's actions ⇒ the policy-gradient expectation is
unchanged (additive action-independent baseline) while the demand-path variance component cancels.
Gate, eval, milestones, and every reported cost are untouched — the learning signal only.
Fail-closed: requires `train_env=ar1`; a wrong-process twin raises `RuntimeError` on episode one.

**A3** `agent.min_train_episodes=N` (default 0 = legacy). The early-stop *condition* is unchanged;
if it fires before the floor, the run continues and prints one deferral line. `EARLY STOP` can only
execute at `ep ≥ N`.

## Correctness proofs (tests/test_a1_baseline.py — all PASS)
| proof | result |
|---|---|
| T1 trajectory invariance | baseline ON vs OFF: obs/actions/costs **bitwise identical** (twin consumes no torch RNG) |
| T2 reward arithmetic | `rew_ON − rew_OFF == (base_cost + β·base_others)/reward_scale`, max err **0.00e+00** |
| T3 variance reduction | episode-return variance **÷2.9** at n=16 CRN episodes; learner↔baseline CRN corr **−0.93** |
| T4 CRN guard | wrong-ρ twin **rejected loudly** ("A1 CRN twin mismatch") |
| existing suites | Phase-2, Phase-3/4, forecaster: ALL PASS (default `none`; goldens untouched) |

A3 smokes: floored stop executes at the floor; deferral branch fired at ep 30 and the run then
**found a new best at ep 70** — the mechanism rescuing a late improvement, in miniature.

## Honest sandbox findings (read before running the probes)
1. `c_loss` drops **~40×** (3e7 → ~7e5): the baseline really absorbs the demand-path variance.
2. **Explained variance stays ≈ 0 — and that is expected, not failure.** After the demand
   component is removed, the residual return is dominated by the policy's own exploration noise,
   which is aleatoric: no critic can explain it. A1's benefit is *gradient-variance reduction*
   (the measured 2.9×, ≈ a 3× batch at zero compute), not an EV number. Do not judge A1 by EV.
3. A 700-episode sandbox probe on seed 61 showed **no early ignition** (gate 2188 → 2336,
   drifting). Inconclusive by budget: seed-60 ignition historically arrived at ep 1000–2000.
   The decision experiments below are full-budget and run on your machine.

## Decision experiments (off-manifest probe tags; ~2.5 h total)
```bat
set WANDB_MODE=disabled
set SIGNAL_CSVLOG=1

:: P1 -- A1 on the frozen seed: raw + nocomm pair, seed 61, full budget
python -c "import subprocess,sys;[subprocess.run([sys.executable,'agents/train_signal.py','agent=signal','seed=61','total_episodes=8000','agent.heldout_every=200','agent.heldout_episodes=8','agent.patience=2000','env.penalty_at_retailer_only=false','agent.train_env=ar1','agent.ar1_rho=0.9','agent.heldout_mode=ar1','agent.comm_topology=retailer_broadcast','agent.msg_content=raw','agent.use_dhat_head=true','agent.baseline_mode=condbs',f'agent.use_comm={c}',f'agent.algorithm=a1_{n}_s61']) for c,n in (('true','raw'),('false','nocomm'))]"

:: P2 -- A3 floor on the frozen seed: raw, baseline OFF, floor 4000
python agents/train_signal.py agent=signal seed=61 total_episodes=8000 agent.heldout_every=200 agent.heldout_episodes=8 agent.patience=2000 agent.min_train_episodes=4000 env.penalty_at_retailer_only=false agent.train_env=ar1 agent.ar1_rho=0.9 agent.heldout_mode=ar1 agent.use_comm=true agent.comm_topology=retailer_broadcast agent.msg_content=raw agent.use_dhat_head=true agent.algorithm=a3_raw_s61

:: P1b -- no-regression: same A1 pair on seed 60 (where ignition already worked)
python -c "import subprocess,sys;[subprocess.run([sys.executable,'agents/train_signal.py','agent=signal','seed=60','total_episodes=8000','agent.heldout_every=200','agent.heldout_episodes=8','agent.patience=2000','env.penalty_at_retailer_only=false','agent.train_env=ar1','agent.ar1_rho=0.9','agent.heldout_mode=ar1','agent.comm_topology=retailer_broadcast','agent.msg_content=raw','agent.use_dhat_head=true','agent.baseline_mode=condbs',f'agent.use_comm={c}',f'agent.algorithm=a1_{n}_s60']) for c,n in (('true','raw'),('false','nocomm'))]"
```

## Read-out
```bat
python -c "import glob,os,subprocess,sys;[subprocess.run([sys.executable,'agents/eval_signal.py','--ckpt',max(glob.glob(f'weights_signal/run_signal_*_{t}/signal_checkpoint_best.pt'),key=os.path.getmtime),'--dump-comm',f'repair_out/a1/{t}','--dump-ar1','0.9','--dump-episodes','200'],stdout=subprocess.DEVNULL) for t in ('a1_raw_s61','a1_nocomm_s61','a3_raw_s61','a1_raw_s60','a1_nocomm_s60') if glob.glob(f'weights_signal/run_signal_*_{t}/signal_checkpoint_best.pt')]"

python -c "import json,os,numpy as np;f=lambda t,s:(np.mean([float(x) for x in json.load(open(f'repair_out/a1/{t}/seed{s}.json')).values()]) if os.path.exists(f'repair_out/a1/{t}/seed{s}.json') else None);o=lambda a,s:(np.mean([float(x) for x in json.load(open(f'repair_out/devcheck200/{a}_s{s}/seed{s}.json')).values()]) if os.path.exists(f'repair_out/devcheck200/{a}_s{s}/seed{s}.json') else None);print('  s61 A1 pair : nocomm %s raw %s  V=%+.1f   (old s61: V=+46.3)'%(f('a1_nocomm_s61',61),f('a1_raw_s61',61),(f('a1_nocomm_s61',61) or 0)-(f('a1_raw_s61',61) or 0)));print('  s61 A3 raw  : %s vs old nocomm %s -> V=%+.1f (old +46.3)'%(f('a3_raw_s61',61),o('r4_nocomm',61),(o('r4_nocomm',61) or 0)-(f('a3_raw_s61',61) or 0)));print('  s60 A1 pair : nocomm %s raw %s  V=%+.1f   (old s60: V=+624.9)'%(f('a1_nocomm_s60',60),f('a1_raw_s60',60),(f('a1_nocomm_s60',60) or 0)-(f('a1_raw_s60',60) or 0)))"

findstr /c:"A3 floor" /c:"EARLY STOP" /c:"new best" weights_signal\.trainlog_a3_raw_s61.txt 2>nul
findstr /c:"held-out mean cost" weights_signal\.trainlog_a3_raw_s61.txt
```

## Decision matrix
| P1 (A1, s61) | P2 (A3, s61) | P1b (A1, s60) | call |
|---|---|---|---|
| V ≥ ~+300 | — | V ≈ +500..700 | **adopt A1** (amendment A5, re-hash, ALL arms retrain) |
| V ≈ ±100 | improvement after ep 2200 | — | A1 insufficient; **adopt A3 floor** (cheap amendment) |
| V ≈ ±100 | flat to 4000 | — | neither binds ignition; keep registered instrument, report the mixture (v_distribution already covers it) |
| any | any | V collapses on s60 | **do not adopt A1** regardless of s61 |

## Guardrail
These tags are off-manifest probes. Adoption of either knob = instrument change ⇒ manifest
amendment + `python scripts/prereg_v3.py` re-hash + **every arm retrained symmetrically** before
the pod. No confirmatory seed (70–94) has run; the window for this is open now and closes at launch.
