# v3.4 — conditioning parity with the WORKING reference (SvenRei/BeerGame, agents/rl/mappo.py)

## The comparison that settles "where is the bug"
Your reference MAPPO learns smoothly for 8k episodes. The systematic diff against SIGNAL:

| # | reference (works) | SIGNAL (ignition lottery) |
|---|---|---|
| 1 | `obs/100` at EVERY network input (actor fc + critic `state/100`) | RAW obs into the belief GRU; RAW 133-dim state into the critic |
| 2 | Categorical head over shared bins; log-prob of the EXECUTED action; orthogonal init g=0.01; its own comment: "The old Normal policy sampled unbounded continuous actions while the env clipped them" | Normal over S; env executes clip(S−IP,0,100) — the exact design the reference REPLACED |
| 3 | SEPARATE optimizers: critic lr 1e-3 (3.3× actor), separate clips | one Adam + one 0.5 clip over actors+critic; measured: 100% of updates clipped, pre-clip norm ~4e3, dominated by ~1.5e6 critic loss |
| 4 | entropy 0.05 annealed; 1000-ep warm-up; near-uniform start (real exploration phase) | entropy 0.0; warm start at S≈74 (deliberate, load-bearing) |

Diff #1 generalizes the v3.3 msg_head saturation bug we already proved: the SAME raw-input
pathology afflicted the belief GRU and the critic all along. It also explains why ALL learning
rode on the single d̂ scalar (the only well-conditioned path), and hence the A18/detach kills.

## What v3.4 ports (config-gated, default OFF = registered instrument, bitwise)
- `agent.obs_norm=true`  : obs AND messages ÷ obs_scale at every actor input (belief GRU +
  base-stock head; messages are in demand units, so the same divisor — scaling obs alone would
  make the channel 100× louder than the state and bias the comm treatment); critic global state
  ÷ `state_scale` (100).
- `agent.split_optimizer=true` : separate critic Adam at `lr_critic` (default 1e-3) with
  SEPARATE gradient clips; `critic_grad_norm` now logged alongside the actor's.

Deliberately NOT ported (each would be an instrument redesign, not a repair): the categorical
action head (#2 — candidate only if parity fails), entropy 0.05 + warm-up (#4 — meaningful with
a cold start; our sweep at 0.003/0.01 on the warm start read as noise).

## Evidence so far
- All suites pass with flags OFF (bitwise-identical registered instrument).
- Wiring proof: `in_norm True | split True | critic lr 0.001`.
- 743-ep probe, seed 61 (the frozen seed), flags ON:
  ACTOR grad_norm median 4.66 (was ~3,900; 100% clipped) — the actor receives its own gradient.
  Gate: new best at ep 200 (2343.5 → 2323.5) — s61 never improved past its first gate before.
  explained_variance still ≈ 0 — now understood as the wrong success metric (the reference
  almost certainly has low EV too; it learns on normalized advantages + exploration).

## Decision pair (your machine, ~2 h)
```bat
set WANDB_MODE=disabled
set SIGNAL_CSVLOG=1
:: seed 61 (frozen seed) raw+nocomm with parity ON
python -c "import subprocess,sys;from concurrent.futures import ThreadPoolExecutor;B=['agent=signal','seed=61','total_episodes=8000','agent.heldout_every=200','agent.heldout_episodes=8','agent.patience=2000','env.penalty_at_retailer_only=false','agent.train_env=ar1','agent.ar1_rho=0.9','agent.heldout_mode=ar1','agent.comm_topology=retailer_broadcast','agent.msg_content=raw','agent.use_dhat_head=true','agent.obs_norm=true','agent.split_optimizer=true'];J=[['agent.use_comm=true','agent.algorithm=v34_raw_s61'],['agent.use_comm=false','agent.algorithm=v34_nocomm_s61']];f=lambda j:subprocess.run([sys.executable,'agents/train_signal.py',*B,*j]);list(ThreadPoolExecutor(max_workers=2).map(f,J))"
:: seed 60 no-regression pair
python -c "import subprocess,sys;from concurrent.futures import ThreadPoolExecutor;B=['agent=signal','seed=60','total_episodes=8000','agent.heldout_every=200','agent.heldout_episodes=8','agent.patience=2000','env.penalty_at_retailer_only=false','agent.train_env=ar1','agent.ar1_rho=0.9','agent.heldout_mode=ar1','agent.comm_topology=retailer_broadcast','agent.msg_content=raw','agent.use_dhat_head=true','agent.obs_norm=true','agent.split_optimizer=true'];J=[['agent.use_comm=true','agent.algorithm=v34_raw_s60'],['agent.use_comm=false','agent.algorithm=v34_nocomm_s60']];f=lambda j:subprocess.run([sys.executable,'agents/train_signal.py',*B,*j]);list(ThreadPoolExecutor(max_workers=2).map(f,J))"
```

## Read-out
```bat
python -c "import glob,os,subprocess,sys;[subprocess.run([sys.executable,'agents/eval_signal.py','--ckpt',max(glob.glob(f'weights_signal/run_signal_*_{t}/signal_checkpoint_best.pt'),key=os.path.getmtime),'--dump-comm',f'repair_out/v34/{t}','--dump-ar1','0.9','--dump-episodes','200'],stdout=subprocess.DEVNULL) for t in ('v34_raw_s61','v34_nocomm_s61','v34_raw_s60','v34_nocomm_s60') if glob.glob(f'weights_signal/run_signal_*_{t}/signal_checkpoint_best.pt')]"
python -c "import json,os,numpy as np;f=lambda t,s:(np.mean([float(x) for x in json.load(open(f'repair_out/v34/{t}/seed{s}.json')).values()]) if os.path.exists(f'repair_out/v34/{t}/seed{s}.json') else None);print('  s61 v3.4: nocomm %s raw %s  V=%+.1f  (registered s61 V=+46.3)'%(f('v34_nocomm_s61',61),f('v34_raw_s61',61),(f('v34_nocomm_s61',61) or 0)-(f('v34_raw_s61',61) or 0)));print('  s60 v3.4: nocomm %s raw %s  V=%+.1f  (registered s60 V=+624.9)'%(f('v34_nocomm_s60',60),f('v34_raw_s60',60),(f('v34_nocomm_s60',60) or 0)-(f('v34_raw_s60',60) or 0)))"
python plot_train_curves.py --seeds 60,61 --arms v34_raw v34_nocomm 2>nul
```

## Decision rule (mirrors A1's; the veto is symmetric)
- s61 V ≥ ~+300 AND s60 V ≈ +500..700  → adopt as amendment A5: re-hash, retrain ALL arms.
- s61 unchanged (~+50)                 → parity alone insufficient; the remaining structural
  divergence is the action head (#2). That is a redesign decision, not a patch.
- s60 collapses                        → do not adopt; registered instrument stands.
