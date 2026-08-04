# v3.1 REPAIR — validation protocol (run before any pod launch)

## What changed
`agents/signal_agent.py`, `base_stock()`: the internal forecast now enters the base-stock head
**detached** — `self.demand_estimate(h).detach()`.

## Why
The registered defect was *"PPO back-propagating into the forecast head via the base-stock head
connection"* (d̂ driven to ≈S/5; pred SD 0.50 vs bench 5.52). A18 in v3.0 removed the whole
connection. Measured consequence at ρ=0.9, seed 60:

| config | nocomm | raw | V | vs frontier 3747.6 |
|---|---|---|---|---|
| v3.0 as registered (d̂ head OFF) | 4965.3 | 5062.6 | **−97** | +35% |
| v2 config (d̂ head ON, capturable) | 4788.6 | 3976.1 | **+813** | +6.1% |
| pilot (8000 eps, n=25) | 4207.8 | 3745.8 | +462 | 0% |

A18 removed the only architecturally-guaranteed source of **state-dependent** ordering, leaving a
static base-stock frozen at initialization. Detaching keeps the conditional structure while making
capture *impossible*: d_head is trained ONLY by the supervised aux MSE (no critic needed), and the
policy consumes it read-only.

## Arm settings that must change (manifest)
`agent.use_dhat_head=true` and `agent.s_init=5` on all SIGNAL arms (replacing v3.0's
`use_dhat_head=false` + `s_init=75`). Message content is unchanged and remains the only
manipulated variable.

## Validation (≈40 min, dev seed 60 only)
```bat
python -c "import subprocess,sys;[subprocess.run([sys.executable,'agents/train_signal.py','agent=signal','seed=60','total_episodes=8000','agent.heldout_every=200','agent.heldout_episodes=8','agent.patience=2000','env.penalty_at_retailer_only=false','agent.train_env=ar1','agent.ar1_rho=0.9','agent.heldout_mode=ar1','agent.comm_topology=retailer_broadcast','agent.use_dhat_head=true','agent.s_init=5','agent.msg_content=raw',f'agent.use_comm={c}',f'agent.algorithm=v31_{n}_s60']) for c,n in (('true','raw'),('false','nocomm'))]"

python -c "import glob,subprocess,sys;[subprocess.run([sys.executable,'agents/eval_signal.py','--ckpt',glob.glob(f'weights_signal/run_signal_*_v31_{n}_s60/signal_checkpoint_best.pt')[0],'--dump-comm',f'repair_out/devcheck/v31_{n}','--dump-ar1','0.9','--dump-episodes','30']) for n in ('nocomm','raw')]"

python -c "import json,numpy as np;c={a:np.mean([float(x) for x in json.load(open(f'repair_out/devcheck/v31_{a}/seed60.json')).values()]) for a in ('nocomm','raw')};print('v3.1: nocomm %.1f  raw %.1f  V = %+.1f'%(c['nocomm'],c['raw'],c['nocomm']-c['raw']))"
```

## Decision rule
- **raw ≈ 3900–4200 and V positive (few hundred)** → repair confirmed. Amend manifest + prereg
  (A3), re-tag, launch. The C3 certification contribution SURVIVES: the detached d̂ is now an
  honest supervised forecast and can be certified post-hoc with the existing gate
  (ratio ≤1.10, |bias| ≤0.5, slope 0.8–1.2, corr ≥0.8, SD ≥50% bench) as evidence the capture
  is gone.
- **raw ≈ 4800–5000 / V ≈ 0** → detaching also removed too much; fall back to routing the
  CERTIFIED FROZEN forecast into the base-stock head on every arm (larger change: threads
  `dhat_ext` through 7 call sites in signal_agent.py + eval_signal.py, plus a train↔eval
  identity test analogous to T11).

## Status of this sandbox check
Golden suite T1–T12 still passes with the detach in place. A 200-episode probe evaluated at
4833.8 — too short to judge (v2cfg needed ~1800 episodes to reach 3976). Full-budget
validation on your machine is the deciding evidence.
