# v3.3 — learned message-head saturation fix

## The bug
`SIGNALActor.message()` for `content="learned"` computed
`msg_gain * tanh(msg_head(cat([obs, h])))` with **unnormalized** `obs`. Observations are natural
units (inventory/pipeline reach 50–100 mid-episode) while `h` is tanh-bounded O(1), so the
pre-tanh activation was already large at initialization and grew during the episode:

| obs | before fix: \|tanh\| | gradient d tanh/d pre |
|---|---|---|
| `[12,0,16,12]` (init) | 0.998 / 0.553 / 0.946 | **0.003** / 0.694 / 0.106 |
| `[60,0,80,20]` (mid-episode) | **1.000 / 1.000 / 1.000** | **~0** |

The channel was a saturated near-constant from step one and could not learn. This was visible in
the v3.2 honesty probe and neither of us read it: retailer channel 0 showed `sat=1.00` with
`corr(demand)=0.003`. Consequence: `r4_learned` reached only V=+228 vs raw's +625 — an artifact of
a dead head, NOT a finding about learned communication.

## The fix
Scale `msg_head`'s obs input by `obs_scale` (default `max_order` = 100) so it is commensurate with
the tanh-bounded GRU state. **The emitted message scale is unchanged** (`msg_gain * tanh`, still
O(10)); only the head's input conditioning changes.

| obs | after fix: \|tanh\| | gradient |
|---|---|---|
| `[12,0,16,12]` | 0.319 / 0.019 / 0.011 | 0.898 / 1.000 / 1.000 |
| `[60,0,80,20]` | 0.402 / 0.078 / 0.060 | 0.839 / 0.994 / 0.996 |

Golden evidence: the learned rung's first emitted message moved from **9.998** (pinned at the gain
of 10 — saturated) to **5.043** (mid-range, responsive).

## Test status
`raw`, `dhat`, `condmean` goldens are **UNCHANGED** — proof the fix is confined to the learned rung.
The `learned` golden was re-baselined (a_loss 54.699→54.261, c_loss 29898→27709, S_sum 2732→2717,
cost_sum 356→345, msg_t3 9.998→5.043) because the old values encoded the saturated head.
All suites pass: T1–T6, T7–T12b, Q1–Q6.

## Validate (~25 min, dev seed 60)
```bat
del weights_signal\.done_r4_learned_s60
set WANDB_MODE=disabled
python run_repair_study.py signal-dev --arms r4_learned --seeds-limit 1 --jobs 1

python -c "import glob,os,subprocess,sys;subprocess.run([sys.executable,'agents/eval_signal.py','--ckpt',max(glob.glob('weights_signal/run_signal_*_r4_learned_s60/signal_checkpoint_best.pt'),key=os.path.getmtime),'--dump-comm','repair_out/devcheck200/r4_learned','--dump-ar1','0.9','--dump-episodes','200'])"

python -c "import glob,os,subprocess,sys;subprocess.run([sys.executable,'agents/eval_signal.py','--ckpt',max(glob.glob('weights_signal/run_signal_*_r4_learned_s60/signal_checkpoint_best.pt'),key=os.path.getmtime),'--messages','--interventions','--ar1','--ar1-rho','0.9','--episodes','200'])"

python -c "import json,os,numpy as np;b=json.load(open('results/baselines_ar_v3.json'))['rungs'];ref=float(b['AR_BestBS']['0.9']);A=('r4_nocomm','r4_raw','r4_arpred','r4_learned','r4_dhatc');c={a:np.mean([float(x) for x in json.load(open(f'repair_out/devcheck200/{a}/seed60.json')).values()]) for a in A if os.path.exists(f'repair_out/devcheck200/{a}/seed60.json')};print('  rung           cost        V     gap');[print('    %-12s %8.1f %+8.1f   %+6.1f%%'%(a,c[a],c['r4_nocomm']-c[a],100*(c[a]/ref-1))) for a in c]"
```

## What to read
In the honesty probe, `sat>0.9` should now be well below 1.00 on most channels and at least one
channel should show a materially higher `corr(demand)` than the old 0.003. In the intervention
probe, the zeroed delta should rise well above the old +238 if the channel is now carrying
information. And V should move up from +228 — toward raw's +625 if the channel learns an
invertible-ish encoding, or stay mid-ladder if a learned code genuinely cannot match a
parameter-free readout. **Either outcome is now a real result rather than an artifact.**
