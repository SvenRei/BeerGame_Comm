# v3.2a — A19 aux-gate correction. Re-run r4_dhatc only.

## What the dev run established (rho=0.9, seed 60)
| arm | cost | V | vs frontier 3747.6 |
|---|---|---|---|
| r4_nocomm | 4788.6 | — | +27.8% |
| r4_raw | 3976.1 | **+812.6** | +6.1% |
| **r4_arpred** | **3889.1** | **+899.6** | **+3.8%** |
| r4_dhatc | 4904.6 | −115.9 | +30.9% |

raw and nocomm reproduce v2cfg bit-for-bit, so the instrument is validated. arpred is the best
arm and sits within 3.8% of the privileged frontier.

## Why dhatc was an artifact, not a finding
`signal_agent.py` zeroed the auxiliary loss whenever `forecast_mode=separate_frozen`. Correct
under A18 (the internal d_head was unused there). Under **A19 the internal d_head drives
actuation on every arm**, so r4_dhatc was the only arm whose level scalar had no supervised
anchor — an uncontrolled actuation difference. Decisive evidence: `arpred` broadcasts
mu + rho*(d-mu) and the certified forecaster predicts essentially the same quantity, yet they
differ by ~1000 cost units. Same information, different actuation treatment.

**Fix:** the gate now applies only when the internal head is genuinely unused
(`separate_frozen AND not use_dhat_head`). Actuation treatment is identical across all arms and
`msg_content` is again the sole manipulated variable. Suites T1–T12b and Q1–Q6 pass.

## Re-run (only dhatc changes; the other three are unaffected — do NOT retrain them)
```bat
del weights_signal\.done_r4_dhatc_s60
set WANDB_MODE=disabled
python run_repair_study.py signal-dev --arms r4_dhatc --seeds-limit 1 --jobs 1

python -c "import glob,subprocess,sys;subprocess.run([sys.executable,'agents/eval_signal.py','--ckpt',glob.glob('weights_signal/run_signal_*_r4_dhatc_s60/signal_checkpoint_best.pt')[0],'--dump-comm','repair_out/devcheck/r4_dhatc','--dump-ar1','0.9','--dump-episodes','30'])"

python -c "import json,numpy as np;c={a:np.mean([float(x) for x in json.load(open(f'repair_out/devcheck/{a}/seed60.json')).values()]) for a in ('r4_nocomm','r4_raw','r4_dhatc','r4_arpred')};[print('  %-11s %8.1f   V=%+8.1f'%(a,c[a],c['r4_nocomm']-c[a])) for a in c];print('  frontier 3747.6 | pilot raw 3745.8 V=+462')"
```

## Reading the result
- **dhatc lands near raw/arpred (V positive, few hundred)** → C-NULL' Branch A territory: a
  certified forecast is roughly as good as raw POS. Launch the pod.
- **dhatc still well below raw with V ~ 0 or negative** → now a genuine finding rather than an
  artifact: the certified frozen forecast is not a useful message even when actuation is
  controlled. That is publishable (it sharpens the trajectory-vs-statistic argument), but send
  the numbers before launching so the registration wording matches.
