# v3.2 — A19 continuity mode. Validation before pod launch.

## The measurement that drove this (rho=0.9, seed 60, identical protocol)
| actuation config | nocomm | raw | V | vs frontier 3747.6 | learning |
|---|---|---|---|---|---|
| A18 clean (no d_hat in S head) | 4965.3 | 5062.6 | −97 | +35% | none (best @ep400) |
| d_hat in S head, **detached** | 4875.2 | 4833.8 | +41 | +29% | none (best @ep200) |
| **d_hat in S head, with gradient** | 4788.6 | **3976.1** | **+813** | **+6.1%** | yes (best @ep1800) |
| registered pilot (n=25, 8000 eps) | 4207.8 | 3745.8 | +462 | 0% | — |

**Diagnosis.** The critic explains ~0 variance, so the low-dimensional policy gradient through
the d_hat scalar is the learner's PRINCIPAL channel for adapting the order-up-to level. A18 and
the detach both sever it, leaving a static base-stock frozen near init — and with identical
frozen actuation on every arm, V ≈ 0 **by construction**.

## What changed in code (v3.2)
1. `signal_agent.base_stock()` — detach REVERTED; d_hat enters the S head with gradient, as in
   the pilot.
2. `signal_agent` A18 guard — **A19 continuity mode enabled**: `use_dhat_head=true` together
   with `forecast_mode=separate_frozen` is now valid. Actuation and message are independent by
   construction (`message()` returns `dhat_ext` when supplied; `base_stock()` always uses the
   internal `demand_estimate(h)`).
3. `tests/test_phase2_integration.py` — T12 still fail-closes raw+frozen and typo modes; **T12b**
   asserts A19 is accepted, the forecaster loads certified and frozen, and the actuation head
   stays active. All suites pass.

## The scientific reframing (this is the paper's story now)
The internal d_hat is a **policy parameterization** — a learned level scalar — not a forecast.
That is exactly why the audit found pred SD 0.50 vs bench 5.52. So the defect was never in the
actuation; it was **broadcasting that internal scalar as if it were a demand forecast**. The
repair belongs in the MESSAGE: `msg_content=dhat` + `forecast_mode=separate_frozen` broadcasts
the CERTIFIED FROZEN forecast (ratio 1.047, bias −0.018, SD 5.616/5.625) while the policy keeps
its internal scalar. Contribution C3 (pretrain → certify → freeze) is unchanged and strengthened:
the mechanism is demonstrated and repaired at its true site rather than amputated. Bonus: the
actuation is now IDENTICAL across all arms, so message content is the only manipulated
variable — a cleaner control than the pilot had.

## Registration
Manifest amendment A3 (supersedes A2) written pre-execution; prereg re-registered.
**New binding hash: 5fc47b7dc24eae43c75c030b272774bd8cd182963697082f6a1d650de31b5d58**

## VALIDATE (≈80 min, dev seed 60 only) — four arms, then eval at rho 0.9
```bat
set WANDB_MODE=disabled
python run_repair_study.py signal-dev --arms r4_nocomm r4_raw r4_dhatc r4_arpred --seeds-limit 1 --jobs 2

python -c "import glob,subprocess,sys;[subprocess.run([sys.executable,'agents/eval_signal.py','--ckpt',glob.glob(f'weights_signal/run_signal_*_{a}_s60/signal_checkpoint_best.pt')[0],'--dump-comm',f'repair_out/devcheck/{a}','--dump-ar1','0.9','--dump-episodes','30']) for a in ('r4_nocomm','r4_raw','r4_dhatc','r4_arpred')]"

python -c "import json,numpy as np;c={a:np.mean([float(x) for x in json.load(open(f'repair_out/devcheck/{a}/seed60.json')).values()]) for a in ('r4_nocomm','r4_raw','r4_dhatc','r4_arpred')};[print('  %-11s %8.1f   V=%+8.1f'%(a,c[a],c['r4_nocomm']-c[a])) for a in c];print('  frontier 3747.6 | pilot raw 3745.8 V=+462')"
```

## Decision rule
- **raw ≈ 3900–4200, V(raw) positive (few hundred)** → instrument reproduces the pilot. Commit,
  re-tag v3.0, launch the pod. V(dhatc) and V(arpred) are then genuine scientific readings, not
  artifacts.
- **raw ≳ 4600 / V ≈ 0** → the A19 arms still are not learning; STOP and report the four numbers.

## Note on delete-before-rerun
`del weights_signal\.done_r4_*` (or delete `weights_signal` entirely) — the sentinels from the
earlier configs must not be reused.
