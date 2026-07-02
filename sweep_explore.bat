@echo off
REM ============================================================================
REM sweep_explore.bat -- find the config that UN-STICKS the SIGNAL base-stock.
REM
REM Hypothesis: S sits at its init (~50) because the shared grad-clip is dominated
REM by the critic loss (raw reward ~ -7500 -> c_loss ~ 9M), starving the actor.
REM This 2x2 isolates the two candidate levers, comm OFF, seed 10:
REM   reward_scale {1, 100}  x  exploration {off, on(entropy+log_std)}   (+ one strong cell)
REM
REM SUCCESS = held-out mean cost drops well below the bias-only plateau (~7500)
REM           toward / below the BAR (4063 on the validation lambdas). Then confirm
REM           with eval_signal that S_mean FANS with lambda (not a flat ~50).
REM
REM NOTE: SIGNAL is CPU-bound (sequential per-step GRU, batch=1), so the RTX 3060
REM       does NOT speed this up -- keep use_gpu=false. To parallelize, run a couple
REM       of these lines in separate terminals (they share CPU cores).
REM
REM USAGE:  sweep_explore.bat 1200     <- episodes/config (default 1200; ~1500 is cleaner)
REM ============================================================================
setlocal
cd /d "%~dp0"

set PY=python
if exist "venv\Scripts\python.exe"  set PY=venv\Scripts\python.exe
if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe

set EP=%1
if "%EP%"=="" set EP=1200
set WANDB_MODE=disabled
set COMMON=agent.use_comm=false seed=10 total_episodes=%EP% agent.heldout_every=100 agent.heldout_episodes=8

echo ============================================================
echo  SIGNAL exploration/scale sweep : %EP% episodes/config, comm OFF, seed 10
echo  watch the "best held-out mean cost" line per config (BAR=4063, plateau~7500)
echo ============================================================

echo.
echo [A] control            reward_scale=1   entropy=0     log_std_init=-0.5
%PY% agents\train_signal.py agent=signal %COMMON% agent.reward_scale=1   agent.entropy_coef=0    agent.log_std_init=-0.5 agent.algorithm=sw_A_control

echo.
echo [B] reward_scale only  reward_scale=100 entropy=0     log_std_init=-0.5
%PY% agents\train_signal.py agent=signal %COMMON% agent.reward_scale=100 agent.entropy_coef=0    agent.log_std_init=-0.5 agent.algorithm=sw_B_rs

echo.
echo [C] exploration only   reward_scale=1   entropy=0.02  log_std_init=1.0
%PY% agents\train_signal.py agent=signal %COMMON% agent.reward_scale=1   agent.entropy_coef=0.02 agent.log_std_init=1.0  agent.algorithm=sw_C_explore

echo.
echo [D] rs + exploration   reward_scale=100 entropy=0.02  log_std_init=1.0
%PY% agents\train_signal.py agent=signal %COMMON% agent.reward_scale=100 agent.entropy_coef=0.02 agent.log_std_init=1.0  agent.algorithm=sw_D_rs_explore

echo.
echo [E] rs + strong explore reward_scale=100 entropy=0.05 log_std_init=1.6
%PY% agents\train_signal.py agent=signal %COMMON% agent.reward_scale=100 agent.entropy_coef=0.05 agent.log_std_init=1.6  agent.algorithm=sw_E_rs_strong

echo.
echo ============================================================
echo  DONE. Compare the five "best held-out mean cost" lines above.
echo  Lowest wins; then confirm S fans with lambda on the winner:
echo     for /d %%D in (weights_signal\run_signal_*_sw_*) do @echo %%D
echo     %PY% agents\eval_signal.py --ckpt weights_signal\run_signal_<id>_sw_<X>\signal_checkpoint_best.pt --regime-episodes 5
echo  (read the C1 table's S_mean column: should climb ~33 -> ~115 across lambda, not sit at ~50)
echo ============================================================
endlocal
