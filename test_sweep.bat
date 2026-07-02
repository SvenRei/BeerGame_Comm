@echo off
REM ============================================================================
REM test_sweep.bat -- signal agent shakedown across EVERY config path.
REM   Runs straight from cmd (or double-click). No need to activate the venv or
REM   be in the right folder first -- this script cd's to its own location
REM   (assumed = project root) and uses the venv python automatically.
REM
REM   Tests that each arm TRAINS end-to-end (rollout + MAPPO update + held-out
REM   gate + checkpoint) without crashing. Does NOT measure value-of-sharing
REM   (that needs agents\eval_signal.py).
REM
REM USAGE:  test_sweep.bat 300      <- FAST path-check first (all 11 arms, minutes)
REM         test_sweep.bat 8000     <- the real run
REM         test_sweep.bat          <- no arg defaults to 300
REM ============================================================================
setlocal

REM --- go to this script's own folder (must be the project root) ---
cd /d "%~dp0"

REM --- pick the python interpreter (prefer the project venv) ---
set PY=python
if exist "venv\Scripts\python.exe"  set PY=venv\Scripts\python.exe
if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe
echo Using interpreter: %PY%

REM --- sanity: the trainer must be reachable from here ---
if not exist "agents\train_signal.py" (
  echo ERROR: agents\train_signal.py not found.
  echo Put test_sweep.bat in the PROJECT ROOT ^(the folder that contains agents\, envs\, conf\^).
  pause
  exit /b 1
)

set EP=%1
if "%EP%"=="" set EP=300
set /a HE=%EP%/4
if %HE% LSS 1 set HE=1
set WANDB_MODE=disabled
set COMMON=seed=10 total_episodes=%EP% agent.heldout_every=%HE% agent.heldout_episodes=3

echo ============================================================
echo  signal shakedown : %EP% episodes/run, held-out gate every %HE%
echo ============================================================

echo.
echo [1/11] comm OFF (baseline, ADJ zeroed)
%PY% agents\train_signal.py agent=signal agent.use_comm=false %COMMON% agent.algorithm=test_nocomm

echo.
echo --- message-content ladder (neighbor topology) ---
echo [2/11] dhat
%PY% agents\train_signal.py agent=signal agent.use_comm=true agent.msg_content=dhat    agent.comm_topology=neighbor %COMMON% agent.algorithm=test_dhat
echo [3/11] ip
%PY% agents\train_signal.py agent=signal agent.use_comm=true agent.msg_content=ip      agent.comm_topology=neighbor %COMMON% agent.algorithm=test_ip
echo [4/11] dhat_ip
%PY% agents\train_signal.py agent=signal agent.use_comm=true agent.msg_content=dhat_ip agent.comm_topology=neighbor %COMMON% agent.algorithm=test_dhatip
echo [5/11] learned
%PY% agents\train_signal.py agent=signal agent.use_comm=true agent.msg_content=learned agent.comm_topology=neighbor %COMMON% agent.algorithm=test_learned

echo.
echo --- topologies (dhat content) ---
echo [6/11] skip
%PY% agents\train_signal.py agent=signal agent.use_comm=true agent.msg_content=dhat agent.comm_topology=skip               %COMMON% agent.algorithm=test_skip
echo [7/11] full
%PY% agents\train_signal.py agent=signal agent.use_comm=true agent.msg_content=dhat agent.comm_topology=full               %COMMON% agent.algorithm=test_full
echo [8/11] retailer_broadcast
%PY% agents\train_signal.py agent=signal agent.use_comm=true agent.msg_content=dhat agent.comm_topology=retailer_broadcast %COMMON% agent.algorithm=test_broadcast
echo [9/11] no_neighbor (placebo)
%PY% agents\train_signal.py agent=signal agent.use_comm=true agent.msg_content=dhat agent.comm_topology=no_neighbor        %COMMON% agent.algorithm=test_noneighbor

echo.
echo --- AR(1) regime (the autocorrelation path) ---
echo [10/11] ar1 rho=0.9, comm ON
%PY% agents\train_signal.py agent=signal agent.use_comm=true agent.msg_content=dhat agent.comm_topology=neighbor agent.train_env=ar1 agent.ar1_rho=0.9 agent.heldout_mode=ar1 %COMMON% agent.algorithm=test_ar1_on
echo [11/11] ar1 rho=0.9, comm OFF (the CRN partner)
%PY% agents\train_signal.py agent=signal agent.use_comm=false agent.train_env=ar1 agent.ar1_rho=0.9 agent.heldout_mode=ar1 %COMMON% agent.algorithm=test_ar1_off

echo.
echo ============================================================
echo  DONE (%EP% episodes/run). Verify every checkpoint exists:
echo     dir /s /b weights_signal\*.pt
echo  Expect 11 lines (one per arm). A missing arm = that path crashed.
echo ============================================================
pause
endlocal