@echo off
REM ============================================================================
REM sweep_comm_value.bat -- the value-of-communication experiments (needs use_dhat_head=true,
REM   i.e. the agent can act on demand). CRN-paired comm vs no-comm at seed 10.
REM
REM Theory (Axsater-Rosling 1993 / Chen 1998): in a SERIAL chain with i.i.d. (Poisson)
REM   demand, neighbour sharing is redundant with the order stream -> comm value ~ 0.
REM   Value should appear (Lee-So-Tang 2000) under AUTOCORRELATION, and for topologies
REM   that give upstream the retailer's signal directly ("more than neighbours").
REM
REM ALL CONFIGURATIONS: no-comm + every topology, on BOTH demand regimes.
REM   PART 1 (Poisson)     : no-comm + 10 topologies   (Axsater null + the controls)
REM   PART 2 (AR1 rho=0.9) : no-comm + 10 topologies   (autocorrelation -> value should appear)
REM   PART 3 (AR1 rho=0.0) : no-comm + retailer_broadcast (the rho-floor control: value ~ 0)
REM
REM   Topology semantics (see agents\topologies.py):
REM     neighbor            chain, range-1 (the Axsater null on Poisson)
REM     skip / full         wider listening (range-2 / all-to-all)
REM     retailer_broadcast  everyone hears the CLEAN retailer demand    (max-favorable, Lee)
REM     manufacturer_broadcast  everyone hears the MOST-distorted belief (directional placebo)
REM     upstream_only       belief flows UP hop-by-hop (theory-correct VMI direction)
REM     downstream_only     belief flows DOWN (wrong-direction placebo, adjacent)
REM     no_neighbor         hears only NON-adjacent stages (placebo)
REM     link_top_only       share at ONE link: manufacturer <- distributor (worst-bullwhip link)
REM     link_bottom_only    share at ONE link: wholesaler   <- retailer    (cleanest link)
REM
REM USAGE:
REM   sweep_comm_value.bat 8       <- SMOKE first: confirms all 28 arms run (seconds each)
REM   sweep_comm_value.bat 8000    <- the real run (default; early-stop trims arms that plateau)
REM
REM EARLY STOP: agent.patience=2000 -> an arm stops after 2000 episodes with NO held-out
REM   improvement. The best-gated checkpoint is ALWAYS kept (train_signal only saves on
REM   improvement and never overwrites with a worse model), so early stop never loses the
REM   best policy -- it just skips the dead tail. Set agent.patience=0 to force full 8k.
REM
REM NOTE: SIGNAL is CPU-bound (use_gpu=false); the RTX 3060 won't speed it up. To parallelize,
REM   run a few of these lines in separate terminals. cv_pois_nocomm duplicates a plain no-comm
REM   Poisson run -- if your signal_b_nocomm_s10 covers it, comment that line out.
REM ============================================================================
setlocal
cd /d "%~dp0"

set PY=python
if exist "venv\Scripts\python.exe"  set PY=venv\Scripts\python.exe
if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe

set EP=%1
if "%EP%"=="" set EP=8000
REM gate cadence: EP/4 but capped at 200 (frequent for real runs, still fires on an 8-ep smoke)
set /a HE=%EP%/4
if %HE% GTR 200 set HE=200
if %HE% LSS 1 set HE=1
set WANDB_MODE=disabled
set COMMON=seed=10 total_episodes=%EP% agent.heldout_every=%HE% agent.heldout_episodes=8 agent.patience=2000

echo ============================================================
echo  COMM-VALUE sweep : %EP% eps/arm, gate every %HE%, seed 10, patience=2000, use_dhat_head=true
echo  watch "best held-out mean cost"; compare comm vs no-comm WITHIN each block
echo ============================================================

echo.
echo --- PART 1: topology on Poisson (neighbour=Axsater null; broadcast=see-through; placebos last) ---
echo [1/28] poisson  no-comm  (baseline)
%PY% agents\train_signal.py agent=signal %COMMON% agent.use_comm=false                                            agent.algorithm=cv_pois_nocomm
echo [2/28] poisson  comm neighbour
%PY% agents\train_signal.py agent=signal %COMMON% agent.use_comm=true agent.comm_topology=neighbor               agent.algorithm=cv_pois_neighbor
echo [3/28] poisson  comm skip
%PY% agents\train_signal.py agent=signal %COMMON% agent.use_comm=true agent.comm_topology=skip                   agent.algorithm=cv_pois_skip
echo [4/28] poisson  comm full
%PY% agents\train_signal.py agent=signal %COMMON% agent.use_comm=true agent.comm_topology=full                   agent.algorithm=cv_pois_full
echo [5/28] poisson  comm retailer_broadcast
%PY% agents\train_signal.py agent=signal %COMMON% agent.use_comm=true agent.comm_topology=retailer_broadcast     agent.algorithm=cv_pois_rbroadcast
echo [6/28] poisson  comm manufacturer_broadcast (placebo)
%PY% agents\train_signal.py agent=signal %COMMON% agent.use_comm=true agent.comm_topology=manufacturer_broadcast agent.algorithm=cv_pois_mbroadcast
echo [7/28] poisson  comm upstream_only (theory-correct direction)
%PY% agents\train_signal.py agent=signal %COMMON% agent.use_comm=true agent.comm_topology=upstream_only          agent.algorithm=cv_pois_upstream
echo [8/28] poisson  comm downstream_only (wrong-direction placebo)
%PY% agents\train_signal.py agent=signal %COMMON% agent.use_comm=true agent.comm_topology=downstream_only        agent.algorithm=cv_pois_downstream
echo [9/28] poisson  comm no_neighbor (placebo)
%PY% agents\train_signal.py agent=signal %COMMON% agent.use_comm=true agent.comm_topology=no_neighbor            agent.algorithm=cv_pois_noneighbor
echo [10/28] poisson comm link_top_only (manufacturer^<-distributor)
%PY% agents\train_signal.py agent=signal %COMMON% agent.use_comm=true agent.comm_topology=link_top_only          agent.algorithm=cv_pois_linktop
echo [11/28] poisson comm link_bottom_only (wholesaler^<-retailer)
%PY% agents\train_signal.py agent=signal %COMMON% agent.use_comm=true agent.comm_topology=link_bottom_only       agent.algorithm=cv_pois_linkbot

echo.
echo --- PART 2: AR(1) autocorrelation rho=0.9 (value should appear; same 10 topologies) ---
set AR9=agent.train_env=ar1 agent.ar1_rho=0.9 agent.heldout_mode=ar1
echo [12/28] ar1 r0.9 no-comm (baseline)
%PY% agents\train_signal.py agent=signal %COMMON% %AR9% agent.use_comm=false                                            agent.algorithm=cv_ar1r9_nocomm
echo [13/28] ar1 r0.9 comm neighbour
%PY% agents\train_signal.py agent=signal %COMMON% %AR9% agent.use_comm=true agent.comm_topology=neighbor               agent.algorithm=cv_ar1r9_neighbor
echo [14/28] ar1 r0.9 comm skip
%PY% agents\train_signal.py agent=signal %COMMON% %AR9% agent.use_comm=true agent.comm_topology=skip                   agent.algorithm=cv_ar1r9_skip
echo [15/28] ar1 r0.9 comm full
%PY% agents\train_signal.py agent=signal %COMMON% %AR9% agent.use_comm=true agent.comm_topology=full                   agent.algorithm=cv_ar1r9_full
echo [16/28] ar1 r0.9 comm retailer_broadcast
%PY% agents\train_signal.py agent=signal %COMMON% %AR9% agent.use_comm=true agent.comm_topology=retailer_broadcast     agent.algorithm=cv_ar1r9_rbroadcast
echo [17/28] ar1 r0.9 comm manufacturer_broadcast (placebo)
%PY% agents\train_signal.py agent=signal %COMMON% %AR9% agent.use_comm=true agent.comm_topology=manufacturer_broadcast agent.algorithm=cv_ar1r9_mbroadcast
echo [18/28] ar1 r0.9 comm upstream_only (theory-correct direction)
%PY% agents\train_signal.py agent=signal %COMMON% %AR9% agent.use_comm=true agent.comm_topology=upstream_only          agent.algorithm=cv_ar1r9_upstream
echo [19/28] ar1 r0.9 comm downstream_only (wrong-direction placebo)
%PY% agents\train_signal.py agent=signal %COMMON% %AR9% agent.use_comm=true agent.comm_topology=downstream_only        agent.algorithm=cv_ar1r9_downstream
echo [20/28] ar1 r0.9 comm no_neighbor (placebo)
%PY% agents\train_signal.py agent=signal %COMMON% %AR9% agent.use_comm=true agent.comm_topology=no_neighbor            agent.algorithm=cv_ar1r9_noneighbor
echo [21/28] ar1 r0.9 comm link_top_only (manufacturer^<-distributor)
%PY% agents\train_signal.py agent=signal %COMMON% %AR9% agent.use_comm=true agent.comm_topology=link_top_only          agent.algorithm=cv_ar1r9_linktop
echo [22/28] ar1 r0.9 comm link_bottom_only (wholesaler^<-retailer)
%PY% agents\train_signal.py agent=signal %COMMON% %AR9% agent.use_comm=true agent.comm_topology=link_bottom_only       agent.algorithm=cv_ar1r9_linkbot

echo.
echo --- PART 3: AR(1) autocorrelation GRADIENT (H1/H2): no-comm vs retailer_broadcast at rho 0,0.3,0.6 ---
echo    (rho=0.9 pair reused from PART 2 = cv_ar1r9_nocomm / cv_ar1r9_rbroadcast; r0/r3/r6/r9 give V(rho):
echo     V should be ~0 at rho=0 (H1 null, TOST) and RISE with rho (H2). retailer_broadcast = clean-signal anchor)
set AR0=agent.train_env=ar1 agent.ar1_rho=0.0 agent.heldout_mode=ar1
set AR3=agent.train_env=ar1 agent.ar1_rho=0.3 agent.heldout_mode=ar1
set AR6=agent.train_env=ar1 agent.ar1_rho=0.6 agent.heldout_mode=ar1
echo [23/28] ar1 r0.0 no-comm
%PY% agents\train_signal.py agent=signal %COMMON% %AR0% agent.use_comm=false                                        agent.algorithm=cv_ar1r0_nocomm
echo [24/28] ar1 r0.0 comm retailer_broadcast
%PY% agents\train_signal.py agent=signal %COMMON% %AR0% agent.use_comm=true agent.comm_topology=retailer_broadcast agent.algorithm=cv_ar1r0_rbroadcast
echo [25/28] ar1 r0.3 no-comm
%PY% agents\train_signal.py agent=signal %COMMON% %AR3% agent.use_comm=false                                        agent.algorithm=cv_ar1r3_nocomm
echo [26/28] ar1 r0.3 comm retailer_broadcast
%PY% agents\train_signal.py agent=signal %COMMON% %AR3% agent.use_comm=true agent.comm_topology=retailer_broadcast agent.algorithm=cv_ar1r3_rbroadcast
echo [27/28] ar1 r0.6 no-comm
%PY% agents\train_signal.py agent=signal %COMMON% %AR6% agent.use_comm=false                                        agent.algorithm=cv_ar1r6_nocomm
echo [28/28] ar1 r0.6 comm retailer_broadcast
%PY% agents\train_signal.py agent=signal %COMMON% %AR6% agent.use_comm=true agent.comm_topology=retailer_broadcast agent.algorithm=cv_ar1r6_rbroadcast

echo.
echo ============================================================
echo  DONE. Quick read = compare "best held-out mean cost" WITHIN each block:
echo    Poisson : every topology ~= no-comm    (Axsater serial null; placebos confirm no leakage)
echo              if retailer_broadcast / upstream_only beat no-comm -> Lee see-through-bullwhip
echo    AR1 r0.9: retailer_broadcast / upstream_only should BEAT no-comm; placebos (manufacturer_
echo              broadcast, downstream_only, no_neighbor) should NOT -> value is the clean signal
echo    AR1 gradient (H1/H2): V(rho)=cost(cv_ar1r{N}_nocomm)-cost(cv_ar1r{N}_rbroadcast) for N in 0,3,6,9
echo              should be ~0 at rho=0 (H1 null) and RISE with rho (H2) -> value tracks autocorrelation
echo  For the statistical V_cost + upstream forecast-error delta, dump per-seed rows and aggregate:
echo    %PY% agents\eval_signal.py --ckpt weights_signal\run_signal_^<id^>_cv_ar1r9_rbroadcast\signal_checkpoint_best.pt   --dump-comm results\ar1r9_comm
echo    %PY% agents\eval_signal.py --ckpt weights_signal\run_signal_^<id^>_cv_ar1r9_nocomm\signal_checkpoint_best.pt --dump-comm results\ar1r9_nocomm
echo    %PY% scripts\comm_stats.py report --comm-dir results\ar1r9_comm --nocomm-dir results\ar1r9_nocomm
echo ============================================================
endlocal
