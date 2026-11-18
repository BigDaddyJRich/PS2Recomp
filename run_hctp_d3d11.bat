@echo off
setlocal
rem D3D11 renderer (the main one). Game folder: F:\HCTP MODS\Vanilla (unmodded SLUS + our MODS overlays and the
rem commentary pack BGM\PBPSVR_LIST.pac).
set "ROOT=%~dp0"
set "GAME=F:\HCTP MODS\Vanilla"
set "RUNNER=%ROOT%out\native-gs\ps2xRuntime\ps2EntryRunner.exe"
cd /d "%GAME%"
if not defined HCTP_RENDERER set "HCTP_RENDERER=d3d11"
set "HCTP_MOVIES=native"
set "HCTP_ROSTER_EXPAND=1"
set "HCTP_REPLAY=1"
set "HCTP_MATRIX=1"
set "HCTP_TITLE_MOVIES=1"
rem community Royal Rumble mod: new entrant every 60 s (0 = stock)
set "HCTP_RUMBLE_60S=1"
set "HCTP_PERF_SPLIT=1"
set "HCTP_PERF_PHASES=1"
set "HCTP_GS_THREAD=1"
set "HCTP_CHEERLOG=1"
set "HCTP_EVTLOG=1"
set "HCTP_FILELOG=1"
set "VENDETTA_UNLOCKALL=1"
set "HCTP_GPU_CHARS_LITE=1"
set "HCTP_GPU_STAGE1=0"
rem Nov 4: tonight's fixes OFF in the shortcut until frame pacing is smooth again
set "HCTP_ZREAD_NODRAIN=1"
set "HCTP_S1_SBRING=0"
set "HCTP_CHAINBUF_POOL=1"
rem Nov 15: SvR06 entrances run SvR06's own pyro engine (hctp_svr06_ctx.cpp; was off = no SvR06 pyro)
if not defined HCTP_SVR06_CTX set "HCTP_SVR06_CTX=1"
rem Nov 15: SvR04 entrances run SvR04's own particle engine (hctp_svr04_ctx.cpp arm mode; Jordan-tested Nov 14 via run_camtest_svr04ctx.bat,
rem was never on here -> Nov 15 test used the old host pyro = smoke/water missing, HBK pyro height)
if not defined HCTP_SVR04_CTX set "HCTP_SVR04_CTX=1"
rem Nov 17 S21: SYM pyro + pyro SE = SYM's own gimmick/particle engine in its own memory (hctp_sym_ctx.cpp; 0 = off)
if not defined HCTP_SYM_CTX set "HCTP_SYM_CTX=1"
rem Nov 15: keep the previous run's log (a hang report needs the run before a restart)
if exist "%~dp0logs\test_nov2.err" copy /y "%~dp0logs\test_nov2.err" "%~dp0logs\test_nov2.prev.err" >nul
if exist "%~dp0logs\test_nov2.out" copy /y "%~dp0logs\test_nov2.out" "%~dp0logs\test_nov2.prev.out" >nul
"%RUNNER%" SLUS_207.87 > "%~dp0logs\test_nov2.out" 2> "%~dp0logs\test_nov2.err"
