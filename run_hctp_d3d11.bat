@echo off
setlocal
rem D3D11 renderer (the main one). Game folder: F:\HCTP MODS\Vanilla (unmodded SLUS + our MODS overlays and the
rem commentary pack BGM\PBPSVR_LIST.pac).
set "ROOT=%~dp0"
set "GAME=F:\HCTP MODS\Vanilla"
set "RUNNER=%ROOT%out\native-gs\ps2xRuntime\ps2EntryRunner.exe"
cd /d "%GAME%"
set "HCTP_RENDERER=d3d11"
set "HCTP_MOVIES=native"
set "HCTP_TITLE_MOVIES=1"
rem community Royal Rumble mod: new entrant every 60 s (0 = stock)
set "HCTP_RUMBLE_60S=1"
set "HCTP_PERF_SPLIT=1"
set "HCTP_PERF_PHASES=1"
set "HCTP_GS_THREAD=1"
"%RUNNER%" SLUS_207.87 > "%~dp0logs\test_nov2.out" 2> "%~dp0logs\test_nov2.err"
