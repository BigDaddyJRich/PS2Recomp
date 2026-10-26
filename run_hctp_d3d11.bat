@echo off
setlocal
rem D3D11 renderer (the main one). Game folder: F:\HCTP MODS\Vanilla (unmodded SLUS + our MODS overlays and the
rem commentary pack BGM\PBPSVR_LIST.pac). HCTP_GAME_DIR=<folder> picks another one (e.g. this tree's vanilla copy).
set "ROOT=%~dp0"
set "GAME=F:\HCTP MODS\Vanilla"
if defined HCTP_GAME_DIR set "GAME=%HCTP_GAME_DIR%"
set "RUNNER=%ROOT%out\native-gs\ps2xRuntime\ps2EntryRunner.exe"
cd /d "%GAME%"
set "HCTP_RENDERER=d3d11"
"%RUNNER%" SLUS_207.87
