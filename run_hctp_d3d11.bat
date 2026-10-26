@echo off
setlocal
rem Same build as run_hctp_native_gs.bat, with the D3D11 renderer (all the GS fixes) instead of the native-GS module.
set "ROOT=%~dp0"
set "GAME=%ROOT%WWE SmackDown! Here Comes the Pain"
set "RUNNER=%ROOT%out\native-gs\ps2xRuntime\ps2EntryRunner.exe"
cd /d "%GAME%"
set "HCTP_RENDERER=d3d11"
"%RUNNER%" SLUS_207.87
