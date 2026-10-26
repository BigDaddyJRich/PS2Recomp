@echo off
rem Claude Oct 26: one diagnostic run for the menu-transition band. No rebuild needed.
rem Run it, go to the screen where the band shows, wait ~10 s on it, then close the window.
rem Log: F:\hctp-newbase-vulcan\logs\native-gs-stats.err
setlocal
set "ROOT=%~dp0"
set "GAME=%ROOT%WWE SmackDown! Here Comes the Pain"
set "BIN=%ROOT%third_party\ps2-native-gs\bin"
set "RUNNER=%ROOT%out\native-gs\ps2xRuntime\ps2EntryRunner.exe"
if not exist "%ROOT%logs" mkdir "%ROOT%logs"
cd /d "%GAME%"
set "PATH=%BIN%;%PATH%"
set "HCTP_NATIVE_GS_MODULE=%BIN%\libps2-native-gs.dll"
set "HCTP_RENDERER=native-gs"
set "VENDETTA_GS=gpu"
set "VENDETTA_FILTER=bilinear"
rem Native GS counts every draw it skips (depth-format targets, aliased targets...) once a second.
set "PS2X_GS_NATIVE_STATS=1"
set "PS2X_GS_NATIVE_READBACK_LOG=1"
set "HCTP_NATIVE_GS_READBACK_DIAG=1"
"%RUNNER%" SLUS_207.87 2> "%ROOT%logs\native-gs-stats.err" > "%ROOT%logs\native-gs-stats.out"
