@echo off
setlocal
set "ROOT=%~dp0"
set "GAME=%ROOT%WWE SmackDown! Here Comes the Pain"
set "BIN=%ROOT%third_party\ps2-native-gs\bin"
set "RUNNER=%ROOT%out\native-gs\ps2xRuntime\ps2EntryRunner.exe"
set "MODULE=%BIN%\libps2-native-gs.dll"
if not exist "%RUNNER%" echo Build first with build_hctp_vulkan.bat & exit /b 1
if not exist "%GAME%\SLUS_207.87" echo Missing vanilla SLUS_207.87 & exit /b 1
if not exist "%MODULE%" echo Missing libps2-native-gs.dll & exit /b 1
cd /d "%GAME%"
set "PATH=%BIN%;%PATH%"
set "HCTP_NATIVE_GS_MODULE=%MODULE%"
set "HCTP_RENDERER=native-gs"
set "VENDETTA_GS=gpu"
set "VENDETTA_FILTER=bilinear"
"%RUNNER%" SLUS_207.87
