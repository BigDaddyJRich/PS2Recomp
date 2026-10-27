@echo off
rem A/B for the native particle program p16: VU1 interpreter draws (real PS2 program), native p16 is compared against it.
rem Output -> logs\p16verify.out / p16verify.err ([nvu1g] p16 match %)
set "HCTP_NATIVE_VERIFY=1"
set "PS2X_GPU_PROF=1"
cd /d "%~dp0"
call "%~dp0run_hctp_d3d11.bat" > "%~dp0logs\p16verify.out" 2> "%~dp0logs\p16verify.err"
