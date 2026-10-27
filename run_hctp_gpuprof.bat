@echo off
rem GPU cost per draw class (PS2X_GPU_PROF). Output -> logs\gpuprof.out / gpuprof.err
set "PS2X_GPU_PROF=1"
cd /d "%~dp0"
call "%~dp0run_hctp_d3d11.bat" > "%~dp0logs\gpuprof.out" 2> "%~dp0logs\gpuprof.err"
