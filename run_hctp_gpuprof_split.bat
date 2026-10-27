@echo off
rem GPU cost per draw class (PS2X_GPU_PROF). Output -> logs\gpuprof_split.out / gpuprof_split.err
set "PS2X_GPU_PROF=1"
set "HCTP_AFAIL_SPLIT=1"
cd /d "%~dp0"
call "%~dp0run_hctp_d3d11.bat" > "%~dp0logs\gpuprof_split.out" 2> "%~dp0logs\gpuprof_split.err"
