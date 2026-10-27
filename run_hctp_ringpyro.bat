@echo off
rem HBK in-ring pyro dips: GPU per-draw profile + CPU sampler. Output -> logs\ringpyro.out / logs\ringpyro.err
set "PS2X_GPU_PROF=1"
set "HCTP_SAMPLER=1"
set "HCTP_SAMPLER_MS=1000"
cd /d "%~dp0"
call "%~dp0run_hctp_d3d11.bat" > "%~dp0logs\ringpyro.out" 2> "%~dp0logs\ringpyro.err"
