@echo off
rem Entrance-load black-screen hang: IOP hot blocks (VENDETTA_IOPPROF) + 1 s CPU sampler.
rem If it hangs, leave it running and tell Claude (it drops iop_dump_now into the game folder for an IOP thread dump).
rem Output -> logs\iophang.out / logs\iophang.err
set "VENDETTA_IOPPROF=1"
set "HCTP_SAMPLER=1"
set "HCTP_SAMPLER_MS=1000"
cd /d "%~dp0"
call "%~dp0run_hctp_d3d11.bat" > "%~dp0logs\iophang.out" 2> "%~dp0logs\iophang.err"
