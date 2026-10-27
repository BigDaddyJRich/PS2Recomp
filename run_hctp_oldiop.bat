@echo off
rem A/B test: same as run_hctp_ringpyro.bat but with the new IOP interrupt-mask fix OFF (old behaviour).
rem Output -> logs\oldiop.out / logs\oldiop.err
set "HCTP_IOP_INTR_MASK=0"
set "HCTP_SAMPLER=1"
set "HCTP_SAMPLER_MS=1000"
cd /d "%~dp0"
call "%~dp0run_hctp_d3d11.bat" > "%~dp0logs\oldiop.out" 2> "%~dp0logs\oldiop.err"
