@echo off
rem Nov 10: native p16 particle program ON (x cull fix, FCAND 0x042108) - test HBK/Goldberg in-ring pyro for streaks.
rem Log tag [nvu1g] p16 (runs / native / fails).
set "HCTP_NATIVE_P16=1"
call "%~dp0run_hctp_d3d11.bat" %*
