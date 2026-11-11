@echo off
rem CAM test + SPU key-on log: run_camtest_spu.bat <entrance event no>
set "HCTP_CAM_FORCE_EV=%1"
set "VENDETTA_SPULOG=1"
call "F:\hctp-newbase-vulcan\run_hctp_d3d11.bat"
