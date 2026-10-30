@echo off
rem Drop-in stand-in for hip/rtc_compile.exe (stale on this box: it hard-rejects gfx115x, and it
rem cannot be rebuilt here). Argument order matches the exe exactly so hip/build-modules.ps1
rem can point -Compiler at this file unchanged:  <out.hsaco> <src.hip> comgr <target> [opts]
python "%~dp0probe_comgr.py" --compile %*
exit /b %ERRORLEVEL%
