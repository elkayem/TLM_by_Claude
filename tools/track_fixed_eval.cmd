@echo off
rem Score each new mixtape checkpoint on the same validation passages, so
rem the trend is not hidden by eval noise (see track_fixed_eval.py).
rem Double-click to start; close the window to stop. Scores go to
rem %USERPROFILE%\tlm-checkpoints\mixtape\fixed-eval.csv
title TLM fixed-passage scores
set "PY=%TLM_PYTHON%"
if "%PY%"=="" set "PY=%USERPROFILE%\tlm-venv\Scripts\python.exe"
cd /d "%~dp0.."
"%PY%" -u tools\track_fixed_eval.py %*
pause
