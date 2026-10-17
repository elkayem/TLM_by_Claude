@echo off
rem Log GPU health every 5 minutes while a long run trains (see gpu_health.ps1).
powershell -NoProfile -ExecutionPolicy Bypass -NoExit -File "%~dp0gpu_health.ps1"
