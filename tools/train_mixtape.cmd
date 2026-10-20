@echo off
rem Start (or resume) the Stage 4 mixtape training run in its own window.
rem Double-click this file. The work is done by train_mixtape.ps1 next to it.
powershell -NoProfile -ExecutionPolicy Bypass -NoExit -File "%~dp0train_mixtape.ps1" %*
