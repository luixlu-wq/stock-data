@echo off
REM Wrapper for the PowerShell scheduler setup so both entry points stay in sync.

set SCRIPT_DIR=%~dp0
powershell.exe -ExecutionPolicy Bypass -File "%SCRIPT_DIR%setup_daily_task.ps1"
pause
