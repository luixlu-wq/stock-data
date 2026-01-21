@echo off
REM Setup Windows Task Scheduler for Daily Paper Trading
REM This creates a scheduled task that runs every day at 4:15 PM EST

echo ========================================
echo Setting up Daily Paper Trading Task
echo ========================================

REM Configuration
set TASK_NAME=DailyPaperTrading
set SCRIPT_PATH=%~dp0daily_paper_trading_qdrant.py
set PYTHON_PATH=C:\Users\luixj\AppData\Local\Programs\Python\Python312\python.exe
set VENV_PYTHON=%~dp0..\..\venv\Scripts\python.exe
set WORKING_DIR=%~dp0..\..

REM Check if task already exists
schtasks /query /tn %TASK_NAME% >nul 2>&1
if %errorlevel% equ 0 (
    echo Task already exists. Deleting old task...
    schtasks /delete /tn %TASK_NAME% /f
)

echo Creating new task...
echo Task Name: %TASK_NAME%
echo Script: %SCRIPT_PATH%
echo Schedule: Daily at 4:15 PM EST
echo.

REM Create the task
REM Run daily at 4:15 PM (16:15)
schtasks /create /tn %TASK_NAME% ^
    /tr "\"%VENV_PYTHON%\" \"%SCRIPT_PATH%\"" ^
    /sc daily ^
    /st 16:15 ^
    /f ^
    /rl highest

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo Task created successfully!
    echo ========================================
    echo.
    echo Task Name: %TASK_NAME%
    echo Schedule: Daily at 4:15 PM
    echo.
    echo To view the task:
    echo   schtasks /query /tn %TASK_NAME% /v /fo list
    echo.
    echo To run manually now:
    echo   schtasks /run /tn %TASK_NAME%
    echo.
    echo To delete the task:
    echo   schtasks /delete /tn %TASK_NAME% /f
    echo.
    echo ========================================
) else (
    echo.
    echo ========================================
    echo ERROR: Failed to create task
    echo ========================================
    echo Please run this script as Administrator
    echo.
)

pause
