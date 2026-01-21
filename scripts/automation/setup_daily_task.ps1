# Setup Windows Task Scheduler for Daily Paper Trading
# Run this script as Administrator
# Usage: .\setup_daily_task.ps1

Write-Host "========================================"  -ForegroundColor Green
Write-Host "Setting up Daily Paper Trading Task" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

# Configuration
$TaskName = "DailyPaperTrading"
$ScriptPath = Join-Path $PSScriptRoot "daily_paper_trading_qdrant.py"
$VenvPython = Join-Path $PSScriptRoot "..\..\venv\Scripts\python.exe"
$WorkingDir = Join-Path $PSScriptRoot "..\..\"

# Resolve paths
$ScriptPath = Resolve-Path $ScriptPath
$VenvPython = Resolve-Path $VenvPython
$WorkingDir = Resolve-Path $WorkingDir

Write-Host ""
Write-Host "Configuration:" -ForegroundColor Cyan
Write-Host "  Task Name: $TaskName"
Write-Host "  Script: $ScriptPath"
Write-Host "  Python: $VenvPython"
Write-Host "  Working Dir: $WorkingDir"
Write-Host "  Schedule: Daily at 4:15 PM EST"
Write-Host ""

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "WARNING: Not running as Administrator" -ForegroundColor Yellow
    Write-Host "The task may not be created properly." -ForegroundColor Yellow
    Write-Host "Please run PowerShell as Administrator and try again." -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to continue anyway (not recommended)"
}

# Delete existing task if it exists
$existingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existingTask) {
    Write-Host "Task already exists. Deleting..." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Old task deleted." -ForegroundColor Green
}

# Create action (what to run)
$Action = New-ScheduledTaskAction `
    -Execute $VenvPython `
    -Argument "`"$ScriptPath`"" `
    -WorkingDirectory $WorkingDir

# Create trigger (when to run) - Daily at 4:15 PM
$Trigger = New-ScheduledTaskTrigger `
    -Daily `
    -At "4:15 PM"

# Create settings
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable

# Create principal (run with highest privileges)
$Principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Highest

# Register the task
try {
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $Action `
        -Trigger $Trigger `
        -Settings $Settings `
        -Principal $Principal `
        -Description "Automated daily paper trading with Qdrant storage. Runs S2_FilterNegative strategy simulation and saves results to vector database." `
        -Force

    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "Task created successfully!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Task Details:" -ForegroundColor Cyan
    Write-Host "  Name: $TaskName"
    Write-Host "  Schedule: Daily at 4:15 PM"
    Write-Host "  Next Run: $($(Get-ScheduledTask -TaskName $TaskName).Triggers[0].StartBoundary)"
    Write-Host ""
    Write-Host "Useful Commands:" -ForegroundColor Cyan
    Write-Host "  View task details:"
    Write-Host "    Get-ScheduledTask -TaskName '$TaskName' | Format-List" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  Run task manually now:"
    Write-Host "    Start-ScheduledTask -TaskName '$TaskName'" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  View task history:"
    Write-Host "    Get-ScheduledTaskInfo -TaskName '$TaskName'" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  Disable task:"
    Write-Host "    Disable-ScheduledTask -TaskName '$TaskName'" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  Enable task:"
    Write-Host "    Enable-ScheduledTask -TaskName '$TaskName'" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  Delete task:"
    Write-Host "    Unregister-ScheduledTask -TaskName '$TaskName' -Confirm:`$false" -ForegroundColor Gray
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green

    # Test task (optional)
    $testRun = Read-Host "Do you want to test run the task now? (y/n)"
    if ($testRun -eq 'y' -or $testRun -eq 'Y') {
        Write-Host ""
        Write-Host "Running task..." -ForegroundColor Cyan
        Start-ScheduledTask -TaskName $TaskName
        Write-Host "Task started. Check logs at: logs/automation/" -ForegroundColor Green
    }

} catch {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "ERROR: Failed to create task" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "Error details:" -ForegroundColor Yellow
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ""
    Write-Host "Make sure you're running PowerShell as Administrator!" -ForegroundColor Yellow
}

Write-Host ""
Read-Host "Press Enter to exit"
