@echo off
REM ============================================
REM QUANTIX AUTO-TRAINER: Windows Scheduler Setup
REM ============================================
REM This script registers a daily Windows Task 
REM that trains your AI models at 6:00 AM every day.
REM Run this ONCE as Administrator.
REM ============================================

set PYTHON_PATH=python
set SCRIPT_PATH=%~dp0auto_trainer.py

echo.
echo ==========================================
echo  QUANTIX: Setting up Daily AI Training
echo ==========================================
echo.
echo Script: %SCRIPT_PATH%
echo Schedule: Daily at 6:00 AM
echo.

schtasks /create /tn "Quantix_AI_DailyTrainer" /tr "%PYTHON_PATH% \"%SCRIPT_PATH%\"" /sc daily /st 06:00 /f

if %errorlevel% equ 0 (
    echo.
    echo [SUCCESS] Daily training task registered!
    echo Your AI models will auto-retrain every morning at 6:00 AM.
    echo.
    echo To remove: schtasks /delete /tn "Quantix_AI_DailyTrainer" /f
    echo To run now: python auto_trainer.py
) else (
    echo.
    echo [ERROR] Failed to register task.
    echo Try running this script as Administrator.
)

pause
