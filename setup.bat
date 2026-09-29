@echo off
REM Poker Vision Setup Script for Windows
REM Run this in PowerShell or Command Prompt

echo ========================================
echo Poker Vision - Windows Setup
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install from python.org
    pause
    exit /b 1
)

REM Create venv
echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create venv
    pause
    exit /b 1
)

REM Activate venv
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install mss numpy opencv-python Pillow ultralytics eval7 pywin32

echo.
echo ========================================
echo Setup complete!
echo.
echo To run:
echo   venv\Scripts\python -m src.main --console
echo.
echo Or for overlay mode (recommended):
echo   venv\Scripts\python -m src.main
echo ========================================

pause
