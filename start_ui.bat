@echo off
title Pathology Active Learning Engine
color 0A
echo.
echo  =====================================================
echo       PATHOLOGY ACTIVE LEARNING ENGINE
echo  =====================================================
echo.

:: Kill any previous instance hogging port 8000
echo  [1/3] Clearing port 8000 if occupied...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
)

:: Small pause after kill
timeout /t 1 /nobreak >nul

echo  [2/3] Starting FastAPI server...
echo.

:: Run app.py directly in THIS window (it auto-opens the browser)
cd /d "%~dp0"
python app.py
