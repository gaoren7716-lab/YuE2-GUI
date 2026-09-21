@echo off
chcp 65001 >nul
cd /d "%~dp0"
title YuE2 Music Generator

if not exist ".venv\Scripts\python.exe" (
  echo.
  echo   ============================================================
  echo     Not installed yet.
  echo   ============================================================
  echo.
  echo   Please double-click  install.bat  first.
  echo.
  pause
  exit /b 1
)

echo.
echo   Starting YuE2 Music Generator ...
echo   Your browser will open automatically.
echo   Keep this window open. Close it to quit.
echo.

.venv\Scripts\python.exe "%~dp0app.py"

echo.
echo   Program stopped.
pause
