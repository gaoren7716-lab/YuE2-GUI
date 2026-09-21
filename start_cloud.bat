@echo off
chcp 65001 >nul
cd /d "%~dp0"
title YuE2 Music Generator (Cloud)

if not exist ".venv\Scripts\python.exe" (
  echo.
  echo   Not installed yet. Run install.bat first.
  echo.
  pause
  exit /b 1
)

echo.
echo   Starting in CLOUD mode - listening on all network interfaces.
echo.
echo   Access it from outside using this machine's public IP
echo   or the port-forwarding address your GPU provider gives you.
echo   Default port: 7861
echo.
echo   WARNING: do not expose this port to the open internet without
echo   a firewall or an SSH tunnel.
echo.

set YUE2_BIND=0.0.0.0
set YUE2_PORT=7861
.venv\Scripts\python.exe "%~dp0app.py"

echo.
echo   Program stopped.
pause
