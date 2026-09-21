@echo off
chcp 65001 >nul
cd /d "%~dp0"
title YuE2 Installer

echo.
echo   ============================================================
echo     YuE2 Music Generator  -  Setup
echo   ============================================================
echo.

set "PYEXE="
for %%V in (3.12 3.13 3.11 3.10) do (
  if not defined PYEXE (
    py -%%V -c "1" >nul 2>nul
    if not errorlevel 1 set "PYEXE=py -%%V"
  )
)
if not defined PYEXE (
  where python >nul 2>nul
  if not errorlevel 1 set "PYEXE=python"
)
if not defined PYEXE (
  if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" (
    set "PYEXE=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
  )
)

if not defined PYEXE (
  echo   [ERROR] Python 3.10 or newer was not found.
  echo.
  echo   Please install it from:
  echo       https://www.python.org/downloads/
  echo.
  echo   IMPORTANT: check "Add python.exe to PATH" during setup.
  echo.
  pause
  exit /b 1
)

echo   Using Python: %PYEXE%
echo.

%PYEXE% "%~dp0install.py" %*
set "RC=%errorlevel%"

echo.
if not "%RC%"=="0" (
  echo   Setup did NOT finish. Read the messages above.
  echo   Running this file again will resume from where it stopped.
) else (
  echo   Setup finished. You can now double-click start.bat
)
echo.
pause
exit /b %RC%
