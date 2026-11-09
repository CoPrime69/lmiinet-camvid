@echo off
rem Runs the whole project with the shared assignments venv. Arguments are passed to run.py.
cd /d "%~dp0"
set "PY=%~dp0..\..\.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo Shared venv python not found at "%PY%"
  pause
  exit /b 1
)
"%PY%" run.py %*
echo.
echo run.py finished with exit code %ERRORLEVEL%
pause
