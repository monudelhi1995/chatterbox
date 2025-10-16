@echo off
REM Activate the project's virtual environment (.venv) if present, then run api_fastapi.py (prefer using the venv python.exe when available)

setlocal
pushd "%~dp0"

if exist ".venv\Scripts\activate.bat" (
  echo Activating virtual environment at %cd%\.venv
  call ".venv\Scripts\activate.bat"
) else (
  echo [WARN] Virtual environment ".venv" not found in %cd%. Running system Python.
)

REM Prefer the venv python executable if present, else fall back to `python` on PATH
if exist ".venv\Scripts\python.exe" (
  echo Running with .venv\Scripts\python.exe
  ".venv\Scripts\python.exe" "api_fastapi.py"
) else (
  echo Running with system Python (python on PATH)
  python "api_fastapi.py"
)

REM Keep the console open so you can see output/errors. Remove the next line if you want the window to close automatically.
pause

popd
endlocal
