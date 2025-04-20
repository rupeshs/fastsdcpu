
@echo off
setlocal
echo Starting FastSD CPU env installation...

set "PYTHON_COMMAND=python"

call python --version > nul 2>&1
if %errorlevel% equ 0 (
    echo Python command check :OK
) else (
    echo "Error: Python command not found,please install Python(Recommended : Python 3.10 or Python 3.11) and try again."
    pause
    exit /b 1
    
)

call uv --version > nul 2>&1
if %errorlevel% equ 0 (
    echo uv command check :OK
) else (
    echo "Error: uv command not found,please install https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_2 and try again."
    pause
    exit /b 1
    
)
:check_python_version
for /f "tokens=2" %%I in ('%PYTHON_COMMAND% --version 2^>^&1') do (
    set "python_version=%%I"
)

echo Python version: %python_version%

uv venv --python 3.11.6 "%~dp0env" 
call "%~dp0env\Scripts\activate.bat" && uv pip install torch --index-url https://download.pytorch.org/whl/cpu 
call "%~dp0env\Scripts\activate.bat" && uv pip install -r "%~dp0requirements.txt"
echo FastSD CPU env installation completed.
pause