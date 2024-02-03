
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

:check_python_version
for /f "tokens=2" %%I in ('%PYTHON_COMMAND% --version 2^>^&1') do (
    set "python_version=%%I"
)

echo Python version: %python_version%

%PYTHON_COMMAND% -m venv "%~dp0env" 
call "%~dp0env\Scripts\activate.bat" && pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu 
call "%~dp0env\Scripts\activate.bat" && pip install -r "%~dp0requirements.txt"
call pip install torchvision==0.15.2 
echo FastSD CPU env installation completed.
pause