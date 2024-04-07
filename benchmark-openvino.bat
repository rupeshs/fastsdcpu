@echo off
setlocal

set "PYTHON_COMMAND=python"

call python --version > nul 2>&1
if %errorlevel% equ 0 (
    echo Python command check :OK
) else (
    echo "Error: Python command not found, please install Python (Recommended : Python 3.10 or Python 3.11) and try again"
    pause
    exit /b 1
    
)

:check_python_version
for /f "tokens=2" %%I in ('%PYTHON_COMMAND% --version 2^>^&1') do (
    set "python_version=%%I"
)

echo Python version: %python_version%

call "%~dp0env\Scripts\activate.bat"  && %PYTHON_COMMAND%  src/app.py -b --use_openvino  --openvino_lcm_model_id "rupeshs/sd-turbo-openvino"