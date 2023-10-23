
@echo off
setlocal
echo Starting fastsdcpu...

set "PYTHON_COMMAND=python"

where python >nul 2>&1
    if %errorlevel% neq 0 (
        echo Error: Python not found, please install Python 3.8 or higher.
        exit /b 1
    ) 

echo Found %PYTHON_COMMAND% command

:check_python_version
for /f "tokens=2" %%I in ('%PYTHON_COMMAND% --version 2^>^&1') do (
    set "python_version=%%I"
)

echo Python version: %python_version%


set PATH=%PATH%;%~dp0env\envs\fastsd-env\Lib\site-packages\openvino\libs
%PYTHON_COMMAND% -m venv "%~dp0env"  && python "%~dp0main.py"