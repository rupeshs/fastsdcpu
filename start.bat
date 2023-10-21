@echo off
echo Starting fastsdcpu
cd "%~dp0env\condabin\"
call activate.bat
micromamba activate "%~dp0env\envs\fastsd-env" && python "%~dp0main.py"

