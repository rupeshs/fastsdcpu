@echo off
echo Starting fastsdcpu
cd "%~dp0env\condabin\"
set PATH=%PATH%;%~dp0env\envs\fastsd-env\Lib\site-packages\openvino\libs
call activate.bat
micromamba activate "%~dp0env\envs\fastsd-env" && python "%~dp0main.py"