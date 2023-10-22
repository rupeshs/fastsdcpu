"%~dp0tools\windows\micromamba.exe" -r "%~dp0env" create -y -f "%~dp0environment.yml"
cd "%~dp0env\condabin\"
call activate.bat
call micromamba activate "%~dp0env\envs\fastsd-env" && pip install -r "%~dp0requirements.txt"
echo Env installation completed.
pause