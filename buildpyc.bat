python -m compileall -b src\
for /r src %f in (*.py) do del "%f"