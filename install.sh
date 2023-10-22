#!/usr/bin/env bash
set -e
PYTHON_COMMAND="python3"

if ! command -v python3 &>/dev/null; then
    if ! command -v python &>/dev/null; then
        echo "Error: Python not found, please install python 3.8 or higher"
        exit 1
    fi
fi

if command -v python &>/dev/null; then
   PYTHON_COMMAND="python"
fi

echo "Found $PYTHON_COMMAND command"

check_python_version() {
    python_version=$($PYTHON_COMMAND --version 2>&1 | awk '{print $2}') 

    if awk -v ver="$python_version" 'BEGIN { if (ver >= 3.8) exit 0; else exit 1; }'; then
        return 0 
    else
        return 1  
    fi
}

python_version=$($PYTHON_COMMAND --version 2>&1 | awk '{print $2}')  
echo "Python version : $python_version"

BASEDIR=$(pwd)

$PYTHON_COMMAND -m venv "$BASEDIR/env"
# shellcheck disable=SC1091
source "$BASEDIR/env/bin/activate"
pip install -r "$BASEDIR/requirements.txt"

if grep -q -F "python3 main.py" "start.sh"; then
      if [ "$PYTHON_COMMAND" != "python3" ]; then
          echo "Updating to python in start.sh"
          sed -i "s/python3/python/g" "start.sh"
      fi 
else
    
    if [ "$PYTHON_COMMAND" != "python" ]; then
          echo "Updating to python3 in start.sh"
          sed -i "s/python/python3/g" "start.sh"
      fi 
fi
chmod +x "start.sh"

read -n1 -r -p "FastSD CPU installation completed,press any key to continue..." key