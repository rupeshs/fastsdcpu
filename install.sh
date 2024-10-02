#!/usr/bin/env bash
echo Starting FastSD CPU env installation...
set -e
PYTHON_COMMAND="python3"

if ! command -v python3 &>/dev/null; then
    if ! command -v python &>/dev/null; then
        echo "Error: Python not found, please install python 3.8 or higher and try again"
        exit 1
    fi
fi

if command -v python &>/dev/null; then
   PYTHON_COMMAND="python"
fi

echo "Found $PYTHON_COMMAND command"

python_version=$($PYTHON_COMMAND --version 2>&1 | awk '{print $2}')  
echo "Python version : $python_version"

BASEDIR=$(pwd)

$PYTHON_COMMAND -m venv "$BASEDIR/env"
# shellcheck disable=SC1091
source "$BASEDIR/env/bin/activate"
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
if [[ "$1" == "--disable-gui" ]]; then
    #! For termux , we don't need Qt based GUI
    packages="$(grep -v "^ *#\|^PyQt5" requirements.txt | grep .)" 
    # shellcheck disable=SC2086
    pip install $packages
else
    pip install -r "$BASEDIR/requirements.txt"
fi

chmod +x "start.sh"
chmod +x "start-webui.sh"
read -n1 -r -p "FastSD CPU installation completed,press any key to continue..." key