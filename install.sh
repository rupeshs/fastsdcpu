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

if ! command -v uv &>/dev/null; then
    echo "Error: uv command not found,please install https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_1 and try again."
    exit 1
fi

BASEDIR=$(pwd)

uv venv --python 3.11.6 "$BASEDIR/env"
# shellcheck disable=SC1091
source "$BASEDIR/env/bin/activate"
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
if [[ "$1" == "--disable-gui" ]]; then
    #! For termux , we don't need Qt based GUI
    packages="$(grep -v "^ *#\|^PyQt5" requirements.txt | grep .)" 
    # shellcheck disable=SC2086
    uv pip install $packages
else
    uv pip install -r "$BASEDIR/requirements.txt"
fi

chmod +x "start.sh"
chmod +x "start-webui.sh"
read -n1 -r -p "FastSD CPU installation completed,press any key to continue..." key