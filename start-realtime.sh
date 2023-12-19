#!/bin/bash

echo "Starting fastsdcpu..."

PYTHON_COMMAND="python"

# Check if Python is installed
if command -v $PYTHON_COMMAND &>/dev/null; then
    echo "Python command check: OK"
else
    echo "Error: Python command not found, please install Python (Recommended: Python 3.10 or Python 3.11) and try again"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$("$PYTHON_COMMAND" --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Add openvino libs to PATH
export PATH=$PATH:$(dirname "$(readlink -f "$0")")/env/lib/python3.10/openvino/libs

# Activate virtual environment
source "$(dirname "$(readlink -f "$0")")/env/bin/activate" && $PYTHON_COMMAND "$(dirname "$(readlink -f "$0")")/src/app.py" --r
