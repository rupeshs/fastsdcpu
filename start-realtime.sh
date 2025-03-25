#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

echo "Starting fastsdcpu..."

PYTHON_COMMAND="python3"

# Check if Python is installed
if command -v $PYTHON_COMMAND >/dev/null 2>&1; then
    echo "Python command check: OK"
else
    echo "Error: Python command not found, please install Python (Recommended: Python 3.10 or Python 3.11) and try again"
    exit 1
fi

# Get Python version
python_version=$($PYTHON_COMMAND --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

echo $PATH
# Activate virtual environment and run the application
source "$(dirname "$0")/env/bin/activate" && $PYTHON_COMMAND "$(dirname "$0")/src/app.py" --r
