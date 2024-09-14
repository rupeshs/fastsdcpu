#!/usr/bin/env bash
cd $(dirname $0)
echo Starting FastSD CPU please wait...
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
# shellcheck disable=SC1091
source "$BASEDIR/env/bin/activate"

## https://qiita.com/ko1nksm/items/7d37852b9fc581b1266e
abort() { echo "$*" >&2; exit 1; }
unknown() { abort "unrecognized option '$1'"; }
required() { [ $# -gt 1 ] || abort "option '$1' requires an argument"; }

OPTION_SHARE=''
OPTION_ROOT_PATH=''

while [ $# -gt 0 ]; do
  case $1 in
    -s | --share ) OPTION_SHARE=' --share' ;;
    -r | --root_path ) required "$@" && shift; OPTION_ROOT_PATH=' --root_path='$1 ;;
    -h | --help ) abort "usage : $(basename $0) [--share] [--root_path=\"/(path)\"]" ;;
    -?*) unknown "$@" ;;
    *) break
  esac
  shift
done

$PYTHON_COMMAND src/app.py -w ${OPTION_SHARE} ${OPTION_ROOT_PATH}
