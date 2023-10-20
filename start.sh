#!/bin/bash
echo Starting fastsdcpu please wait...
BASEDIR=$(pwd)
eval "$(micromamba shell hook --shell=bash)"
micromamba activate $BASEDIR/env/envs/fastsd-env && python $BASEDIR/main.py