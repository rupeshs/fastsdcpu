#!/bin/bash
BASEDIR=$(pwd)
chmod +x $BASEDIR/tools/mac/micromamba
chmod +x start.sh
$BASEDIR/tools/mac/micromamba -r $BASEDIR/env create -y -f $BASEDIR/environment.yml
$BASEDIR/tools/mac/micromamba shell init -s bash -p $BASEDIR/umamba
read -n1 -r -p "FastSD CPU installation completed,press any key to continue..." key