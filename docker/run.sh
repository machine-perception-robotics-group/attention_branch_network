#!/bin/bash

# run.sh
# Copyright (c) Tsubasa HIRAKAWA, 2020

# usage: ./run.sh [container name] [volume mount 1] [volume mount 2] ...


# check command line args
if [ $# -lt 1 ]; then
    echo "ERROR: less arguments"
    echo "Usage: ./run.sh [container name] [volume mount 1] [volume mount 2] ..."
    exit
fi


echo "run docker image (cumprg/abn:pt040_epoint) ..."
echo "    container name: ${1}"


if [ $# -gt 1 ]; then
    for var in ${@:2}; do
        mounts+="-v ${var} "
        echo "    mount point: ${var}"
    done
else
    mounts=""
    echo "    mount point: N/A"
fi


nvidia-docker run -ti --rm -u $(id -u):$(id -g)\
    --ipc=host --name=${1} ${mounts} \
    cumprg/abn:pt040_epoint

