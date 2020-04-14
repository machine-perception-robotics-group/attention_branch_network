#!/bin/bash

# build.sh
# Copyright (c) Tsubasa HIRAKAWA, 2020


# build docker image (with entrypoint) ##########
chmod a+x entrypoint.sh
nvidia-docker build --tag=cumprg/abn:pt040_epoint --force-rm=true --file=./Dockerfile_torch040_entrypoint .


# build docker image (w/o entrypoint) ###########
nvidia-docker build --tag=cumprg/abn:pt040 --force-rm=true --file=./Dockerfile_torch040 .
