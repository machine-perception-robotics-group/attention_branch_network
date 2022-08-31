#!/bin/bash

# build.sh
# Copyright (c) Tsubasa HIRAKAWA, 2022


# build docker image (torch 1.13.0, with entrypoint) ##########
chmod a+x entrypoint.sh
nvidia-docker build --tag=cumprg/abn:pt1130_epoint --force-rm=true --file=./Dockerfile_torch1130_entrypoint .


# build docker image (torch 0.4.0, with entrypoint) ##########
chmod a+x entrypoint.sh
nvidia-docker build --tag=cumprg/abn:pt040_epoint --force-rm=true --file=./Dockerfile_torch040_entrypoint .


# build docker image (torch 0.4.0, w/o entrypoint) ###########
nvidia-docker build --tag=cumprg/abn:pt040 --force-rm=true --file=./Dockerfile_torch040 .
