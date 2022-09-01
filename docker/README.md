# Docker for Attention Branch Network


## Change Log

* 15 Apr 2020: Add Docker environment (torch 0.4.0)
* 01 Sep 2022: Update PyTorch version (torch 1.12.0)


## Download image from DockerHub

We prepare two docker images for a certain PyTorch version.
1. A docker image that runs the daemon as **a root user**.
2. A docker image that runs the daemon as **a genral user** with the same user ID in host OS.

You can download docker images by following commands:

### torch 1.12.0

The base image is `nvcr.io/nvidia/pytorch:22.05-py3` downloaded from NGC container.

If you want to run docker daemon as *root user.

**Docker image as a root user**
```bash
docker pull nvcr.io/nvidia/pytorch:22.05-py3
```

**Docker image as a general user**
```bash
docker pull cumprg/abn:pt1120_epoint
```

### torch 0.4.0

**Docker image as a root user**
```bash
docker pull cumprg/abn:pt040
```

**Docker image as a general user**
The detailed usage is described in the bottom (see Run).
```bash
docker pull cumprg/abn:pt040_epoint
```

## Build image

If you want to build on your environment by yourself, please run following command.

```bash
./build.sh
```


## Run docker with entrypoint

By using `cumprg/abn:pt040_epoint` or `cumprg/abn:pt1120_epoint`, you can run docker with a general user with the same user ID in host OS.

First, you need change `imagename` in `runs.sh` as follows:

```bash
# docker image name (please choose one of the followings)
# imagename="cumprg/abn:pt040_epoint"
imagename="cumprg/abn:pt1120_epoint"
```

Then, you can run a docker daemon by following command. (User ID is automatically set.)

```bash
./run.sh [container name] [volume mount 1] [volume mount 2] ...
```

If you want run manually, please add user ID option. For example, 

```bash
nvidia-docker run -ti --rm -u [uid]:[gid] [docker image name]
```
