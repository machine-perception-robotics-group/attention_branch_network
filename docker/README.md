# Docker for Attention Branch Network


## Download image from DockerHub

You can download docker images by following commands:

This image runs docker daemon as root user.
```bash
docker pull cumprg/abn:pt040
```

This image runs docker daemon as a general user with the same user ID in host OS.
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

By using `cumprg/abn:pt040_epoint`, you can run docker with a general user with the same user ID in host OS.

You can run a docker daemon by following command. (User ID is automatically set.)

```bash
./run.sh [container name] [volume mount 1] [volume mount 2] ...
```

If you want run manually, please add user ID option. For example, 

```bash
nvidia-docker run -ti --rm -u [uid]:[gid] cumprg/abn:pt040_epoint
```
