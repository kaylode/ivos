```bash
$ cd ivos
$ DOCKER_BUILDKIT=1 docker build -t ivos:latest .
$ docker run -it --rm --gpus 0 -v /path/to/this/repo/:/home/hcmus/workspace/ ivos bash
```
