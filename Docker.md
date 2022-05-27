Modify `run.sh` to customize the inference process. Mount your data directory to `/input/` in the container to predict on your own data.

```bash
$ cd ivos
# Build the docker image with specific dockerfile path
$ DOCKER_BUILDKIT=1 docker build -t ivos:latest -f docker/selab/Dockerfile .
$ docker run --rm --gpus 0 -v /path/to/input/data/:/input/ ivos
```
