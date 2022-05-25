Modify `run.sh` to customize the inference process. Mount your data directory to `/input/` in the container to predict on your own data.

```bash
$ cd ivos
$ DOCKER_BUILDKIT=1 docker build -t ivos:latest .
$ docker run --rm --gpus 0 -v /path/to/input/data/:/input/ ivos
```
