Modify `tools/scripts/predict.sh` to customize the inference process. Mount your data directory to `/input/` in the container to predict on your own data.

```bash
$ cd ivos
# Build the docker image with specific dockerfile path
$ docker run --rm --gpus 0 -v /path/to/input/data/:/input/ ivos
```

For FLARE22 submission, please follow below instructions:

- Go to root directory, first download trained weights using `tools/scripts/download_weights.sh`
- Then prepare your testing data and an output folder 
- Next, build docker by `$ DOCKER_BUILDKIT=1 docker build -t hcmus:latest -f docker/Dockerfile .`
- After the docker is done, execute it by running

```bash
docker run --gpus "device=1" --name hcmus --rm -v $PWD/<path to image folder>/:/home/root/workspace/inputs/ -v $PWD/<path to output folder>/:/home/root/workspace/outputs/ hcmus:latest /bin/bash -c "sh tools/scripts/predict.sh"
```

- Results wil be saved at the specified folder.