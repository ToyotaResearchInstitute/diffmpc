#!/bin/bash
docker run --rm -it \
 -v "$PWD"/../..:/workspace \
 -v "$PWD"/entrypoint_benchmark.sh:/entrypoint_benchmark.sh \
 -e DISPLAY=$DISPLAY \
 -v /tmp/.X11-unix:/tmp/.X11-unix \
 --gpus all \
 trajax bash -c "/entrypoint_benchmark.sh bash"
