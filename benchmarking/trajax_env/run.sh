#!/bin/bash
docker run --rm -it \
 -v "$PWD"/../..:/workspace \
 -v "$PWD"/entrypoint.sh:/entrypoint.sh \
 -e DISPLAY=$DISPLAY \
 -v /tmp/.X11-unix:/tmp/.X11-unix \
 --gpus all \
 trajax bash -c "/entrypoint.sh bash"
