#!/bin/bash

IMAGE_PATH="utils/containers/container.sif"
DOCKER_IMAGE="norquistdylan/nes-25:latest"

singularity build "$IMAGE_PATH" "docker://$DOCKER_IMAGE"
