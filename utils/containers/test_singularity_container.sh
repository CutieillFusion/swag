#!/bin/bash

IMAGE_PATH="utils/containers/container.sif"
TEST_SCRIPT="utils/misc/test_singularity_container.py"

singularity exec -B /data/ai_club/nes_2025/swag:/data/ai_club/nes_2025/swag "$IMAGE_PATH" python "$TEST_SCRIPT" requirements.txt