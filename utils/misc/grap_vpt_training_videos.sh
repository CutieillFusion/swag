#!/bin/bash

mapfile -t ids < utils/misc/vpt_video_ids.txt

for id in "${ids[@]}"; do
    if [ ! -f "/data/ai_club/nes_2025/swag/vpt/data/raw/video_${id}.mp4" ]; then
        cp /data/ai_club/nes_2025/ur-dylan-josiah/models/nes_cropped_training_videos/video_${id}_cropped.mp4 /data/ai_club/nes_2025/swag/vpt/data/raw
        mv /data/ai_club/nes_2025/swag/vpt/data/raw/video_${id}_cropped.mp4 /data/ai_club/nes_2025/swag/vpt/data/raw/video_${id}.mp4
    fi
done
