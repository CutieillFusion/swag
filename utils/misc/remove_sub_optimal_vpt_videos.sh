#!/bin/bash

mapfile -t keep_folders < utils/misc/optimal_vpt_video_ids.txt

cd /data/ai_club/nes_2025/swag/vpt/data/numpy

echo "Folders that will be removed:"
for item in */; do
    if [ -d "$item" ]; then
        folder_name=${item%/}
        keep=false
        for keep_folder in "${keep_folders[@]}"; do
            if [ "$folder_name" = "$keep_folder" ]; then
                keep=true
                break
            fi
        done
        if [ "$keep" = false ]; then
            echo "$folder_name"
        fi
    fi
done

read -p "Do you want to proceed with removing all other folders? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Operation cancelled."
    exit 1
fi

for item in */; do
    if [ -d "$item" ]; then
        folder_name=${item%/}
        keep=false
        for keep_folder in "${keep_folders[@]}"; do
            if [ "$folder_name" = "$keep_folder" ]; then
                keep=true
                break
            fi
        done
        if [ "$keep" = false ]; then
            echo "Removing folder: $folder_name"
            rm -rf "$folder_name"
        fi
    fi
done

echo "Cleanup complete!"