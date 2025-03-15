#!/bin/bash
#SBATCH --job-name=SWAG_1
#SBATCH --output=slurm/misc/%j.out
#SBATCH --error=slurm/misc/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=undergrad_research
#SBATCH --partition=dgx
#SBATCH --cpus-per-task=40
#SBATCH --time=7-00:00:00
#SBATCH --mem=500GB

export PYTHONPATH=/data/ai_club/nes_2025/swag:$PYTHONPATH

mapfile -t ids < utils/misc/idm_video_ids.txt

for id in "${ids[@]}"; do
    rm -rf idm/data/numpy/$id
done

for id in "${ids[@]}"; do
    ~/python3.12/bin/python3.12 /data/ai_club/nes_2025/swag/utils/convert_to_numpy.py --video_id $id --input_dir idm/data/raw --output_dir idm/data/numpy --labels True
done