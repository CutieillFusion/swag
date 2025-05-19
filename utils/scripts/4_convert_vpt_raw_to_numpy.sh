#!/bin/bash
#SBATCH --job-name=SWAG_4
#SBATCH --output=slurm/misc/%j.out
#SBATCH --error=slurm/misc/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=undergrad_research
#SBATCH --partition=dgx
#SBATCH --cpus-per-task=40
#SBATCH --time=5-00:00:00
#SBATCH --mem=400GB

export PYTHONPATH=/data/ai_club/nes_2025/swag:$PYTHONPATH

mapfile -t ids < utils/misc/vpt_video_ids.txt

for id in "${ids[@]}"; do
    rm -rf vpt/data/numpy/$id
done

for id in "${ids[@]}"; do
    /data/ai_club/nes_2025/swag/.venv/bin/python /data/ai_club/nes_2025/swag/utils/convert_to_numpy.py --video_id $id --input_dir vpt/data/raw --output_dir vpt/data/numpy --labels False
done

# sbatch utils/scripts/4_convert_vpt_raw_to_numpy.sh