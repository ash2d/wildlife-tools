#!/bin/bash
#SBATCH --job-name=train_mgd
#SBATCH --output=logs/train_mgd_%j/o.out
#SBATCH --error=logs/train_mgd_%j/e.err
#SBATCH --time=18:00:00
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclude=gpuhost015

# Navigate to your project directory
cd /home/users/dash/guppies/embeddings/wildlife-tools

mkdir -p logs
# Activate the virtual environment
source /home/users/dash/guppies/embeddings/wildlife-tools/.venv/bin/activate

cd exploring

# Run your Python script with desired arguments
srun  uv run train_megadescriptor_arcface.py --csv '/home/users/dash/guppies/embeddings/wildlife-tools/exploring/csvs/top_10000_individuals.csv' --root '/gws/nopw/j04/iecdt/dash/cropped_images/id/w_random_masks' --epochs 10 
