#!/bin/bash
#SBATCH --job-name=train_dino
#SBATCH --output=logs/train_dino_%j/o.out
#SBATCH --error=logs/train_dino_%j/e.err
#SBATCH --time=18:00:00
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
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
srun uv run train_dinov2_infonce_wandb.py --epochs 50 --csv '/home/users/dash/guppies/embeddings/wildlife-tools/exploring/csvs/top_1000_individuals_database.csv' --root '/gws/nopw/j04/iecdt/dash/cropped_images/id/w_random_masks'
