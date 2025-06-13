#!/bin/bash
#SBATCH --job-name=train_dino_ssl_100000
#SBATCH --output=logs/train_dino_ssl_100000%j/o.out
#SBATCH --error=logs/train_dino_ssl_100000%j/e.err
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
srun uv run train_dinov2_selfsupervised.py --csv '/home/users/dash/guppies/embeddings/wildlife-tools/exploring/csvs/no_id_100000.csv' --epochs 15 --root '/gws/nopw/j04/iecdt/dash/cropped_images/no_id/images_w_box' --output-dir '/gws/nopw/j04/iecdt/dash/embeddings/models' --log-file '/home/users/dash/guppies/embeddings/wildlife-tools/exploring/logs/dinossl/3.txt'