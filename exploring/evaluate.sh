#!/bin/bash
#SBATCH --job-name=train_mgd
#SBATCH --output=logs/evaluate_%j/o.out
#SBATCH --error=logs/evaluate_%j/e.err
#SBATCH --time=10:00:00
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
srun  uv run evaluate_reid.py --model "hf-hub:BVRA/MegaDescriptor-T-224" --save-path '/gws/nopw/j04/iecdt/dash/embeddings/models/MGD/MGD_30000ids_arcface/model_epoch_10.pt/checkpoint.pth' --database /home/users/dash/guppies/embeddings/wildlife-tools/exploring/csvs/top_30000_individuals_database.csv --query /home/users/dash/guppies/embeddings/wildlife-tools/exploring/csvs/top_30000_individuals_query.csv --root '/gws/nopw/j04/iecdt/dash/cropped_images/id/w_random_masks' --output '30000ids_results.txt'