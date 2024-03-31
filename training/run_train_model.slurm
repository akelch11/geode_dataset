#!/bin/bash
#SBATCH --job-name=run_train_model   # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=01:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-user=akelch@princeton.edu
#SBATCH --gpus-per-node=1
#SBATCH --mem=10G

module purge
module load anaconda3/2023.9
conda activate torch-thesis
 ! python geode_eval/training/revised_train_linear_model.py --region Africa