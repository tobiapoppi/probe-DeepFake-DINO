#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=students-dev
#SBATCH --gres=gpu:1
#SBATCH --output=result_eval.out
#SBATCH --error=result_eval.out

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate dino38

. ./init_bash.sh

python -u eval_linear_deepfake.py --arch vit_small --patch_size 8 --evaluate --transforms_pipeline --output_dir /homes/tpoppi/probe-DeepFake-DINO/first_full_training/
