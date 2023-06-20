#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=students-prod
#SBATCH --gres=gpu:1	
#SBATCH --output=result.out       
#SBATCH --error=result.out       

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate dino38

. ./init_bash.sh

python -u eval_linear_deepfake.py --arch vit_small --patch_size 8
