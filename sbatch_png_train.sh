#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=students-prod
#SBATCH --gres=gpu:1
#SBATCH --output=png_11_luglio.out
#SBATCH --error=png_11_luglio.out

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate dino38

. ./init_bash.sh

python -u -W ignore eval_linear_deepfake_png.py --arch vit_small --patch_size 8 --output_dir /homes/tpoppi/probe-DeepFake-DINO/checkpoints/png_11_luglio_ridotto
