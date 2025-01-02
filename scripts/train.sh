#!/bin/bash
#SBATCH -A nklab           # Set Account name
#SBATCH --job-name=train_tbe
#SBATCH -c 16
#SBATCH --mem=125gb
#SBATCH -t 2-00:00              # Runtime in D-HH:MM
#SBATCH --gres=gpu:1
#XX SBATCH --nodelist=ax[17,26]
#SBATCH --exclude=ax[01-07,09-10,19]
#SBATCH --mail-user=eh2976@columbia.edu
#SBATCH --mail-type=ALL
#SBATCH --output=train_tbe_%j.out

cd /engram/nklab/algonauts/ethan/whole_brain_encoder/
source /home/eh2976/.bashrc

conda activate xformers
ml load gcc/10.4 # needed for torch compile to work properly

python main.py --subj 2 --enc_output_layer 1 --hemi rh --lr 0.0001 --run 3948277