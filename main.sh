#!/bin/bash
#SBATCH -A nklab           # Set Account name
#SBATCH --job-name=train_tbe
#SBATCH -c 22
#SBATCH --mem-per-cpu=4gb
#XX SBATCH -c 20
#XX SBATCH --mem-per-cpu=7gb
#SBATCH -t 2-00:00              # Runtime in D-HH:MM
#SBATCH --gres=gpu:1
#XX SBATCH --nodelist=ax01
#SBATCH --mail-user=eh2976@columbia.edu
#SBATCH --mail-type=ALL
#SBATCH --output=train_tbe_%j.out

cd /engram/nklab/algonauts/ethan/transformer_brain_encoder/
source /home/eh2976/.bashrc

conda activate xformers

python main.py --lr 0.005 --hemi lh --axis posterior --subj 1