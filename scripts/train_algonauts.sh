#!/bin/bash
#SBATCH -A nklab           # Set Account name
#SBATCH --job-name=algonauts
#SBATCH -c 16
#SBATCH --mem=64gb
#SBATCH -t 2-00:00              # Runtime in D-HH:MM
#SBATCH --gres=gpu:1
#XX SBATCH --nodelist=ax26
#SBATCH --mail-user=eh2976@columbia.edu
#SBATCH --mail-type=ALL
#SBATCH --output=algonauts_%j.out

cd /engram/nklab/algonauts/ethan/transformer_brain_encoder/
source /home/eh2976/.bashrc

conda activate xformers
ml load gcc/10.4 # needed for torch compile to work properly

python main_algonauts.py --run 4 --hemi lh --output_path /engram/nklab/algonauts/ethan/transformer_brain_encoder/checkpoints_algonauts --epochs 20 --parcel_dir /engram/nklab/algonauts/ethan/parcelling/results/algonauts_streams