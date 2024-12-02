#!/bin/bash
#SBATCH -A nklab           # Set Account name
#SBATCH --job-name=plot_results
#SBATCH -c 22
#SBATCH --mem-per-cpu=4gb
#SBATCH -t 2-00:00              # Runtime in D-HH:MM
#SBATCH --gres=gpu:1
#XX SBATCH --nodelist=ax01
#SBATCH --mail-user=eh2976@columbia.edu
#SBATCH --mail-type=ALL
#SBATCH --output=plot_results_%j.out

cd /engram/nklab/algonauts/ethan/transformer_brain_encoder/
source /home/eh2976/.bashrc

conda activate xformers
ml load gcc/10.4 # needed for torch compile to work properly

# python eval_model.py --subj $@

conda activate pycortex

python plot_run_results.py $@