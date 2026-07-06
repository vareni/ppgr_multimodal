#!/bin/bash
#SBATCH --job-name=eval_mlp_micro
#SBATCH --output=./logs/%A_eval_mlp_micro.out
#SBATCH --error=./logs/%A_eval_mlp_micro.err
#SBATCH --time=00:40:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --nodelist=kessel

source ~/.bashrc
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate thesis_env

#export CUDA_LAUNCH_BLOCKING=1

/home/guests/varvara_kondratyeva/.conda/envs/thesis_env/bin/python -u joint_model_eval.py --rgbd --glucose --microbiome \
--cgm_model 'CGMHead'  # --microbiome



