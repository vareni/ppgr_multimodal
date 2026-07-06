#!/bin/bash
#SBATCH --job-name=check_iego
#SBATCH --output=./logs/check_iego.out
#SBATCH --error=./logs/check_iego.err
#SBATCH --time=00:40:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --nodes=1
#SBATCH --nodelist=iego

source ~/.bashrc
#source /opt/miniconda3/etc/profile.d/conda.sh
conda activate vlm_env

export PYTHONNOUSERSITE=1

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

python - <<'PY'
import torch
print("torch:", torch.__version__, torch.version.cuda)
print("cuda:", torch.cuda.is_available())
print("count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
    free, total = torch.cuda.mem_get_info(i)
    print("free/total GB:", free / 1024**3, total / 1024**3)
PY